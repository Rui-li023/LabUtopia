"""Exercise the actual composite terminal methods without launching Isaac Sim."""

import ast
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.conical_grasp import ConicalNeckGrasp

source = ast.parse((Path(__file__).resolve().parents[1] / "controllers/open_transport_pour_controller.py").read_text())
phase = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "TaskPhase")
controller = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_OpenTransportPourController")
methods = [node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name in ("_step_collect", "_step_infer", "_check_success", "_place_liquid_conical", "_get_phase_action")]
scope = {"Enum": Enum, "Any": Any, "Dict": dict, "Tuple": tuple, "np": np, "R": R, "liquid_enabled": lambda cfg: True}
exec(compile(ast.fix_missing_locations(ast.Module(body=[phase, *methods], type_ignores=[])), "actual_otp_methods", "exec"), scope)


class ConicalTerminalTests(unittest.TestCase):
    def test_pick_uses_configured_neck_depth_and_blocked_jaw_target(self):
        forward = Mock(return_value=("action", "record"))
        owner = SimpleNamespace(
            cfg=SimpleNamespace(), current_phase=scope["TaskPhase"].PICKING2,
            every_controller_index=1, pick_controller2=Mock(), gripper_control=Mock(),
            object_utils=SimpleNamespace(get_object_size=lambda object_path: [0.093, 0.093, 0.164524]),
            _conical_neck_grasp=ConicalNeckGrasp(depth=0.023, closed_finger_position=0.012),
        )
        state = {"liquid_transfers": {"pour": {"source_mouth": [0.1, -0.4, 0.9]}}}
        with patch.dict(scope, forward_liquid_pick=forward):
            self.assertEqual(scope["_get_phase_action"](owner, state), ("action", "record"))
        self.assertEqual(forward.call_args.kwargs["grasp_depth"], 0.023)
        self.assertEqual(forward.call_args.kwargs["grip_distance"], 0.012)
        self.assertNotIn("liquid", state)
        self.assertEqual(forward.call_args.args[1]["object_name"], "conical_bottle02")

    def test_transport_waypoints_are_bounded_and_slip_stops_motion(self):
        hand = np.array([0.2, -0.1, 0.98])
        motion = SimpleNamespace(
            articulation_rmp=SimpleNamespace(get_active_joints_subset=lambda: SimpleNamespace(get_joint_positions=lambda: np.zeros(7))),
            rmp_flow=SimpleNamespace(get_end_effector_pose=lambda joints: (hand, np.eye(3))),
        )
        place = SimpleNamespace(
            _cspace_controller=motion, _event=0, _events_dt=[0.01] * 6,
            forward=Mock(return_value=(None, None)), _null_action=lambda n: None,
            _build_record_array=lambda action, joints: np.zeros(8),
        )
        owner = SimpleNamespace(
            place_controller2=place, gripper_control=Mock(), _conical_place_orientation=None,
            _conical_release_target=None, _conical_transport_waypoint=None, _conical_place_grasp_distance=None,
            _conical_transport_delta=np.zeros(3),
        )
        state = {"joint_positions": np.zeros(9),
                 "liquid_transfers": {"pour": {"source_mouth": [0.2, -0.1, 1.0], "source_up": [0, 0, 1]}},
                 "placement_supports": {"/World/target_plat2": [0.1, -0.5, 0.7198]}}
        scope["_place_liquid_conical"](owner, state)
        target = place.forward.call_args.kwargs["place_position"] + [0, 0, 0.1]
        self.assertLessEqual(np.linalg.norm(target - hand), 0.000010001)
        self.assertFalse(place.approach_arrival_allowed)
        previous_step = owner._conical_transport_delta.copy()
        for _ in range(60):
            scope["_place_liquid_conical"](owner, state)
            step = owner._conical_transport_delta.copy()
            self.assertLessEqual(np.linalg.norm(step), 0.000500001)
            self.assertLessEqual(np.linalg.norm(step - previous_step), 0.000010001)
            previous_step = step
        place.forward.reset_mock()
        state["liquid_transfers"]["pour"]["source_mouth"][2] -= 0.12
        scope["_place_liquid_conical"](owner, state)
        self.assertEqual(place._event, len(place._events_dt))
        self.assertEqual(place._phase_failure_reason, "Bottle slipped during final transport")
        place.forward.assert_not_called()

    def test_terminal_requires_released_bottles_on_supports(self):
        centers = {
            "/World/beaker2": np.array([0.0, 0, 0.830]),
            "/World/target_plat": np.array([0.0, 0, 0.775]),
            "/World/conical_bottle02": np.array([0.1, 0, 0.862]),
            "/World/target_plat2": np.array([0.1, 0, 0.775]),
            "/World/MuffleFurnace/handle": np.array([0.75, 0.1, 0.85]),
        }
        sizes = {path: np.array([0.1, 0.1, height]) for path, height in (
            ("/World/beaker2", 0.1), ("/World/target_plat", 0.01),
            ("/World/conical_bottle02", 0.164), ("/World/target_plat2", 0.01),
        )}
        owner = SimpleNamespace(
            state={"joint_positions": np.array([0] * 7 + [0.04, 0.04]),
                   "liquid_transfers": {"pour": {"source_up": [0.0, 0.0, 1.0]}},
                   "placement_supports": {"/World/target_plat": [0, 0, 0.780], "/World/target_plat2": [0.1, 0, 0.780]}}, cfg=SimpleNamespace(), mode="collect",
            object_utils=SimpleNamespace(get_geometry_center=lambda object_path: centers[object_path],
                                         get_object_size=lambda object_path: sizes[object_path]),
            _handle_init=np.array([0.7, 0.1, 0.85]), _z_init_beaker2=0.8, _zmax_beaker2=1.0,
            _z_init_conical=0.8, _zmax_conical=1.0, _tilt_peak_conical=120,
        )
        self.assertTrue(scope["_check_success"](owner))
        owner.state["liquid_transfers"]["pour"]["source_up"] = [1.0, 0.0, 0.0]
        self.assertFalse(scope["_check_success"](owner))
        owner.state["liquid_transfers"]["pour"]["source_up"] = [0.0, 0.0, 1.0]
        owner.state["joint_positions"][-2:] = 0.02
        self.assertFalse(scope["_check_success"](owner))
        owner.state["joint_positions"][-2:] = 0.04
        centers["/World/conical_bottle02"][2] += 0.02
        self.assertFalse(scope["_check_success"](owner))

    def test_constructor_preserves_randomized_first_phase(self):
        class Parent:
            def __init__(self, cfg, robot):
                self.current_phase = scope["TaskPhase"].PICKING2
                self.randomized_sequence = [self.current_phase, scope["TaskPhase"].POURING]

        init = next(node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        node = ast.ClassDef(name="ActualInit", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[init], decorator_list=[])
        local = dict(scope, Parent=Parent)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "actual_otp_init", "exec"), local)
        owner = local["ActualInit"](None, None)
        self.assertEqual(owner.current_phase, scope["TaskPhase"].PICKING2)
        self.assertIsNone(owner._conical_place_orientation)
        self.assertIsNone(owner._conical_release_target)

    def owner(self, success):
        return SimpleNamespace(
            current_phase=scope["TaskPhase"].TRANSPORTING2,
            active_controller=SimpleNamespace(is_done=lambda: True),
            record_liquid_handoff=Mock(), _get_next_phase=lambda: None,
            _check_success=lambda: success, liquid_success=lambda: success,
            cfg=SimpleNamespace(), data_collector=Mock(), _last_success=False,
            _last_failure_reason="", success_steps=set(scope["TaskPhase"]) - {scope["TaskPhase"].FINISHED},
            N_PHASES=6, _check_phase_success=lambda state: True,
        )

    def test_completed_motion_without_liquid_never_writes_successful_episode(self):
        owner = self.owner(False)
        self.assertEqual(scope["_step_collect"](owner, {"joint_positions": [0] * 9}), (None, True, False))
        owner.data_collector.write_cached_data.assert_not_called()
        self.assertEqual(owner._last_failure_reason, "liquid_transfer_failed")
        owner.record_liquid_handoff.assert_called_once()

    def test_valid_transfer_and_placement_can_be_saved(self):
        owner = self.owner(True)
        self.assertEqual(scope["_step_collect"](owner, {"joint_positions": [0] * 9}), (None, True, True))
        owner.data_collector.write_cached_data.assert_called_once()

    def test_inference_phase_completion_does_not_bypass_liquid_gate(self):
        owner = self.owner(False)
        self.assertEqual(scope["_step_infer"](owner, {}), (None, True, False))


if __name__ == "__main__":
    unittest.main()
