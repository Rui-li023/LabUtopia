"""Test measured clearance / absolute wrist-roll logic without a GPU runtime."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

source = ast.parse((Path(__file__).resolve().parents[1] / "controllers/atomic_actions/liquid_pour_controller.py").read_text())
controller = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "LiquidPourController")
forward = next(node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
scope = {"np": np, "Rotation": Rotation, "logger": Mock()}
exec(compile(ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])), "actual_liquid_motion", "exec"), scope)


class LiquidPourMotionTests(unittest.TestCase):
    def setup_motion(self, return_lift=0.08):
        angle = np.deg2rad(120)
        rotation = Rotation.from_rotvec([0, angle, 0])
        self.radial = rotation.apply([1.0, 0, 0])
        self.state = {"joint_positions": np.zeros(9), "liquid": {
            "source_up": rotation.apply([0.0, 0, 1]), "source_basis": rotation.as_matrix().T,
            "source_mouth": np.array([0, 0, 0.9]) - self.radial * 0.02,
            "target_mouth": np.array([0.0, 0.0, 0.8]), "source_radius": 0.02,
            "source_fraction": 0.0, "target_up": [0, 0, 1],
        }}
        motion = SimpleNamespace(
            articulation_rmp=SimpleNamespace(get_active_joints_subset=lambda: SimpleNamespace(get_joint_positions=lambda: np.zeros(7))),
            rmp_flow=SimpleNamespace(get_end_effector_pose=lambda joints: (self.state["liquid"]["source_mouth"] + [0.03, 0, 0], rotation.as_matrix())),
            forward=Mock(return_value="action"),
        )
        self.owner = SimpleNamespace(
            _cspace_controller=motion, cfg=SimpleNamespace(liquid=SimpleNamespace(
                pour_lip_height=0.12, pour_upright_clearance=0.08, pour_return_lift=return_lift,
                axisymmetric_source=True, fixed_grasp_orientation=True)),
            angle=angle, _event=3, frames=0, stable=0, _is_done=False,
            _return_clearance_ready=False, _return_clearance_stable=0,
            _return_retreat_direction=None,
            _initial_grasp_distance=None, _phase_failure_reason="",
            direction=np.array([1.0, 0, 0]), radial_local=np.array([1.0, 0, 0]),
            waypoint=np.array([0.0, 0.0, 0.92]), _grasp_orientation=Rotation.identity(),
            _build_record_array=lambda action, joints: np.zeros(8), _null_action=lambda n: None,
        )

    def test_return_does_not_rotate_until_measured_clearance_is_stable(self):
        self.setup_motion()
        initial_angle = self.owner.angle
        for _ in range(40):
            scope["forward"](self.owner, self.state)
        self.assertEqual(self.owner.angle, initial_angle)
        self.assertFalse(self.owner._return_clearance_ready)
        self.state["liquid"]["source_mouth"] = np.array([0.0, 0.0, 1.0]) - self.radial * 0.02
        for _ in range(30):
            scope["forward"](self.owner, self.state)
        self.assertTrue(self.owner._return_clearance_ready)
        self.assertEqual(self.owner.angle, initial_angle)
        scope["forward"](self.owner, self.state)
        self.assertLess(self.owner.angle, initial_angle)

    def test_unreachable_return_clearance_fails_closed(self):
        self.setup_motion()
        self.owner.frames = 600
        scope["forward"](self.owner, self.state)
        self.assertTrue(self.owner._is_done)
        self.assertEqual(self.owner._phase_failure_reason, "Return-clearance lift did not converge")

    def test_recorded_pour_keeps_minimum_dwell_even_after_emptying(self):
        self.setup_motion()
        self.owner._event = 2
        self.owner.cfg.liquid.pour_min_drain_steps = 240
        self.owner.cfg.liquid.pour_drain_steps = 600
        self.owner.frames = 40
        scope["forward"](self.owner, self.state)
        self.assertEqual(self.owner._event, 2)
        self.owner.frames = 239
        scope["forward"](self.owner, self.state)
        self.assertEqual(self.owner._event, 3)

    def test_return_can_clear_receiver_towards_robot_before_rotation(self):
        self.setup_motion()
        self.owner.cfg.liquid.pour_return_retreat = 0.12
        self.owner._cspace_controller._default_position = np.array([-0.4, 0, 0.71])
        initial_angle = self.owner.angle
        scope["forward"](self.owner, self.state)
        np.testing.assert_allclose(self.owner._return_retreat_direction, [-1, 0, 0])
        self.assertLess(self.owner.waypoint[0], 0)
        self.assertEqual(self.owner.angle, initial_angle)

    def test_default_beaker_motion_does_not_gain_new_return_phase(self):
        self.setup_motion(return_lift=0.0)
        initial_angle = self.owner.angle
        scope["forward"](self.owner, self.state)
        self.assertLess(self.owner.angle, initial_angle)

    def test_absolute_wrist_reference_does_not_follow_source_axial_spin(self):
        self.setup_motion()
        scope["forward"](self.owner, self.state)
        expected = self.owner._cspace_controller.forward.call_args.kwargs["target_end_effector_orientation"].copy()
        self.state["liquid"]["source_basis"] = Rotation.from_euler("xyz", [20, 120, 90], degrees=True).as_matrix()
        scope["forward"](self.owner, self.state)
        actual = self.owner._cspace_controller.forward.call_args.kwargs["target_end_effector_orientation"]
        np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
