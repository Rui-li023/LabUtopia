import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_pick_controller():
    class ArticulationAction:
        def __init__(self, joint_positions=None):
            self.joint_positions = joint_positions

    class BaseRobot:
        pass

    package_name = "_pick_adaptive_test"
    stub_modules = {
        "isaacsim": _package("isaacsim"),
        "isaacsim.core": _package("isaacsim.core"),
        "isaacsim.core.utils": _package("isaacsim.core.utils"),
        "isaacsim.core.utils.stage": types.SimpleNamespace(get_stage_units=lambda: 1.0),
        "isaacsim.core.utils.types": types.SimpleNamespace(ArticulationAction=ArticulationAction),
        "robots": _package("robots"),
        "robots.base_robot": types.SimpleNamespace(
            BaseRobot=BaseRobot,
            GRIPPER_CLOSED=0,
            GRIPPER_OPEN=1,
        ),
        package_name: _package(package_name),
    }

    root = Path(__file__).resolve().parents[1] / "controllers" / "atomic_actions"
    with patch.dict(sys.modules, stub_modules):
        base_name = f"{package_name}.atomic_base_controller"
        base_spec = importlib.util.spec_from_file_location(base_name, root / "atomic_base_controller.py")
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules[base_name] = base_module
        base_spec.loader.exec_module(base_module)

        pick_name = f"{package_name}.pick_controller"
        pick_spec = importlib.util.spec_from_file_location(pick_name, root / "pick_controller.py")
        pick_module = importlib.util.module_from_spec(pick_spec)
        sys.modules[pick_name] = pick_module
        pick_spec.loader.exec_module(pick_module)

    return pick_module.PickController


PickController = _load_pick_controller()


class AdaptivePickPhaseTest(unittest.TestCase):
    def test_orientation_gate_requires_consecutive_stable_arrivals(self):
        controller = self.make_controller()
        controller.adaptive_phase_completion = True
        controller.arrival_stable_steps = 2
        controller.arrival_orientation_ok = False
        for _ in range(3):
            controller._advance_if_arrived(True, np.zeros(3), np.zeros(3))
        self.assertEqual(controller._event, 0)
        controller.arrival_orientation_ok = True
        controller._advance_if_arrived(True, np.zeros(3), np.zeros(3))
        self.assertEqual(controller._event, 0)
        controller.arrival_orientation_ok = False
        controller._advance_if_arrived(True, np.zeros(3), np.zeros(3))
        self.assertEqual(controller._arrival_steps, 0)
        controller.arrival_orientation_ok = True
        for _ in range(2):
            controller._advance_if_arrived(True, np.zeros(3), np.zeros(3))
        self.assertEqual(controller._event, 1)

    def contact_controller(self, gap):
        controller = self.make_controller()
        controller._event = 4
        controller.close_contact_gap_range = (0.016, 0.036)
        controller.close_stable_steps = 3
        controller.close_timeout_steps = 8
        controller._close_steps = 0
        controller._close_stable = 0
        controller._previous_close_gap = None
        controller._robot = types.SimpleNamespace(
            get_joint_positions=lambda: np.array([gap / 2, gap / 2]),
            gripper=types.SimpleNamespace(joint_dof_indicies=np.array([0.0, 1.0])),
        )
        return controller

    def test_contact_close_waits_for_stable_gap(self):
        controller = self.contact_controller(0.030)
        for _ in range(3):
            controller._advance_state()
            self.assertEqual(controller._event, 4)
        controller._advance_state()
        self.assertEqual(controller._event, 5)

    def test_slow_close_can_require_full_command_before_lifting(self):
        controller = self.contact_controller(0.030)
        controller.close_command_step_m = 0.0001
        controller.close_ramp_requires_target = True
        for _ in range(5):
            controller._advance_state()
            self.assertEqual(controller._event, 4)
        controller._close_command_finished = True
        for _ in range(3):
            controller._advance_state()
        self.assertEqual(controller._event, 5)

    def test_empty_or_unclosed_gripper_cannot_lift(self):
        for gap in (0.0, 0.060):
            controller = self.contact_controller(gap)
            for _ in range(8):
                controller._advance_state()
            self.assertEqual(controller._event, 4)
            self.assertIn("stable contact", controller.get_failure_reason())

    def test_contact_gate_can_wait_for_object_without_erasing_jaw_contact(self):
        controller = self.contact_controller(0.030)
        controller.close_contact_stable = False
        for _ in range(5):
            controller._advance_state()
        self.assertEqual(controller._event, 4)
        self.assertEqual(controller._close_stable, 0)
        self.assertEqual(controller._jaw_contact_steps, 4)
        controller.close_contact_stable = True
        for _ in range(3):
            controller._advance_state()
        self.assertEqual(controller._event, 5)

    def test_close_hold_uses_fixed_approach_target_not_moving_object(self):
        controller = self.make_controller()
        controller._event = 4
        controller._robot_position = np.zeros(3)
        controller._robot = MagicMock()
        controller._cspace_controller = MagicMock()
        controller.hold_pose_during_close = True
        controller._last_target = np.array([0.2, 0.3, 0.76])
        controller._execute_phase(
            np.array([0.25, 0.4, 0.75]),
            np.array([0, 1, 0, 0]),
            np.zeros(9),
            "pipette",
            MagicMock(),
            np.array([0.2, 0.3, 0.76]),
            0.02,
            0.30,
            0.0,
            0.0,
        )
        np.testing.assert_array_equal(
            controller._cspace_controller.forward.call_args.kwargs["target_end_effector_position"],
            controller._last_target,
        )
        controller._robot.close_gripper.assert_called_once()

    def test_optional_close_ramp_records_bounded_distance_commands(self):
        controller = self.make_controller()
        controller._event = 4
        controller._robot_position = np.zeros(3)
        controller._robot = MagicMock()
        controller._robot.gripper.joint_dof_indicies = [7, 8]
        controller.close_command_step_m = 0.0004
        controller._close_command_target = None
        controller.lift_offset_xyz = None
        controller.lift_along_tool = False
        joints = np.array([0] * 7 + [0.04, 0.04])
        commands = []
        for _ in range(60):
            controller._execute_phase(np.zeros(3), np.array([0, 1, 0, 0]), joints, "conical_bottle02",
                                      MagicMock(), np.zeros(3), 0.0, 0.2, 0.0, 0.019)
            commands.append(controller._robot.close_gripper_to_distance.call_args.args[0])
        self.assertAlmostEqual(commands[0], 0.0396)
        self.assertAlmostEqual(commands[-1], 0.019)
        self.assertLessEqual(float(np.max(np.abs(np.diff(commands)))), 0.000400001)
        self.assertAlmostEqual(controller._last_gripper_target_m, commands[-1])

    def make_controller(self):
        controller = object.__new__(PickController)
        controller.adaptive_phase_completion = True
        controller.arrival_stable_steps = 2
        controller._arrival_steps = 0
        controller._transitioned_this_step = False
        controller._phase_failure_reason = ""
        controller._last_target = None
        controller._last_gripper_position = None
        controller._last_position_error = None
        controller._event = 0
        controller._t = 0.0
        controller._events_dt = [0.5] * 7
        return controller

    def test_requires_consecutive_stable_arrivals(self) -> None:
        controller = self.make_controller()

        controller._advance_if_arrived(True, [0.001, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(controller._event, 0)
        self.assertEqual(controller._arrival_steps, 1)

        controller._transitioned_this_step = False
        controller._advance_if_arrived(False, [0.02, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(controller._arrival_steps, 0)

        controller._advance_if_arrived(True, [0.001, 0.0, 0.0], [0.0, 0.0, 0.0])
        controller._transitioned_this_step = False
        controller._advance_if_arrived(True, [0.001, 0.0, 0.0], [0.0, 0.0, 0.0])

        self.assertEqual(controller._event, 1)
        self.assertEqual(controller._arrival_steps, 0)
        self.assertTrue(controller._transitioned_this_step)

    def test_motion_phase_timeout_reports_context(self) -> None:
        controller = self.make_controller()

        controller._advance_if_arrived(False, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
        controller._advance_state()
        controller._transitioned_this_step = False
        controller._advance_if_arrived(False, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
        controller._advance_state()

        self.assertTrue(controller.is_done())
        self.assertIn("move_above", controller.get_failure_reason())
        self.assertIn("0.200 m", controller.get_failure_reason())

    def test_reset_clears_timeout_and_locked_target(self) -> None:
        controller = self.make_controller()
        controller._is_done = True
        controller._locked_picking_position = [1.0, 2.0, 3.0]
        controller._cspace_controller = None
        controller._default_n_phases = 7
        controller._randomization_sampled = True

        controller.reset()

        self.assertFalse(controller.is_done())
        self.assertEqual(controller._event, 0)
        self.assertEqual(controller.get_failure_reason(), "")
        self.assertIsNone(controller._locked_picking_position)


class OrientationHandoffTest(unittest.TestCase):
    """The quaternion ordering handed to the motion policy.

    Every task controller written before 2026-08-24 passes ``R.from_euler(...).as_quat()``
    (scipy's x, y, z, w) and its grasp euler was TUNED against that array reaching
    RMPFlow unpermuted -- the same thing the place / pour / press / stir atomic
    controllers still do. Permuting it to Isaac's (w, x, y, z) is the arithmetically
    correct handoff, but it silently rotates every one of those tuned grasps, so it
    has to stay opt-in for the configs that were tuned WITH it (the multi-arm sweep
    and the `_v2` grasp-frame configs).
    """

    QUAT_XYZW = [0.1, 0.2, 0.3, 0.9]

    def test_legacy_handoff_is_the_default_and_does_not_permute(self) -> None:
        controller = object.__new__(PickController)
        self.assertTrue(
            controller.legacy_orientation_handoff,
            "legacy handoff must be the default: it is what every pre-existing task was tuned against",
        )
        self.assertEqual(list(controller._to_isaac_quat(self.QUAT_XYZW)), self.QUAT_XYZW)

    def test_opting_out_permutes_xyzw_to_isaac_wxyz(self) -> None:
        controller = object.__new__(PickController)
        controller.legacy_orientation_handoff = False
        self.assertEqual(list(controller._to_isaac_quat(self.QUAT_XYZW)), [0.9, 0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
