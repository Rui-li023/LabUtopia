"""Exercise the actual Franka command-sync method without starting Isaac."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def load_method(name):
    source = Path(__file__).resolve().parents[1] / "robots/franka/franka.py"
    tree = ast.parse(source.read_text())
    robot = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Franka")
    method = next(node for node in robot.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np, "ArticulationAction": SimpleNamespace, "GRIPPER_OPEN": 1, "GRIPPER_CLOSED": 0}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


SYNC = load_method("sync_gripper_from_action")
ENABLE = load_method("enable_gripper_force_servo")
APPLY = load_method("apply_gripper_effort")
OPEN = load_method("open_gripper")
CLOSE = load_method("close_gripper")


class FrankaGripperSyncTests(unittest.TestCase):
    def test_position_endpoints_issue_exact_recorded_targets(self):
        robot = self.robot("position")
        robot._gripper_open_position = [0.05, 0.05]
        robot._gripper = Mock()
        for method, target, state in ((OPEN, 0.05, 1), (CLOSE, 0.0, 0)):
            for _ in range(3):
                method(robot)
                action = robot._gripper.apply_action.call_args.args[0]
                np.testing.assert_array_equal(action.joint_positions, [target, target])
                self.assertEqual(robot._gripper_cmd_opening, target)
                self.assertEqual(robot._gripper_state, state)
            robot._gripper.open.assert_not_called()
            robot._gripper.close.assert_not_called()

    def test_position_endpoint_does_not_alias_configured_array(self):
        robot = self.robot("position")
        robot._gripper = Mock()
        robot._gripper_open_position = np.array([0.05, 0.05])
        OPEN(robot)
        robot._gripper.apply_action.call_args.args[0].joint_positions[0] = 0.0
        np.testing.assert_array_equal(robot._gripper_open_position, [0.05, 0.05])

    def test_force_and_velocity_endpoints_keep_persistent_actuation(self):
        for mode in ("force", "velocity"):
            robot = self.robot(mode)
            robot._gripper = Mock()
            robot._gripper_closing_force = 35.0
            robot._gripper_closing_speed = 0.02
            for method, sign in ((OPEN, 1), (CLOSE, -1)):
                method(robot)
                if mode == "force":
                    self.assertEqual(robot._current_gripper_effort, sign * 35.0)
                else:
                    self.assertEqual(robot._current_gripper_velocity, sign * 0.02)
            robot._gripper.apply_action.assert_not_called()

    def test_separate_opening_speed_preserves_slow_closure(self):
        robot = self.robot()
        robot.get_joint_positions = lambda: np.zeros(9)
        robot.apply_action = Mock()
        ENABLE(robot, speed=0.02, opening_speed=0.05, opening_width=0.04)
        robot._current_gripper_effort = 35
        APPLY(robot)
        np.testing.assert_allclose(robot.apply_action.call_args.args[0].joint_velocities, [0.05, 0.05])
        robot._current_gripper_effort = -35
        APPLY(robot)
        np.testing.assert_allclose(robot.apply_action.call_args.args[0].joint_velocities, [-0.02, -0.02])
        ENABLE(robot, speed=0.02, opening_width=0.04)
        self.assertEqual(robot._gripper_force_servo_opening_speed, 0.02)

    def robot(self, mode="force"):
        return SimpleNamespace(
            _gripper_control_mode=mode,
            _gripper_state=0,
            _current_gripper_effort=0.0,
            _gripper_open_position=[0.04, 0.04],
            _gripper_closed_position=[0.0, 0.0],
            gripper=SimpleNamespace(joint_dof_indicies=[7, 8]),
            open_gripper=Mock(),
            close_gripper=Mock(),
        )

    def test_repeated_open_initializes_persistent_drive(self):
        robot = self.robot()
        robot.open_gripper.side_effect = lambda: setattr(robot, "_current_gripper_effort", 35.0)
        for _ in range(2):
            SYNC(robot, SimpleNamespace(joint_positions=[0.0] * 7 + [0.04, 0.04]))
        self.assertEqual(robot._current_gripper_effort, 35.0)
        self.assertEqual(robot.open_gripper.call_count, 2)
        robot.close_gripper.assert_not_called()

    def test_repeated_close_reasserts_force_or_velocity(self):
        for mode in ("force", "velocity"):
            robot = self.robot(mode)
            robot._gripper_state = 1
            for _ in range(2):
                SYNC(robot, SimpleNamespace(joint_positions=[0.0] * 9))
            self.assertEqual(robot.close_gripper.call_count, 2)
            robot.open_gripper.assert_not_called()

    def test_uncommanded_fingers_and_position_mode_are_untouched(self):
        for action in (
            None,
            SimpleNamespace(joint_positions=None),
            SimpleNamespace(joint_positions=[0.0] * 7),
            SimpleNamespace(joint_positions=[0.0] * 7 + [None, None]),
        ):
            robot = self.robot()
            SYNC(robot, action)
            robot.open_gripper.assert_not_called()
            robot.close_gripper.assert_not_called()
        robot = self.robot("position")
        SYNC(robot, SimpleNamespace(joint_positions=[0.0] * 9))
        robot.close_gripper.assert_not_called()
