"""Exercise production replay/trajectory methods without starting Isaac Sim."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

import numpy as np


def load_methods(path: str, class_name: str, names: set[str]):
    source = Path(__file__).resolve().parents[1] / path
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "np": np,
        "Any": Any,
        "Optional": Optional,
        "ArticulationAction": SimpleNamespace,
        "logger": Mock(),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    return {name: namespace[name] for name in names}


Trajectory = type(
    "Trajectory",
    (),
    load_methods(
        "controllers/robot_controllers/trajectory_controller.py",
        "FrankaTrajectoryController",
        {
            "_map_gripper_state_to_positions",
            "_gripper_action_extras",
            "generate_trajectory",
            "get_next_action",
            "is_trajectory_complete",
        },
    ),
)
Controller = type(
    "Controller",
    (),
    load_methods("controllers/base_controller.py", "BaseController", {"_step_replay"}),
)


class ReplayActionDeliveryTests(unittest.TestCase):
    def controller(self, actions):
        trajectory = Trajectory()
        trajectory._action_sequence = []
        trajectory._action_sequence_index = 0
        trajectory._use_interpolation = False
        trajectory._gripper_open_joint_pos = np.array([0.05, 0.05])
        trajectory._gripper_closed_joint_pos = np.zeros(2)
        trajectory._GRIP_MARGIN = 0.0
        controller = Controller()
        controller._current_actions = np.asarray(actions, dtype=float).reshape(-1, 8)
        controller._current_action_step = 0
        controller.trajectory_controller = trajectory
        controller.state = {}
        controller._replay_handle_attach = Mock()
        controller._check_success = Mock(return_value=False)
        controller._advance_replay_episode = Mock()
        controller.check_success_counter = 0
        controller.REQUIRED_SUCCESS_STEPS = 2
        controller.replay_run_to_end = False
        controller._replay_reached_success = False
        controller._replay_settle_budget = 2
        controller._replay_settle_used = 0
        controller.reset_needed = False
        return controller

    def test_every_waypoint_including_changed_final_gripper_is_delivered(self):
        actions = np.array([[0.1] * 7 + [0.0], [0.2] * 7 + [0.72], [0.3] * 7 + [0.0]])
        controller = self.controller(actions)
        for waypoint in actions:
            action, done, success = controller._step_replay({})
            self.assertFalse(done)
            self.assertFalse(success)
            np.testing.assert_array_equal(action.joint_positions[:7], waypoint[:7])
            np.testing.assert_allclose(action.joint_positions[7:], 0.05 * (1.0 - waypoint[7]))
            self.assertEqual(controller._replay_settle_used, 0)
        self.assertIsNone(controller._step_replay({})[0])
        self.assertEqual(controller._replay_settle_used, 1)

    def test_single_frame_is_applied_before_settling(self):
        controller = self.controller([[0.0] * 7 + [0.72]])
        action, done, _ = controller._step_replay({})
        self.assertIsNotNone(action)
        self.assertFalse(done)
        np.testing.assert_allclose(action.joint_positions[7:], [0.014, 0.014])
        self.assertEqual(controller._replay_settle_used, 0)

    def test_full_recording_mode_applies_last_action_even_after_success(self):
        controller = self.controller([[0.1] * 7 + [0.72], [0.2] * 7 + [0.0]])
        controller.replay_run_to_end = True
        controller.REQUIRED_SUCCESS_STEPS = 1
        controller._check_success.return_value = True
        for _ in range(2):
            action, done, _ = controller._step_replay({})
            self.assertIsNotNone(action)
            self.assertFalse(done)
        controller._advance_replay_episode.assert_not_called()
        self.assertEqual(controller._step_replay({}), (None, True, True))
        controller._advance_replay_episode.assert_called_once()

    def test_normal_mode_can_stop_at_sustained_success(self):
        controller = self.controller([[0.1] * 7 + [0.0]] * 4)
        controller._check_success.return_value = True
        self.assertFalse(controller._step_replay({})[1])
        self.assertEqual(controller._step_replay({}), (None, True, True))

    def test_exhausted_failure_waits_only_the_bounded_settle_budget(self):
        controller = self.controller([[0.1] * 7 + [0.0]])
        controller._step_replay({})
        for _ in range(2):
            self.assertEqual(controller._step_replay({}), (None, False, False))
        self.assertEqual(controller._step_replay({}), (None, True, False))
        controller._advance_replay_episode.assert_called_once()

    def test_empty_recording_produces_no_action(self):
        controller = self.controller([])
        self.assertEqual(controller._step_replay({}), (None, False, False))


if __name__ == "__main__":
    unittest.main()
