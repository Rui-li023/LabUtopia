import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf

from experiments.pipette.extract_pipette_init_state import MARKER, extract_states
from experiments.pipette.run_pipette_fixture import (
    make_fixture_reset,
    read_arm_drive_properties,
    read_arm_dynamics,
    screen_solver_factory,
    smooth_joint_path,
    with_contact_guard,
)


class InitialFixtureTests(unittest.TestCase):
    def test_contact_guard_is_terminal_before_commands_in_every_phase(self):
        for phase in ("picking", "placing", "retreating", "finished", "next_object_handoff"):
            original = Mock()
            contact = {"paths": ["/robot/finger", "/previous_tool/body"], "impulse_norm": 0.01}
            active = SimpleNamespace(
                current_phase=phase,
                data_collector=SimpleNamespace(clear_cache=Mock()),
                _last_success=True,
                _last_failure_reason="",
            )
            state = {"joint_positions": np.zeros(9)}
            result = with_contact_guard(original, lambda contact=contact: contact, "finished")(active, state)
            self.assertEqual(result, (None, True, False))
            self.assertIs(active.state, state)
            self.assertEqual(active.current_phase, "finished")
            self.assertFalse(active._last_success)
            self.assertIsInstance(active._last_failure_reason, str)
            self.assertIn("/previous_tool/body", active._last_failure_reason)
            active.data_collector.clear_cache.assert_called_once()
            original.assert_not_called()

    def test_no_contact_preserves_the_exact_original_action_and_bookkeeping(self):
        original_result = (object(), False, False)
        original = Mock(return_value=original_result)
        active, state = SimpleNamespace(), {"joint_positions": np.zeros(9)}
        result = with_contact_guard(original, lambda: None, "finished")(active, state)
        self.assertIs(result, original_result)
        original.assert_called_once_with(active, state)
        self.assertEqual(vars(active), {})

    def test_arm_observer_copies_actual_signals_without_actuation(self):
        values = np.arange(9, dtype=float)
        props = np.zeros(9, dtype=[("maxEffort", float), ("driveMode", int)])
        props["maxEffort"] = 35
        robot = SimpleNamespace(
            dof_properties=props,
            dof_names=[f"joint{i}" for i in range(9)],
            _articulation_view=SimpleNamespace(get_friction_coefficients=Mock(return_value=values[None, :])),
            get_enabled_self_collisions=Mock(return_value=1),
            get_joint_velocities=Mock(return_value=values),
            get_measured_joint_efforts=Mock(return_value=values),
            apply_action=Mock(),
        )
        applied = SimpleNamespace(joint_positions=values, joint_velocities=values, joint_efforts=values)
        properties = read_arm_drive_properties(robot)
        dynamics = read_arm_dynamics(robot, applied)
        values[:] = -99
        props["maxEffort"] = 999
        for key in dynamics:
            np.testing.assert_array_equal(dynamics[key], np.arange(9))
        np.testing.assert_array_equal(properties["dof_properties"]["maxEffort"], np.full(9, 35))
        np.testing.assert_array_equal(properties["joint_friction"], np.arange(9)[None, :])
        self.assertTrue(properties["enabled_self_collisions"])
        robot.apply_action.assert_not_called()

    def test_arm_observer_does_not_invent_missing_effort_or_command(self):
        robot = SimpleNamespace(
            get_joint_velocities=Mock(return_value=None), get_measured_joint_efforts=Mock(return_value=None)
        )
        self.assertTrue(all(value is None for value in read_arm_dynamics(robot, None).values()))

    def test_probe_path_preserves_waypoints_and_stays_inside_each_segment(self):
        path = np.array([[0.0] * 7, [0.5] * 7, [-0.2] * 7])
        samples = smooth_joint_path(path)
        np.testing.assert_array_equal(samples[0], path[0])
        np.testing.assert_allclose(samples[200], path[1], atol=1e-14)
        np.testing.assert_allclose(samples[-1], path[-1], atol=1e-14)
        self.assertTrue(np.all(np.diff(samples[:201], axis=0) >= 0))
        self.assertTrue(np.all(np.diff(samples[200:], axis=0) <= 0))
        self.assertLess(np.max(np.abs(np.diff(samples, axis=0))), 0.005)

    def test_probe_path_rejects_invalid_waypoints_and_rate(self):
        for path in ([[0] * 7], [[0] * 6] * 2, [[float("nan")] * 7] * 2):
            with self.assertRaises(ValueError):
                smooth_joint_path(path)
        for rate in (0, -1, 199, 801, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                smooth_joint_path([[0] * 7, [1] * 7], rate)

    def test_candidate_rate_preserves_path_and_bounds_nominal_speed(self):
        path = np.array([[0.0] * 7, [0.5, -1.0, 0.2, 0.0, 0.7, -0.3, 0.1], [0.0] * 7])
        slow = smooth_joint_path(path, 400)
        candidate = smooth_joint_path(path, 200)
        self.assertLess(len(candidate), len(slow))
        np.testing.assert_array_equal(candidate[0], path[0])
        np.testing.assert_allclose(candidate[200], path[1], atol=1e-14)
        np.testing.assert_allclose(candidate[-1], path[-1], atol=1e-14)
        # At this fixture's 60 Hz physics clock, quintic peak derivative is
        # 1.875: this bounds commands, NOT actual tracking or contact stability.
        self.assertLessEqual(np.max(np.abs(np.diff(candidate, axis=0))) * 60, 1.875 * 60 / 200)
        self.assertTrue(np.all(candidate >= np.minimum(path[0], path[1]) - 1e-14))
        self.assertTrue(np.all(candidate <= np.maximum(path[0], path[1]) + 1e-14))

    def test_clearance_filters_each_fresh_solver_without_changing_joint_solution(self):
        solution = [0.0] * 7
        factory = Mock(
            side_effect=lambda: SimpleNamespace(compute_inverse_kinematics=Mock(return_value=(solution, True)))
        )
        checker = Mock(side_effect=[False, True])
        stats = {"queries": 0, "rejected": 0}
        wrapped = screen_solver_factory(factory, checker, stats)
        first, second = wrapped(), wrapped()
        self.assertIsNot(first, second)
        returned, ok = first.compute_inverse_kinematics("hand", warm_start=solution)
        self.assertIs(returned, solution)
        self.assertFalse(ok)
        self.assertTrue(second.compute_inverse_kinematics("hand")[1])
        self.assertEqual(stats, {"queries": 2, "rejected": 1})
        self.assertEqual(factory.call_count, 2)

    def test_restore_is_once_only_collect_only_and_copies_input(self):
        fixture = {"init_state": {"object_poses": {"/tool": {"position": [1, 2, 3]}}}}
        cfg = OmegaConf.create(
            {
                "mode": "collect",
                "task": {
                    "obj_paths": [{"path": "/tool"}],
                    "replay_restore_orientation": True,
                },
            }
        )
        task = SimpleNamespace(cfg=cfg, reset_with_init_state=Mock())
        reset = make_fixture_reset(fixture)
        reset(task)
        saved = task.reset_with_init_state.call_args.args[0]
        self.assertEqual(saved, fixture["init_state"])
        saved["object_poses"]["/tool"]["position"][0] = 99
        self.assertEqual(fixture["init_state"]["object_poses"]["/tool"]["position"][0], 1)
        with self.assertRaises(ValueError):
            reset(task)
        task._fixture_restored = False
        task.cfg.mode = "replay"
        with self.assertRaises(ValueError):
            reset(task)
        self.assertEqual(task.reset_with_init_state.call_count, 1)

    def test_mismatched_scene_rejected_before_writes(self):
        task = SimpleNamespace(
            cfg=OmegaConf.create({"mode": "collect", "task": {"obj_paths": [{"path": "/different"}]}}),
            reset_with_init_state=Mock(),
        )
        with self.assertRaises(ValueError):
            make_fixture_reset({"init_state": {"object_poses": {"/tool": {}}}})(task)
        task.reset_with_init_state.assert_not_called()

    def test_multiline_numpy_arrays_and_braces_inside_strings(self):
        text = MARKER + (
            "{'object_poses': {'/tool': {'position': [1, 2, 3]}}, 'extra': {'label': '} { text'},\n"
            "'robot_init_joint_positions': array([1.0,\n 2.0], dtype=float32)}\nnext log\n"
        )
        states = extract_states(text)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["robot_init_joint_positions"], [1.0, 2.0])
        self.assertEqual(states[0]["extra"]["label"], "} { text")

    def test_repeated_scene_numbers_do_not_drop_failed_attempts(self):
        text = "\n".join(MARKER + "{'object_poses': {}, 'index': " + str(i) + "}" for i in range(3))
        self.assertEqual([state["index"] for state in extract_states(text)], [0, 1, 2])

    def test_executable_expressions_and_unexpected_array_arguments_rejected(self):
        for expression in ("__import__('os').getcwd()", "array([1], bad=True)", "array([1], dtype=object)"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                extract_states(MARKER + "{'object_poses': {}, 'unsafe': " + expression + "}")
