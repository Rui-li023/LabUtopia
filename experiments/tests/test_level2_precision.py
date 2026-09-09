"""Precision placement contracts, without starting Isaac Sim."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation


def load_controllers():
    root = Path(__file__).resolve().parents[2] / "controllers"
    package = "_level2_precision_test"
    modules = {}
    for name in (package, package + ".atomic_actions", "isaacsim", "isaacsim.core", "isaacsim.core.utils", "robots"):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    modules.update(
        {
            "isaacsim.core.utils.stage": types.SimpleNamespace(get_stage_units=lambda: 1.0),
            "isaacsim.core.utils.types": types.SimpleNamespace(ArticulationAction=MagicMock),
            "isaacsim.core.utils.rotations": types.SimpleNamespace(euler_angles_to_quat=MagicMock()),
            "robots.base_robot": types.SimpleNamespace(BaseRobot=object, GRIPPER_CLOSED=0, GRIPPER_OPEN=1),
            package + ".base_controller": types.SimpleNamespace(
                BaseController=type("Base", (), {"REQUIRED_SUCCESS_STEPS": 60})
            ),
            package + ".atomic_actions.pick_controller": types.SimpleNamespace(PickController=MagicMock),
        }
    )
    loaded = []
    with patch.dict(sys.modules, modules):
        for name in (
            "atomic_actions.atomic_base_controller",
            "atomic_actions.place_controller",
            "flask_to_cork_controller",
            "stopper_flask_controller",
            "pipette_kinematics",
            "pipette_rack_controller",
            "pipette_rack_sequence_controller",
        ):
            spec = importlib.util.spec_from_file_location(package + "." + name, root / (name.replace(".", "/") + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
    return (
        loaded[1].PlaceController,
        loaded[2].FlaskToCorkTaskController,
        loaded[2].Phase,
        loaded[3].StopperFlaskTaskController,
        loaded[5].PipetteRackTaskController,
        loaded[6].PipetteRackSequenceTaskController,
    )


PlaceController, FlaskController, Phase, StopperController, PipetteController, SequenceController = load_controllers()


class PipetteGripMetricTests(unittest.TestCase):
    def test_horizontal_feedback_does_not_pull_toward_axial_endpoint(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(
            task=types.SimpleNamespace(pipette_track_insertion_axis=True, pipette_front_insertion=True)
        )
        controller.place_controller = types.SimpleNamespace(
            _event=0, _approach_complete=True, approach_offset=np.array([-0.12, 0.0, 0.0])
        )
        controller._command_seat_position = lambda state: np.array([0.2, -0.2, 0.9])
        state = {
            "object_reference_position": np.array([0.08, -0.206, 0.91]),
            "target_quaternion": Rotation.identity().as_quat(),
            "object_quaternion": Rotation.from_euler("z", 1, degrees=True).as_quat(),
        }
        lateral, angular = controller._placement_feedback(state)
        np.testing.assert_allclose(lateral, [0, 0.006], atol=1e-12)
        np.testing.assert_allclose(angular, [0, 0, -np.radians(1)], atol=1e-12)
        # An oblique slot must also remove all along-slot correction.
        controller.place_controller.approach_offset = np.array([-0.1, -0.1, 0])
        lateral, _ = controller._placement_feedback(state)
        self.assertAlmostEqual(float(np.dot(lateral, [1, 1])), 0)
        controller.place_controller._event = 1
        np.testing.assert_allclose(controller._placement_feedback(state)[0], [0.12, 0.006])
        for event in (2, 3, 4):
            controller.place_controller._event = event
            self.assertEqual(controller._placement_feedback(state), (None, None))

    def test_horizontal_feedback_is_opt_in_and_rejects_invalid_axis(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(pipette_front_insertion=True))
        controller.place_controller = types.SimpleNamespace(_event=0, _approach_complete=True)
        self.assertEqual(controller._placement_feedback({}), (None, None))
        controller.cfg.task.pipette_track_insertion_axis = True
        controller._command_seat_position = lambda state: np.zeros(3)
        for direction in ([0, 0, 0], [np.nan, 0, 0]):
            controller.place_controller.approach_offset = direction
            with self.assertRaisesRegex(ValueError, "approach direction"):
                controller._placement_feedback({"object_reference_position": np.zeros(3)})
        controller.place_controller._approach_complete = False
        self.assertEqual(controller._placement_feedback({}), (None, None))

    def test_physical_hand_metric_does_not_replace_fk_slip_guard(self):
        controller = object.__new__(PipetteController)
        controller.current_phase = Phase.PICKING
        controller.pick_controller = types.SimpleNamespace(_event=5)
        controller._grasp_reference = None
        controller._grasp_trace_steps = 0
        controller.rmp_controller = MagicMock()
        controller.rmp_controller.rmp_flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        state = {
            "object_quaternion": Rotation.identity().as_quat(),
            "physical_hand_quaternion": Rotation.identity().as_quat(),
        }
        with patch.dict("os.environ", {"LABUTOPIA_PIPETTE_GRASP_DEBUG": "0"}):
            self.assertAlmostEqual(controller._trace_grasp(state), 0)
            state["object_quaternion"] = Rotation.from_euler("z", 3, degrees=True).as_quat()
            state["physical_hand_quaternion"] = Rotation.from_euler("z", 2, degrees=True).as_quat()
            self.assertAlmostEqual(controller._trace_grasp(state), 3)
            self.assertAlmostEqual(controller._physical_postclose_peak_deg, 1)
            self.assertAlmostEqual(controller._grasp_phase_peaks["picking"], 3)
            controller.current_phase = Phase.PLACING
            controller.place_controller = types.SimpleNamespace(_event=2)
            state["object_quaternion"] = Rotation.from_euler("z", 4, degrees=True).as_quat()
            self.assertAlmostEqual(controller._trace_grasp(state), 4)
            self.assertAlmostEqual(controller._grasp_phase_peaks["placing_2"], 4)
            self.assertAlmostEqual(controller._postclose_peak_rotation_deg, 4)

    def test_closure_settling_is_reported_without_relaxing_original_slip_guard(self):
        controller = object.__new__(PipetteController)
        controller.current_phase = Phase.PICKING
        controller.pick_controller = types.SimpleNamespace(_event=4, _jaw_contact_steps=10)
        controller.place_controller = types.SimpleNamespace(_event=0)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(pipette_stable_close=True))
        controller._grasp_reference = None
        controller._grasp_trace_steps = 0
        controller.rmp_controller = MagicMock()
        controller.rmp_controller.rmp_flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        state = {"object_quaternion": Rotation.identity().as_quat()}
        with patch.dict("os.environ", {"LABUTOPIA_PIPETTE_GRASP_DEBUG": "0"}):
            self.assertAlmostEqual(controller._trace_grasp(state), 0)
            state["object_quaternion"] = Rotation.from_euler("z", 2, degrees=True).as_quat()
            self.assertAlmostEqual(controller._trace_grasp(state), 2)
            self.assertIsNone(controller._postclose_grasp_reference)
            controller.pick_controller._event = 5
            self.assertAlmostEqual(controller._trace_grasp(state), 2)
            state["object_quaternion"] = Rotation.from_euler("z", 2.4, degrees=True).as_quat()
            # The returned value still drives the unchanged 8-degree guard.
            self.assertAlmostEqual(controller._trace_grasp(state), 2.4)
            self.assertAlmostEqual(controller._grasp_peak_rotation_deg, 2.4)
            self.assertAlmostEqual(controller._postclose_peak_rotation_deg, 0.4)

    def test_peak_tracks_relative_rotation_and_ignores_released_motion(self):
        controller = object.__new__(PipetteController)
        controller.current_phase = Phase.PICKING
        controller.pick_controller = types.SimpleNamespace(_event=5)
        controller.place_controller = types.SimpleNamespace(_event=0)
        controller._grasp_reference = None
        controller._grasp_trace_steps = 0
        controller.rmp_controller = MagicMock()
        flow = controller.rmp_controller.rmp_flow
        flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        state = {"object_quaternion": Rotation.identity().as_quat()}
        with patch.dict("os.environ", {"LABUTOPIA_PIPETTE_GRASP_DEBUG": "0"}):
            self.assertAlmostEqual(controller._trace_grasp(state), 0)
            # Moving the hand and object together must not count as slip.
            flow.get_end_effector_pose.return_value = (
                np.zeros(3),
                Rotation.from_euler("z", 5, degrees=True).as_matrix(),
            )
            state["object_quaternion"] = Rotation.from_euler("z", 5, degrees=True).as_quat()
            self.assertAlmostEqual(controller._trace_grasp(state), 0)
            state["object_quaternion"] = Rotation.from_euler("z", 8, degrees=True).as_quat()
            self.assertAlmostEqual(controller._trace_grasp(state), 3)
            state["object_quaternion"] = Rotation.from_euler("z", 6, degrees=True).as_quat()
            self.assertAlmostEqual(controller._trace_grasp(state), 1)
            self.assertAlmostEqual(controller._grasp_peak_rotation_deg, 3)
            controller.current_phase = Phase.PLACING
            controller.place_controller._event = 3
            state["object_quaternion"] = Rotation.from_euler("z", 90, degrees=True).as_quat()
            self.assertIsNone(controller._trace_grasp(state))
            self.assertAlmostEqual(controller._grasp_peak_rotation_deg, 3)
            self.assertTrue(controller._grasp_reported)
            controller.current_phase = Phase.PICKING
            controller._grasp_reference = None
            self.assertAlmostEqual(controller._trace_grasp(state), 0)
            self.assertAlmostEqual(controller._grasp_peak_rotation_deg, 0)


class FourPipetteCompletionTests(unittest.TestCase):
    tool_count = 4

    def controller(self):
        controller = object.__new__(SequenceController)
        controller.cfg = types.SimpleNamespace(
            task=types.SimpleNamespace(
                placement_x_tolerance=0.018,
                placement_y_tolerance=0.008,
                placement_z_tolerance=0.012,
                pipette_paths=[f"/tool_{i}" for i in range(self.tool_count)],
                slot_indices=list(range(self.tool_count)),
            )
        )
        controller._all_initial_sizes = [np.array([0.26, 0.03, 0.04])] * self.tool_count
        controller._all_previous_positions = None
        controller._scene_state = {
            "gripper_position": np.array([-0.1, 0.1, 1.3]),
            "target_axis": np.array([0, 0, 1]),
            "pipettes": [
                {
                    "object_path": f"/tool_{i}",
                    "object_position": np.array([0.2, i * 0.038, 0.91]),
                    "target_seat_position": np.array([0.2, i * 0.038, 0.91]),
                    "object_grasp_position": np.array([0.2, i * 0.038, 1.0]),
                    "object_size": np.array([0.07, 0.03, 0.26]),
                    "object_axis": np.array([0, 0, 1]),
                    "slot_index": i,
                }
                for i in range(self.tool_count)
            ],
        }
        controller.data_collector = MagicMock()
        return controller

    def test_all_configured_tools_required_in_collect_and_replay(self):
        for mode in ("collect", "replay"):
            for failed_index in range(self.tool_count):
                controller = self.controller()
                controller.mode = mode
                controller._active_pipette = self.tool_count - 1
                controller._scene_state["pipettes"][failed_index]["object_position"][2] -= 0.1
                controller._check_success()
                self.assertFalse(controller._check_success())

    def test_all_stable_released_objects_pass_same_geometry_gate(self):
        controller = self.controller()
        self.assertFalse(controller._check_success())
        self.assertTrue(controller._check_success())
        controller._scene_state["gripper_position"] = controller._scene_state["pipettes"][2][
            "object_grasp_position"
        ].copy()
        self.assertFalse(controller._check_success())

    def test_intermediate_handoff_does_not_write_or_clear_episode(self):
        controller = self.controller()
        controller._active_pipette = 0
        controller.pick_controller = MagicMock()
        controller.place_controller = MagicMock()
        controller.place_controller._last_record_positions = np.arange(8, dtype=float)
        controller.rmp_controller = MagicMock()
        result = controller._on_placement_complete({"joint_positions": np.zeros(9)})
        self.assertEqual(result, (None, False, False))
        self.assertEqual(controller._active_pipette, 1)
        np.testing.assert_array_equal(controller.pick_controller._last_record_positions, np.arange(8))
        controller.data_collector.write_cached_data.assert_not_called()
        controller.data_collector.clear_cache.assert_not_called()

    def test_last_handoff_writes_once(self):
        controller = self.controller()
        controller._active_pipette = self.tool_count - 1
        self.assertEqual(controller._on_placement_complete({"joint_positions": np.zeros(9)}), (None, True, True))
        controller.data_collector.write_cached_data.assert_called_once()

    def test_complete_sequence_requires_sixty_consecutive_frames(self):
        controller = self.controller()
        controller.mode = "infer"
        controller.current_phase = Phase.PLACING
        controller._active_pipette = self.tool_count - 1
        controller._placement_steps = 0
        controller._last_success = False
        self.assertFalse(controller.is_success())  # Initial motion sample.
        for _ in range(59):
            self.assertFalse(controller.is_success())
        self.assertTrue(controller.is_success())

    def test_missing_tool_or_initial_size_cannot_pass_success(self):
        for field in ("pipettes", "initial_sizes"):
            controller = self.controller()
            if field == "pipettes":
                controller._scene_state[field].pop()
            else:
                controller._all_initial_sizes.pop()
            self.assertFalse(controller._check_success())
            self.assertFalse(controller._check_success())

    def test_observations_must_match_configured_assignment(self):
        controller = self.controller()
        controller._active_pipette = 0
        state = controller._scene_state
        with patch.object(PipetteController, "step", return_value="action") as step:
            self.assertEqual(controller.step(state), "action")
            step.assert_called_once()
        state["pipettes"][-1]["slot_index"] = 99
        with self.assertRaisesRegex(ValueError, "configured"):
            controller.step(state)

    def test_language_matches_configured_tool_count(self):
        controller = self.controller()
        controller.mode = "infer"
        controller._active_pipette = self.tool_count - 1
        word = "three" if self.tool_count == 3 else "four"
        self.assertIn(f"all {word}", controller.get_language_instruction())
        self.assertIn(f"pipette {self.tool_count} of {word}", controller._sample_phase_instruction(Phase.PLACING))


class ThreePipetteCompletionTests(FourPipetteCompletionTests):
    tool_count = 3


class PrecisionPlaceTests(unittest.TestCase):
    def controller(self, precision=True):
        robot = MagicMock()
        robot.num_gripper_joints = 2
        return PlaceController("test", MagicMock(), robot=robot, precision=precision)

    def test_timeout_does_not_open_gripper(self):
        controller = self.controller()
        controller._event = 1
        controller._t = 0.999
        controller._advance_state()
        self.assertTrue(controller.is_done())
        self.assertIn("timed out", controller.get_failure_reason())
        controller._robot.open_gripper.assert_not_called()

    def test_hover_requires_height_before_descent(self):
        controller = self.controller()
        goal = np.array([0.2, 0.1, 0.8])
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal, 0.15, 0.003)
        self.assertEqual(controller._event, 0)
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal + np.array([0, 0, 0.15]), 0.15, 0.003)
        self.assertEqual(controller._event, 1)

    def test_release_retreat_is_above_actual_release(self):
        controller = self.controller()
        controller._event = 3
        goal = np.array([0.2, 0.1, 0.8])
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal, 0.15, 0.2)
        np.testing.assert_allclose(controller.target_position, [0.2, 0.1, 1.12])

    def test_axial_release_retreat_preserves_actual_release_height(self):
        controller = self.controller()
        controller._event = 3
        controller.retreat_offset = np.array([-0.12, 0.0, 0.0])
        goal = np.array([0.2, 0.1, 0.8])
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal, 0.15, 0.014)
        np.testing.assert_allclose(controller.target_position, [0.08, 0.1, 0.814])
        controller._robot.open_gripper.assert_called_once()

    def test_return_to_transit_finishes_before_placement_handoff(self):
        controller = self.controller()
        controller._event = 4
        controller.target_position = np.array([0.1, -0.1, 0.98])
        controller.post_retreat_position = np.array([0.05, 0.05, 1.12])
        first = controller.target_position.copy()
        controller._execute_phase(first, None, np.zeros(9), MagicMock(), first, 0.1, 0.014)
        self.assertEqual(controller._event, 4)
        self.assertTrue(controller._post_retreat_started)
        np.testing.assert_array_equal(controller.target_position, controller.post_retreat_position)
        controller._execute_phase(first, None, np.zeros(9), MagicMock(), controller.target_position, 0.1, 0.014)
        self.assertEqual(controller._event, 5)

    def test_front_approach_aligns_before_entering_fork(self):
        controller = self.controller()
        controller.approach_offset = np.array([-0.14, 0, 0])
        goal = np.array([0.2, -0.2, 0.9])
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), np.array([0.06, -0.2, 1.0]), 0.1, 0.055)
        self.assertTrue(controller._approach_complete)
        self.assertEqual(controller._event, 0)
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), np.array([0.2, -0.2, 1.0]), 0.1, 0.055)
        self.assertEqual(controller._event, 1)

    def test_precision_has_no_pose_noise(self):
        controller = self.controller()
        controller._ensure_randomization()
        self.assertEqual(controller._place_offset_z_noise, 0)
        self.assertEqual(controller._orientation_noise_deg, 0)

    def test_retreat_clearance_has_its_own_tolerance(self):
        controller = self.controller()
        controller._event = 4
        controller.target_position = np.array([0.2, 0.1, 1.2])
        actual = np.array([0.2, 0.115, 1.19])
        controller._execute_phase(np.zeros(3), None, np.zeros(9), MagicMock(), actual, 0.1, 0.05)
        self.assertEqual(controller._event, 4)
        controller.retreat_position_threshold = 0.025
        controller._execute_phase(np.zeros(3), None, np.zeros(9), MagicMock(), actual, 0.1, 0.05)
        self.assertEqual(controller._event, 5)

    def test_fork_height_clearance_does_not_relax_lateral_alignment(self):
        controller = self.controller()
        controller.precision_position_threshold = 0.008
        controller.precision_height_threshold = 0.020
        controller._event = 1
        goal = np.array([0.2, 0.1, 0.9])
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal + np.array([0.012, 0, 0.015]), 0.15, 0)
        self.assertEqual(controller._event, 1)
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal + np.array([0.004, 0, 0.015]), 0.15, 0)
        self.assertEqual(controller._event, 2)

    def test_legacy_timeout_still_advances(self):
        controller = self.controller(precision=False)
        controller._event = 1
        controller._t = 0.999
        controller._advance_state()
        self.assertEqual(controller._event, 2)
        self.assertEqual(controller.get_failure_reason(), "")

    def test_feature_gate_does_not_release_when_only_tcp_arrived(self):
        controller = self.controller()
        controller._event = 1
        goal = np.array([0.2, 0.1, 0.9])
        controller.placement_feature_target = np.array([0.2, 0.1, 0.85])
        controller.placement_feature_position = np.array([0.2, 0.1, 0.95])
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal, 0.1, 0)
        self.assertEqual(controller._event, 1)
        controller.placement_feature_position = controller.placement_feature_target.copy()
        controller.placement_feature_aligned = False
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal, 0.1, 0)
        self.assertEqual(controller._event, 1)
        controller.placement_feature_aligned = True
        controller._execute_phase(goal, None, np.zeros(9), MagicMock(), goal, 0.1, 0)
        self.assertEqual(controller._event, 2)


class PlacementGoalTests(unittest.TestCase):
    def test_loaded_transit_records_closed_before_place_start(self):
        controller = object.__new__(PipetteController)
        hub = np.array([0.05, 0.05, 1.12])
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(pipette_transit_position=hub))
        controller.current_phase = Phase.PLACING
        controller._trace_grasp = lambda state: None
        controller._transit_phase = controller._transit_steps = controller._transit_stable = 0
        controller._transit_orientation = np.array([1, 0, 0, 0])
        robot = MagicMock()
        robot.num_gripper_joints = 2
        controller.place_controller = PlaceController("test", MagicMock(), robot=robot, precision=True)
        place = controller.place_controller
        robot.get_gripper_commanded_opening.return_value = 0.0
        self.assertEqual(place._last_gripper_state, 1)
        self.assertEqual(place._current_gripper_state, 0)
        action = types.SimpleNamespace(joint_positions=np.zeros(7), joint_indices=np.arange(7))
        place._cspace_controller.forward.return_value = action
        controller.rmp_controller = MagicMock()
        controller.rmp_controller.rmp_flow.get_end_effector_pose.return_value = (hub, np.eye(3))
        with patch.object(PlaceController.__mro__[1], "record_commanded_gripper", True):
            _, record = controller._forward_active({"joint_positions": np.zeros(9), "gripper_position": hub})
        self.assertEqual(record[-1], 1.0)  # Recorded channel is closedness, not enum.
        self.assertEqual(place._last_gripper_state, 0)
        self.assertTrue(place._start)

    def test_depth_compensation_does_not_move_success_target(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(pipette_insertion_depth_offset=0.004))
        state = {"target_seat_position": np.array([0.2, 0.1, 0.91]), "target_quaternion": [0, 0, 0, 1]}
        np.testing.assert_allclose(controller._command_seat_position(state), [0.204, 0.1, 0.91])
        np.testing.assert_allclose(controller._goal_position(state), [0.2, 0.1, 0.91])

    def test_preflight_rejects_unreachable_path_without_robot_commands(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(pipette_front_insertion=True))
        controller.robot = MagicMock()
        controller.robot._articulation_view.get_dof_limits.return_value = np.array([[[-3, 3]] * 9])
        controller.rmp_controller = MagicMock()
        controller.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = (
            np.zeros(7)
        )
        solver = controller.rmp_controller.rmp_flow.get_kinematics_solver()
        solver.compute_inverse_kinematics.return_value = (np.zeros(7), False)
        controller._pick_orientation = lambda **kwargs: np.array([1, 0, 0, 0])
        state = {
            "object_grasp_position": np.array([0.1, 0.1, 0.8]),
            "object_reference_position": np.array([0.1, 0.1, 0.75]),
            "target_seat_position": np.array([0.2, -0.1, 0.91]),
            "object_quaternion": [0, 0, 0, 1],
            "target_quaternion": [0, 0, 0, 1],
        }
        with self.assertRaisesRegex(ValueError, "No continuous grasp-to-seat"):
            controller._plan_grasp_branch(state, np.zeros(3))
        controller.robot.apply_action.assert_not_called()
        controller.robot.set_joint_positions.assert_not_called()

    def test_handoff_records_the_physics_hold_frame(self):
        controller = object.__new__(FlaskController)
        controller.active_controller = MagicMock()
        controller.data_collector = MagicMock()
        controller.get_language_instruction = lambda: "hold"
        controller.get_task_index = lambda: 0
        state = {"camera_data": {}, "joint_positions": np.zeros(9)}
        controller._cache_hold_step(state)
        controller.data_collector.cache_step.assert_called_once()
        controller.data_collector.write_cached_data.assert_not_called()

    def test_in_hand_slip_fails_before_compensating_the_pose(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(max_grasp_rotation_deg=8.0))
        controller.current_phase = Phase.PICKING
        controller.pick_controller = MagicMock()
        controller._trace_grasp = lambda state: 12.0
        controller._forward_active({"joint_positions": np.zeros(9)})
        self.assertIn("slipped inside", controller.pick_controller._phase_failure_reason)
        controller.pick_controller.forward.assert_not_called()

    def controller(self):
        controller = object.__new__(FlaskController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace())
        controller.state = {
            "object_position": np.array([0.2, 0.1, 0.85]),
            "target_seat_position": np.array([0.2, 0.1, 0.85]),
            "gripper_position": np.array([0.2, 0.1, 1.05]),
            "object_up": np.array([0, 0, 1]),
        }
        controller._previous_object_position = np.array([0.2, 0.1, 0.85])
        return controller

    def test_rejects_flask_on_table_below_support(self):
        controller = self.controller()
        controller.state["object_position"][2] -= 0.09
        self.assertFalse(controller._check_placed_on_target())

    def test_original_flask_grasp_uses_neck_marker_and_preserves_legacy_default(self):
        for use_marker in (False, True):
            controller = self.controller()
            controller.cfg.task.original_flask_neck_grasp = use_marker
            controller.current_phase = Phase.PICKING
            controller.pick_controller = MagicMock()
            controller.gripper_control = MagicMock()
            controller._gripper_distance = 0.014
            state = dict(
                controller.state,
                object_neck_position=np.array([0.203, 0.099, 0.88]),
                joint_positions=np.zeros(9),
                object_size=np.array([0.08, 0.08, 0.155]),
                object_name="round_bottom_flask",
            )
            controller._forward_active(state)
            expected = state["object_neck_position"] if use_marker else state["object_position"] + [0, 0, 0.02]
            kwargs = controller.pick_controller.forward.call_args.kwargs
            np.testing.assert_allclose(kwargs["picking_position"], expected)
            self.assertEqual(kwargs["gripper_distances"], 0.014)
            self.assertEqual(kwargs["after_offset_z"], 0.14 if use_marker else 0.25)

    def test_original_flask_grasp_fails_explicitly_without_marker(self):
        controller = self.controller()
        controller.cfg.task.original_flask_neck_grasp = True
        controller.current_phase = Phase.PICKING
        with self.assertRaisesRegex(ValueError, "NeckGrasp"):
            controller._forward_active(controller.state)

    def test_original_flask_alignment_is_bounded_and_frozen_after_descent(self):
        controller = self.controller()
        controller.place_controller = types.SimpleNamespace(_event=1)
        controller.rmp_controller = MagicMock()
        controller.rmp_controller.rmp_flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        state = dict(controller.state)
        state["object_up"] = Rotation.from_euler("y", 20, degrees=True).apply([0, 0, 1])
        state["gripper_position"] = state["object_position"] + state["object_up"] * 0.03
        goal, orientation = controller._original_flask_place_pose(state)
        correction = Rotation.from_quat(orientation[[1, 2, 3, 0]])
        self.assertAlmostEqual(np.degrees(correction.magnitude()), 3)
        self.assertFalse(controller.place_controller.placement_feature_aligned)
        np.testing.assert_allclose(controller.place_controller.placement_feature_target, [0.2, 0.1, 0.853])
        np.testing.assert_allclose(goal, state["target_seat_position"] + correction.apply(state["object_up"] * 0.03))
        controller.place_controller._event = 3
        state["object_position"] = np.zeros(3)
        later_goal, later_orientation = controller._original_flask_place_pose(state)
        np.testing.assert_allclose(later_goal, goal)
        np.testing.assert_allclose(later_orientation, orientation)
        controller.rmp_controller.rmp_flow.get_end_effector_pose.assert_called_once()

    def test_original_flask_feedback_stops_instead_of_chasing_a_dropped_object(self):
        controller = self.controller()
        controller.place_controller = types.SimpleNamespace(_event=0, _events_dt=[0.01] * 6)
        controller._placement_hand_orientation = None
        controller._grasp_offset = np.array([0, 0, 0.03])
        controller.rmp_controller = MagicMock()
        goal, orientation = controller._original_flask_place_pose(controller.state)
        self.assertIn("left the gripper", controller.place_controller._phase_failure_reason)
        self.assertEqual(controller.place_controller._event, 6)
        np.testing.assert_allclose(goal, [0.2, 0.1, 0.88])
        self.assertTrue(np.all(np.isfinite(orientation)))
        controller.rmp_controller.rmp_flow.get_end_effector_pose.assert_not_called()

    def test_rejects_held_tipped_or_moving_flask(self):
        for field, value in (
            ("gripper_position", [0.2, 0.1, 0.875]),
            ("object_up", [1, 0, 0]),
            ("object_position", [0.205, 0.1, 0.85]),
        ):
            with self.subTest(field=field):
                controller = self.controller()
                controller.state[field] = np.array(value)
                self.assertFalse(controller._check_placed_on_target())

    def test_accepts_settled_upright_released_flask(self):
        controller = self.controller()
        self.assertTrue(controller._check_placed_on_target())
        self.assertEqual(controller._last_failure_reason, "")

    def test_inference_requires_consecutive_success_frames(self):
        controller = self.controller()
        controller.mode = "infer"
        controller.current_phase = Phase.PLACING
        controller._placement_steps = 0
        for _ in range(59):
            self.assertFalse(controller.is_success())
        controller.state["object_up"] = np.array([1, 0, 0])
        self.assertFalse(controller.is_success())
        self.assertEqual(controller._placement_steps, 0)
        controller.state["object_up"] = np.array([0, 0, 1])
        for _ in range(59):
            self.assertFalse(controller.is_success())
        self.assertTrue(controller.is_success())

    def test_collect_and_replay_result_reads_do_not_recheck_success(self):
        for mode in ("collect", "replay"):
            for success in (False, True):
                with self.subTest(mode=mode, success=success):
                    controller = self.controller()
                    controller.mode = mode
                    controller._last_success = success
                    controller._check_placed_on_target = MagicMock(side_effect=AssertionError("rechecked"))
                    self.assertEqual(controller.is_success(), success)
                    controller._check_placed_on_target.assert_not_called()

    def test_stopper_goal_uses_mouth_and_insertion_depth(self):
        controller = object.__new__(StopperController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(insertion_depth=0.008))
        controller.initial_size = np.array([0.028, 0.028, 0.03])
        goal = controller._goal_position({"target_position": [0.2, 0.1, 0.8], "target_size": [0.08, 0.08, 0.14]})
        np.testing.assert_allclose(goal, [0.2, 0.1, 0.877])

    def test_flask_uprighting_rotates_grasp_offset(self):
        controller = self.controller()
        up = np.array([0.5, 0, np.sqrt(0.75)])
        state = {"object_up": up, "object_position": np.zeros(3), "gripper_position": up * 0.05}
        np.testing.assert_allclose(controller._capture_grasp_offset(state), [0, 0, 0.05], atol=1e-8)

    def test_pipette_on_table_cannot_pass_orientation_gate(self):
        controller = object.__new__(PipetteController)
        state = {"object_size": [0.38, 0.04, 0.06], "object_axis": [1, 0, 0], "target_axis": [0, 0, 1]}
        self.assertFalse(controller._placement_orientation_ok(state))
        state.update(object_size=[0.17, 0.04, 0.38], object_axis=[0, 0, 1])
        self.assertTrue(controller._placement_orientation_ok(state))
        state["object_axis"] = [0, 0, -1]
        self.assertFalse(controller._placement_orientation_ok(state))

    def test_pipette_matches_head_direction_without_inverting(self):
        controller = object.__new__(PipetteController)
        state = {
            "object_axis": [1, 0, 0],
            "target_axis": [0, 0, 1],
            "gripper_position": np.array([0.05, 0, 0]),
            "object_position": np.zeros(3),
        }
        np.testing.assert_allclose(controller._capture_grasp_offset(state), [0, 0, 0.05], atol=1e-8)

    def test_pipette_can_rest_along_fork_but_not_on_adjacent_slot(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace())
        target = np.array([0.20, -0.20, 0.94])
        self.assertTrue(controller._placement_xy_ok(target + np.array([0.036, 0, 0]), target))
        self.assertFalse(controller._placement_xy_ok(target + np.array([0, 0.14, 0]), target))

    def test_pipette_refines_offset_after_rotation(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace())
        controller.current_phase = Phase.PLACING
        controller.place_controller = MagicMock(_event=1)
        controller._placement_offset_refined = False
        controller._placement_orientation = np.array([1, 0, 0, 0])
        controller.gripper_control = MagicMock()
        state = {
            "object_position": np.array([0.2, 0.1, 1.05]),
            "gripper_position": np.array([0.21, 0.1, 1.12]),
            "target_seat_position": np.array([0.2, -0.2, 0.92]),
            "joint_positions": np.zeros(9),
        }
        controller._forward_active(state)
        np.testing.assert_allclose(controller._grasp_offset, [0.01, 0, 0.07])
        np.testing.assert_allclose(
            controller.place_controller.forward.call_args.kwargs["place_position"], [0.21, -0.2, 0.99]
        )

    def test_reference_pipette_uses_measured_hand_and_scalar_first_handoff(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace())
        controller.rmp_controller = MagicMock()
        controller.place_controller = MagicMock()
        hand = Rotation.from_euler("xyz", [140, 20, -30], degrees=True)
        source = Rotation.from_euler("xyz", [10, 65, 20], degrees=True)
        target = Rotation.from_euler("z", 5, degrees=True)
        rmp = controller.rmp_controller
        rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = np.zeros(7)
        tcp_local = np.array([0, 0, 0.0034])
        rmp.rmp_flow.get_end_effector_pose.return_value = (
            np.array([0.20, 0.1, 1.05]) - hand.apply(tcp_local),
            hand.as_matrix(),
        )
        rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.return_value = (np.zeros(7), True)
        state = {
            "object_axis": source.apply([0, 0, 1]),
            "target_axis": target.apply([0, 0, 1]),
            "object_quaternion": source.as_quat(),
            "target_quaternion": target.as_quat(),
            "gripper_position": np.array([0.20, 0.1, 1.05]),
            "object_position": np.array([0.10, 0.1, 1.03]),
            "object_reference_position": np.array([0.18, 0.1, 1.02]),
            "target_seat_position": np.array([0.18, -0.1, 0.826]),
        }
        correction = target * source.inv()
        offset = controller._capture_grasp_offset(state)
        np.testing.assert_allclose(offset, correction.apply([0.02, 0, 0.03]))
        result = Rotation.from_quat(controller._placement_orientation[[1, 2, 3, 0]])
        np.testing.assert_allclose(result.as_matrix(), (correction * hand).as_matrix(), atol=1e-8)
        ik_goal = rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.call_args.args[1]
        tcp_goal = controller._command_seat_position(state) + offset + [0, 0, controller._release_clearance()]
        np.testing.assert_allclose(ik_goal, tcp_goal - result.apply(tcp_local))
        np.testing.assert_allclose(
            controller.place_controller._cspace_controller.tcp_local_offset, tcp_local, atol=1e-12
        )

    def test_lost_pipette_is_not_used_to_refine_release_target(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(reference_geometry=True))
        controller.current_phase = Phase.PLACING
        controller.place_controller = MagicMock(_event=1, _events_dt=[0.01] * 6)
        controller._placement_offset_refined = False
        controller._placement_orientation = np.array([1, 0, 0, 0])
        controller._grasp_offset = np.array([0.01, 0, 0.04])
        controller.gripper_control = MagicMock()
        state = {
            "object_position": np.array([0.2, 0.1, 0.75]),
            "object_reference_position": np.array([0.2, 0.1, 0.75]),
            "object_grasp_position": np.array([0.22, 0.1, 0.76]),
            "gripper_position": np.array([0.21, -0.1, 1.12]),
            "target_seat_position": np.array([0.2, -0.2, 0.826]),
            "joint_positions": np.zeros(9),
        }
        controller._forward_active(state)
        self.assertIn("grasp lost", controller.place_controller._phase_failure_reason)
        self.assertFalse(controller._placement_offset_refined)
        np.testing.assert_allclose(controller._grasp_offset, [0.01, 0, 0.04])

    def test_pipette_ik_compensates_tcp_frame_translation(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        rmp = MagicMock()
        rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = np.zeros(7)
        rmp.rmp_flow.get_end_effector_pose.return_value = (np.array([0.2, 0, 1.0]), np.eye(3))
        solver = rmp.rmp_flow.get_kinematics_solver.return_value
        solver.compute_inverse_kinematics.return_value = (np.zeros(7), True)
        motion = motion_class(rmp, np.zeros(7), [0, 0, 0.0034])
        motion.forward(np.array([0.2, 0, 1.0034]), np.array([1, 0, 0, 0]))
        np.testing.assert_allclose(solver.compute_inverse_kinematics.call_args.args[1], [0.2, 0, 1.0])

    def test_pipette_motion_rejects_unreachable_waypoint(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        rmp = MagicMock()
        rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.return_value = (np.zeros(7), False)
        motion = motion_class(rmp, np.zeros(7))
        with self.assertRaisesRegex(ValueError, "not IK reachable"):
            motion.forward(np.array([2, 0, 1]), np.array([1, 0, 0, 0]))

    def test_pipette_joint_transfer_can_be_slowed_without_changing_goal(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        rmp = MagicMock()
        rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = np.zeros(7)
        rmp.rmp_flow.get_end_effector_pose.return_value = (np.array([0.2, 0, 1.0]), np.eye(3))
        rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.return_value = (np.ones(7), True)
        normal = motion_class(rmp, np.zeros(7))
        slow = motion_class(rmp, np.zeros(7), transfer_time_scale=2.0)
        for motion in (normal, slow):
            motion.forward(np.array([0.3, 0, 1.0]), np.array([1, 0, 0, 0]))
        self.assertEqual(slow.duration, 2 * normal.duration)
        np.testing.assert_array_equal(slow.goal, normal.goal)
        np.testing.assert_array_equal(slow.seed, normal.seed)

    def test_pipette_transfer_rejects_invalid_time_scale(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        for scale in (0, 0.5, np.nan, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                motion_class(MagicMock(), np.zeros(7), transfer_time_scale=scale)

    def test_cartesian_translation_and_rotation_honor_transfer_time_scale(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        for angle in (0, np.pi / 2):
            with self.subTest(angle=angle):
                rmp = MagicMock()
                rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = np.zeros(7)
                rmp.rmp_flow.get_end_effector_pose.return_value = (np.array([0.2, 0, 1.0]), np.eye(3))
                rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.return_value = (np.zeros(7), True)
                motions = [
                    motion_class(
                        rmp,
                        np.zeros(7),
                        cartesian_first=True,
                        joint_bounds=np.tile([-2, 2], (7, 1)),
                        transfer_time_scale=scale,
                    )
                    for scale in (1.0, 2.0)
                ]
                quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
                for motion in motions:
                    motion.forward(np.array([0.32, 0, 1.0]), quat)
                self.assertEqual(motions[1].duration, 2 * motions[0].duration)
                np.testing.assert_array_equal(motions[1].goal, motions[0].goal)

    def test_small_ik_steps_cannot_bypass_runtime_joint_margin(self):
        namespace = PipetteController._capture_grasp_offset.__globals__
        motion_class = namespace["PipettePoseMotion"]
        rmp = MagicMock()
        start = np.array([1.9, 0, 0, 0, 0, 0, 0])
        unsafe = start.copy()
        unsafe[0] = 1.98  # Small 0.08-rad step, but only 0.02-rad limit clearance.
        rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = start.copy()
        rmp.rmp_flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.return_value = (unsafe, True)
        bounds = np.tile([-2.0, 2.0], (7, 1))
        motion = motion_class(rmp, start, cartesian_first=True, joint_bounds=bounds, joint_margin=0.1)
        recovery = MagicMock(return_value=(start.copy(), True))
        with patch.dict(namespace, {"continuous_inverse_kinematics": recovery}):
            action = motion.forward(np.zeros(3), [1, 0, 0, 0])
        self.assertEqual(recovery.call_count, 1)  # The actual Cartesian step.
        self.assertEqual(recovery.call_args.kwargs["joint_margin"], 0.1)
        np.testing.assert_array_equal(recovery.call_args.args[5], bounds)
        np.testing.assert_allclose(action.joint_positions, start)

    def test_bounded_cartesian_motion_uses_actual_branch_not_discarded_endpoint(self):
        namespace = PipetteController._capture_grasp_offset.__globals__
        motion_class = namespace["PipettePoseMotion"]
        rmp = MagicMock()
        rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = np.zeros(7)
        rmp.rmp_flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        solver = rmp.rmp_flow.get_kinematics_solver.return_value
        goal = np.array([0.2, 0, 0])

        def solve(frame, position, orientation, **kwargs):
            return np.zeros(7), not np.allclose(position, goal)

        solver.compute_inverse_kinematics.side_effect = solve
        motion = motion_class(
            rmp, np.ones(7), cartesian_first=True, joint_bounds=np.tile([-2, 2], (7, 1)), joint_margin=0.1
        )
        motion.forward(goal, [1, 0, 0, 0])
        self.assertEqual(solver.compute_inverse_kinematics.call_count, 1)
        np.testing.assert_array_equal(solver.compute_inverse_kinematics.call_args.kwargs["warm_start"], np.zeros(7))
        self.assertFalse(np.allclose(solver.compute_inverse_kinematics.call_args.args[1], goal))
        # This only allows a valid local step; an unreachable arrival still fails.
        motion.step = motion.duration
        with (
            patch.dict(namespace, {"continuous_inverse_kinematics": MagicMock(return_value=(np.zeros(7), False))}),
            self.assertRaisesRegex(ValueError, "no continuous IK"),
        ):
            motion.forward(goal, [1, 0, 0, 0])

    def test_tracking_bias_cannot_spend_ik_margin(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        rmp = MagicMock()
        target = np.array([1.89, 0, 0, 0, 0, 0, 0])
        rmp.articulation_rmp.get_active_joints_subset().get_joint_positions.return_value = target.copy()
        rmp.rmp_flow.get_end_effector_pose.return_value = (np.zeros(3), np.eye(3))
        rmp.rmp_flow.get_kinematics_solver().compute_inverse_kinematics.return_value = (target.copy(), True)
        motion = motion_class(
            rmp, target, cartesian_first=True, joint_bounds=np.tile([-2, 2], (7, 1)), joint_margin=0.1
        )
        motion.tracking_bias[0] = 0.03
        action = motion.forward(np.zeros(3), [1, 0, 0, 0])
        self.assertAlmostEqual(action.joint_positions[0], 1.9)
        self.assertAlmostEqual(motion.tracking_bias[0], 0.01)

    def test_runtime_margin_rejects_invalid_physical_intervals(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        for bounds in (np.zeros((7, 2)), np.ones((6, 2)), np.full((7, 2), np.nan)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                motion_class(MagicMock(), np.zeros(7), joint_bounds=bounds)
        for margin in (0, 0.3, np.nan):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                motion_class(MagicMock(), np.zeros(7), joint_margin=margin)

    def test_symmetric_pick_swaps_jaws_but_preserves_downward_approach(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(task=types.SimpleNamespace(pipette_pick_pitch=180))
        first = Rotation.from_quat(controller._pick_orientation()[[1, 2, 3, 0]])
        controller.cfg.task.pipette_pick_yaw = 180
        swapped = Rotation.from_quat(controller._pick_orientation()[[1, 2, 3, 0]])
        np.testing.assert_allclose(first.apply([0, 0, 1]), [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(swapped.apply([0, 0, 1]), first.apply([0, 0, 1]), atol=1e-12)
        np.testing.assert_allclose(swapped.apply([0, 1, 0]), -first.apply([0, 1, 0]), atol=1e-12)

    def test_object_aligned_grasp_locks_before_closure(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(
            task=types.SimpleNamespace(
                pipette_pick_pitch=180,
                pipette_pick_yaw=180,
                pipette_object_aligned_grasp=True,
            )
        )
        controller.pick_controller = types.SimpleNamespace(_event=1)
        controller._pick_target_orientation = None
        source = Rotation.from_euler("z", 5, degrees=True) * Rotation.from_euler("y", 90, degrees=True)
        controller.state = {"object_quaternion": source.as_quat()}
        target = controller._pick_orientation()
        hand = Rotation.from_quat(target[[1, 2, 3, 0]])
        np.testing.assert_allclose(hand.apply([0, 1, 0]), source.apply([0, 1, 0]), atol=1e-12)
        controller.state["object_quaternion"] = Rotation.from_euler("z", 20, degrees=True).as_quat()
        controller.pick_controller._event = 5
        np.testing.assert_array_equal(controller._pick_orientation(), target)

    def test_preflight_aligns_to_settled_object_before_caching_plan(self):
        controller = object.__new__(PipetteController)
        controller.cfg = types.SimpleNamespace(
            task=types.SimpleNamespace(pipette_pick_pitch=180, pipette_object_aligned_grasp=True)
        )
        controller.pick_controller = types.SimpleNamespace(_event=0)
        controller._pick_target_orientation = None
        controller.state = {"object_quaternion": Rotation.identity().as_quat()}
        nominal = controller._pick_orientation()
        self.assertIsNone(controller._pick_target_orientation)
        correction = Rotation.from_euler("xyz", [4, -10, 3], degrees=True)
        source = correction * Rotation.from_euler("y", 90, degrees=True)
        aligned = controller._pick_orientation(settled_state={"object_quaternion": source.as_quat()})
        expected = correction * Rotation.from_quat(nominal[[1, 2, 3, 0]])
        np.testing.assert_allclose(Rotation.from_quat(aligned[[1, 2, 3, 0]]).as_matrix(), expected.as_matrix())
        controller._planned_pick_orientation = aligned.copy()
        controller.pick_controller._event = 5
        np.testing.assert_array_equal(controller._pick_orientation(), aligned)

    def test_pipette_motion_reset_clears_feedback(self):
        motion_class = PipetteController._capture_grasp_offset.__globals__["PipettePoseMotion"]
        rmp = MagicMock()
        motion = motion_class(rmp, np.zeros(7))
        motion.goal = np.ones(7)
        motion.step = 123
        motion.tracking_bias[:] = 0.01
        motion.angular_integral[:] = 0.1
        motion.lateral_error = np.ones(2)
        motion.reset()
        self.assertIsNone(motion.goal)
        self.assertIsNone(motion.lateral_error)
        self.assertEqual(motion.step, 0)
        np.testing.assert_array_equal(motion.tracking_bias, np.zeros(7))
        np.testing.assert_array_equal(motion.angular_integral, np.zeros(3))

    def test_pipette_head_grasp_is_not_mistaken_for_release(self):
        controller = object.__new__(PipetteController)
        controller.state = {
            "object_reference_position": np.array([0.2, 0.1, 0.846]),
            "object_grasp_position": np.array([0.2, 0.1, 0.926]),
            "gripper_position": np.array([0.2, 0.1, 0.926]),
        }
        with patch.object(FlaskController, "_check_placed_on_target", return_value=True):
            self.assertFalse(controller._check_placed_on_target())
            self.assertIn("grasp feature", controller._last_failure_reason)
            controller.state["gripper_position"][2] += 0.12
            self.assertTrue(controller._check_placed_on_target())


if __name__ == "__main__":
    unittest.main()
