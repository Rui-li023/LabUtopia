import copy
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.pipette.summarize_pipette_grip import summarize_trace
from experiments.support.grasp_telemetry import capture_grasp_sample, observed_step, summarize_grasp_samples


def sample(angle=0.0, *, closed=True, hand_angle=0.0, episode=0):
    return {
        "type": "sample",
        "episode": episode,
        "closed_command": closed,
        "tcp": [0, 0, 1],
        "finger_positions": [0.014, 0.014],
        "hand_quaternion_xyzw": Rotation.from_euler("z", hand_angle, degrees=True).as_quat().tolist(),
        "objects": [
            {
                "path": "/pipette",
                "grasp_position": [0, 0, 1],
                "quaternion_xyzw": Rotation.from_euler("z", angle, degrees=True).as_quat().tolist(),
            }
        ],
    }


class GraspTelemetryTests(unittest.TestCase):
    def test_truncated_trace_and_skipped_replay_steps_are_visible(self):
        rows = [dict(sample(), mode="replay", replay_action_step=i) for i in (0, 1, 3)]
        report = summarize_trace([{"type": "metadata"}, *rows])
        self.assertFalse(report["trace_complete"])
        self.assertEqual(report["continuity_errors"], [{"episode": 0, "step": 3}])
        marker = {"type": "shutdown_requested", "last_step_terminal": True}
        self.assertTrue(summarize_trace([{"type": "metadata"}, *rows, marker])["trace_complete"])

    def test_settled_reference_cannot_be_created_at_placement(self):
        rows = [sample(i) for i in range(25)]
        for _ in range(40):
            row = sample(25)
            row["tcp"] = [0, 0, 1.1]
            row["objects"][0]["grasp_position"] = [0, 0, 1.1]
            rows.append(row)
        report = summarize_grasp_samples(rows)
        self.assertIsNone(report["peak_settled_rotation_deg"])
        self.assertGreater(report["peak_contact_rotation_deg"], 10)

    def test_world_rotation_with_hand_is_not_slip(self):
        rows = [sample() for _ in range(40)] + [sample(90, hand_angle=90), sample(closed=False)]
        report = summarize_grasp_samples(rows)
        self.assertAlmostEqual(report["peak_settled_rotation_deg"], 0)
        self.assertTrue(report["segments"][0]["release_observed"])

    def test_grasp_does_not_switch_to_a_closer_neighbor_mid_segment(self):
        rows = [sample() for _ in range(40)] + [sample(5)]
        for row in rows:
            row["objects"].append({"path": "/neighbor", "grasp_position": [1, 0, 1], "quaternion_xyzw": [0, 0, 0, 1]})
        rows[-1]["objects"][0]["grasp_position"] = [0.01, 0, 1]
        rows[-1]["objects"][1]["grasp_position"] = [0, 0, 1]
        report = summarize_grasp_samples(rows)
        self.assertEqual(report["segments"][0]["object_path"], "/pipette")
        self.assertAlmostEqual(report["peak_contact_rotation_deg"], 5)

    def test_fixed_reference_keeps_peak_and_excludes_after_open(self):
        rows = [sample() for _ in range(40)] + [sample(5), sample(1), sample(90, closed=False)]
        self.assertAlmostEqual(summarize_grasp_samples(rows)["peak_settled_rotation_deg"], 5)

    def test_missing_stationary_contact_is_not_zero_slip(self):
        rows = [sample(i) for i in range(40)]
        report = summarize_grasp_samples(rows)
        self.assertIsNone(report["peak_settled_rotation_deg"])
        self.assertFalse(report["segments"][0]["release_observed"])

    def test_episode_reset_does_not_merge_grasps(self):
        rows = [sample() for _ in range(40)] + [sample(30, episode=1) for _ in range(40)]
        report = summarize_grasp_samples(rows)
        self.assertEqual(report["closed_segments"], 2)
        self.assertAlmostEqual(report["peak_settled_rotation_deg"], 0)

    def test_observer_copies_state_and_returns_exact_original_result(self):
        controller = SimpleNamespace(
            _episode_num=0,
            mode="replay",
            robot=SimpleNamespace(_gripper_state=1, gripper=SimpleNamespace(joint_dof_indicies=[7, 8])),
        )
        state = {
            "object_path": "/pipette",
            "object_quaternion": np.array([0, 0, 0, 1.0]),
            "object_grasp_position": np.array([0, 0, 1.0]),
            "gripper_position": np.array([0, 0, 1.0]),
            "physical_hand_quaternion": np.array([0, 0, 0, 1.0]),
            "joint_positions": np.zeros(9),
        }
        before = copy.deepcopy(state)
        result = (object(), False, False)
        calls = []
        emitted = []

        def original(obj, observation):
            calls.append((obj, observation))
            return result

        self.assertIs(observed_step(original, emitted.append)(controller, state), result)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], state)
        for key in state:
            np.testing.assert_equal(state[key], before[key])
        state["object_quaternion"][0] = 1
        self.assertEqual(emitted[0]["objects"][0]["quaternion_xyzw"], [0, 0, 0, 1])
        state["physical_hand_quaternion"][0] = np.nan
        with self.assertRaises(ValueError):
            capture_grasp_sample(controller, state)
