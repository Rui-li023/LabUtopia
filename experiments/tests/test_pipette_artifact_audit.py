"""Artifact audits must not hide partial traces, missing tools, or corrupt rows."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from experiments.pipette.audit_pipette_artifacts import audit_trace, audit_trajectory


def trace():
    sample = {
        "type": "sample",
        "mode": "replay",
        "episode": 0,
        "replay_action_step": 0,
        "closed_command": True,
        "tcp": [0, 0, 1],
        "finger_positions": [0.014, 0.014],
        "hand_quaternion_xyzw": [0, 0, 0, 1],
        "done": False,
        "success": False,
        "objects": [{"path": "/tool", "grasp_position": [0, 0, 1], "quaternion_xyzw": [0, 0, 0, 1]}],
    }
    rows = [dict(copy.deepcopy(sample), replay_action_step=i) for i in range(40)]
    rows.append(dict(copy.deepcopy(sample), replay_action_step=40, closed_command=False, done=True, success=True))
    return [
        {"type": "metadata", "read_only_observer": True, "source_sha256": {"observer.py": "hash"}},
        *rows,
        {"type": "shutdown_requested", "last_step_terminal": True},
    ]


RUN = {"valid_run": True, "exit_code": 0, "attempt_count": 1, "success_count": 1}
HASHES = {"observer.py": "hash"}


class ArtifactAuditTests(unittest.TestCase):
    def test_complete_trace_checks_without_asserting_acceptance(self):
        report = audit_trace(trace(), RUN, ["/tool"], HASHES)
        self.assertTrue(report["checks_passed"])
        self.assertNotIn("accepted", report)
        self.assertEqual(report["grasp_report"]["closed_segments"], 1)

    def test_partial_duplicate_missing_tool_and_changed_observer_fail(self):
        rows = trace()
        duplicate = [*rows[:-1], rows[-2], rows[-1]]
        for records, paths, hashes in (
            (rows[:-1], ["/tool"], HASHES),
            (duplicate, ["/tool"], HASHES),
            (rows, ["/tool", "/missing"], HASHES),
            (rows, ["/tool"], {"observer.py": "changed"}),
        ):
            with self.subTest(paths=paths, length=len(records), hashes=hashes):
                self.assertFalse(audit_trace(records, RUN, paths, hashes)["checks_passed"])

    def test_unknown_stationary_reference_stays_null(self):
        rows = trace()
        for index, row in enumerate(rows[1:-2]):
            angle = index * 0.01
            row["objects"][0]["quaternion_xyzw"] = [0, 0, np.sin(angle / 2), np.cos(angle / 2)]
        report = audit_trace(rows, RUN, ["/tool"], HASHES)
        self.assertTrue(report["checks_passed"])
        self.assertIsNone(report["grasp_report"]["peak_settled_rotation_deg"])
        self.assertGreater(report["grasp_report"]["peak_contact_rotation_deg"], 0)

    def test_finite_rows_binary_commands_and_decoded_video_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0000.h5"
            (path.parent / "image.mp4").touch()
            with h5py.File(path, "w") as dataset:
                dataset["actions"] = np.zeros((3, 8))
                dataset["agent_pose"] = np.zeros((3, 8))
                dataset["init_state/object_pose_positions"] = np.zeros((2, 3))
                dataset["init_state/object_pose_orientations"] = np.tile([0, 0, 0, 1], (2, 1))
            with patch(
                "experiments.pipette.audit_pipette_artifacts.probe_video",
                return_value={"width": 256, "height": 256, "nb_frames": 3},
            ):
                self.assertTrue(audit_trajectory(path, [256, 256])["checks_passed"])
                with h5py.File(path, "r+") as dataset:
                    dataset["actions"][0, -1] = 0.5
                    dataset["agent_pose"][1, 0] = float("nan")
                report = audit_trajectory(path, [256, 256])
                self.assertFalse(report["checks_passed"])
                self.assertFalse(report["checks"]["binary_gripper_commands"])
                self.assertFalse(report["checks"]["finite_matching_8d_rows"])
            with patch(
                "experiments.pipette.audit_pipette_artifacts.probe_video",
                return_value={"width": 256, "height": 256, "nb_frames": 2},
            ):
                self.assertFalse(audit_trajectory(path, [256, 256])["checks"]["videos_match_actions"])
