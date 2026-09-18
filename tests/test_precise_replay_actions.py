"""Preserve physical command precision without changing the policy HDF5 schema."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_collectors.data_collector import DataCollector, _write_episode_data
from data_collectors.mobile_data_collector import MobileDataCollector
from utils.replay_data_loader import ReplayDataLoader, ReplayIntegrityError


class PreciseReplayActionsTests(unittest.TestCase):
    def test_mobile_collector_preserves_precise_channel_and_phases(self):
        with tempfile.TemporaryDirectory() as root:
            collector = MobileDataCollector([], save_dir=root, max_workers=1, record_precise_replay_actions=True)
            try:
                command = np.zeros(11, dtype=np.float64)
                command[10] = 1 - float(np.float32(0.014)) / 0.05
                collector.cache_step({}, np.zeros(11), action=command, phase=5)
                extra = collector._extra_datasets()
                self.assertEqual(extra["replay_actions"].dtype, np.dtype("float64"))
                np.testing.assert_array_equal(extra["replay_actions"][0], command)
                np.testing.assert_array_equal(extra["phase"], [5])
            finally:
                collector.close()

    def record(self, root, precise):
        collector = DataCollector([], save_dir=root, max_workers=1, record_precise_replay_actions=precise)
        try:
            target = np.float32(0.014)
            command = np.zeros(8, dtype=np.float64)
            command[7] = 1.0 - float(target) / 0.05
            collector.cache_step({}, np.zeros(9), action=command)
            directory = Path(root) / "dataset/episode_0000"
            _write_episode_data(
                str(directory),
                "episode_0000",
                {},
                np.asarray(collector.temp_agent_pose),
                np.asarray(collector.temp_actions),
                extra_datasets=collector._extra_datasets(),
            )
            return directory / "episode_0000.h5", target, command
        finally:
            collector.close()

    def test_precise_channel_roundtrips_to_identical_float32_drive_target(self):
        with tempfile.TemporaryDirectory() as root:
            path, target, command = self.record(root, True)
            with h5py.File(path) as data:
                self.assertEqual(data["actions"].dtype, np.dtype("float32"))
                self.assertEqual(data["agent_pose"].dtype, np.dtype("float32"))
                self.assertEqual(data["replay_actions"].dtype, np.dtype("float64"))
                self.assertEqual(data["replay_action_version"][()], 1)
                lossy = np.float32(0.05 + float(data["actions"][0, 7]) * -0.05)
                self.assertNotEqual(lossy, target)
            loaded = ReplayDataLoader(str(Path(root) / "dataset"))[0]
            np.testing.assert_array_equal(loaded.actions[0], command)
            restored = np.float32(0.05 + loaded.actions[0, 7] * -0.05)
            self.assertEqual(restored, target)

    def test_legacy_float32_actions_are_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, command = self.record(root, False)
            with h5py.File(path) as data:
                self.assertNotIn("replay_actions", data)
            loaded = ReplayDataLoader(str(Path(root) / "dataset"))[0]
            self.assertEqual(loaded.actions.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(loaded.actions[0], command.astype(np.float32))

    def test_precise_recording_rejects_missing_actions(self):
        with tempfile.TemporaryDirectory() as root:
            collector = DataCollector([], save_dir=root, record_precise_replay_actions=True)
            try:
                collector.cache_step({}, np.zeros(9))
                with self.assertRaisesRegex(ValueError, "every observation"):
                    collector._extra_datasets()
            finally:
                collector.close()

    def test_invalid_precise_records_fail_instead_of_dropping_episodes(self):
        for invalid in ("version", "shape", "nan", "dtype"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as root:
                path, _, _ = self.record(root, True)
                with h5py.File(path, "a") as data:
                    if invalid == "version":
                        data["replay_action_version"][()] = 2
                    elif invalid == "nan":
                        data["replay_actions"][0, 7] = np.nan
                    else:
                        del data["replay_actions"]
                        data.create_dataset(
                            "replay_actions",
                            data=np.zeros(
                                (2, 8) if invalid == "shape" else (1, 8),
                                dtype=np.float64 if invalid == "shape" else np.float32,
                            ),
                        )
                with self.assertRaises(ReplayIntegrityError):
                    ReplayDataLoader(str(Path(root) / "dataset"))


if __name__ == "__main__":
    unittest.main()
