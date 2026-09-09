import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

from experiments.pipette.evaluate_pipette_pairs import DEFAULT_ACCEPTANCE_THRESHOLD, run_attempt
from experiments.pipette.summarize_pipette_pairs import summarize


def manifest(seed, successes, threshold=0.7):
    return {
        "config": "level4",
        "asset_sha256": "asset",
        "task_config_sha256": "config",
        "code_sha256": {},
        "observation_resolution": [256, 256],
        "threshold": threshold,
        "inputs_unchanged": True,
        "attempts": [
            {
                "collect": {"seed": seed, "attempt_count": 10, "success_count": successes, "valid_run": True},
                "replay": {"success_count": successes, "valid_run": True},
            }
        ],
    }


class PairedEvaluationTests(unittest.TestCase):
    def test_rendered_replay_keeps_observer_and_uses_requested_aspect_ratio(self):
        cfg = OmegaConf.create({"mode": "replay", "max_episodes": 1, "task": {}, "multi_run": {}})
        with tempfile.TemporaryDirectory() as directory:

            def run(command, **kwargs):
                kwargs["stdout"].write("Episode Stats: Success Rate = 1/1\n")
                self.assertIn("experiments.pipette.observe_pipette_grip", command)
                self.assertNotIn("--no-video", command)
                self.assertEqual(command[command.index("--render-width") + 1], "960")
                self.assertEqual(command[command.index("--render-height") + 1], "720")
                return SimpleNamespace(returncode=0)

            with patch("experiments.pipette.evaluate_pipette_pairs.subprocess.run", side_effect=run):
                result = run_attempt(
                    cfg,
                    "replay",
                    123,
                    Path(directory) / "replay",
                    10,
                    resolution=960,
                    observe_grip=True,
                    save_video=True,
                    render_height=720,
                )
            self.assertTrue(result["valid_run"])
            self.assertEqual(result["rendered_videos"], [])  # fake log does not create real video evidence

    def test_current_goal_requires_strictly_over_ninety_percent_for_both_levels(self):
        self.assertEqual(DEFAULT_ACCEPTANCE_THRESHOLD, 0.90)
        batches = [manifest(seed, 9, threshold=0.90) for seed in (1, 2, 3)]
        self.assertFalse(summarize(batches)["accepted"])
        batches[0] = manifest(1, 10, threshold=0.90)
        self.assertTrue(summarize(batches)["accepted"])

    def test_observer_keeps_main_arguments_and_records_trace_path(self):
        cfg = OmegaConf.create({"mode": "replay", "max_episodes": 1, "task": {}, "multi_run": {}})
        with tempfile.TemporaryDirectory() as directory:

            def run(command, **kwargs):
                kwargs["stdout"].write("Episode Stats: Success Rate = 1/1\n")
                self.assertIn("experiments.pipette.observe_pipette_grip", command)
                self.assertIn("--headless", command)
                self.assertIn("--no-video", command)  # default batch behavior stays unchanged
                self.assertEqual(command[command.index("--") + 1], "--config-dir")
                return SimpleNamespace(returncode=0)

            with patch("experiments.pipette.evaluate_pipette_pairs.subprocess.run", side_effect=run):
                result = run_attempt(cfg, "replay", 123, Path(directory) / "replay", 10, observe_grip=True)
            self.assertEqual(result["success_count"], 1)
            self.assertTrue(result["valid_run"])
            self.assertTrue(result["grip_trace"].endswith("grip.jsonl"))

    def test_partial_or_crashed_run_cannot_receive_success_credit(self):
        cfg = OmegaConf.create({"mode": "replay", "max_episodes": 1, "task": {}, "multi_run": {}})
        for exit_code, stats in ((-15, "1/1"), (0, "1/2")):
            with self.subTest(exit_code=exit_code, stats=stats), tempfile.TemporaryDirectory() as directory:

                def run(command, stats=stats, exit_code=exit_code, **kwargs):
                    kwargs["stdout"].write("Episode Stats: Success Rate = " + stats + "\n")
                    return SimpleNamespace(returncode=exit_code)

                with patch("experiments.pipette.evaluate_pipette_pairs.subprocess.run", side_effect=run):
                    result = run_attempt(cfg, "replay", 123, Path(directory) / "replay", 10)
                self.assertFalse(result["valid_run"])
                self.assertEqual(result["success_count"], 0)

    def test_threshold_is_strict_and_failures_remain_denominator(self):
        self.assertFalse(summarize([manifest(1, 7), manifest(2, 7), manifest(3, 7)])["accepted"])
        result = summarize([manifest(1, 8), manifest(2, 7), manifest(3, 7)])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["rates"]["paired_end_to_end"], [22, 30])

    def test_rejects_repeated_seed_or_changed_config(self):
        with self.assertRaisesRegex(ValueError, "Repeated"):
            summarize([manifest(1, 10), manifest(1, 10)])
        other = manifest(2, 10)
        other["task_config_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            summarize([manifest(1, 10), other])

    def test_incomplete_or_small_batches_cannot_pass(self):
        self.assertFalse(summarize([manifest(1, 10)])["accepted"])
        batches = [manifest(i, 10) for i in range(3)]
        incomplete = copy.deepcopy(batches)
        incomplete[1]["attempts"][0]["replay"]["valid_run"] = False
        self.assertFalse(summarize(incomplete)["accepted"])
        batches[1]["inputs_unchanged"] = False
        self.assertFalse(summarize(batches)["accepted"])
