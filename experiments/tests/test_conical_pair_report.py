"""Statistics must not drop failed collections or accept truncated replay."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from experiments.liquid.evaluate_conical_pairs import REPO, counts, run_child, summarize_run


class ConicalPairReportTests(unittest.TestCase):
    def summary(self, **updates):
        row = {
            "done": True,
            "success": True,
            "component_only": True,
            "kinematic_attachment": None,
            "replay_actions_consumed": 100,
        }
        row.update(updates)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            (root / "run/conical_telemetry.jsonl").write_text(json.dumps(row) + "\n")
            return summarize_run(root, component=True, expected_actions=100)

    def test_complete_unattached_replay_passes(self):
        self.assertTrue(self.summary()["success"])

    def test_truncated_or_wrong_scope_or_attached_replay_fails(self):
        for changes in (
            {"replay_actions_consumed": 99},
            {"done": False},
            {"success": False},
            {"component_only": False},
            {"kinematic_attachment": "/World/Bottle"},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(self.summary(**changes)["success"])

    def test_failed_collections_remain_in_paired_denominator(self):
        report = counts(
            [
                {"collect": {"success": False}},
                {"collect": {"success": True}, "replay": {"success": True}},
                {"collect": {"success": True}, "replay": {"success": False}},
            ]
        )
        self.assertEqual(report["collect_rate"], 2 / 3)
        self.assertEqual(report["replay_conditional_rate"], 1 / 2)
        self.assertEqual(report["paired_rate"], 1 / 3)
        self.assertEqual(report["paired_attempts"], 3)

    def test_missing_evidence_is_not_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = summarize_run(Path(temporary), component=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure"], "incomplete_run")
        self.assertIsNone(counts([])["paired_rate"])

    def test_absolute_output_uses_hydra_relative_config_path(self):
        cfg = OmegaConf.create({"multi_run": {"run_dir": ""}, "hydra": {"run": {"dir": ""}}})
        child = Mock()
        child.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("experiments.liquid.evaluate_conical_pairs.subprocess.Popen", return_value=child) as launch,
        ):
            directory = Path(temporary) / "collect"
            run_child(cfg, directory, component=True, timeout=1)
            command = launch.call_args.args[0]
            config_path = Path(command[command.index("--config-dir") + 1])
            self.assertFalse(config_path.is_absolute())
            self.assertEqual((REPO / config_path).resolve(), directory)


if __name__ == "__main__":
    unittest.main()
