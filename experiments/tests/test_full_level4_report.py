"""Full-task acceptance must count every attempt and require all six phases."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from experiments.liquid.evaluate_conical_pairs import run_child
from experiments.liquid.evaluate_full_level4 import summarize_full_run, update_counts


class FullLevel4ReportTests(unittest.TestCase):
    def test_four_of_five_accepts_only_after_all_finish(self):
        report = {
            "planned_seeds": list(range(5)),
            "required_successes": 4,
            "status": "running",
            "rows": [{"status": "complete", "collect": {"success": True}} for _ in range(4)],
        }
        update_counts(report)
        self.assertFalse(report["accepted"])
        report["rows"].append({"status": "complete", "collect": {"success": False}})
        report["status"] = "complete"
        update_counts(report)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["counts"]["completed_success_rate"], 0.8)
        self.assertEqual(report["counts"]["failures"], 1)
        report["rows"][0]["collect"]["success"] = False
        update_counts(report)
        self.assertFalse(report["accepted"])

    def test_interruption_is_unfinished_not_success(self):
        report = {
            "planned_seeds": list(range(5)),
            "required_successes": 4,
            "status": "interrupted",
            "rows": [{"status": "complete", "collect": {"success": True}} for _ in range(4)]
            + [{"status": "interrupted"}],
        }
        update_counts(report)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["counts"]["unfinished"], 1)

    def test_success_requires_full_scope_and_all_phases(self):
        phases = ["opening", "picking1", "transporting", "picking2", "pouring", "transporting2"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            (root / "run.log").write_text(f"Generated random task sequence: {phases}\n")
            for observed, component, expected in (
                (phases, False, True),
                (phases[3:], False, False),
                (phases, True, False),
            ):
                rows = [
                    {
                        "phase": phase,
                        "component_only": component,
                        "kinematic_attachment": None,
                        "done": phase == "finished",
                        "success": phase == "finished",
                    }
                    for phase in [*observed, "finished"]
                ]
                (root / "run/conical_telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                with self.subTest(observed=observed, component=component):
                    self.assertEqual(summarize_full_run(root)["success"], expected)

    def test_full_runner_disables_component_shortcut_and_limits_attempts(self):
        cfg = OmegaConf.create({"multi_run": {"run_dir": ""}, "hydra": {"run": {"dir": ""}}})
        child = Mock()
        child.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("experiments.liquid.evaluate_conical_pairs.subprocess.Popen", return_value=child) as launch,
        ):
            run_child(cfg, Path(temporary) / "collect", component=False, timeout=1)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--max-attempts") + 1], "1")
            self.assertEqual(command[command.index("--max-episodes") + 1], "1")
            self.assertEqual(launch.call_args.kwargs["env"]["CONICAL_ONLY"], "0")
            self.assertNotIn("CONICAL_PHYSICS_ONLY", launch.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
