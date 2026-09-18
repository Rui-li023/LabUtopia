"""The stock smoke-test YAML copies must keep the physical-liquid success gate."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


class PourLiquidConfigTests(unittest.TestCase):
    def test_all_pour_configs_enable_liquid_even_in_smoke_tests(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("single_config_runner", root / "tests/test_single_config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = (
            "level1_pour",
            "level2_pour_liquid",
            "level2_pour_liquid_v2",
            "level3_pour_liquid",
            "level4_clean_beaker",
            "level4_liquid_mixing",
            "level4_open_transport_pour",
            "level5_close_pour",
            "level5_close_pour_gr00t",
            "level5_close_pour_lingbot",
            "level5_close_pour_openpi",
            "level5_close_pour_smolvla",
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "test.yaml"
                module.modify_config(root / "config" / f"{name}.yaml", destination)
                cfg = yaml.safe_load(destination.read_text())
                self.assertTrue(cfg["liquid"]["enabled"])
                self.assertEqual(cfg["liquid"]["min_target_fraction"], 0.8)
                self.assertEqual(cfg["liquid"]["max_outside_fraction"], 0.05)
                self.assertEqual(cfg["liquid"]["settle_steps"], 60)
                self.assertTrue(cfg["collector"]["record_precise_replay_actions"])
                self.assertEqual(cfg["collector"]["type"], "mock")
                self.assertEqual(cfg["mode"], "collect")
                self.assertTrue((root / cfg["usd_path"]).is_file())
                self.assertGreaterEqual(cfg["task"]["max_steps"], 2400)
                original = yaml.safe_load((root / "config" / f"{name}.yaml").read_text())
                self.assertEqual(cfg["cameras"], original["cameras"])


if __name__ == "__main__":
    unittest.main()
