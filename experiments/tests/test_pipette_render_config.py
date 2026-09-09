"""Hydra render presets must expose runnable task/controller keys at root."""

import unittest
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG = ROOT / "config"
EXPERIMENT_CONFIG = ROOT / "experiments" / "config"


class PipetteRenderConfigTests(unittest.TestCase):
    def test_overview_only_changes_camera_and_points_at_workspace(self):
        search_path = f"hydra.searchpath=[file://{EXPERIMENT_CONFIG}]"
        with initialize_config_dir(config_dir=str(PRODUCTION_CONFIG), version_base=None):
            baseline = compose(config_name="level4_four_pipettes_to_rack")
            rendered = compose(
                config_name="level4_four_pipettes_to_rack",
                overrides=[search_path, "+render=pipette_overview_camera"],
            )
        original = OmegaConf.to_container(baseline, resolve=False)
        altered = OmegaConf.to_container(rendered, resolve=False)
        original.pop("cameras")
        cameras = altered.pop("cameras")
        self.assertEqual(original, altered)
        self.assertEqual(len(cameras), 1)
        camera = cameras[0]
        self.assertEqual(camera["prim_path"], "/World/PipetteOverviewCamera")
        self.assertEqual(camera["resolution"], [960, 720])
        q = np.asarray(camera["orientation"])
        forward = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0, 0, -1])
        desired = np.array([-0.08, 0.2, 1.0]) - camera["translation"]
        np.testing.assert_allclose(forward, desired / np.linalg.norm(desired), atol=1e-10)

    def test_render_presets_compose_as_complete_replay_tasks(self):
        search_path = f"hydra.searchpath=[file://{PRODUCTION_CONFIG}]"
        with initialize_config_dir(config_dir=str(EXPERIMENT_CONFIG), version_base=None):
            for level, kind in ((2, "pipette_rack"), (4, "pipette_rack_sequence")):
                name = "level2_pipette_to_rack" if level == 2 else "level4_four_pipettes_to_rack"
                with self.subTest(name=name):
                    cfg = compose(config_name="render/" + name, overrides=[search_path])
                    self.assertEqual(cfg.mode, "replay")
                    self.assertEqual(cfg.task_type, kind)
                    self.assertEqual(cfg.controller_type, kind)
                    self.assertEqual(cfg.max_episodes, 1)
                    self.assertNotIn("render", cfg)
                    if level == 4:
                        self.assertEqual(len(cfg.task.pipette_paths), 4)
                        self.assertIn("tall_four_guide36", cfg.usd_path)
