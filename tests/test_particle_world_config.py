"""Exercise actual clock setup methods without launching SimulationApp."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_clock_methods(scope: dict) -> None:
    source = ast.parse((ROOT / "utils/particle_fluid.py").read_text())
    configure = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "configure_particle_world"
    )
    source = ast.parse((ROOT / "utils/pour_liquid.py").read_text())
    owner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PourLiquid")
    owner.body = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "configure_world"]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[configure, owner], type_ignores=[])),
            "actual_particle_clock_methods",
            "exec",
        ),
        scope,
    )


class ParticleWorldConfigTests(unittest.TestCase):
    def setUp(self):
        self.world = Mock()
        self.world.is_playing.return_value = False
        self.stage = Mock()
        self.prim = Mock()
        self.prim.IsA.return_value = True
        self.stage.Traverse.return_value = [self.prim]
        self.api = Mock()
        self.scope = {
            "Any": Any,
            "np": np,
            "omni": SimpleNamespace(physx=Mock()),
            "UsdPhysics": SimpleNamespace(Scene=object()),
            "PhysxSchema": SimpleNamespace(PhysxSceneAPI=SimpleNamespace(Apply=Mock(return_value=self.api))),
        }
        load_clock_methods(self.scope)

    def test_default_clock_preserves_other_pour_tasks(self):
        self.scope["PourLiquid"].configure_world(self.world, self.stage)
        self.world.set_simulation_dt.assert_called_once_with(physics_dt=1 / 480, rendering_dt=1 / 60)
        self.api.CreateTimeStepsPerSecondAttr.assert_called_once_with(480)

    def test_task_override_changes_substeps_not_control_or_render_clock(self):
        cfg = OmegaConf.create({"liquid": {"physics_dt": 1 / 960}})
        self.scope["PourLiquid"].configure_world(self.world, self.stage, cfg)
        self.world.set_simulation_dt.assert_called_once_with(physics_dt=1 / 960, rendering_dt=1 / 60)
        self.api.CreateTimeStepsPerSecondAttr.assert_called_once_with(960)

    def test_absent_override_preserves_default(self):
        self.scope["PourLiquid"].configure_world(self.world, self.stage, OmegaConf.create({"liquid": {}}))
        self.world.set_simulation_dt.assert_called_once_with(physics_dt=1 / 480, rendering_dt=1 / 60)

    def test_invalid_clock_rejected_before_mutating_world(self):
        for dt in (0, -1, float("nan"), float("inf"), 1 / 90, 1):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.scope["PourLiquid"].configure_world(
                    self.world, self.stage, OmegaConf.create({"liquid": {"physics_dt": dt}})
                )
        self.world.stop.assert_not_called()
        self.world.set_simulation_dt.assert_not_called()

    def test_running_world_is_rejected(self):
        self.world.is_playing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "before simulation starts"):
            self.scope["PourLiquid"].configure_world(self.world, self.stage)
        self.world.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
