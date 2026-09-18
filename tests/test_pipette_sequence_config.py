"""Three-tool scene contract; no simulator startup or changed asset geometry."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class ThreePipetteConfigTests(unittest.TestCase):
    def test_only_name_and_tool_assignment_differ_from_four_tool_task(self):
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            four = OmegaConf.to_container(compose(config_name="level4_four_pipettes_to_rack"))
            three = OmegaConf.to_container(compose(config_name="level4_three_pipettes_to_rack"))
        self.assertTrue((ROOT / three["usd_path"]).is_file())
        self.assertEqual(three["task"]["pipette_paths"], four["task"]["pipette_paths"][:3])
        self.assertEqual(three["task"]["slot_indices"], [0, 1, 2])
        self.assertEqual(three["task"]["obj_paths"], four["task"]["obj_paths"][:-1])
        four["name"] = three["name"]
        for key in ("pipette_paths", "slot_indices", "obj_paths"):
            four["task"][key] = three["task"][key]
        self.assertEqual(three, four)  # Includes physics, materials and every success tolerance.

    def test_render_and_force_candidate_compose_with_three_tools(self):
        # render/ and physics/ presets live in experiments/config and inherit the
        # production config/ tree through Hydra's search path, the same way
        # `experiments/main.py --config-dir experiments/config` composes them.
        experiments_config = ROOT / "experiments" / "config"
        with initialize_config_dir(config_dir=str(experiments_config), version_base=None):
            overrides = [f"hydra.searchpath=[file://{ROOT / 'config'}]"]
            rendered = compose(config_name="render/level4_three_pipettes_to_rack", overrides=overrides)
            direct = compose(config_name="physics/level4_three_pipettes_to_rack_50n", overrides=overrides)
        # hydra.searchpath never resolves the *primary* config, so the production
        # task has to be the entry point when the physics group is overridden.
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            candidate = compose(
                config_name="level4_three_pipettes_to_rack",
                overrides=[
                    f"hydra.searchpath=[file://{experiments_config}]",
                    "+physics=pipette_single_actuator_50n_candidate",
                ],
            )
        self.assertEqual(rendered.mode, "replay")
        self.assertEqual(len(rendered.task.pipette_paths), 3)
        self.assertEqual(rendered.task_type, "pipette_rack_sequence")
        self.assertEqual(rendered.controller_type, "pipette_rack_sequence")
        self.assertNotIn("render", rendered)
        self.assertEqual(OmegaConf.to_container(direct), OmegaConf.to_container(candidate))
        self.assertEqual(direct.robot.gripper.closing_force, 50)


class SequenceTaskContractTests(unittest.TestCase):
    def task(self, count=3):
        source_path = ROOT / "tasks/pipette_rack_sequence_task.py"
        node = next(n for n in ast.parse(source_path.read_text()).body if isinstance(n, ast.ClassDef))
        base = type("Base", (), {"setup_objects": MagicMock(), "reset_with_init_state": MagicMock()})
        source, sdf = MagicMock(), MagicMock()
        source.GetPrimAtPath.return_value.GetAttribute.return_value.Get.return_value = 4
        namespace = {
            "Any": Any,
            "PipetteRackTask": base,
            "Usd": SimpleNamespace(Stage=SimpleNamespace(Open=lambda _: source)),
            "Sdf": sdf,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
        task = namespace["PipetteRackSequenceTask"]()
        task.cfg = SimpleNamespace(
            usd_path="unused.usda",
            task=SimpleNamespace(pipette_paths=[f"/tool_{i}" for i in range(count)], slot_indices=list(range(count))),
        )
        task.stage = MagicMock()
        return task, base, sdf

    def test_setup_clones_only_configured_extra_bodies(self):
        for count in (3, 4):
            with self.subTest(count=count):
                task, base, sdf = self.task(count)
                task.setup_objects()
                self.assertEqual(sdf.CopySpec.call_count, count - 1)
                self.assertEqual(task.pipette_paths, [f"/tool_{i}" for i in range(count)])
                base.setup_objects.assert_called_once()

    def test_invalid_counts_and_slots_rejected_before_cloning(self):
        for count, slots in ((2, [0, 1]), (5, [0, 1, 2, 3, 4]), (3, [0, 0, 1]), (3, [0, 1, 4]), (3, [0, 1, 2, 3])):
            with self.subTest(count=count, slots=slots):
                task, base, sdf = self.task(count)
                task.cfg.task.slot_indices = slots
                with self.assertRaises(ValueError):
                    task.setup_objects()
                sdf.CopySpec.assert_not_called()
                base.setup_objects.assert_not_called()

    def test_replay_rejects_four_tool_recording_before_any_state_writes(self):
        task, base, _ = self.task()
        task.setup_objects()
        state = {"extra": {"pipette_slots": {f"/tool_{i}": i for i in range(4)}}}
        with self.assertRaisesRegex(ValueError, "assignment differs"):
            task.reset_with_init_state(state)
        base.reset_with_init_state.assert_not_called()
        del state["extra"]["pipette_slots"]["/tool_3"]
        task.reset_with_init_state(state)
        base.reset_with_init_state.assert_called_once_with(state)


if __name__ == "__main__":
    unittest.main()
