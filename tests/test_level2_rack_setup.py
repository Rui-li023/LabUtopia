"""Exercise the real scene-setup method without starting SimulationApp."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from utils.gripper_contact import PipetteContactParameters


class RackSetupTests(unittest.TestCase):
    def task(self, settings=None, whole_rack_friction=None):
        source = Path(__file__).resolve().parents[1] / "tasks/pipette_rack_task.py"
        tree = ast.parse(source.read_text())
        task_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        setup = next(
            node for node in task_class.body if isinstance(node, ast.FunctionDef) and node.name == "setup_objects"
        )
        base_setup, material_setup = Mock(), Mock()
        base = type("Base", (), {"setup_objects": lambda task: base_setup()})
        task_class.bases = [ast.Name(id="Base", ctx=ast.Load())]
        task_class.body = [setup]
        namespace = {
            "Base": base,
            "os": os,
            "np": np,
            "logger": Mock(),
            "configure_acrylic_rack_contact": material_setup,
            "PipetteContactParameters": PipetteContactParameters,
            "validate_single_mimic_finger_drive": Mock(return_value={}),
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[task_class], type_ignores=[])), str(source), "exec"),
            namespace,
        )
        task = namespace[task_class.name]()
        task.stage = object()
        task.cfg = OmegaConf.create({"task": {"obj_paths": [{"path": "/tool"}, {"path": "/rack"}]}})
        if settings is not None:
            task.cfg.task.rack_contact = settings
        if whole_rack_friction is not None:
            task.cfg.task.obj_paths[1].friction = whole_rack_friction
        return task, base_setup, material_setup

    def test_material_is_applied_after_objects_exist(self):
        task, base, material = self.task({"static_friction": 0.3, "dynamic_friction": 0.2})
        material.side_effect = lambda *args, **kwargs: base.assert_called_once_with()
        with patch.dict(os.environ, {"LABUTOPIA_PIPETTE_CONTACTS": "0"}):
            task.setup_objects()
        material.assert_called_once_with(task.stage, "/rack", static_friction=0.3, dynamic_friction=0.2)

    def test_old_configs_do_not_receive_new_material(self):
        task, base, material = self.task()
        with patch.dict(os.environ, {"LABUTOPIA_PIPETTE_CONTACTS": "0"}):
            task.setup_objects()
        base.assert_called_once_with()
        material.assert_not_called()

    def test_mixed_contact_scopes_fail_before_scene_writes(self):
        task, base, material = self.task({"static_friction": 0.3, "dynamic_friction": 0.3}, 0.3)
        with self.assertRaisesRegex(ValueError, "whole-rack friction"):
            task.setup_objects()
        base.assert_not_called()
        material.assert_not_called()

    def single_drive_task(self):
        task, base, material = self.task()
        task.cfg.robot = {
            "gripper": {"control_mode": "force", "closing_force": 50, "force_semantics": "single_mimic_drive"}
        }
        task.cfg.task.force_gripper_servo = True
        task.robot = Mock(prim_path_str="/World/Franka")
        topology = task.setup_objects.__globals__["validate_single_mimic_finger_drive"]
        return task, base, material, topology

    def test_single_drive_topology_rejected_before_servo_or_contact_changes(self):
        task, base, material, topology = self.single_drive_task()
        topology.side_effect = ValueError("wrong topology")
        with self.assertRaisesRegex(ValueError, "wrong topology"):
            task.setup_objects()
        task.robot.enable_gripper_force_servo.assert_not_called()
        base.assert_not_called()
        material.assert_not_called()

    def test_single_drive_cannot_silently_fall_back_to_raw_effort(self):
        task, base, _, topology = self.single_drive_task()
        task.cfg.task.force_gripper_servo = False
        with self.assertRaisesRegex(ValueError, "bounded velocity"):
            task.setup_objects()
        topology.assert_not_called()
        task.robot.enable_gripper_force_servo.assert_not_called()
        base.assert_not_called()

    def test_single_drive_topology_is_checked_before_enabling_servo(self):
        task, base, _, topology = self.single_drive_task()
        task.robot.enable_gripper_force_servo.side_effect = lambda **kwargs: topology.assert_called_once_with(
            task.stage, "/World/Franka"
        )
        with patch.dict(os.environ, {"LABUTOPIA_PIPETTE_CONTACTS": "0"}):
            task.setup_objects()
        task.robot.enable_gripper_force_servo.assert_called_once()
        base.assert_called_once()
