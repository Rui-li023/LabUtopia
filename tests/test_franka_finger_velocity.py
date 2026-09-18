"""Both physical fingers are speed limited, and replay cannot mix limits."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import numpy as np

from utils.franka_mimic_force import validate_franka_finger_speed

ROOT = Path(__file__).resolve().parents[1]


def method(path: str, name: str, scope: dict):
    source = ast.parse((ROOT / path).read_text())
    node = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, "exec"), scope)
    return scope[name]


class FrankaFingerVelocityTests(unittest.TestCase):
    def setUp(self):
        self.owner = SimpleNamespace(
            cfg=SimpleNamespace(task=SimpleNamespace(gripper_drive_force=60.0, gripper_max_joint_velocity=0.05)),
            stage=Mock(),
            robot=Mock(),
        )
        self.owner.robot.prim_path = "/World/Franka"
        self.owner.robot.gripper.joint_dof_indicies = [7, 8]
        self.owner.stage.GetPrimAtPath.side_effect = lambda path: path
        self.joint_api = Mock()
        self.scope = {
            "np": np,
            "logger": Mock(),
            "UsdPhysics": Mock(),
            "PhysxSchema": SimpleNamespace(PhysxJointAPI=self.joint_api, PhysxArticulationAPI=Mock()),
            "validate_single_mimic_finger_drive": Mock(),
            "validate_franka_finger_speed": validate_franka_finger_speed,
        }
        self.configure = method("tasks/open_transport_pour_task.py", "_configure_gripper_drive", self.scope)

    def test_both_fingers_limited_but_only_lead_is_driven(self):
        self.configure(self.owner)
        view = self.owner.robot._articulation_view
        np.testing.assert_array_equal(view.set_max_joint_velocities.call_args.args[0], [[0.05, 0.05]])
        np.testing.assert_array_equal(view.set_max_joint_velocities.call_args.kwargs["joint_indices"], [7, 8])
        np.testing.assert_array_equal(view.set_gains.call_args.kwargs["joint_indices"], [7])
        self.assertEqual(
            self.joint_api.Apply.call_args_list,
            [
                call("/World/Franka/panda_hand/panda_finger_joint1"),
                call("/World/Franka/panda_hand/panda_finger_joint2"),
            ],
        )

    def test_invalid_speed_fails_before_drive_writes(self):
        for speed in (0, -1, float("inf"), float("nan"), 0.051, 10):
            self.owner.cfg.task.gripper_max_joint_velocity = speed
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                self.configure(self.owner)
        self.owner.robot._articulation_view.set_gains.assert_not_called()
        self.joint_api.Apply.assert_not_called()

    def test_absent_option_preserves_asset_speed_limits(self):
        del self.owner.cfg.task.gripper_max_joint_velocity
        self.configure(self.owner)
        self.owner.robot._articulation_view.set_max_joint_velocities.assert_not_called()
        self.joint_api.Apply.assert_not_called()


class FingerVelocityReplayTests(unittest.TestCase):
    def setUp(self):
        self.reset = method("utils/pour_liquid.py", "reset", {"Any": Any, "np": np})
        self.owner = SimpleNamespace(
            routes=(),
            cfg=SimpleNamespace(task=SimpleNamespace(gripper_drive_force=60.0, gripper_max_joint_velocity=0.05)),
            spacing=0.004,
            gripper_contact={},
            collision_hash="unchanged-colliders",
            world=SimpleNamespace(get_physics_dt=lambda: 1 / 960),
            fluid=Mock(),
            _seed_points=lambda: np.array([[0.1, -0.4, 0.8]], dtype=np.float32),
        )

    def test_new_snapshot_records_limit_and_replays_unchanged(self):
        snapshot = self.reset(self.owner, source_path="/Source")
        self.assertEqual(snapshot["gripper_max_joint_velocity"], 0.05)
        self.assertEqual(self.reset(self.owner, source_path="/Source", snapshot=snapshot), snapshot)

    def test_other_tasks_keep_legacy_snapshot_without_limit(self):
        del self.owner.cfg.task.gripper_max_joint_velocity
        snapshot = self.reset(self.owner, source_path="/Source")
        self.assertNotIn("gripper_max_joint_velocity", snapshot)
        self.assertEqual(self.reset(self.owner, source_path="/Source", snapshot=snapshot), snapshot)

    def test_different_physics_timestep_is_rejected(self):
        snapshot = self.reset(self.owner, source_path="/Source")
        snapshot["physics_dt"] = 1 / 480
        self.owner.fluid.reset.reset_mock()
        with self.assertRaisesRegex(ValueError, "physics timestep differs"):
            self.reset(self.owner, source_path="/Source", snapshot=snapshot)
        self.owner.fluid.reset.assert_not_called()

    def test_old_or_different_limit_is_rejected_before_fluid_reset(self):
        snapshot = self.reset(self.owner, source_path="/Source")
        self.owner.fluid.reset.reset_mock()
        for old in (
            {k: v for k, v in snapshot.items() if k != "gripper_max_joint_velocity"},
            {**snapshot, "gripper_max_joint_velocity": 10.0},
        ):
            with self.assertRaisesRegex(ValueError, "velocity limit differs"):
                self.reset(self.owner, source_path="/Source", snapshot=old)
        self.owner.fluid.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
