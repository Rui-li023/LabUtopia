"""Versioned replay snapshots preserve state and drive targets without stepping."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pxr import Gf, Usd, UsdGeom


def load_snapshot_module():
    source = Path(__file__).resolve().parents[1] / "utils/replay_physics_snapshot.py"
    spec = importlib.util.spec_from_file_location("_replay_snapshot_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "isaacsim.core.prims": SimpleNamespace(SingleRigidPrim=Mock(), SingleArticulation=Mock()),
            "isaacsim.core.utils.types": SimpleNamespace(ArticulationAction=SimpleNamespace),
        },
    ):
        spec.loader.exec_module(module)
    return module


snapshot_io = load_snapshot_module()
TRANSLATION_OPS = snapshot_io._translation_ops


class ReplayPhysicsSnapshotTests(unittest.TestCase):
    def setUp(self):
        translation_patch = patch.object(snapshot_io, "_translation_ops", return_value={})
        translation_patch.start()
        self.addCleanup(translation_patch.stop)
        self.robot = Mock()
        self.robot._gripper_control_mode = "position"
        self.robot.get_world_pose.return_value = (np.array([0, 0, 0.71]), np.array([1, 0, 0, 0]))
        self.robot.get_joint_positions.return_value = np.arange(9) / 10
        self.robot.get_joint_velocities.return_value = np.arange(9) / 100
        self.robot.get_applied_action.return_value = SimpleNamespace(
            joint_positions=np.arange(9) / 11,
            joint_velocities=np.arange(9) / 101,
            joint_efforts=np.arange(9) / 13,
        )
        self.body = Mock()
        self.body.get_world_pose.return_value = (np.array([0.2, 0.1, 0.8]), np.array([0.7, 0.7, 0, 0]))
        self.body.get_linear_velocity.return_value = np.array([0, 0, -0.01])
        self.body.get_angular_velocity.return_value = np.array([0.02, 0, 0])
        self.task = SimpleNamespace(
            robot=self.robot,
            world=Mock(),
            _object_rigid_body_offsets={"/World/flask": [("/World/flask", None)]},
        )

    def capture(self):
        with patch.object(snapshot_io, "SingleRigidPrim", return_value=self.body):
            return snapshot_io.capture_pre_warmup(self.task)

    def test_scene_door_uses_articulation_joint_state_not_link_teleports(self):
        self.task.LIQUID_REPLAY_ARTICULATIONS = ("/World/door",)
        self.task._object_rigid_body_offsets["/World/door"] = [("/World/door/handle", None)]
        door = Mock()
        door.get_world_pose.return_value = ([0.7, 0.1, 0.78], [1, 0, 0, 0])
        door.get_joint_positions.return_value = [-0.1]
        door.get_joint_velocities.return_value = [0.02]
        with patch.object(snapshot_io, "SingleArticulation", return_value=door):
            state = self.capture()
            self.assertNotIn("/World/door/handle", state["rigid_bodies"])
            self.assertEqual(state["articulations"]["/World/door"]["joint_positions"], [-0.1])
            with patch.object(snapshot_io, "SingleRigidPrim", return_value=self.body):
                snapshot_io.restore_pre_warmup(self.task, state)
            np.testing.assert_array_equal(door.set_joint_positions.call_args.args[0], [-0.1])
            np.testing.assert_array_equal(door.set_joint_velocities.call_args.args[0], [0.02])

    def test_capture_has_one_versioned_timing_and_json_roundtrips(self):
        snapshot = json.loads(json.dumps(self.capture(), allow_nan=False))
        self.assertEqual(snapshot["version"], 1)
        self.assertEqual(snapshot["timing"], "pre_warmup")
        self.assertEqual(snapshot["gripper_command"], "absolute_position_v1")
        np.testing.assert_array_equal(snapshot["robot"]["joint_velocities"], np.arange(9) / 100)
        np.testing.assert_array_equal(snapshot["robot"]["drive_positions"], np.arange(9) / 11)
        np.testing.assert_array_equal(snapshot["rigid_bodies"]["/World/flask"]["orientation_wxyz"], [0.7, 0.7, 0, 0])
        self.task.world.step.assert_not_called()
        self.robot.set_joint_positions.assert_not_called()
        self.body.set_world_pose.assert_not_called()

    def test_restore_preserves_velocities_targets_and_rigid_pose_without_stepping(self):
        snapshot = json.loads(json.dumps(self.capture()))
        with patch.object(snapshot_io, "SingleRigidPrim", return_value=self.body):
            snapshot_io.restore_pre_warmup(self.task, snapshot)
        np.testing.assert_array_equal(self.robot.set_joint_positions.call_args.args[0], np.arange(9) / 10)
        np.testing.assert_array_equal(self.robot.set_joint_velocities.call_args.args[0], np.arange(9) / 100)
        action = self.robot.apply_action.call_args.args[0]
        np.testing.assert_array_equal(action.joint_positions, np.arange(9) / 11)
        np.testing.assert_array_equal(action.joint_velocities, np.arange(9) / 101)
        np.testing.assert_array_equal(action.joint_efforts, np.arange(9) / 13)
        np.testing.assert_array_equal(self.body.set_world_pose.call_args.kwargs["orientation"], [0.7, 0.7, 0, 0])
        np.testing.assert_array_equal(self.body.set_linear_velocity.call_args.args[0], [0, 0, -0.01])
        np.testing.assert_array_equal(self.body.set_angular_velocity.call_args.args[0], [0.02, 0, 0])
        self.task.world.step.assert_not_called()

    def test_optional_drive_channels_remain_uncommanded(self):
        self.robot.get_applied_action.return_value.joint_velocities = None
        self.robot.get_applied_action.return_value.joint_efforts = None
        snapshot = self.capture()
        with patch.object(snapshot_io, "SingleRigidPrim", return_value=self.body):
            snapshot_io.restore_pre_warmup(self.task, snapshot)
        action = self.robot.apply_action.call_args.args[0]
        self.assertIsNone(action.joint_velocities)
        self.assertIsNone(action.joint_efforts)

    def test_static_support_local_translation_preserves_double_precision(self):
        position = [0.201234567890123, -0.315678901234567, 0.734177190123456]
        op = Mock()
        op.Get.return_value = Gf.Vec3d(*position)
        with patch.object(snapshot_io, "_translation_ops", return_value={"/World/support": op}):
            snapshot = json.loads(json.dumps(self.capture()))
            with patch.object(snapshot_io, "SingleRigidPrim", return_value=self.body):
                snapshot_io.restore_pre_warmup(self.task, snapshot)
        self.assertEqual(list(op.Set.call_args.args[0]), position)
        self.assertIsInstance(op.Set.call_args.args[0], Gf.Vec3d)
        self.task.world.step.assert_not_called()

    def test_native_usd_pivot_stack_is_preserved(self):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/World/support")
        translate = root.AddTranslateOp()
        position = Gf.Vec3d(0.7897656971731393, 6.127797612104619, 0.7197252837275829)
        translate.Set(position)
        root.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(0.54, 5.93, 0))
        root.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
        root.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
        order = list(root.GetXformOpOrderAttr().Get())
        task = SimpleNamespace(stage=stage, _object_rigid_body_offsets={"/World/support": []})
        ops = TRANSLATION_OPS(task)
        saved = json.loads(json.dumps(list(ops["/World/support"].Get())))
        translate.Set(Gf.Vec3d(0))
        ops["/World/support"].Set(type(ops["/World/support"].Get())(*saved))
        self.assertEqual(translate.Get(), position)
        self.assertEqual(list(root.GetXformOpOrderAttr().Get()), order)

    def test_rejects_unknown_versions_or_body_layout(self):
        snapshot = self.capture()
        snapshot["version"] = 2
        with self.assertRaisesRegex(ValueError, "version"):
            snapshot_io.restore_pre_warmup(self.task, snapshot)
        snapshot["version"] = 1
        snapshot["rigid_bodies"] = {}
        with (
            patch.object(snapshot_io, "SingleRigidPrim", return_value=self.body),
            self.assertRaisesRegex(ValueError, "rigid bodies"),
        ):
            snapshot_io.restore_pre_warmup(self.task, snapshot)
        self.robot.set_joint_positions.assert_not_called()

    def test_rejects_unsupported_actuation_and_nonfinite_capture(self):
        self.robot._gripper_control_mode = "force"
        with self.assertRaisesRegex(ValueError, "position grippers"):
            self.capture()
        self.robot._gripper_control_mode = "position"
        self.robot.get_joint_velocities.return_value[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite vectors"):
            self.capture()


if __name__ == "__main__":
    unittest.main()
