"""De-instancing must fix the whole robot, not alter assets or scene physics."""

import unittest
from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from utils.robot_rendering import prepare_robot_visuals


class RobotRenderingTests(unittest.TestCase):
    def fixture(self):
        asset = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(asset, "/Link").GetPrim()
        asset.SetDefaultPrim(root)
        mesh = UsdGeom.Mesh.Define(asset, "/Link/geometry")
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        stage = Usd.Stage.CreateInMemory()
        for robot in ("Arm", "Other"):
            UsdGeom.Xform.Define(stage, f"/World/{robot}")
            for i in range(9):
                link = stage.DefinePrim(f"/World/{robot}/link{i}")
                link.GetReferences().AddReference(asset.GetRootLayer().identifier)
                link.SetInstanceable(True)
        return asset, stage

    def test_all_links_visible_geometry_and_physics_preserved(self):
        asset, stage = self.fixture()
        original = asset.GetRootLayer().ExportToString()
        metadata = prepare_robot_visuals(stage, SimpleNamespace(is_stopped=lambda: True), "/World/Arm")
        self.assertEqual(len(metadata["changed_paths"]), 9)
        for i in range(9):
            link = stage.GetPrimAtPath(f"/World/Arm/link{i}")
            self.assertFalse(link.IsInstance())
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/Arm/link{i}/geometry"))
            self.assertEqual(len(mesh.GetPointsAttr().Get()), 3)
            self.assertEqual(mesh.ComputeVisibility(), "inherited")
            self.assertTrue(UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get())
            self.assertTrue(stage.GetPrimAtPath(f"/World/Other/link{i}").IsInstance())
        self.assertEqual(asset.GetRootLayer().ExportToString(), original)
        repeated = prepare_robot_visuals(stage, SimpleNamespace(is_stopped=lambda: True), "/World/Arm")
        self.assertEqual(repeated["changed_paths"], [])

    def test_rejects_running_world_and_broad_or_missing_paths(self):
        asset, stage = self.fixture()
        original = stage.GetRootLayer().ExportToString()
        with self.assertRaises(RuntimeError):
            prepare_robot_visuals(stage, SimpleNamespace(is_stopped=lambda: False), "/World/Arm")
        for path in ("/", "/World", "relative", "/World/Missing"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                prepare_robot_visuals(stage, SimpleNamespace(is_stopped=lambda: True), path)
        self.assertEqual(stage.GetRootLayer().ExportToString(), original)


if __name__ == "__main__":
    unittest.main()
