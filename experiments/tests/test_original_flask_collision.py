"""The flask support fix preserves original visuals and uses real mesh surfaces."""

import unittest
from pathlib import Path

import numpy as np

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD bindings unavailable")
class OriginalFlaskCollisionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002"
        self.original = Usd.Stage.Open(str(root / "lab_002.usd"))
        self.fixed = Usd.Stage.Open(str(root / "flask_support_collision/lab_002_flask_support.usda"))

    def test_original_meshes_materials_transforms_and_dynamics_unchanged(self):
        for root in ("round_bottom_flask", "pipeclay_triangle", "cork_ring"):
            for prim in Usd.PrimRange(self.original.GetPrimAtPath("/World/" + root)):
                fixed = self.fixed.GetPrimAtPath(prim.GetPath())
                self.assertTrue(fixed.IsValid())
                self.assertEqual(prim.GetTypeName(), fixed.GetTypeName())
                for attr in prim.GetAttributes():
                    name = attr.GetName()
                    if name in {
                        "physics:approximation",
                        "physics:collisionEnabled",
                        "physxCollision:contactOffset",
                        "physxCollision:restOffset",
                    }:
                        continue
                    self.assertEqual(str(attr.Get()), str(fixed.GetAttribute(name).Get()), (str(prim.GetPath()), name))
                for relation in prim.GetRelationships():
                    self.assertEqual(relation.GetTargets(), fixed.GetRelationship(relation.GetName()).GetTargets())

    def test_support_meshes_are_static_exact_and_dynamic_flask_still_convex(self):
        for root in ("pipeclay_triangle", "cork_ring"):
            prim = self.fixed.GetPrimAtPath("/World/" + root + "/mesh_000")
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), "none")
            self.assertIn("PhysxCollisionAPI", prim.GetMetadata("apiSchemas").GetAppliedItems())
            self.assertAlmostEqual(prim.GetAttribute("physxCollision:contactOffset").Get(), 0.0005)
            self.assertFalse(self.fixed.GetPrimAtPath("/World/" + root).HasAPI(UsdPhysics.RigidBodyAPI))
        flask = self.fixed.GetPrimAtPath("/World/round_bottom_flask")
        self.assertTrue(UsdPhysics.RigidBodyAPI(flask).GetRigidBodyEnabledAttr().Get())
        self.assertEqual(
            UsdPhysics.MeshCollisionAPI(self.fixed.GetPrimAtPath(flask.GetPath().AppendChild("mesh_001")))
            .GetApproximationAttr()
            .Get(),
            "convexDecomposition",
        )
        self.assertFalse(UsdPhysics.RigidBodyAPI(flask).GetKinematicEnabledAttr().Get())

    def test_original_y_up_convention_and_seats_match_actual_supported_region(self):
        xf = UsdGeom.XformCache()
        root = self.fixed.GetPrimAtPath("/World/round_bottom_flask")
        axis = root.GetAttribute("reference:upAxis").Get()
        world_axis = xf.GetLocalToWorldTransform(root).TransformDir(Gf.Vec3d(*axis)).GetNormalized()
        self.assertAlmostEqual(world_axis[2], 1)
        for name, height in (("pipeclay_triangle", 0.13892829), ("cork_ring", 0.09598935)):
            root = self.fixed.GetPrimAtPath("/World/" + name)
            b = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(root).ComputeAlignedRange()
            seat = self.fixed.GetPrimAtPath(root.GetPath().AppendChild("FlaskSeat"))
            self.assertAlmostEqual(
                xf.GetLocalToWorldTransform(seat).ExtractTranslation()[2] - b.GetMin()[2], height, places=7
            )

    def test_neck_marker_tracks_original_straight_neck_without_added_geometry(self):
        root = self.fixed.GetPrimAtPath("/World/round_bottom_flask")
        bbox = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(root).ComputeAlignedRange()
        marker = self.fixed.GetPrimAtPath(root.GetPath().AppendChild("NeckGrasp"))
        delta = UsdGeom.XformCache().GetLocalToWorldTransform(marker).ExtractTranslation() - bbox.GetMidpoint()
        self.assertAlmostEqual(delta[0], 0)
        self.assertAlmostEqual(delta[1], 0)
        self.assertAlmostEqual(delta[2], 0.03)
        self.assertFalse(marker.HasAPI(UsdPhysics.CollisionAPI))
        for name in ("round_bottom_flask", "pipeclay_triangle", "cork_ring"):
            meshes = [
                {str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/" + name)) if p.IsA(UsdGeom.Mesh)}
                for stage in (self.original, self.fixed)
            ]
            self.assertEqual(*meshes)

    def test_optional_glass_mass_is_derived_from_original_solid_volume(self):
        stage = Usd.Stage.Open(str(Path(self.fixed.GetRootLayer().realPath).parent / "lab_002_flask_glass_mass.usda"))
        glass = stage.GetPrimAtPath("/World/round_bottom_flask/mesh_001")
        mesh = UsdGeom.Mesh(glass)
        np.testing.assert_array_equal(mesh.GetFaceVertexCountsAttr().Get(), 3)
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(glass)
        points = np.array([matrix.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()])
        faces = (points - points.mean(0))[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
        volume = np.einsum("ij,ij->i", faces[:, 0], np.cross(faces[:, 1], faces[:, 2])).sum() / 6
        self.assertAlmostEqual(volume, glass.GetAttribute("reference:solidMeshVolumeM3").Get())
        self.assertGreater(volume, 0)
        self.assertAlmostEqual(UsdPhysics.MassAPI(glass).GetMassAttr().Get(), volume * 2230)
        # This optional layer may change mass only, not physical/visual surfaces.
        for prim in Usd.PrimRange(self.fixed.GetPrimAtPath("/World/round_bottom_flask")):
            changed = stage.GetPrimAtPath(prim.GetPath())
            for attr in prim.GetAttributes():
                if str(prim.GetPath()) == str(glass.GetPath()) and attr.GetName() == "physics:mass":
                    continue
                self.assertEqual(str(attr.Get()), str(changed.GetAttribute(attr.GetName()).Get()))
            for relation in prim.GetRelationships():
                self.assertEqual(relation.GetTargets(), changed.GetRelationship(relation.GetName()).GetTargets())

    def test_refined_candidate_preserves_all_visual_attributes_and_stays_dynamic(self):
        stage = Usd.Stage.Open(str(Path(self.fixed.GetRootLayer().realPath).parent / "lab_002_flask_physics.usda"))
        for root in ("round_bottom_flask", "pipeclay_triangle", "cork_ring"):
            for prim in Usd.PrimRange(self.fixed.GetPrimAtPath("/World/" + root)):
                changed = stage.GetPrimAtPath(prim.GetPath())
                self.assertEqual(changed.GetTypeName(), prim.GetTypeName())
                for attr in prim.GetAttributes():
                    if not attr.GetName().startswith(("physics:", "physx")):
                        self.assertEqual(str(attr.Get()), str(changed.GetAttribute(attr.GetName()).Get()))
                for relation in prim.GetRelationships():
                    self.assertEqual(relation.GetTargets(), changed.GetRelationship(relation.GetName()).GetTargets())
        root = stage.GetPrimAtPath("/World/round_bottom_flask")
        self.assertTrue(UsdPhysics.RigidBodyAPI(root).GetRigidBodyEnabledAttr().Get())
        self.assertFalse(UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get())
        mesh = stage.GetPrimAtPath(root.GetPath().AppendChild("mesh_001"))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), "convexDecomposition")
        self.assertEqual(mesh.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get(), 128)


if __name__ == "__main__":
    unittest.main()
