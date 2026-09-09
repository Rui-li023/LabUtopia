"""USD composition and collision regression checks (requires pxr)."""

import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class GlasswareAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002/level2_glassware/lab_002_glassware.usda"
        )
        cls.stage = Usd.Stage.Open(str(path))

    def test_scene_can_be_referenced_by_default_prim(self):
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/World")
        root.GetReferences().AddReference(self.stage.GetRootLayer().identifier)
        self.assertTrue(stage.GetPrimAtPath("/World/round_bottom_flask/GlassShell"))
        self.assertTrue(stage.GetPrimAtPath("/World/table"))

    def test_flask_has_explicit_mass_and_open_neck(self):
        prim = self.stage.GetPrimAtPath("/World/round_bottom_flask")
        self.assertAlmostEqual(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), 0.08)
        centroid = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()
        self.assertGreater(centroid[2], 0.0)
        self.assertLess(centroid[2], 0.04)
        self.assertFalse(self.stage.GetPrimAtPath(str(prim.GetPath()) + "/Group").IsActive())
        colliders = [p for p in prim.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual(len(colliders), 25)
        for p in colliders:
            if p.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(p).GetPointsAttr().Get()
                # Each neck sector stays away from the open central bore.
                self.assertTrue(all((v[0] ** 2 + v[1] ** 2) ** 0.5 > 0.012 for v in points))

    def test_supports_have_colliders_and_seat_markers(self):
        for name in ("cork_ring", "pipeclay_triangle"):
            with self.subTest(name=name):
                prim = self.stage.GetPrimAtPath("/World/" + name)
                self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
                self.assertTrue(any(p.HasAPI(UsdPhysics.CollisionAPI) for p in prim.GetChildren()))
                self.assertTrue(self.stage.GetPrimAtPath(str(prim.GetPath()) + "/FlaskSeat"))

    def test_glass_shell_is_closed_manifold(self):
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath("/World/round_bottom_flask/GlassShell"))
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        edges = {}
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[cursor : cursor + count]
            cursor += count
            for a, b in zip(face, [*face[1:], face[0]]):
                edge = tuple(sorted((a, b)))
                edges[edge] = edges.get(edge, 0) + 1
        self.assertTrue(edges)
        self.assertEqual(set(edges.values()), {2})


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class ReferencePipetteAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002/realistic_pipette"
        cls.stage = Usd.Stage.Open(str(cls.directory / "lab_002_pipette.usda"))

    def test_published_rack_envelope_and_slots(self):
        rack = self.stage.GetPrimAtPath("/World/pipette_rack")
        bounds = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(rack).ComputeAlignedRange()
        for actual, expected in zip(bounds.GetSize(), (0.172, 0.296, 0.206)):
            self.assertAlmostEqual(actual, expected, delta=0.001)
        for actual, expected in zip(rack.GetAttribute("reference:slotWidths").Get(), (0.028, 0.020)):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertFalse(self.stage.GetPrimAtPath("/World/pipette_01/CollarCollision"))

    def test_export_preserves_complete_uv_and_normal_topology(self):
        for root in ("pipette_01", "pipette_rack"):
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/" + root)):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                with self.subTest(path=str(prim.GetPath())):
                    mesh = UsdGeom.Mesh(prim)
                    count = len(mesh.GetFaceVertexIndicesAttr().Get())
                    self.assertEqual(len(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").ComputeFlattened()), count)
                    self.assertEqual(len(mesh.GetNormalsAttr().Get()), count)
                    self.assertTrue(mesh.GetDoubleSidedAttr().Get())
                    self.assertTrue(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])

    def test_pbr_maps_are_portable_and_exist(self):
        maps = set()
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/RealisticPipetteMaterials")):
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.path.endswith(".png"):
                    self.assertFalse(Path(value.path).is_absolute())
                    self.assertTrue((self.directory / value.path).is_file())
                    maps.add(Path(value.path).name)
        self.assertEqual(
            maps,
            {
                "pipette_basecolor.png",
                "pipette_roughness.png",
                "pipette_normal.png",
                "acrylic_roughness.png",
                "acrylic_normal.png",
            },
        )

    def test_rack_feet_are_not_inside_bench_collider(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "config/level2_pipette_to_rack.yaml")
        bench = (
            UsdGeom.BBoxCache(0, ["default"])
            .ComputeWorldBound(self.stage.GetPrimAtPath("/World/Cube"))
            .ComputeAlignedRange()
        )
        self.assertGreaterEqual(cfg.task.obj_paths[1].position_range.z[0], bench.GetMax()[2])

    def test_upper_and_lower_slot_centres_remain_open(self):
        for name, x in (("Upper_28mm_slots", -0.019), ("Lower_20mm_slots", -0.036)):
            mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath("/World/pipette_rack/" + name))
            points = np.asarray(mesh.GetPointsAttr().Get())
            faces = points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
            a = faces[:, 0, :2]
            b, c = faces[:, 1, :2] - a, faces[:, 2, :2] - a
            delta = np.array([x, 0.1125]) - a
            determinant = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
            nonvertical = np.abs(determinant) > 1e-12
            u, v = np.zeros(len(faces)), np.zeros(len(faces))
            u[nonvertical] = (delta[:, 0] * c[:, 1] - delta[:, 1] * c[:, 0])[nonvertical] / determinant[nonvertical]
            v[nonvertical] = (b[:, 0] * delta[:, 1] - b[:, 1] * delta[:, 0])[nonvertical] / determinant[nonvertical]
            self.assertFalse(np.any(nonvertical & (u >= 0) & (v >= 0) & (u + v <= 1)), name)

    def test_pipette_contact_uses_visual_mesh_not_added_support(self):
        body = self.stage.GetPrimAtPath("/World/pipette_01/Body")
        self.assertIn("PhysxCollisionAPI", body.GetMetadata("apiSchemas").GetAppliedItems())
        self.assertEqual(UsdPhysics.MeshCollisionAPI(body).GetApproximationAttr().Get(), "convexDecomposition")
        self.assertAlmostEqual(body.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Get(), 0.5)
        material, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial(materialPurpose="physics")
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        self.assertAlmostEqual(physics.GetStaticFrictionAttr().Get(), 0.9, places=6)
        self.assertAlmostEqual(physics.GetDynamicFrictionAttr().Get(), 0.7, places=6)
        self.assertAlmostEqual(body.GetAttribute("physxCollision:torsionalPatchRadius").Get(), 0.003, places=6)


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class EnlargedPipetteAssetTests(ReferencePipetteAssetTests):
    @classmethod
    def setUpClass(cls):
        parent = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002"
        cls.directory = parent / "realistic_pipette_260mm"
        cls.stage = Usd.Stage.Open(str(cls.directory / "lab_002_pipette.usda"))
        cls.original = Usd.Stage.Open(str(parent / "realistic_pipette/lab_002_pipette.usda"))

    def test_body_is_uniformly_enlarged_without_changing_uvs(self):
        old = self.original.GetPrimAtPath("/World/pipette_01/Body")
        new = self.stage.GetPrimAtPath("/World/pipette_01/Body")
        points = np.asarray(UsdGeom.Mesh(new).GetPointsAttr().Get())
        np.testing.assert_allclose(points, np.asarray(UsdGeom.Mesh(old).GetPointsAttr().Get()) * (260 / 220), atol=1e-7)
        self.assertAlmostEqual(float(np.ptp(points[:, 2])), 0.260, delta=0.0001)
        # Rebuilt bevel UV interpolation can vary by <0.2 texel on a 2K map;
        # check unchanged layout, not bit-identical Blender float arithmetic.
        np.testing.assert_allclose(
            UsdGeom.PrimvarsAPI(new).GetPrimvar("st").ComputeFlattened(),
            UsdGeom.PrimvarsAPI(old).GetPrimvar("st").ComputeFlattened(),
            atol=1e-4,
            rtol=0,
        )

    def test_grasp_and_reference_markers_follow_new_size(self):
        for name, expected in (
            ("Grasp", np.array([-0.006, 0, 0.190]) * (260 / 220)),
            ("Reference", [0, 0, 0.130]),
        ):
            point = self.stage.GetPrimAtPath("/World/pipette_01/" + name).GetAttribute("xformOp:translate").Get()
            np.testing.assert_allclose(point, expected)

    def test_stand_visual_and_collision_meshes_were_not_enlarged(self):
        for prim in self.original.GetPrimAtPath("/World/pipette_rack").GetChildren():
            if prim.IsA(UsdGeom.Mesh):
                np.testing.assert_array_equal(
                    UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                    UsdGeom.Mesh(self.stage.GetPrimAtPath(prim.GetPath())).GetPointsAttr().Get(),
                )


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class FourSlotPipetteStandTests(unittest.TestCase):
    def test_four_slots_have_physical_finger_clearance_without_resizing_pipette(self):
        root = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002"
        stage = Usd.Stage.Open(str(root / "realistic_pipette_260mm_tall_four_guide36/lab_002_pipette.usda"))
        original = Usd.Stage.Open(str(root / "realistic_pipette_260mm_tall_wide_guide36/lab_002_pipette.usda"))
        rack = stage.GetPrimAtPath("/World/pipette_rack")
        self.assertEqual(rack.GetAttribute("reference:slotCount").Get(), 4)
        self.assertAlmostEqual(rack.GetAttribute("reference:slotPitch").Get(), 0.065)
        self.assertFalse(stage.GetPrimAtPath("/World/pipette_rack/Seat4"))
        ys = [
            stage.GetPrimAtPath(f"/World/pipette_rack/Seat{i}").GetAttribute("xformOp:translate").Get()[1]
            for i in range(4)
        ]
        np.testing.assert_allclose(np.diff(ys), 0.065)
        for attribute in ("points", "faceVertexIndices"):
            np.testing.assert_array_equal(
                stage.GetPrimAtPath("/World/pipette_01/Body").GetAttribute(attribute).Get(),
                original.GetPrimAtPath("/World/pipette_01/Body").GetAttribute(attribute).Get(),
            )
        np.testing.assert_allclose(
            stage.GetPrimAtPath("/World/pipette_01/Body").GetAttribute("primvars:st").Get(),
            original.GetPrimAtPath("/World/pipette_01/Body").GetAttribute("primvars:st").Get(),
            atol=1e-4,
            rtol=0,
        )
        for prim in rack.GetChildren():
            if prim.IsA(UsdGeom.Mesh):
                self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
                self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), "none")


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class TallPipetteStandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002"
        cls.stage = Usd.Stage.Open(str(cls.root / "realistic_pipette_260mm_tall/lab_002_pipette.usda"))

    def test_body_was_not_deformed_to_fit_stand(self):
        old = Usd.Stage.Open(str(self.root / "realistic_pipette_260mm/lab_002_pipette.usda"))
        for attr in ("points", "faceVertexCounts", "faceVertexIndices"):
            np.testing.assert_allclose(
                self.stage.GetPrimAtPath("/World/pipette_01/Body").GetAttribute(attr).Get(),
                old.GetPrimAtPath("/World/pipette_01/Body").GetAttribute(attr).Get(),
                atol=1e-7,
            )

    def test_sequence_clone_retains_a_typed_dynamic_root(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")
        path = "/World/sequence_pipette_02"
        self.assertTrue(
            Sdf.CopySpec(self.stage.GetRootLayer(), Sdf.Path("/World/pipette_01"), stage.GetRootLayer(), Sdf.Path(path))
        )
        spec = stage.GetRootLayer().GetPrimAtPath(path)
        spec.specifier = Sdf.SpecifierDef
        spec.typeName = "Xform"
        prim = stage.GetPrimAtPath(path)
        self.assertTrue(prim.IsA(UsdGeom.Xform))
        self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(stage.GetPrimAtPath(path + "/Body").HasAPI(UsdPhysics.CollisionAPI))

    def test_all_six_slots_have_distinct_seat_markers(self):
        points = [
            self.stage.GetPrimAtPath(f"/World/pipette_rack/Seat{i}").GetAttribute("xformOp:translate").Get()
            for i in range(6)
        ]
        np.testing.assert_allclose(np.diff(np.asarray(points)[:, 1]), 0.038)
        self.assertTrue(all(p[2] > 0.18 for p in points))

    def test_every_visible_rack_mesh_is_a_collider(self):
        meshes = [p for p in self.stage.GetPrimAtPath("/World/pipette_rack").GetChildren() if p.IsA(UsdGeom.Mesh)]
        self.assertGreaterEqual(len(meshes), 8)
        for mesh in meshes:
            self.assertTrue(mesh.HasAPI(UsdPhysics.CollisionAPI))


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class ActivePipetteGuideAssetTests(unittest.TestCase):
    def test_active_guides_preserve_body_and_have_physical_clearance(self):
        root = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002"
        original = Usd.Stage.Open(str(root / "realistic_pipette_260mm/lab_002_pipette.usda"))
        original_body = original.GetPrimAtPath("/World/pipette_01/Body")
        for variant, pitch, lower_width in (
            ("realistic_pipette_260mm_tall_guide33", 0.038, 0.033),
            ("realistic_pipette_260mm_tall_wide_guide33", 0.045, 0.033),
            ("realistic_pipette_260mm_tall_wide_guide36", 0.045, 0.036),
        ):
            with self.subTest(variant=variant):
                stage = Usd.Stage.Open(str(root / variant / "lab_002_pipette.usda"))
                body = stage.GetPrimAtPath("/World/pipette_01/Body")
                np.testing.assert_allclose(
                    body.GetAttribute("points").Get(), original_body.GetAttribute("points").Get(), atol=1e-7
                )
                rack = stage.GetPrimAtPath("/World/pipette_rack")
                np.testing.assert_allclose(rack.GetAttribute("reference:slotWidths").Get(), [0.033, lower_width])
                seats = [
                    stage.GetPrimAtPath(f"/World/pipette_rack/Seat{i}").GetAttribute("xformOp:translate").Get()
                    for i in range(6)
                ]
                np.testing.assert_allclose(np.diff(np.asarray(seats)[:, 1]), pitch)
                for prim in rack.GetChildren():
                    if prim.IsA(UsdGeom.Mesh):
                        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
                self.assertFalse(stage.GetPrimAtPath("/World/pipette_01/CollarCollision"))


if __name__ == "__main__":
    unittest.main()
