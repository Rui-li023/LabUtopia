"""User-requested cork appearance must not change geometry or contact physics."""

import unittest
from pathlib import Path

try:
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD bindings unavailable")
class FlaskCorkMaterialTests(unittest.TestCase):
    TARGETS = {"/World/round_bottom_flask/Group/mesh_003", "/World/cork_ring/mesh_000"}

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/lab_002/flask_support_collision"
        cls.before = Usd.Stage.Open(str(cls.directory / "lab_002_flask_support.usda"))
        cls.after = Usd.Stage.Open(str(cls.directory / "lab_002_flask_cork.usda"))

    def test_only_the_two_requested_bindings_change(self):
        changed = set()
        for prim in self.before.Traverse():
            other = self.after.GetPrimAtPath(prim.GetPath())
            self.assertEqual(prim.GetTypeName(), other.GetTypeName())
            self.assertEqual(prim.GetMetadata("apiSchemas"), other.GetMetadata("apiSchemas"))
            for attr in prim.GetAttributes():
                self.assertEqual(str(attr.Get()), str(other.GetAttribute(attr.GetName()).Get()), str(attr.GetPath()))
            for rel in prim.GetRelationships():
                actual = other.GetRelationship(rel.GetName()).GetTargets()
                if actual != rel.GetTargets():
                    self.assertEqual(rel.GetName(), "material:binding")
                    changed.add(str(prim.GetPath()))
                    self.assertEqual(actual, [Sdf.Path("/World/FlaskCorkMaterials/Cork")])
        self.assertEqual(changed, self.TARGETS)

    def test_no_geometry_or_physical_material_added(self):
        for prim in self.after.Traverse():
            if not self.before.GetPrimAtPath(prim.GetPath()):
                self.assertTrue(str(prim.GetPath()).startswith("/World/FlaskCorkMaterials"))
                self.assertIn(prim.GetTypeName(), ("Scope", "Material", "Shader"))
                self.assertFalse(prim.HasAPI(UsdPhysics.MaterialAPI))
                self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
                self.assertFalse(prim.HasAPI(UsdPhysics.MassAPI))
        for path in self.TARGETS:
            for stage in (self.before, self.after):
                material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial(
                    materialPurpose="physics"
                )
                self.assertFalse(material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI))

    def test_both_renderers_have_complete_portable_cork_maps(self):
        material = UsdShade.Material(self.after.GetPrimAtPath("/World/FlaskCorkMaterials/Cork"))
        self.assertTrue(material.GetSurfaceOutput().HasConnectedSource())
        self.assertTrue(material.GetSurfaceOutput("mdl").HasConnectedSource())
        textures = set()
        for prim in Usd.PrimRange(material.GetPrim()):
            for attr in prim.GetAttributes():
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath) and value.path.endswith(".png"):
                    self.assertFalse(Path(value.path).is_absolute())
                    self.assertTrue(Path(value.resolvedPath).is_file())
                    textures.add(Path(value.path).name)
        self.assertEqual(
            textures,
            {"Cork001_1K-PNG_Color.png", "Cork001_1K-PNG_NormalGL.png", "Cork001_1K-PNG_Roughness.png"},
        )


if __name__ == "__main__":
    unittest.main()
