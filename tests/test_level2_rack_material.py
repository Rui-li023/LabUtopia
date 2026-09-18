"""Real USD assets retain visuals, geometry, and silicone feet under contact tuning."""

import unittest
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from utils.pipette_rack_material import configure_acrylic_rack_contact
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD Python bindings are unavailable")
class RackMaterialTests(unittest.TestCase):
    def stage(self, variant="tall_four"):
        asset = Path(__file__).resolve().parents[1] / (
            f"assets/chemistry_lab/lab_002/realistic_pipette_260mm_{variant}_guide36/lab_002_pipette.usda"
        )
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World").GetReferences().AddReference(str(asset))
        self.assertEqual(stage.GetRootLayer().realPath, "")
        return stage

    def test_active_stands_use_original_transparent_finish_and_existing_maps(self):
        for variant in ("tall_four", "tall_wide"):
            with self.subTest(variant=variant):
                stage = self.stage(variant)
                clear_path = "/World/RealisticPipetteMaterials/Clear_PMMA"
                shader = UsdShade.Shader(stage.GetPrimAtPath(clear_path + "/MDL"))
                self.assertEqual(shader.GetSourceAsset("mdl").path, "OmniGlass.mdl")
                self.assertAlmostEqual(shader.GetInput("glass_ior").Get(), 1.49, places=6)
                self.assertAlmostEqual(shader.GetInput("frosting_roughness").Get(), 0.07, places=6)
                self.assertFalse(shader.GetInput("thin_walled").Get())
                for name in ("roughness_texture", "normal_map_texture"):
                    texture = shader.GetInput(name).Get()
                    self.assertTrue(Path(texture.resolvedPath).is_file())
                    self.assertIn("realistic_pipette_260mm/textures/", texture.resolvedPath)
                self.assertFalse(stage.GetPrimAtPath("/World/RealisticPipetteMaterials/White_PMMA"))
                for name in ("Base_plate", "Inclined_back", "Upper_33mm_slots", "Lower_36mm_slots"):
                    panel = stage.GetPrimAtPath("/World/pipette_rack/" + name)
                    self.assertEqual(
                        str(UsdShade.MaterialBindingAPI(panel).ComputeBoundMaterial()[0].GetPath()), clear_path
                    )
                    self.assertTrue(UsdPhysics.CollisionAPI(panel).GetCollisionEnabledAttr().Get())

    def test_scoped_material_preserves_both_stands(self):
        for variant in ("tall_four", "tall_wide"):
            with self.subTest(variant=variant):
                stage = self.stage(variant)
                rack = stage.GetPrimAtPath("/World/pipette_rack")
                colliders = [p for p in Usd.PrimRange(rack) if p.HasAPI(UsdPhysics.CollisionAPI)]
                before = {
                    str(p.GetPath()): (
                        str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath()),
                        tuple(UsdGeom.Mesh(p).GetExtentAttr().Get()),
                        len(UsdGeom.Mesh(p).GetPointsAttr().Get()),
                    )
                    for p in colliders
                }
                body = stage.GetPrimAtPath("/World/pipette_01/Body")
                body_material = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial("physics")[0].GetPath()
                changed = configure_acrylic_rack_contact(
                    stage, str(rack.GetPath()), static_friction=0.3, dynamic_friction=0.3
                )
                self.assertEqual(len(changed), 4)
                self.assertEqual(len(colliders), 8)
                for prim in colliders:
                    binding = UsdShade.MaterialBindingAPI(prim)
                    visual = binding.ComputeBoundMaterial()[0]
                    material = binding.ComputeBoundMaterial("physics")[0]
                    self.assertEqual(str(visual.GetPath()), before[str(prim.GetPath())][0])
                    self.assertEqual(tuple(UsdGeom.Mesh(prim).GetExtentAttr().Get()), before[str(prim.GetPath())][1])
                    self.assertEqual(len(UsdGeom.Mesh(prim).GetPointsAttr().Get()), before[str(prim.GetPath())][2])
                    self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
                    if prim.GetName().startswith("Foot_"):
                        self.assertNotIn(str(prim.GetPath()), changed)
                        self.assertEqual(material.GetPath(), visual.GetPath())
                        self.assertFalse(material.GetPrim().HasAPI(UsdPhysics.MaterialAPI))
                    else:
                        self.assertAlmostEqual(
                            UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get(), 0.3
                        )
                self.assertEqual(
                    UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial("physics")[0].GetPath(), body_material
                )
                self.assertAlmostEqual(
                    UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/pipette_01")).GetMassAttr().Get(), 0.09
                )
                self.assertFalse(rack.HasAPI(UsdPhysics.RigidBodyAPI))

    def test_invalid_values_and_incomplete_racks_fail_before_authoring(self):
        stage = self.stage()
        for static, dynamic in ((float("nan"), 0.3), (0.3, 0.4), (-0.1, -0.2), (1.3, 0.3)):
            with self.subTest(static=static, dynamic=dynamic), self.assertRaises(ValueError):
                configure_acrylic_rack_contact(
                    stage, "/World/pipette_rack", static_friction=static, dynamic_friction=dynamic
                )
        self.assertFalse(stage.GetPrimAtPath("/World/pipette_rack/AcrylicContact"))
        stage.GetPrimAtPath("/World/pipette_rack/Lower_36mm_slots").SetActive(False)
        with self.assertRaises(ValueError):
            configure_acrylic_rack_contact(stage, "/World/pipette_rack", static_friction=0.3, dynamic_friction=0.3)
        self.assertFalse(stage.GetPrimAtPath("/World/pipette_rack/AcrylicContact"))

    def test_reapplying_parameters_is_idempotent(self):
        stage = self.stage()
        first = configure_acrylic_rack_contact(stage, "/World/pipette_rack", static_friction=0.3, dynamic_friction=0.2)
        second = configure_acrylic_rack_contact(stage, "/World/pipette_rack", static_friction=0.3, dynamic_friction=0.2)
        self.assertEqual(first, second)
        material = stage.GetPrimAtPath("/World/pipette_rack/AcrylicContact")
        self.assertAlmostEqual(UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Get(), 0.2)
