"""Substep configuration must cover referenced scenes without disabling them."""

import unittest

try:
    from pxr import PhysxSchema, Usd, UsdPhysics

    from experiments.support.physics_scene_steps import synchronize_physics_scene_steps
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, "USD Python bindings unavailable")
class PhysicsSceneStepTests(unittest.TestCase):
    def test_synchronizes_both_scenes_without_reassigning_bodies_or_changing_physics(self):
        stage = Usd.Stage.CreateInMemory()
        world = UsdPhysics.Scene.Define(stage, "/physicsScene")
        lab = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        PhysxSchema.PhysxSceneAPI.Apply(world.GetPrim()).CreateTimeStepsPerSecondAttr(240)
        PhysxSchema.PhysxSceneAPI.Apply(lab.GetPrim()).CreateTimeStepsPerSecondAttr(60)
        lab.CreateGravityMagnitudeAttr(9.81)
        body = UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/World/tool", "Xform"))
        body.CreateSimulationOwnerRel().SetTargets([lab.GetPath()])
        body.CreateRigidBodyEnabledAttr(True)
        collider = UsdPhysics.CollisionAPI.Apply(stage.DefinePrim("/World/tool/mesh", "Mesh"))
        collider.CreateCollisionEnabledAttr(True)
        records = synchronize_physics_scene_steps(stage, 240)
        self.assertEqual(
            {r["path"]: r["previous_hz"] for r in records}, {"/physicsScene": 240, "/World/PhysicsScene": 60}
        )
        for prim in (world.GetPrim(), lab.GetPrim()):
            self.assertTrue(prim.IsActive())
            self.assertEqual(PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get(), 240)
        self.assertAlmostEqual(lab.GetGravityMagnitudeAttr().Get(), 9.81, places=5)
        self.assertEqual(body.GetSimulationOwnerRel().GetTargets(), [lab.GetPath()])
        self.assertTrue(body.GetRigidBodyEnabledAttr().Get())
        self.assertTrue(collider.GetCollisionEnabledAttr().Get())

    def test_repeat_is_idempotent_and_reports_actual_previous_rate(self):
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.Scene.Define(stage, "/scene")
        synchronize_physics_scene_steps(stage, 120)
        before = stage.GetRootLayer().ExportToString()
        self.assertEqual(
            synchronize_physics_scene_steps(stage, 120),
            [{"path": "/scene", "previous_hz": 120, "configured_hz": 120}],
        )
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_invalid_rate_or_missing_scene_fails_before_writes(self):
        stage = Usd.Stage.CreateInMemory()
        before = stage.GetRootLayer().ExportToString()
        for rate in (0, 59, 481, 240.0, True, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                synchronize_physics_scene_steps(stage, rate)
            self.assertEqual(stage.GetRootLayer().ExportToString(), before)
        with self.assertRaisesRegex(ValueError, "without a PhysicsScene"):
            synchronize_physics_scene_steps(stage, 240)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)
