"""Robot inertia correction must preserve mass, COM, geometry and replay parity."""

import unittest
from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from utils.robot_mass_properties import prepare_robot_mass_properties, validate_robot_mass_replay


class RobotMassPropertiesTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.world = SimpleNamespace(is_stopped=lambda: True)
        UsdGeom.Xform.Define(self.stage, "/World/Arm")
        self.link = UsdGeom.Xform.Define(self.stage, "/World/Arm/link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(self.link)
        self.mass = UsdPhysics.MassAPI.Apply(self.link)
        self.mass.CreateMassAttr(1.13)
        self.mass.CreateCenterOfMassAttr(Gf.Vec3f(0.01, 0.0, 0.0))
        self.mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-7))
        self.mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        self.shape = UsdGeom.Cube.Define(self.stage, "/World/Arm/link/collider")
        self.shape.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(self.shape.GetPrim())

    def prepare(self):
        return prepare_robot_mass_properties(self.stage, self.world, "/World/Arm", "collision_geometry")

    def test_opt_in_requests_auto_inertia_and_preserves_mass_com_geometry(self):
        original_mass = self.mass.GetMassAttr().Get()
        original_com = self.mass.GetCenterOfMassAttr().Get()
        record = self.prepare()
        self.assertEqual(self.mass.GetDiagonalInertiaAttr().Get(), Gf.Vec3f(0.0))
        self.assertEqual(self.mass.GetPrincipalAxesAttr().Get(), Gf.Quatf(0.0))
        self.assertEqual(self.mass.GetMassAttr().Get(), original_mass)
        self.assertEqual(self.mass.GetCenterOfMassAttr().Get(), original_com)
        self.assertEqual(self.shape.GetSizeAttr().Get(), 0.1)
        self.assertEqual(record["bodies"][0]["mass_kg"], original_mass)
        self.assertGreater(record["bodies"][0]["original_diagonal_inertia"][0], 0)

    def test_default_is_byte_identical_no_op(self):
        before = self.stage.GetRootLayer().ExportToString()
        self.assertEqual(prepare_robot_mass_properties(self.stage, self.world, "/World/Arm"), {})
        self.assertEqual(before, self.stage.GetRootLayer().ExportToString())

    def test_validation_finishes_before_any_inertia_is_changed(self):
        bad = UsdGeom.Xform.Define(self.stage, "/World/Arm/missing_collision").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(bad)
        UsdPhysics.MassAPI.Apply(bad).CreateMassAttr(1.0)
        before = self.mass.GetDiagonalInertiaAttr().Get()
        with self.assertRaisesRegex(ValueError, "no exposed enabled collision"):
            self.prepare()
        self.assertEqual(self.mass.GetDiagonalInertiaAttr().Get(), before)

    def test_invalid_scope_live_world_and_units_fail(self):
        with self.assertRaises(ValueError):
            prepare_robot_mass_properties(self.stage, self.world, "/World", "collision_geometry")
        with self.assertRaises(ValueError):
            prepare_robot_mass_properties(self.stage, self.world, "/World/Arm", "invented_mode")
        self.world.is_stopped = lambda: False
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.world.is_stopped = lambda: True
        UsdGeom.SetStageMetersPerUnit(self.stage, 0.01)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_strict_replay_rejects_changed_or_missing_inertia_configuration(self):
        corrected = self.prepare()
        validate_robot_mass_replay({}, {})
        validate_robot_mass_replay(corrected, corrected)
        for recorded, current in (({}, corrected), (corrected, {})):
            with self.assertRaisesRegex(ValueError, "collect a fresh dataset"):
                validate_robot_mass_replay(recorded, current)


if __name__ == "__main__":
    unittest.main()
