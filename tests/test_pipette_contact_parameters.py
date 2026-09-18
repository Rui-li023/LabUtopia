import unittest
from types import SimpleNamespace

from utils.gripper_contact import PipetteContactParameters


class ContactParameterTests(unittest.TestCase):
    def test_single_actuator_force_is_explicit_and_uses_bounded_servo(self):
        cfg = SimpleNamespace(
            robot=SimpleNamespace(
                gripper=SimpleNamespace(closing_force=50, force_semantics="single_mimic_drive", control_mode="force")
            ),
            task=SimpleNamespace(force_gripper_servo=True, gripper_force_servo_mode="velocity"),
        )
        parameters = PipetteContactParameters.from_config(cfg)
        self.assertEqual(parameters.closing_force_drive_cap, 50)
        self.assertEqual(parameters.force_semantics, "single_mimic_drive")
        for owner, key, value in (
            (cfg.robot.gripper, "control_mode", "velocity"),
            (cfg.task, "force_gripper_servo", False),
            (cfg.task, "gripper_force_servo_mode", "position"),
        ):
            original = getattr(owner, key)
            setattr(owner, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "bounded velocity"):
                PipetteContactParameters.from_config(cfg)
            setattr(owner, key, original)

    def test_single_actuator_compliance_remains_force_bounded(self):
        with self.assertRaisesRegex(ValueError, "50000"):
            PipetteContactParameters(
                closing_force_per_finger=50, force_semantics="single_mimic_drive", contact_stiffness=35000
            )
        parameters = PipetteContactParameters(
            closing_force_per_finger=50, force_semantics="single_mimic_drive", contact_stiffness=50000
        )
        self.assertEqual(parameters.closing_force_drive_cap / parameters.contact_stiffness, 0.001)

    def test_opening_speed_is_independent_and_opt_in(self):
        cfg = SimpleNamespace(
            robot=SimpleNamespace(gripper=SimpleNamespace(closing_speed=0.02, opening_speed=0.05)),
            task=SimpleNamespace(),
        )
        parameters = PipetteContactParameters.from_config(cfg)
        self.assertEqual(parameters.closing_speed, 0.02)
        self.assertEqual(parameters.opening_speed, 0.05)
        self.assertIsNone(PipetteContactParameters().opening_speed)

    def test_explicit_contact_offset_is_opt_in_and_bounded(self):
        cfg = SimpleNamespace(robot=SimpleNamespace(), task=SimpleNamespace(gripper_pad_contact_offset=0.0005))
        self.assertEqual(PipetteContactParameters().contact_offset, 0)
        self.assertEqual(PipetteContactParameters.from_config(cfg).contact_offset, 0.0005)

    def test_finite_force_based_pad_compliance(self):
        cfg = SimpleNamespace(
            robot=SimpleNamespace(gripper=SimpleNamespace(closing_speed=0.02, closing_force=35)),
            task=SimpleNamespace(gripper_pad_contact_stiffness=100000, gripper_pad_contact_damping=100),
        )
        parameters = PipetteContactParameters.from_config(cfg)
        self.assertEqual(parameters.contact_stiffness, 100000)
        self.assertEqual(parameters.contact_damping, 100)
        self.assertAlmostEqual(parameters.closing_force_per_finger / parameters.contact_stiffness, 0.00035)

    def test_defaults_preserve_baseline(self):
        cfg = SimpleNamespace(robot=SimpleNamespace(), task=SimpleNamespace())
        self.assertEqual(PipetteContactParameters.from_config(cfg), PipetteContactParameters())

    def test_reads_actual_gripper_speed_and_pad_material(self):
        cfg = SimpleNamespace(
            robot=SimpleNamespace(gripper=SimpleNamespace(closing_speed=0.02, closing_force=35)),
            task=SimpleNamespace(gripper_pad_static_friction=1.0, gripper_pad_dynamic_friction=0.8),
        )
        self.assertEqual(PipetteContactParameters.from_config(cfg), PipetteContactParameters(0.02, 35, 1.0, 0.8))

    def test_rejects_unbounded_or_nonphysical_values(self):
        for kwargs in (
            {"closing_speed": 0},
            {"closing_speed": 0.051},
            {"closing_force_per_finger": 35.1},
            {"closing_force_per_finger": float("nan")},
            {"static_friction": float("inf")},
            {"dynamic_friction": -0.1},
            {"dynamic_friction": 1.0, "static_friction": 0.9},
            {"static_friction": 5.0},
            {"contact_stiffness": 100},
            {"contact_stiffness": float("inf")},
            {"contact_damping": 100},
            {"contact_stiffness": 100000, "contact_damping": -1},
            {"contact_stiffness": 100000, "contact_damping": 1001},
            {"contact_offset": -0.001},
            {"contact_offset": 0.00001},
            {"contact_offset": 0.006},
            {"contact_offset": float("nan")},
            {"opening_speed": 0},
            {"opening_speed": 0.051},
            {"opening_speed": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipetteContactParameters(**kwargs)
