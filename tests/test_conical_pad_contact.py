import unittest
from types import SimpleNamespace

from utils.gripper_contact import CompliantFingerPadParameters, finger_pad_snapshot


class ConicalPadContactTests(unittest.TestCase):
    def test_default_contact_is_finite_and_submillimeter_at_30_n(self):
        params = CompliantFingerPadParameters()
        self.assertLessEqual(30 / params.stiffness, 0.001)
        self.assertLessEqual(params.min_torsional_radius, 0.004)

    def test_rejects_unphysical_contact_parameters(self):
        for options in ({"min_torsional_radius": 0.01}, {"stiffness": 100},
                        {"damping": float("nan")}, {"static_friction": 100},
                        {"contact_offset": 0.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CompliantFingerPadParameters(**options)

    def test_no_configuration_preserves_legacy_snapshot(self):
        self.assertEqual(finger_pad_snapshot(SimpleNamespace(task=SimpleNamespace())), {})

    def test_snapshot_contains_all_contact_settings(self):
        cfg = SimpleNamespace(task=SimpleNamespace(gripper_drive_force=30, gripper_contact={}))
        saved = finger_pad_snapshot(cfg)
        self.assertEqual(saved["stiffness"], 60000)
        self.assertEqual(saved["dynamic_friction"], 0.7)
        cfg.task.gripper_contact = {"min_torsional_radius": 0.002}
        self.assertNotEqual(saved, finger_pad_snapshot(cfg))


if __name__ == "__main__":
    unittest.main()
