import unittest
from types import SimpleNamespace

from utils.conical_grasp import ConicalNeckGrasp
from utils.gripper_contact import finger_pad_snapshot


class ConicalGraspTests(unittest.TestCase):
    def test_centered_face_stays_off_original_rim_and_shoulder(self):
        grasp = ConicalNeckGrasp()
        center = 0.164524 - grasp.depth
        self.assertGreater(center - 0.0105, 0.129728)
        self.assertLess(center + 0.0105, 0.155611)
        # Narrowest neck diameter about 44.9 mm: the blocked target preserves
        # enough drive error for a 60 N cap at 10000 N/m, unlike old 19 mm.
        self.assertGreater((0.0449 / 2 - grasp.closed_finger_position) * 10000, 60)

    def test_force_contact_remains_inside_existing_compression_envelope(self):
        cfg = SimpleNamespace(task=SimpleNamespace(gripper_drive_force=60, gripper_contact={}))
        self.assertLessEqual(60 / finger_pad_snapshot(cfg)["stiffness"], 0.001)

    def test_config_and_invalid_grasps(self):
        cfg = SimpleNamespace(task=SimpleNamespace(conical_grasp={"depth": 0.025}))
        self.assertEqual(ConicalNeckGrasp.from_config(cfg).depth, 0.025)
        for options in (
            {"depth": 0.012},
            {"depth": 0.060},
            {"depth": float("nan")},
            {"closed_finger_position": 0.03},
            {"closed_finger_position": -0.01},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ConicalNeckGrasp(**options)


if __name__ == "__main__":
    unittest.main()
