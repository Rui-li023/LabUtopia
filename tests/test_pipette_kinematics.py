"""No simulator needed: exercise bounded local recovery using exact toy FK."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location(
    "pipette_kinematics", Path(__file__).resolve().parents[1] / "controllers/pipette_kinematics.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ToySolver:
    def compute_inverse_kinematics(self, *args, **kwargs):
        return np.full(7, 2.0), True  # Deliberately jumps to a remote branch.

    def get_cspace_position_limits(self):
        return np.full(7, -3.0), np.full(7, 3.0)

    def compute_forward_kinematics(self, frame, joints):
        return joints[:3], Rotation.from_rotvec(joints[3:6]).as_matrix()


class ContinuousIKTests(unittest.TestCase):
    def test_larger_preflight_margin_reserves_tracking_headroom(self):
        bounds = np.tile([-0.2, 0.2], (7, 1))
        args = (ToySolver(), "tcp", np.array([0.12, 0, 0]), np.array([1, 0, 0, 0]), np.zeros(7), bounds)
        self.assertTrue(module.continuous_inverse_kinematics(*args)[1])
        solution, ok = module.continuous_inverse_kinematics(*args, joint_margin=0.1)
        self.assertFalse(ok)
        self.assertLessEqual(solution[0], 0.1)

    def test_cannot_relax_physical_margin(self):
        for margin in (0, -0.1, 0.251, float("nan")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                module.continuous_inverse_kinematics(
                    ToySolver(), "tcp", np.zeros(3), np.array([1, 0, 0, 0]), np.zeros(7), joint_margin=margin
                )

    def test_recovers_nearby_pose_without_branch_jump(self):
        solution, ok = module.continuous_inverse_kinematics(
            ToySolver(), "tcp", np.array([0.01, 0.02, 0.03]), np.array([1, 0, 0, 0]), np.zeros(7)
        )
        self.assertTrue(ok)
        np.testing.assert_allclose(solution[:3], [0.01, 0.02, 0.03], atol=0.0005)
        self.assertLessEqual(np.max(np.abs(solution)), 0.35)

    def test_rejects_pose_outside_local_trust_region(self):
        _, ok = module.continuous_inverse_kinematics(
            ToySolver(), "tcp", np.array([1.0, 0, 0]), np.array([1, 0, 0, 0]), np.zeros(7)
        )
        self.assertFalse(ok)

    def test_preserves_physical_joint_margin(self):
        bounds = np.tile([-0.1, 0.1], (7, 1))
        solution, ok = module.continuous_inverse_kinematics(
            ToySolver(), "tcp", np.array([0.08, 0, 0]), np.array([1, 0, 0, 0]), np.zeros(7), bounds
        )
        self.assertFalse(ok)
        self.assertLessEqual(solution[0], 0.05)
