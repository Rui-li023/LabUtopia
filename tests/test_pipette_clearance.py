"""Read-only geometric planning screens do not weaken physical success tests."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

_spec = importlib.util.spec_from_file_location(
    "_pipette_clearance_test", Path(__file__).resolve().parents[1] / "controllers/pipette_clearance.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
RackArmClearance, sphere_box_clearances = _module.RackArmClearance, _module.sphere_box_clearances


class SphereBoxTests(unittest.TestCase):
    def test_convex_mesh_cover_retains_surface_and_volume_without_filling_entire_box(self):
        vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        centers, radii = _module.cover_convex_mesh_with_spheres(vertices, cell_size=0.02)
        box, frame = _module.oriented_mesh_bounds(vertices)
        box_centers, _ = _module.cover_box_with_spheres(box, frame, cell_size=0.02)
        self.assertLess(len(centers), len(box_centers))
        barycentric = np.random.default_rng(35).dirichlet(np.ones(4), 1000)
        points = np.vstack([vertices, barycentric @ vertices])
        gap = np.linalg.norm(points[:, None, :] - centers, axis=2) - radii
        self.assertTrue(np.all(np.min(gap, axis=1) <= 1e-12))

    def test_carried_cover_contains_corners_and_volume_after_rotation(self):
        bounds = np.array([[-0.018, -0.015, 0.0], [0.018, 0.015, 0.26]])
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.7, 0.4]).as_matrix()
        frame[:3, 3] = [0.1, 0.2, 0.9]
        centers, radii = _module.cover_box_with_spheres(bounds, frame)
        corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
        points = np.vstack([corners, np.random.default_rng(35).uniform(bounds[0], bounds[1], (1000, 3))])
        points = points @ frame[:3, :3].T + frame[:3, 3]
        gaps = np.linalg.norm(points[:, None, :] - centers, axis=2) - radii
        self.assertTrue(np.all(np.min(gaps, axis=1) < 1e-12))
        self.assertEqual(len(centers), 52)

    def test_carried_cover_rejects_invalid_resolution_and_unbounded_size(self):
        for size in (0, -1, np.nan, 0.004, 0.051):
            with self.subTest(size=size), self.assertRaises(ValueError):
                _module.cover_box_with_spheres([[0, 0, 0], [0.1, 0.1, 0.1]], np.eye(4), cell_size=size)
        with self.assertRaises(ValueError):
            _module.cover_box_with_spheres([[0, 0, 0], [2, 2, 2]], np.eye(4))

    def test_oriented_box_retains_all_mesh_vertices_without_filling_diagonal_gaps(self):
        points = np.array([[x, y, z] for x in (-1, 1) for y in (-0.05, 0.05) for z in (-0.02, 0.02)])
        points = Rotation.from_euler("z", 45, degrees=True).apply(points) + np.array([0, 0, 1])
        box, frame = _module.oriented_mesh_bounds(points)
        self.assertGreater(np.linalg.det(frame[:3, :3]), 0)
        local = (points - frame[:3, 3]) @ frame[:3, :3]
        self.assertTrue(np.all(local >= box[0] - 1e-9))
        self.assertTrue(np.all(local <= box[1] + 1e-9))
        checker = RackArmClearance(None, {"hand": [{"center": [0, 0, 0], "radius": 0.02}]}, [box], box_frames=[frame])
        self.assertGreater(checker._clearances(np.array([[0.65, -0.65, 1]]), [0.02])[0, 0], 0.8)
        aabb = [np.min(points, axis=0), np.max(points, axis=0)]
        self.assertLess(sphere_box_clearances([[0.65, -0.65, 1]], [0.02], [aabb])[0, 0], 0)

    def test_planning_obstacle_is_copy_only_and_preserves_world_pose(self):
        frame = np.eye(4)
        frame[:3, 3] = [2, 3, 4]
        obstacle = _module.PlanningBox([[-1, -2, -3], [1, 2, 3]], frame, padding=0.002)
        position, quat = obstacle.get_world_pose()
        np.testing.assert_array_equal(position, [2, 3, 4])
        np.testing.assert_array_equal(quat, [1, 0, 0, 0])
        np.testing.assert_allclose(obstacle.get_local_scale(), [2.004, 4.004, 6.004])
        position[:] = 0
        np.testing.assert_array_equal(obstacle.get_world_pose()[0], [2, 3, 4])

    def test_outside_tangent_inside_and_corner_distances(self):
        centers = [[2, 0, 0], [1.1, 0, 0], [0, 0, 0], [2, 2, 0]]
        actual = sphere_box_clearances(centers, [0.1] * 4, [[[-1] * 3, [1] * 3]])[:, 0]
        np.testing.assert_allclose(actual, [0.9, 0, -1.1, np.sqrt(2) - 0.1], atol=1e-12)

    def test_all_obstacle_boxes_are_checked(self):
        actual = sphere_box_clearances([[3, 0, 0]], [0.1], [[[-1] * 3, [1] * 3], [[2, -1, -1], [4, 1, 1]]])
        self.assertEqual(actual.shape, (1, 2))
        self.assertGreater(actual[0, 0], 0)
        self.assertLess(actual[0, 1], 0)

    def test_invalid_geometry_is_rejected(self):
        for centers, radii, boxes in (
            ([[0, 0, 0]], [-1], [[[-1] * 3, [1] * 3]]),
            ([[np.nan, 0, 0]], [1], [[[-1] * 3, [1] * 3]]),
            ([[0, 0, 0]], [1], [[[1] * 3, [-1] * 3]]),
            ([], [], [[[-1] * 3, [1] * 3]]),
        ):
            with self.subTest(centers=centers), self.assertRaises(ValueError):
                sphere_box_clearances(centers, radii, boxes)


class RackArmTests(unittest.TestCase):
    def test_transformed_link_spheres_and_margin(self):
        calls = []

        def fk(frame, joints):
            calls.append((frame, joints.copy()))
            return [0, 1.15, 0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

        checker = RackArmClearance(
            SimpleNamespace(compute_forward_kinematics=fk),
            {"hand": [{"center": [0.2, 0, 0], "radius": 0.1}]},
            [[[-1] * 3, [1] * 3]],
            margin=0.005,
        )
        joints = np.zeros(7)
        self.assertAlmostEqual(checker.minimum_clearance(joints), 0.25)
        self.assertTrue(checker(joints))
        np.testing.assert_array_equal(joints, np.zeros(7))
        self.assertEqual(len(calls), 2)
        checker.boxes[0, 1, 1] = 1.30
        self.assertFalse(checker(joints))

    def test_loads_actual_franka_collision_description(self):
        root = Path(__file__).resolve().parents[1]
        checker = RackArmClearance.from_descriptor(
            None, root / "robots/franka/rmpflow/robot_descriptor.yaml", [[[-1] * 3, [1] * 3]]
        )
        self.assertIn("panda_hand", checker.frames)
        self.assertIn("panda_link6", checker.frames)
        self.assertEqual(len(checker.frames), 9)

    def test_missing_spheres_or_invalid_margin_fail_closed(self):
        for spheres, margin in (({}, 0.005), ({"hand": []}, 0.005), ({}, -0.001)):
            with self.assertRaises(ValueError):
                RackArmClearance(None, spheres, [[[-1] * 3, [1] * 3]], margin)


class ClearanceIKTests(unittest.TestCase):
    def test_uses_redundancy_without_moving_target_or_relaxing_joint_margin(self):
        solver = SimpleNamespace(
            compute_forward_kinematics=lambda frame, q: (q[:3], Rotation.from_rotvec(q[3:6]).as_matrix())
        )

        class Checker:
            margin = 0.005

            def minimum_clearance(self, joints):
                return joints[6] - 0.1

            def __call__(self, joints):
                return self.minimum_clearance(joints) >= self.margin

        bounds = np.tile([-2, 2], (7, 1))
        seed = np.zeros(7)
        result, ok = _module.recover_clearance_ik(solver, Checker(), "hand", np.zeros(3), [1, 0, 0, 0], seed, bounds)
        self.assertTrue(ok)
        np.testing.assert_allclose(result[:6], 0, atol=1e-8)
        self.assertGreaterEqual(result[6], 0.105)
        self.assertLessEqual(result[6], 0.35)
        np.testing.assert_array_equal(seed, np.zeros(7))
        # An unreachable clearance requirement cannot be turned into success.
        bounds[6] = [-0.2, 0.2]
        _, ok = _module.recover_clearance_ik(solver, Checker(), "hand", np.zeros(3), [1, 0, 0, 0], seed, bounds)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
