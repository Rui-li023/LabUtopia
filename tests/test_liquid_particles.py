"""Pure seeding, allocation safety and byte-identical legacy pour lattices."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.liquid_metrics import BeakerInterior, ConicalInterior
from utils.liquid_particles import ParticleFluidParameters, particle_state, sample_container, sample_volume


class LiquidParticleTests(unittest.TestCase):
    def test_world_position_and_frame_rotation(self):
        points = sample_volume("box", center=[1, -2, 3], size=[0.02, 0.04, 0.06], spacing=0.01)
        self.assertEqual(points.shape, (48, 3))
        np.testing.assert_allclose(points.mean(0), [1, -2, 3])
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix().T
        matrix[3, :3] = [-4, 5, 6]
        transformed = sample_volume("box", center=[1, -2, 3], size=[0.02, 0.04, 0.06], spacing=0.01, matrix=matrix)
        np.testing.assert_allclose(transformed, points @ matrix[:3, :3] + matrix[3, :3])

    def test_sphere_cylinder_and_separate_locations(self):
        for shape in ("sphere", "cylinder"):
            a = sample_volume(shape, radius=0.02, height=0.04, spacing=0.004)
            b = sample_volume(shape, center=[0.3, -0.1, 0.8], radius=0.02, height=0.04, spacing=0.004)
            np.testing.assert_allclose(b - a, np.tile([0.3, -0.1, 0.8], (len(a), 1)))
            self.assertTrue(np.all(np.linalg.norm(a[:, :2], axis=1) < 0.02))
            if shape == "sphere":
                self.assertTrue(np.all(np.linalg.norm(a, axis=1) < 0.02))

    def test_scaled_frame_preserves_world_spacing(self):
        matrix = np.diag([2.0, 3.0, 4.0, 1.0])
        points = sample_volume("box", size=[0.02, 0.02, 0.02], matrix=matrix, spacing=0.01)
        for axis in range(3):
            np.testing.assert_allclose(np.diff(np.unique(points[:, axis])), 0.01)

    def test_bad_seed_inputs_and_huge_grids_fail_before_allocation(self):
        bad = [
            dict(shape="unknown"),
            dict(shape="sphere"),
            dict(shape="cylinder", radius=0.01),
            dict(shape="box", size=[1, 1, 1], spacing=0),
            dict(shape="box", size=[1, 1, 1], spacing=1e-200),
            dict(shape="box", size=[1, 1, 1], max_particles=float("inf")),
            dict(shape="box", size=[1, 1, -1]),
            dict(shape="box", size=[0.001] * 3, spacing=0.004),
            dict(shape="sphere", radius=0.01, center=[0, 0, float("nan")]),
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sample_volume(**args)
        for matrix in (np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                sample_volume("sphere", radius=0.01, matrix=matrix)
        matrix = np.eye(4)
        matrix[0, 1] = 0.4
        with self.assertRaisesRegex(ValueError, "Sheared"):
            sample_volume("sphere", radius=0.01, matrix=matrix)

    def test_particle_state_velocity_broadcast_and_validation(self):
        points = sample_volume("sphere", radius=0.01)
        positions, velocities = particle_state(points, [0, 0, -0.2])
        self.assertEqual(positions.dtype, np.float32)
        np.testing.assert_array_equal(velocities, np.tile(np.float32([0, 0, -0.2]), (len(points), 1)))
        for positions, velocities in (
            ([], None),
            ([[np.inf, 0, 0]], None),
            ([[0, 0, 0]], [[0, 0, 0], [0, 0, 0]]),
            ([[0, 0, 0]], [0, np.nan, 0]),
            ([[1e300, 0, 0]], None),
        ):
            with self.assertRaises(ValueError):
                particle_state(positions, velocities)

    def test_parameters_are_validated_and_json_compatible(self):
        self.assertEqual(ParticleFluidParameters().snapshot()["color"], [0.025, 0.22, 0.48])
        color = [0.1, 0.2, 0.3]
        parameters = ParticleFluidParameters(color=color)
        color[0] = 0.9
        self.assertEqual(parameters.color, (0.1, 0.2, 0.3))
        for args in (
            dict(spacing=0),
            dict(density=-1),
            dict(viscosity=np.nan),
            dict(color=[1, 2, 3]),
            dict(isosurface="false"),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ParticleFluidParameters(**args)

    @staticmethod
    def legacy_points(interior, matrix, legacy):
        # Frozen algorithm before extraction into the general tool. Coordinate
        # order and boundary rounding both affect contact-sensitive GPU replay.
        spacing, fill_height = 0.004, 0.025
        if legacy:
            scale = np.linalg.norm(matrix[1, :3])
            step = spacing / scale
            radius = interior.radius - step
            x = np.arange(-radius, radius + step / 2, step)
            z = np.arange(-radius, radius + step / 2, step)
            y = np.arange(interior.bottom + step, interior.bottom + fill_height / scale, step)
            points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
            points = points[points[:, 0] ** 2 + points[:, 2] ** 2 < radius**2]
            points[:, 0] += interior.center_x
            points[:, 2] += interior.center_z
        else:
            scales = np.linalg.norm(matrix[:3, :3], axis=1)
            steps = spacing / scales
            x = np.arange(-interior.radius + steps[0], interior.radius, steps[0])
            z = np.arange(-interior.radius + steps[2], interior.radius, steps[2])
            y = np.arange(
                interior.bottom + steps[1],
                min(interior.top - steps[1], interior.bottom + fill_height / scales[1]),
                steps[1],
            )
            points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
            points[:, 0] += interior.center_x
            points[:, 2] += interior.center_z
            profile = np.asarray(interior.lower_profile)
            radius = np.minimum(interior.radius, np.interp(points[:, 1], profile[:, 0], profile[:, 1]))
            radial = np.hypot(points[:, 0] - interior.center_x, points[:, 2] - interior.center_z)
            points = points[radial < radius - max(steps[0], steps[2])]
        return points @ matrix[:3, :3] + matrix[3, :3]

    def test_old_container_lattices_are_exactly_preserved(self):
        for interior, scales, legacy in (
            (BeakerInterior(), [0.0005] * 3, True),
            (BeakerInterior(), [0.0007, 0.0006, 0.0008], False),
            (ConicalInterior(), [0.0001] * 3, False),
        ):
            for angles in ([0, 0, 0], [90, 0, 17], [32, 11, -29]):
                matrix = np.eye(4)
                matrix[:3, :3] = np.diag(scales) @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix().T
                matrix[3, :3] = [0.7, -0.3, 0.9]
                points = sample_container(interior, matrix, spacing=0.004, fill_height=0.025, legacy_cylinder=legacy)
                np.testing.assert_array_equal(points, self.legacy_points(interior, matrix, legacy))
                if not legacy:
                    self.assertTrue(interior.contains(points, matrix).all())


if __name__ == "__main__":
    unittest.main()
