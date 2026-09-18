"""Original-mesh collision patches leave the entire beaker opening unobstructed."""

import json
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


class PourCollisionTests(unittest.TestCase):
    def test_conical_original_shell_has_open_neck_and_real_mass(self):
        root = Path(__file__).resolve().parents[2]
        data = json.loads((root / "assets/chemistry_lab/lab_001/pour_liquid/conical_collision.json").read_text())
        self.assertEqual(len(data["patches"]), 193)
        mass = data["native_glass_volume"] * 0.0001**3 * 2230
        self.assertGreater(mass, 0.10)
        # Original shell is thick (~4.8 mm); retain its measured glass mass.
        self.assertAlmostEqual(mass, 0.384381609, places=6)
        centerline = np.column_stack([np.zeros(100), np.linspace(50, 1650, 100), np.zeros(100)])
        for patch in data["patches"]:
            points = np.asarray(patch["points"])
            self.assertTrue((points.min(0) >= [-462.873, -0.001, -462.870]).all())
            self.assertTrue((points.max(0) <= [462.867, 1645.242, 462.869]).all())
            hull = ConvexHull(points)
            self.assertFalse((centerline @ hull.equations[:, :3].T + hull.equations[:, 3] <= 0).all(1).any(), patch["name"])

    def test_real_glass_mass_and_open_interior(self):
        root = Path(__file__).resolve().parents[2]
        data = json.loads((root / "assets/chemistry_lab/lab_001/pour_liquid/beaker_collision.json").read_text())
        self.assertEqual(len(data["patches"]), 73)
        self.assertAlmostEqual(data["native_glass_volume"] * 0.0005**3 * 2230, 0.0743891, places=6)
        centerline = np.column_stack([np.full(80, 53.346556), np.linspace(-80, 95, 80), np.zeros(80)])
        for patch in data["patches"]:
            points = np.asarray(patch["points"])
            self.assertTrue((points.min(0) >= np.array([-33.631, -86.689, -75.333])).all())
            self.assertTrue((points.max(0) <= np.array([128.68, 94.114, 75.333])).all())
            hull = ConvexHull(points)
            inside = (centerline @ hull.equations[:, :3].T + hull.equations[:, 3] <= 0).all(1)
            self.assertFalse(inside.any(), patch["name"])


if __name__ == "__main__":
    unittest.main()
