"""Liquid success must measure contained liquid, not merely wrist rotation."""

import unittest
from types import SimpleNamespace

import numpy as np

from utils.liquid_metrics import (
    BeakerInterior,
    ConicalInterior,
    StableLiquidGate,
    liquid_accounting,
    liquid_transfer_ok,
)


class LiquidMetricsTests(unittest.TestCase):
    def setUp(self):
        self.interior = BeakerInterior(center_x=0, radius=1, bottom=0, top=2)
        self.source = np.eye(4)
        self.target = np.eye(4)
        self.target[3, 0] = 4
        self.cfg = SimpleNamespace(liquid=SimpleNamespace(enabled=True, settle_steps=3))

    def measure(self, points, count=None):
        return liquid_accounting(
            np.asarray(points),
            len(points) if count is None else count,
            self.source,
            self.target,
            self.interior,
        )

    def test_real_transfer(self):
        metrics = self.measure([[4, 1, 0]] * 9 + [[0, 1, 0]])
        self.assertTrue(liquid_transfer_ok({"liquid": metrics}, self.cfg))
        self.assertEqual(metrics["target_fraction"], 0.9)
        self.assertEqual(metrics["outside_fraction"], 0)

    def test_spill_uses_initial_denominator(self):
        metrics = self.measure([[4, 1, 0]] * 8 + [[10, -1, 0]] * 2)
        self.assertEqual(metrics["target_fraction"], 0.8)
        self.assertFalse(liquid_transfer_ok({"liquid": metrics}, self.cfg))

    def test_empty_or_missing_is_not_success(self):
        for state in ({}, {"liquid": self.measure([])}, {"liquid": {"valid": True}}):
            self.assertFalse(liquid_transfer_ok(state, self.cfg))

    def test_lost_nan_or_infinite_particles_invalid(self):
        for points in ([[4, 1, 0]], [[np.nan, 1, 0]] * 2, [[np.inf, 1, 0]] * 2):
            self.assertFalse(self.measure(points, count=2)["valid"])

    def test_air_above_rim_and_aabb_corner_are_not_in_cup(self):
        metrics = self.measure([[4, 3, 0], [4.9, 1, 0.9], [4, -0.1, 0]])
        self.assertEqual(metrics["outside_count"], 3)

    def test_no_double_count_in_overlapping_containers(self):
        self.target = self.source.copy()
        metrics = self.measure([[0, 1, 0]])
        self.assertEqual(metrics["source_count"], 1)
        self.assertEqual(metrics["target_count"], 0)

    def test_native_rounded_base_is_not_a_cylinder_corner(self):
        interior = BeakerInterior()
        points = np.array([[53.346556 + 65, -82, 0], [53.346556, -82, 0]])
        self.assertEqual(interior.contains(points, np.eye(4)).tolist(), [False, True])

    def test_mesh_transform_includes_rotation_translation_and_scale(self):
        self.target[:3, :3] = [[0, 0, 2], [2, 0, 0], [0, 2, 0]]
        self.assertEqual(self.measure([[6, 0, 0]])["target_count"], 1)

    def test_stability_resets_after_spill_or_tilt(self):
        metrics = self.measure([[4, 1, 0]] * 9 + [[0, 1, 0]])
        metrics.update(source_up=[0, 0, 1], target_up=[0, 0, 1])
        state = {"liquid": metrics}
        gate = StableLiquidGate()
        self.assertFalse(gate.check(state, self.cfg, True))
        self.assertFalse(gate.check(state, self.cfg, True))
        metrics["source_up"] = [1, 0, 0]
        self.assertFalse(gate.check(state, self.cfg, True))
        self.assertEqual(gate.steps, 0)
        metrics["source_up"] = [0, 0, 1]
        for expected in (False, False, True):
            self.assertEqual(gate.check(state, self.cfg, True), expected)
        self.assertFalse(gate.check(state, self.cfg, False))

    def test_disabled_is_backward_compatible(self):
        self.assertTrue(liquid_transfer_ok({}, SimpleNamespace()))

    def test_conical_neck_and_shoulder_are_not_a_base_width_cylinder(self):
        interior = ConicalInterior()
        points = np.array([[0, 60, 0], [350, 60, 0], [250, 1100, 0], [175, 1400, 0],
                           [190, 1400, 0], [0, 40, 0], [0, 1650, 0]])
        self.assertEqual(interior.contains(points, np.eye(4)).tolist(), [True, True, False, True, False, False, False])
        self.assertLess(interior.mouth_radius, interior.radius)

    def test_different_source_and_receiver_interiors(self):
        target = np.eye(4)
        target[3, 0] = 2000
        points = np.array([[0, 1400, 0], [250, 1400, 0], [2053.346556, 10, 0]])
        metrics = liquid_accounting(points, 3, np.eye(4), target, ConicalInterior(), BeakerInterior())
        self.assertEqual([metrics[k] for k in ("source_count", "target_count", "outside_count")], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
