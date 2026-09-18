"""Composite liquid gates are independent of expert phases and replay timing."""

import unittest
from types import SimpleNamespace

from utils.liquid_transfers import LiquidRouteMonitor, liquid_routes


def config(kind):
    return SimpleNamespace(
        task_type=kind,
        task=SimpleNamespace(pick_object_path="/World/beaker", pour_target=SimpleNamespace(prim_path="/World/target")),
        liquid=SimpleNamespace(enabled=True, settle_steps=2),
    )


def received():
    return {
        "valid": True,
        "initial_count": 100,
        "target_count": 95,
        "source_count": 0,
        "outside_count": 5,
        "target_fraction": 0.95,
        "source_fraction": 0,
        "outside_fraction": 0.05,
        "source_up": [0, 0, 1],
        "target_up": [0, 0, 1],
    }


class LiquidRoutesTests(unittest.TestCase):
    def test_mixing_requires_all_three_individual_cohorts(self):
        monitor = LiquidRouteMonitor(config("liquid_mixing"))
        state = {"liquid_transfers": {"mix_5": received(), "mix_4": received()}}
        for _ in range(4):
            monitor.update(state)
        self.assertFalse(monitor.success())
        state["liquid_transfers"]["mix_3"] = received()
        monitor.update(state)
        self.assertFalse(monitor.success())
        monitor.update(state)
        self.assertTrue(monitor.success())

    def test_late_spill_revokes_terminal_success(self):
        monitor = LiquidRouteMonitor(config("liquid_mixing"))
        state = {"liquid_transfers": {name: received() for name in ("mix_5", "mix_4", "mix_3")}}
        monitor.update(state)
        monitor.update(state)
        self.assertTrue(monitor.success())
        state["liquid_transfers"]["mix_5"]["outside_fraction"] = 0.2
        monitor.update(state)
        self.assertFalse(monitor.success())

    def test_cleaning_reuses_one_cohort_and_requires_intermediate_transfer(self):
        cfg = config("clean_beaker")
        routes = liquid_routes(cfg)
        self.assertEqual(routes[0].initial_source, routes[1].initial_source)
        monitor = LiquidRouteMonitor(cfg)
        drain = {"liquid_transfers": {"drain": received()}}
        for _ in range(4):
            monitor.update(drain)
        self.assertFalse(monitor.success())
        rinse = {"liquid_transfers": {"rinse": received()}}
        monitor.update(rinse)
        monitor.update(rinse)
        self.assertEqual(monitor.passed, {"rinse"})
        monitor.update(drain)
        monitor.update(drain)
        self.assertTrue(monitor.success())

    def test_reset_does_not_reuse_previous_success(self):
        monitor = LiquidRouteMonitor(config("mobile_pour"))
        state = {"liquid_transfers": {"pour": received()}}
        monitor.update(state)
        monitor.update(state)
        self.assertTrue(monitor.success())
        monitor.reset()
        self.assertFalse(monitor.success())
        monitor.update(state)
        self.assertFalse(monitor.success())

    def test_missing_observation_fails_closed(self):
        monitor = LiquidRouteMonitor(config("mobile_pour"))
        for _ in range(10):
            monitor.update({})
        self.assertFalse(monitor.success())

    def test_conical_routes_to_actual_receiver(self):
        route, = liquid_routes(config("open_transport_pour"))
        self.assertEqual(route.source, "/World/conical_bottle02")
        self.assertEqual(route.target, "/World/beaker1")
        self.assertEqual(route.initial_source, route.source)


if __name__ == "__main__":
    unittest.main()
