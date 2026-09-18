"""Explicit liquid routes and mode-independent, per-cohort transfer gates."""

from dataclasses import dataclass
from typing import Any

from utils.liquid_metrics import StableLiquidGate, liquid_transfer_ok


@dataclass(frozen=True)
class LiquidRoute:
    name: str
    source: str
    target: str
    initial_source: str
    predecessor: str = ""


def liquid_routes(cfg: Any) -> tuple[LiquidRoute, ...]:
    kind = str(cfg.task_type)
    if kind == "clean_beaker":
        return (
            LiquidRoute("rinse", "/World/beaker_hard_2", "/World/beaker_hard_1", "/World/beaker_hard_2"),
            LiquidRoute("drain", "/World/beaker_hard_1", "/World/target_beaker", "/World/beaker_hard_2", "rinse"),
        )
    if kind == "liquid_mixing":
        return tuple(
            LiquidRoute(f"mix_{i}", f"/World/beaker_0{i}", "/World/beaker_4", f"/World/beaker_0{i}") for i in (5, 4, 3)
        )
    if kind == "mobile_pour":
        source = str(cfg.task.pick_object_path)
        return (LiquidRoute("pour", source, str(cfg.task.pour_target.prim_path), source),)
    if kind == "open_transport_pour":
        return (LiquidRoute("pour", "/World/conical_bottle02", "/World/beaker1", "/World/conical_bottle02"),)
    raise ValueError(f"No composite liquid routes registered for {kind}")


class LiquidRouteMonitor:
    """Serial routes retain checkpoints; terminal receivers must still hold liquid.

    Each mixing ingredient is measured against its OWN original particle count.
    No controller phase or angle-only signal can substitute for liquid arrival.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.routes = liquid_routes(cfg)
        self.gates = {route.name: StableLiquidGate() for route in self.routes}
        self.passed: set[str] = set()
        self.current: set[str] = set()
        self.terminal = {r.name for r in self.routes} - {r.predecessor for r in self.routes if r.predecessor}

    def reset(self) -> None:
        for gate in self.gates.values():
            gate.reset()
        self.passed.clear()
        self.current.clear()

    def update(self, state: dict[str, Any]) -> None:
        self.current.clear()
        observations = state.get("liquid_transfers", {})
        previous = self.passed.copy()
        for route in self.routes:
            metrics = observations.get(route.name, {})
            # A serial second transfer cannot be certified before the first.
            eligible = not route.predecessor or route.predecessor in previous
            if self.gates[route.name].check({"liquid": metrics}, self.cfg, lifted=eligible):
                self.current.add(route.name)
                self.passed.add(route.name)

    def success(self) -> bool:
        return len(self.passed) == len(self.routes) and self.terminal <= self.current

    def transfer_complete(self, name: str, state: dict[str, Any]) -> bool:
        return name in self.passed and liquid_transfer_ok(
            {"liquid": state.get("liquid_transfers", {}).get(name, {})}, self.cfg
        )
