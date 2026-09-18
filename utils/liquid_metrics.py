"""Geometry-only liquid accounting, shared by collect, replay and inference.

Fractions always use the initial particle count, never just the particles that
survived a spill. Coordinates use USD's row-vector matrix convention.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class BeakerInterior:
    """Measured interior of lab_001's beaker mesh, in native mesh coordinates."""

    center_x: float = 53.346556
    center_z: float = 0.0
    radius: float = 66.789001
    bottom: float = -83.688225
    top: float = 90.0
    # Inner-wall rings of the rounded base. A plain cylinder would also count
    # liquid outside the curved glass near the floor.
    lower_profile: tuple[tuple[float, float], ...] = (
        (-83.688225, 52.776886),
        (-82.588, 58.232431),
        (-79.585, 62.686194),
        (-75.132, 65.688327),
        (-69.676, 66.789001),
    )

    @property
    def mouth_radius(self) -> float:
        return self.lower_profile[-1][1]

    def contains(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        inverse = np.linalg.inv(matrix)
        local = points @ inverse[:3, :3] + inverse[3, :3]
        radial_squared = (local[:, 0] - self.center_x) ** 2 + (local[:, 2] - self.center_z) ** 2
        profile = np.asarray(self.lower_profile)
        radius = np.minimum(self.radius, np.interp(local[:, 1], profile[:, 0], profile[:, 1]))
        return (radial_squared < radius**2) & (local[:, 1] > self.bottom) & (local[:, 1] < self.top)


@dataclass(frozen=True)
class ConicalInterior(BeakerInterior):
    """Original conical_bottle02 inner rings (native Y up, world scale 0.0001).

    The full profile includes the narrowing shoulder and open neck; a cylinder
    would wrongly count liquid outside the cone as retained.
    """

    center_x: float = 0.0
    center_z: float = 0.0
    radius: float = 414.755
    bottom: float = 48.280
    top: float = 1645.240
    lower_profile: tuple[tuple[float, float], ...] = (
        (48.280, 329.605), (55.610, 375.482), (77.600, 403.713),
        (115.770, 414.755), (171.630, 409.067), (338.430, 372.844),
        (709.400, 292.280), (1074.020, 213.560), (1221.780, 182.870),
        (1295.620, 177.863), (1438.470, 176.193), (1576.320, 181.089),
        (1635.130, 195.774), (1645.240, 214.641),
    )


def liquid_accounting(
    points: np.ndarray,
    initial_count: int,
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    interior: BeakerInterior | None = None,
    target_interior: BeakerInterior | None = None,
) -> dict[str, Any]:
    """Count target/source/outside (including in-flight), failing closed on bad data."""
    result = {
        "valid": False,
        "initial_count": initial_count,
        "target_count": 0,
        "source_count": 0,
        "outside_count": initial_count,
        "target_fraction": 0.0,
        "source_fraction": 0.0,
        "outside_fraction": 1.0,
    }
    points = np.asarray(points, dtype=float)
    if initial_count <= 0 or points.shape != (initial_count, 3) or not np.isfinite(points).all():
        return result
    interior = interior or BeakerInterior()
    try:
        if not np.isfinite(source_matrix).all() or not np.isfinite(target_matrix).all():
            return result
        source = interior.contains(points, source_matrix)
        target = (target_interior or interior).contains(points, target_matrix) & ~source
    except (ValueError, np.linalg.LinAlgError):
        return result
    ns, nt = int(source.sum()), int(target.sum())
    no = initial_count - ns - nt
    result.update(
        valid=True,
        source_count=ns,
        target_count=nt,
        outside_count=no,
        source_fraction=ns / initial_count,
        target_fraction=nt / initial_count,
        outside_fraction=no / initial_count,
    )
    return result


def liquid_enabled(cfg: Any) -> bool:
    return bool(getattr(getattr(cfg, "liquid", None), "enabled", False))


def liquid_transfer_ok(state: dict[str, Any], cfg: Any) -> bool:
    """A missing/empty liquid observation cannot pass a liquid-enabled task."""
    if not liquid_enabled(cfg):
        return True
    metrics = state.get("liquid", {})
    options = cfg.liquid
    return bool(
        metrics.get("valid", False)
        and metrics.get("initial_count", 0) > 0
        and metrics.get("target_fraction", 0.0) >= float(getattr(options, "min_target_fraction", 0.80))
        and metrics.get("source_fraction", 1.0) <= float(getattr(options, "max_source_fraction", 0.15))
        and metrics.get("outside_fraction", 1.0) <= float(getattr(options, "max_outside_fraction", 0.05))
    )


class StableLiquidGate:
    """Require real transfer, upright containers and a consecutive settling hold."""

    def __init__(self) -> None:
        self.steps = 0

    def reset(self) -> None:
        self.steps = 0

    def check(self, state: dict[str, Any], cfg: Any, lifted: bool) -> bool:
        metrics = state.get("liquid", {})
        source_up = np.asarray(metrics.get("source_up", [0, 0, 0]))
        target_up = np.asarray(metrics.get("target_up", [0, 0, 0]))
        valid = (
            liquid_transfer_ok(state, cfg)
            and lifted
            and source_up.shape == (3,)
            and target_up.shape == (3,)
            and source_up[2] >= np.cos(np.deg2rad(20))
            and target_up[2] >= np.cos(np.deg2rad(10))
        )
        self.steps = self.steps + 1 if valid else 0
        required = int(getattr(cfg.liquid, "settle_steps", 60))
        if self.steps == required:
            logger.info(
                "[LIQUID-SUCCESS] initial={} target={} source={} outside={} hold={}",
                metrics["initial_count"],
                metrics["target_count"],
                metrics["source_count"],
                metrics["outside_count"],
                self.steps,
            )
        return self.steps >= required
