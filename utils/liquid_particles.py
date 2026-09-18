"""Deterministic liquid seeding in any world position, without Isaac imports.

Matrices follow USD's row-vector convention (translation in the last row).
Spacing is in world metres. Shape dimensions/centres are in the supplied frame;
omit the matrix to specify everything in world metres. No particles are moved
after seeding by this module: normal physics owns their motion.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

MAX_PARTICLES = 250_000


@dataclass(frozen=True)
class ParticleFluidParameters:
    spacing: float = 0.004
    density: float = 1000.0
    viscosity: float = 0.01
    cohesion: float = 0.01
    friction: float = 0.05
    surface_tension: float = 0.0
    damping: float = 0.0
    isosurface: bool = True
    color: tuple[float, float, float] = (0.025, 0.22, 0.48)

    def __post_init__(self) -> None:
        for name in ("spacing", "density", "viscosity", "cohesion", "friction", "surface_tension", "damping"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name in ("spacing", "density") and value == 0):
                raise ValueError(f"Invalid particle fluid parameter: {name}")
        color = np.asarray(self.color)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError("Fluid color must contain three values in [0, 1]")
        if not isinstance(self.isosurface, bool):
            raise ValueError("isosurface must be a boolean")
        # YAML supplies lists: detach from the caller's mutable list so the
        # recorded material cannot silently change after the fluid is created.
        object.__setattr__(self, "color", tuple(float(component) for component in color))

    def snapshot(self) -> dict[str, Any]:
        result = asdict(self)
        result["color"] = list(self.color)
        return result


def _limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= MAX_PARTICLES:
        raise ValueError(f"Particle limit must be an integer in [1, {MAX_PARTICLES}]")


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def seed_frame(matrix: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate an affine, orthogonal (optionally scaled) USD transform."""
    matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Seed frame must be a finite 4x4 USD matrix")
    if not np.allclose(matrix[:, 3], [0, 0, 0, 1]):
        raise ValueError("Seed frame must use affine USD row-vector convention")
    scales = np.linalg.norm(matrix[:3, :3], axis=1)
    if np.any(scales < 1e-12):
        raise ValueError("Seed frame must be non-singular")
    axes = matrix[:3, :3] / scales[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6):
        raise ValueError("Sheared seed frames do not preserve particle spacing")
    return matrix, scales


def _axis(start: float, stop: float, step: float, limit: int) -> np.ndarray:
    if stop <= start:
        return np.empty(0)
    count = (stop - start) / step
    if not np.isfinite(count) or count > limit:
        raise ValueError("Seed grid exceeds the particle allocation limit")
    return np.arange(start, stop, step)


def _grid(axes: list[np.ndarray], limit: int) -> np.ndarray:
    _limit(limit)
    if math.prod(map(len, axes)) > limit:
        raise ValueError("Seed grid exceeds the particle allocation limit")
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _world_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if not len(points):
        raise ValueError("Seed volume is empty at this particle spacing")
    result = points @ matrix[:3, :3] + matrix[3, :3]
    if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError("World particle positions are outside the finite PhysX range")
    return result


def sample_volume(
    shape: str,
    *,
    center: Any = (0, 0, 0),
    spacing: float = 0.004,
    size: Any = None,
    radius: float | None = None,
    height: float | None = None,
    matrix: Any = None,
    max_particles: int = MAX_PARTICLES,
) -> np.ndarray:
    """Sample a box, sphere or Z-axis cylinder at a world/local-frame position.

    A custom frame can rotate, translate and scale the volume. For several
    positions, concatenate the returned arrays; the runtime validates the total.
    This seeds a volume, not a container collision model or continuous emitter.
    """
    _limit(max_particles)
    spacing = _positive(spacing, "spacing")
    center = _vector(center, "center")
    transform, scales = seed_frame(matrix)
    if shape == "box":
        dimensions = _vector(size, "box size")
        if np.any(dimensions <= 0):
            raise ValueError("Box dimensions must be positive")
    elif shape in ("sphere", "cylinder"):
        if radius is None:
            raise ValueError("Sphere/cylinder requires radius")
        radius = _positive(radius, "radius")
        if shape == "cylinder" and height is None:
            raise ValueError("Cylinder requires height")
        dimensions = np.array(
            [2 * radius, 2 * radius, 2 * radius if shape == "sphere" else _positive(height, "height")]
        )
    else:
        raise ValueError(f"Unsupported particle volume: {shape}")
    steps = spacing / scales
    axes = [_axis(-extent / 2 + step / 2, extent / 2, step, max_particles) for extent, step in zip(dimensions, steps)]
    points = _grid(axes, max_particles)
    if shape == "sphere":
        points = points[np.sum(points**2, axis=1) < radius**2]
    elif shape == "cylinder":
        points = points[np.sum(points[:, :2] ** 2, axis=1) < radius**2]
    return _world_points(points + center, transform)


def sample_container(
    interior: Any,
    matrix: Any,
    *,
    spacing: float,
    fill_height: float,
    legacy_cylinder: bool = False,
    max_particles: int = MAX_PARTICLES,
) -> np.ndarray:
    """Seed an explicitly calibrated Y-up interior at any USD mesh transform.

    ``interior`` supplies center_x/z, radius, bottom/top and lower_profile rings.
    Legacy mode preserves the old single-beaker lattice exactly for replay.
    This never infers the hollow interior from an arbitrary mesh's bounding box.
    """
    _limit(max_particles)
    spacing = _positive(spacing, "spacing")
    fill_height = _positive(fill_height, "fill_height")
    transform, scales = seed_frame(matrix)
    values = [interior.center_x, interior.center_z, interior.radius, interior.bottom, interior.top]
    profile = np.asarray(interior.lower_profile, dtype=float)
    if (
        not np.isfinite(values).all()
        or interior.radius <= 0
        or interior.top <= interior.bottom
        or profile.ndim != 2
        or profile.shape[1] != 2
        or len(profile) < 1
        or not np.isfinite(profile).all()
        or np.any(profile[:, 1] <= 0)
        or np.any(np.diff(profile[:, 0]) <= 0)
    ):
        raise ValueError("Container requires a finite, ordered interior profile")
    if legacy_cylinder:
        legacy_scale = np.linalg.norm(transform[1, :3])
        step = spacing / legacy_scale
        radius = interior.radius - step
        axes = [
            _axis(-radius, radius + step / 2, step, max_particles),
            _axis(interior.bottom + step, interior.bottom + fill_height / legacy_scale, step, max_particles),
            _axis(-radius, radius + step / 2, step, max_particles),
        ]
        points = _grid(axes, max_particles)
        points = points[points[:, 0] ** 2 + points[:, 2] ** 2 < radius**2]
        points[:, 0] += interior.center_x
        points[:, 2] += interior.center_z
    else:
        steps = spacing / scales
        axes = [
            _axis(-interior.radius + steps[0], interior.radius, steps[0], max_particles),
            _axis(
                interior.bottom + steps[1],
                min(interior.top - steps[1], interior.bottom + fill_height / scales[1]),
                steps[1],
                max_particles,
            ),
            _axis(-interior.radius + steps[2], interior.radius, steps[2], max_particles),
        ]
        points = _grid(axes, max_particles)
        points[:, 0] += interior.center_x
        points[:, 2] += interior.center_z
        radii = np.minimum(interior.radius, np.interp(points[:, 1], profile[:, 0], profile[:, 1]))
        radial = np.hypot(points[:, 0] - interior.center_x, points[:, 2] - interior.center_z)
        points = points[radial < radii - max(steps[0], steps[2])]
    return _world_points(points, transform)


def particle_state(positions: Any, velocities: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate a finite, bounded state; broadcast a constant initial velocity."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not 0 < len(positions) <= MAX_PARTICLES:
        raise ValueError("Particles require a nonempty, bounded Nx3 position array")
    velocities = np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)
    if velocities.shape == (3,):
        velocities = np.broadcast_to(velocities, positions.shape)
    if velocities.shape != positions.shape:
        raise ValueError("Particle velocities must be a 3-vector or match positions")
    if any(
        not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float32).max)
        for value in (positions, velocities)
    ):
        raise ValueError("Particle state must fit finite float32 values")
    return positions.astype(np.float32), velocities.astype(np.float32)
