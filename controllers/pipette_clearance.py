"""Conservative arm/rack screening for planning, never simulator collision edits.

The robot spheres are the existing motion-description approximation. Rack
boxes must enclose the actual rack in world coordinates. A positive result
is a planning clearance estimate, not proof that arbitrary meshes are free
of collision; physical contact reporting remains the final check.
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def sphere_box_clearances(centers, radii, boxes):
    """Signed surface distances for N spheres against M axis-aligned boxes."""
    centers, radii, boxes = (np.asarray(value, dtype=float) for value in (centers, radii, boxes))
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or radii.shape != (len(centers),)
        or boxes.ndim != 3
        or boxes.shape[1:] != (2, 3)
        or not len(centers)
        or not len(boxes)
        or not all(np.isfinite(value).all() for value in (centers, radii, boxes))
        or np.any(radii <= 0)
        or np.any(boxes[:, 1] <= boxes[:, 0])
    ):
        raise ValueError("Expected finite sphere centers/radii and nonempty world-space boxes")
    midpoint = np.mean(boxes, axis=1)
    half_extent = (boxes[:, 1] - boxes[:, 0]) / 2
    outside = np.abs(centers[:, None, :] - midpoint) - half_extent
    signed = np.linalg.norm(np.maximum(outside, 0), axis=2) + np.minimum(np.max(outside, axis=2), 0)
    return signed - radii[:, None]


def oriented_mesh_bounds(points):
    """PCA-aligned box enclosing every vertex; never shrink a physical mesh."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Expected finite 3D mesh vertices")
    origin = np.mean(points, axis=0)
    _, axes = np.linalg.eigh(np.cov((points - origin).T))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    local = (points - origin) @ axes
    bounds = np.array([np.min(local, axis=0), np.max(local, axis=0)])
    if np.any(bounds[1] - bounds[0] <= 1e-9):
        raise ValueError("Rack collider must have nonzero thickness")
    frame = np.eye(4)
    frame[:3, :3], frame[:3, 3] = axes, origin
    return bounds, frame


class PlanningBox:
    """In-memory obstacle adapter for Lula; never creates or edits USD prims."""

    def __init__(self, bounds, frame, padding=0.0):
        bounds, frame = np.asarray(bounds, dtype=float), np.asarray(frame, dtype=float)
        if (
            bounds.shape != (2, 3)
            or frame.shape != (4, 4)
            or not np.isfinite(bounds).all()
            or not np.isfinite(frame).all()
            or np.any(bounds[1] <= bounds[0])
            or not np.isfinite(padding)
            or padding < 0
            or not np.allclose(frame[3], [0, 0, 0, 1])
            or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3))
            or np.linalg.det(frame[:3, :3]) <= 0
        ):
            raise ValueError("Expected a proper rigid planning box frame and positive extents")
        self._center = frame[:3, :3] @ np.mean(bounds, axis=0) + frame[:3, 3]
        self._orientation = Rotation.from_matrix(frame[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self._size = bounds[1] - bounds[0] + 2 * padding

    def get_world_pose(self):
        return self._center.copy(), self._orientation.copy()

    def get_size(self):
        return 1.0

    def get_local_scale(self):
        return self._size.copy()


def cover_box_with_spheres(bounds, frame, *, cell_size=0.02):
    """Cover an entire measured OBB volume for carried-object path planning.

    Each rectangular cell is enclosed by its circumsphere. Unlike fitting
    spheres only to vertices, this also covers triangle interiors. Spheres
    deliberately overestimate geometry and never create physical colliders.
    """
    PlanningBox(bounds, frame)  # Validate finite proper frame and positive extents.
    bounds, frame = np.asarray(bounds, dtype=float), np.asarray(frame, dtype=float)
    if not np.isfinite(cell_size) or not 0.005 <= cell_size <= 0.05:
        raise ValueError("Planning sphere cell size must be in [5, 50] mm")
    extent = bounds[1] - bounds[0]
    counts = np.ceil(extent / cell_size)
    if not np.isfinite(counts).all() or np.any(counts > 10000) or np.prod(counts) > 10000:
        raise ValueError("Carried-object planning cover would exceed 10000 spheres")
    counts = counts.astype(int)
    spacing = extent / counts
    indices = np.indices(counts).reshape(3, -1).T
    centers = bounds[0] + (indices + 0.5) * spacing
    centers = centers @ frame[:3, :3].T + frame[:3, 3]
    radii = np.full(len(centers), np.linalg.norm(spacing) / 2)
    return centers, radii


def cover_convex_mesh_with_spheres(points, *, cell_size=0.01):
    """Cover the entire convex hull, excluding grid cells provably outside it.

    A cell is discarded only if wholly outside a hull half-space. Thus every
    hull point and every original triangle interior remains covered; this is
    still conservative and does not introduce gaps from vertex-only fitting.
    """
    bounds, frame = oriented_mesh_bounds(points)
    centers, radii = cover_box_with_spheres(bounds, frame, cell_size=cell_size)
    local_points = (np.asarray(points) - frame[:3, 3]) @ frame[:3, :3]
    local_centers = (centers - frame[:3, 3]) @ frame[:3, :3]
    planes = ConvexHull(local_points).equations
    half = (bounds[1] - bounds[0]) / np.ceil((bounds[1] - bounds[0]) / cell_size) / 2
    minimum = local_centers @ planes[:, :3].T + planes[:, 3] - np.abs(planes[:, :3]) @ half
    retain = np.all(minimum <= 1e-10, axis=1)
    return centers[retain], radii[retain]


class RackArmClearance:
    """Evaluate existing robot collision spheres using read-only world-frame FK."""

    def __init__(self, solver, spheres: dict, boxes, margin=0.005, box_frames=None):
        if not np.isfinite(margin) or not 0 <= margin <= 0.03:
            raise ValueError("Arm/rack screening margin must be in [0, 0.03] m")
        self.solver = solver
        self.boxes = np.asarray(boxes, dtype=float).copy()
        self.margin = float(margin)
        self.box_frames = None if box_frames is None else np.asarray(box_frames, dtype=float).copy()
        if self.box_frames is not None and (
            self.box_frames.shape != (len(self.boxes), 4, 4)
            or not np.isfinite(self.box_frames).all()
            or not np.allclose(self.box_frames[:, 3], [0, 0, 0, 1])
            or not np.allclose(
                np.transpose(self.box_frames[:, :3, :3], (0, 2, 1)) @ self.box_frames[:, :3, :3], np.eye(3)
            )
        ):
            raise ValueError("Expected rigid orthonormal frames for each rack box")
        self.frames = {}
        for frame, entries in spheres.items():
            centers = np.asarray([entry["center"] for entry in entries], dtype=float)
            radii = np.asarray([entry["radius"] for entry in entries], dtype=float)
            sphere_box_clearances(centers, radii, self.boxes)  # validate before any planning
            self.frames[frame] = (centers, radii)
        if not self.frames:
            raise ValueError("No robot collision spheres available for arm/rack screening")

    @classmethod
    def from_descriptor(cls, solver, descriptor: Path, boxes, margin=0.005, box_frames=None):
        description = yaml.safe_load(Path(descriptor).read_text())
        spheres = {}
        for group in description["collision_spheres"]:
            for frame, entries in group.items():
                if frame in spheres:
                    raise ValueError(f"Duplicate collision-sphere frame: {frame}")
                spheres[frame] = entries
        return cls(solver, spheres, boxes, margin, box_frames)

    def _clearances(self, centers, radii):
        if self.box_frames is None:
            return sphere_box_clearances(centers, radii, self.boxes)
        return np.column_stack(
            [
                sphere_box_clearances((centers - frame[:3, 3]) @ frame[:3, :3], radii, box[None])[:, 0]
                for box, frame in zip(self.boxes, self.box_frames)
            ]
        )

    def limiting_contact(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Expected seven finite arm joints")
        limiting = {"clearance": float("inf")}
        for frame, (centers, radii) in self.frames.items():
            position, matrix = self.solver.compute_forward_kinematics(frame, joints)
            world_centers = centers @ np.asarray(matrix).T + np.asarray(position).reshape(3)
            distances = self._clearances(world_centers, radii)
            sphere, box = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[sphere, box] < limiting["clearance"]:
                limiting = {
                    "clearance": float(distances[sphere, box]),
                    "frame": frame,
                    "sphere": int(sphere),
                    "box": int(box),
                    "center": world_centers[sphere].tolist(),
                    "radius": float(radii[sphere]),
                }
        return limiting

    def minimum_clearance(self, joints):
        return self.limiting_contact(joints)["clearance"]

    def __call__(self, joints):
        return self.minimum_clearance(joints) >= self.margin


def recover_clearance_ik(solver, checker, frame, position, quaternion, seed, bounds, joint_margin=0.1):
    """Move a colliding IK seed locally in the redundant arm space.

    A rejected result is never a command. Preserve the original pose tolerances,
    joint margin and 0.35 rad local trust region, with explicit clearance checks.
    """
    seed, bounds = np.asarray(seed, dtype=float), np.asarray(bounds, dtype=float)
    if (
        seed.shape != (7,)
        or bounds.shape != (7, 2)
        or not np.isfinite(seed).all()
        or not np.isfinite(bounds).all()
        or not np.isfinite(joint_margin)
        or not 0.05 <= joint_margin <= 0.25
    ):
        raise ValueError("Expected finite physical joint bounds and a safe local IK seed")
    lower = np.maximum(bounds[:, 0] + joint_margin, seed - 0.35)
    upper = np.minimum(bounds[:, 1] - joint_margin, seed + 0.35)
    if np.any(lower >= upper):
        return seed.copy(), False
    target = np.asarray(position, dtype=float)
    rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])

    def residual(joints):
        actual, matrix = solver.compute_forward_kinematics(frame, joints)
        angular = (rotation * Rotation.from_matrix(matrix).inv()).as_rotvec()
        # A 1 mm optimization buffer avoids convergence just inside the hard
        # clearance boundary. The final test still uses the configured margin.
        violation = max(0.0, checker.margin + 0.001 - checker.minimum_clearance(joints))
        return np.r_[np.asarray(actual).reshape(3) - target, 0.15 * angular, 10 * violation, 0.0001 * (joints - seed)]

    result = least_squares(
        residual,
        np.clip(seed, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=50,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    error = residual(result.x)
    valid = np.linalg.norm(error[:3]) <= 0.0005 and np.linalg.norm(error[3:6]) / 0.15 <= 0.005
    return result.x, bool(valid and checker(result.x))
