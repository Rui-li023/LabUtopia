"""Task-owned PhysX PBD liquid; no dependency on the external RoboDojo checkout.

Inspired by RoboDojo's fluid objects. Uses NVIDIA's installed particleUtils API.
Visual glass meshes and materials are untouched; Blender-cut collision patches
of the original glass are instantiated only in the scene's runtime edit layer.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pxr import PhysxSchema, UsdGeom, UsdPhysics

from utils.gripper_contact import finger_pad_snapshot
from utils.liquid_metrics import BeakerInterior, ConicalInterior, liquid_accounting
from utils.liquid_particles import sample_container
from utils.liquid_transfers import liquid_routes
from utils.particle_fluid import ParticleFluid, ParticleFluidParameters, configure_particle_world


class PourLiquid:
    ROOT = "/LabUtopiaPourLiquid"

    @staticmethod
    def configure_world(world: Any, stage: Any, cfg: Any = None) -> None:
        """Configure GPU substeps before cameras, preserving the 60 Hz action clock.

        Tasks may refine the physics clock without changing their motion speed.
        Recorded liquid snapshots reject replay with a different timestep.
        """
        liquid = getattr(cfg, "liquid", None)
        physics_dt = float(getattr(liquid, "physics_dt", 1 / 480))
        configure_particle_world(world, stage, physics_dt=physics_dt)

    def __init__(self, stage: Any, world: Any, cfg: Any) -> None:
        self.stage, self.world, self.cfg = stage, world, cfg
        self.gripper_contact = finger_pad_snapshot(cfg)
        self.interior = BeakerInterior()
        self.interiors: dict[str, BeakerInterior] = {}
        self.collision_hashes: dict[str, str] = {}
        self.routes = liquid_routes(cfg) if str(cfg.task_type) != "pick_pour" else ()
        if self.routes:
            self.source_paths = list(dict.fromkeys(route.initial_source + "/mesh" for route in self.routes))
            self.target_path = self.routes[0].target + "/mesh"
            containers = list(
                dict.fromkeys(path + "/mesh" for route in self.routes for path in (route.source, route.target))
            )
        else:
            self.source_paths = [str(obj.path) + "/mesh" for obj in cfg.task.obj_paths]
            self.target_path = str(cfg.target_path) + "/mesh"
            containers = [*self.source_paths, self.target_path]
        self.container_paths = containers
        self.cohorts: dict[str, tuple[int, int]] = {}
        self.source_path = self.source_paths[0]
        self.initial_count = 0
        self.spacing = float(getattr(cfg.liquid, "particle_spacing", 0.004))
        self.fill_height = float(getattr(cfg.liquid, "fill_height", 0.025))
        if not 0.002 <= self.spacing <= 0.006 or not 0.01 <= self.fill_height <= 0.05:
            raise ValueError("Liquid spacing/fill height outside calibrated beaker range")
        for name, default in (
            ("min_target_fraction", 0.8),
            ("max_source_fraction", 0.15),
            ("max_outside_fraction", 0.05),
        ):
            value = float(getattr(cfg.liquid, name, default))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"liquid.{name} must be a finite fraction in [0, 1]")
        if int(getattr(cfg.liquid, "settle_steps", 60)) < 1:
            raise ValueError("Liquid stability hold must be positive")
        context = world.get_physics_context()
        self.scene_path = context.prim_path
        # PBD fluids require GPU physics even when articulation observations use numpy.
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr("GPU")
        for path in containers:
            self._configure_container(path)
        hashes = set(self.collision_hashes.values())
        # Keep historical all-beaker v1/v2 snapshots compatible. Mixed vessels
        # must certify EVERY collider, not only the last configured receiver.
        self.collision_hash = next(iter(hashes)) if len(hashes) == 1 else hashlib.sha256(
            json.dumps(self.collision_hashes, sort_keys=True).encode()
        ).hexdigest()
        self.isosurface = bool(getattr(cfg.liquid, "isosurface", True))
        initial = self._seed_points()
        self.fluid = ParticleFluid(
            stage, scene_path=str(self.scene_path), root_path=self.ROOT, positions=initial,
            parameters=ParticleFluidParameters(spacing=self.spacing, isosurface=self.isosurface),
        )
        self.system_path, self.particles = self.fluid.system_path, self.fluid.particles
        logger.info("[liquid] GPU PBD enabled; original glass + hollow compound collisions; {} particles", len(initial))

    def _configure_container(self, path: str) -> None:
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Liquid requires original lab_001 beaker mesh: {path}")
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        # Do not silently use this calibrated interior for unrelated bottles.
        if points.shape == (1698, 3) and np.allclose(
            points.min(0), [-33.630287, -86.688225, -75.332466], atol=0.002
        ):
            kind, interior = "beaker", BeakerInterior()
        elif points.shape == (1214, 3) and np.allclose(
            points.min(0), [-462.871399, 0.000095, -462.867584], atol=0.002
        ) and np.allclose(points.max(0), [462.864532, 1645.240723, 462.866821], atol=0.002):
            kind, interior = "conical", ConicalInterior()
        else:
            raise ValueError(f"Uncalibrated liquid container geometry: {path}")
        self.interiors[path] = interior
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
        asset = Path(__file__).resolve().parents[1] / f"assets/chemistry_lab/lab_001/pour_liquid/{kind}_collision.json"
        collision_data = json.loads(asset.read_text())
        self.collision_hashes[path] = hashlib.sha256(asset.read_bytes()).hexdigest()
        patches = collision_data["patches"]
        scale_volume = abs(np.linalg.det(self.matrix(path)[:3, :3]))
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(collision_data["native_glass_volume"] * scale_volume * 2230.0)
        mass.CreateDensityAttr(0.0)
        for patch in patches:
            collision_mesh = UsdGeom.Mesh.Define(self.stage, path + "/LiquidCollision/" + patch["name"])
            collision_mesh.CreatePointsAttr(patch["points"])
            collision_mesh.CreateFaceVertexCountsAttr(patch["counts"])
            collision_mesh.CreateFaceVertexIndicesAttr(patch["indices"])
            collision_mesh.CreateSubdivisionSchemeAttr("none")
            collision_mesh.CreateVisibilityAttr("invisible")
            child = collision_mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexHull")
            collision = PhysxSchema.PhysxCollisionAPI.Apply(child)
            collision.CreateContactOffsetAttr(0.0005)
            collision.CreateRestOffsetAttr(0.0)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateSolverPositionIterationCountAttr(16)
        body.CreateSolverVelocityIterationCountAttr(4)

    def pause_for_reset(self) -> None:
        self.fluid.pause()

    def matrix(self, path: str) -> np.ndarray:
        return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)))

    def _initial_points(self, path: str) -> np.ndarray:
        return sample_container(
            self.interiors[path] if self.routes else self.interior, self.matrix(path),
            spacing=self.spacing, fill_height=self.fill_height, legacy_cylinder=not self.routes,
        )

    def _seed_points(self) -> np.ndarray:
        if not self.routes:
            return self._initial_points(self.source_path)
        parts = []
        offset = 0
        for path in self.source_paths:
            points = self._initial_points(path)
            if not len(points):
                raise ValueError(f"Empty liquid cohort: {path}")
            self.cohorts[path] = (offset, offset + len(points))
            offset += len(points)
            parts.append(points)
        return np.concatenate(parts)

    def reset(self, source_path: str | None = None, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.routes:
            self.source_path = source_path + "/mesh"
        if snapshot is None:
            positions = self._seed_points()
            velocities = np.zeros_like(positions)
        else:
            version = 2 if self.routes else 1
            if snapshot.get("version") != version or snapshot.get("source_path") != self.source_path:
                raise ValueError("Incompatible recorded liquid initial state")
            if self.routes:
                if snapshot.get("routes") != [asdict(route) for route in self.routes]:
                    raise ValueError("Replay liquid transfer routes differ from collection")
                self.cohorts = {path: tuple(bounds) for path, bounds in snapshot.get("cohorts", {}).items()}
                cursor = 0
                for path in self.source_paths:
                    start, stop = self.cohorts.get(path, (-1, -1))
                    if start != cursor or not isinstance(stop, int) or stop <= start:
                        raise ValueError("Invalid recorded liquid cohort boundaries")
                    cursor = stop
                if set(self.cohorts) != set(self.source_paths) or cursor != len(snapshot["positions"]):
                    raise ValueError("Recorded liquid cohorts do not cover exactly all particles")
            if snapshot.get("particle_spacing") != self.spacing:
                raise ValueError("Replay liquid resolution differs from collection")
            if snapshot.get("gripper_drive_force") != getattr(self.cfg.task, "gripper_drive_force", None):
                raise ValueError("Replay gripper force calibration differs from collection")
            if snapshot.get("gripper_max_joint_velocity") != getattr(self.cfg.task, "gripper_max_joint_velocity", None):
                raise ValueError("Replay gripper velocity limit differs from collection; collect a fresh dataset")
            if snapshot.get("gripper_contact", {}) != self.gripper_contact:
                raise ValueError("Replay finger-pad contact differs from collection")
            if (
                snapshot.get("collision_hash") != self.collision_hash
                or snapshot.get("physics_dt") != self.world.get_physics_dt()
            ):
                raise ValueError("Replay liquid collision calibration or physics timestep differs from collection")
            positions = np.asarray(snapshot["positions"], dtype=np.float32)
            velocities = np.asarray(snapshot["velocities"], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape != velocities.shape or not len(positions):
            raise ValueError("Invalid or empty liquid initial state")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Liquid initial state must be finite")
        self.initial_count = len(positions)
        positions = positions.astype(np.float32)
        velocities = velocities.astype(np.float32)
        self.fluid.reset(positions, velocities)
        return {
            "version": 2 if self.routes else 1,
            "source_path": self.source_path,
            "particle_spacing": self.spacing,
            "collision_hash": self.collision_hash,
            "physics_dt": self.world.get_physics_dt(),
            "positions": positions.tolist(),
            "velocities": velocities.tolist(),
            **({"gripper_drive_force": float(self.cfg.task.gripper_drive_force)}
               if getattr(self.cfg.task, "gripper_drive_force", None) is not None else {}),
            **({"gripper_max_joint_velocity": float(self.cfg.task.gripper_max_joint_velocity)}
               if getattr(self.cfg.task, "gripper_max_joint_velocity", None) is not None else {}),
            **({"gripper_contact": self.gripper_contact} if self.gripper_contact else {}),
            **({"routes": [asdict(route) for route in self.routes], "cohorts": self.cohorts} if self.routes else {}),
        }

    def observe(self) -> dict[str, Any]:
        points = np.asarray(self.particles.GetPositionsAttr().Get(), dtype=float)
        if self.routes:
            valid = points.shape == (self.initial_count, 3) and np.isfinite(points).all()
            result = {}
            for route in self.routes:
                start, stop = self.cohorts[route.initial_source + "/mesh"]
                result[route.name] = self._observe_pair(
                    points[start:stop] if valid else np.empty((0, 3)),
                    stop - start,
                    route.source + "/mesh",
                    route.target + "/mesh",
                )
            return result
        return self._observe_pair(points, self.initial_count, self.source_path, self.target_path)

    def _observe_pair(self, points: np.ndarray, count: int, source_path: str, target_path: str) -> dict[str, Any]:
        source, target = self.matrix(source_path), self.matrix(target_path)
        source_interior, target_interior = self.interiors[source_path], self.interiors[target_path]
        metrics = liquid_accounting(points, count, source, target, source_interior, target_interior)
        source_mouth = np.array([source_interior.center_x, source_interior.top, source_interior.center_z])
        target_mouth = np.array([target_interior.center_x, target_interior.top, target_interior.center_z])
        metrics.update(
            source_mouth=source_mouth @ source[:3, :3] + source[3, :3],
            target_mouth=target_mouth @ target[:3, :3] + target[3, :3],
            source_up=source[1, :3] / np.linalg.norm(source[1, :3]),
            source_basis=source[:3, :3] / np.linalg.norm(source[:3, :3], axis=1)[:, None],
            target_up=target[1, :3] / np.linalg.norm(target[1, :3]),
            source_radius=source_interior.mouth_radius * np.linalg.norm(source[0, :3]),
            target_radius=target_interior.mouth_radius * np.linalg.norm(target[0, :3]),
        )
        return metrics
