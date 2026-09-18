"""Reusable Isaac Sim particle fluid, independent of tasks/controllers/assets.

Import after SimulationApp starts. Configure the world before cameras/physics
initialize. Positions and velocities are world-space metres and metres/second;
each instance owns a unique USD root and never follows a container kinematically.
"""

from typing import Any

import carb
import numpy as np
import omni.physx
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from utils.liquid_particles import ParticleFluidParameters, particle_state


def configure_particle_world(
    world: Any, stage: Any, *, physics_dt: float = 1 / 480, rendering_dt: float = 1 / 60
) -> None:
    """Opt into GPU PBD and a fixed substep clock before scene initialization."""
    if (
        not np.isfinite([physics_dt, rendering_dt]).all()
        or physics_dt <= 0
        or rendering_dt < physics_dt
        or not np.isclose(rendering_dt / physics_dt, round(rendering_dt / physics_dt))
    ):
        raise ValueError("Particle physics needs positive timesteps and an integer substep ratio")
    if world.is_playing():
        raise RuntimeError("Configure particle physics before simulation starts")
    world.stop()
    omni.physx.get_physx_interface().overwrite_gpu_setting(1)
    context = world.get_physics_context()
    context.enable_gpu_dynamics(True)
    context.set_broadphase_type("GPU")
    world.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            api.CreateTimeStepsPerSecondAttr(round(1 / physics_dt))
            api.CreateEnableGPUDynamicsAttr(True)
            api.CreateBroadphaseTypeAttr("GPU")


class ParticleFluid:
    """A bounded particle set with configurable location, material and reset I/O.

    Callers supply already-safe seed positions and real container collisions.
    This utility does not modify container geometry or decide task success.
    """

    def __init__(
        self,
        stage: Any,
        *,
        scene_path: str,
        root_path: str,
        positions: Any,
        velocities: Any = None,
        parameters: ParticleFluidParameters | None = None,
    ) -> None:
        self.stage = stage
        self.parameters = parameters or ParticleFluidParameters()
        positions, velocities = particle_state(positions, velocities)
        root = Sdf.Path(root_path)
        if not root.IsAbsolutePath() or not root.IsPrimPath() or str(root) in ("/", "/World"):
            raise ValueError("Fluid root must be a specific absolute prim path")
        if stage.GetPrimAtPath(root).IsValid():
            raise ValueError(f"Fluid root already exists; refusing to overwrite: {root_path}")
        scene = stage.GetPrimAtPath(scene_path)
        if not scene.IsA(UsdPhysics.Scene):
            raise ValueError(f"Missing physics scene: {scene_path}")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise ValueError("ParticleFluid requires a metre-scale stage")
        if not PhysxSchema.PhysxSceneAPI(scene).GetEnableGPUDynamicsAttr().Get():
            raise ValueError("GPU PBD must be configured before creating liquid")
        self.root_path, self.scene_path = str(root), str(scene.GetPath())
        # Reset the local stack: even a translated/scaled parent must not move
        # world-space particles or attach them to a moving container.
        UsdGeom.Xform.Define(stage, root).SetResetXformStack(True)
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/updateToUsd", True)
        settings.set_bool("/physics/updateParticlesToUsd", True)
        self.system_path = root.AppendChild("System")
        rest = self.parameters.spacing / 2
        particleUtils.add_physx_particle_system(
            stage,
            self.system_path,
            simulation_owner=Sdf.Path(self.scene_path),
            contact_offset=rest * 3.0,
            rest_offset=rest,
            particle_contact_offset=rest / 0.6,
            solid_rest_offset=rest,
            fluid_rest_offset=rest,
            solver_position_iterations=16,
            max_velocity=5.0,
            max_depenetration_velocity=2.0,
        )
        material_path = str(root.AppendChild("Water"))
        particleUtils.add_pbd_particle_material(
            stage,
            material_path,
            friction=self.parameters.friction,
            viscosity=self.parameters.viscosity,
            cohesion=self.parameters.cohesion,
            surface_tension=self.parameters.surface_tension,
            damping=self.parameters.damping,
        )
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.parameters.color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.13)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(self.system_path), material_path)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(self.system_path)).Bind(material)
        if self.parameters.isosurface:
            particleUtils.add_physx_particle_isosurface(
                stage,
                self.system_path,
                enabled=True,
                grid_spacing=rest,
                surface_distance=rest * 1.5,
                num_mesh_smoothing_passes=2,
                num_mesh_normal_smoothing_passes=2,
            )
        prim = particleUtils.add_physx_particleset_pointinstancer(
            stage,
            root.AppendChild("Particles"),
            positions=Vt.Vec3fArray.FromNumpy(positions),
            velocities=Vt.Vec3fArray.FromNumpy(velocities),
            particle_system_path=self.system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=self.parameters.density * self.parameters.spacing**3,
            density=0.0,
        )
        self.particles = UsdGeom.PointInstancer(prim)
        prototype = UsdGeom.Sphere(stage.GetPrimAtPath(root.AppendPath("Particles/particlePrototype0")))
        prototype.GetRadiusAttr().Set(rest)
        UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(material)
        if self.parameters.isosurface:
            prototype.GetVisibilityAttr().Set("invisible")

    def pause(self) -> None:
        PhysxSchema.PhysxParticleSystem(self.stage.GetPrimAtPath(self.system_path)).GetParticleSystemEnabledAttr().Set(
            False
        )

    def reset(self, positions: Any, velocities: Any = None) -> None:
        """Reset an episode's liquid once; never use this to guide a pour."""
        positions, velocities = particle_state(positions, velocities)
        self.pause()
        self.particles.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
        self.particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(velocities))
        self.particles.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(positions)))
        PhysxSchema.PhysxParticleSystem(self.stage.GetPrimAtPath(self.system_path)).GetParticleSystemEnabledAttr().Set(
            True
        )

    def snapshot(self) -> dict[str, Any]:
        positions, velocities = particle_state(
            self.particles.GetPositionsAttr().Get(), self.particles.GetVelocitiesAttr().Get()
        )
        return {
            "version": 1,
            "root_path": self.root_path,
            "scene_path": self.scene_path,
            "parameters": self.parameters.snapshot(),
            "positions": positions.tolist(),
            "velocities": velocities.tolist(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if (
            snapshot.get("version") != 1
            or snapshot.get("root_path") != self.root_path
            or snapshot.get("scene_path") != self.scene_path
            or snapshot.get("parameters") != self.parameters.snapshot()
        ):
            raise ValueError("Recorded fluid identity or physical parameters do not match")
        self.reset(snapshot["positions"], snapshot["velocities"])
