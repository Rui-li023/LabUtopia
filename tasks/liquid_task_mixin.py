"""Task-owned lifecycle for composite/mobile liquid scenes."""

from typing import Any

from loguru import logger
from pxr import Usd, UsdGeom, UsdPhysics

from utils.liquid_metrics import liquid_enabled
from utils.pour_liquid import PourLiquid
from utils.replay_physics_snapshot import capture_pre_warmup, restore_pre_warmup
from utils.robot_mass_properties import prepare_robot_mass_properties, validate_robot_mass_replay
from utils.robot_rendering import prepare_robot_visuals


class LiquidTaskMixin:
    LIQUID_REPLAY_SUPPORTS: tuple[str, ...] = ()

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        if liquid_enabled(cfg):
            PourLiquid.configure_world(world, stage, cfg)
            self._liquid_robot_rendering = prepare_robot_visuals(stage, world, robot.prim_path)
            self._liquid_robot_mass_properties = prepare_robot_mass_properties(
                stage, world, robot.prim_path, str(getattr(cfg.task, "robot_inertia_mode", "asset"))
            )
        super().__init__(cfg, world, stage, robot)
        self.liquid = PourLiquid(stage, world, cfg) if liquid_enabled(cfg) else None
        if self.liquid is not None:
            roots = [path.rsplit("/", 1)[0] for path in self.liquid.container_paths]
            for path in [*roots, *self.LIQUID_REPLAY_SUPPORTS]:
                root = stage.GetPrimAtPath(path)
                inverse = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
                self._object_rigid_body_offsets[path] = [
                    (
                        str(prim.GetPath()),
                        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse,
                    )
                    for prim in Usd.PrimRange(root)
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ]

    def reset(self) -> None:
        if self.liquid is not None:
            self.liquid.pause_for_reset()
        super().reset()
        if self.liquid is not None:
            # Finish the scene's placement/navigation reset BEFORE filling.
            self._episode_init_state["extra"]["robot_rendering"] = self._liquid_robot_rendering
            if self._liquid_robot_mass_properties:
                self._episode_init_state["extra"]["robot_mass_properties"] = self._liquid_robot_mass_properties
            for path in self.liquid.container_paths:
                self._record_object_pose(path.rsplit("/", 1)[0])
            for _ in range(round(1.5 / self.world.get_physics_dt())):
                self.world.step(render=False)
            self._episode_init_state["extra"]["physics_pre_warmup"] = capture_pre_warmup(self)
            self._episode_init_state["extra"]["liquid"] = self.liquid.reset()

    def reset_with_init_state(self, init_state: dict) -> None:
        if self.liquid is not None:
            self.liquid.pause_for_reset()
            extra = init_state.get("extra", {})
            if "liquid" not in extra or "physics_pre_warmup" not in extra:
                raise ValueError("Liquid replay requires freshly collected liquid + physics initial state")
            validate_robot_mass_replay(extra.get("robot_mass_properties", {}), self._liquid_robot_mass_properties)
        super().reset_with_init_state(init_state)
        if self.liquid is not None:
            restore_pre_warmup(self, extra["physics_pre_warmup"])
            self.liquid.reset(snapshot=extra["liquid"])
            self._episode_init_state["extra"]["robot_rendering"] = self._liquid_robot_rendering

    def step(self) -> dict[str, Any] | None:
        state = super().step()
        if state is not None and self.liquid is not None:
            state["liquid_transfers"] = self.liquid.observe()
            if self.frame_idx % 120 == 0:
                for name, metrics in state["liquid_transfers"].items():
                    logger.info(
                        "[liquid-route] {} target={:.1%} source={:.1%} outside/inflight={:.1%}",
                        name,
                        metrics["target_fraction"],
                        metrics["source_fraction"],
                        metrics["outside_fraction"],
                    )
        return state
