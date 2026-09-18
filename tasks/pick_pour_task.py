from typing import Any

from isaacsim.core.utils.prims import set_prim_visibility
from loguru import logger
from pxr import Usd, UsdGeom, UsdPhysics

from utils.liquid_metrics import liquid_enabled
from utils.pour_liquid import PourLiquid
from utils.replay_physics_snapshot import capture_pre_warmup, restore_pre_warmup
from utils.robot_rendering import prepare_robot_visuals

from .base_task import BaseTask


class PickPourTask(BaseTask):
    """Pick-and-pour task: pick source beaker, tilt to pour into target.

    Source object cycles across multiple beaker variants (visibility managed);
    target object position is randomised from ``cfg.task.left_pos``.
    """

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        if liquid_enabled(cfg):
            PourLiquid.configure_world(world, stage, cfg)
            self._liquid_robot_rendering = prepare_robot_visuals(stage, world, robot.prim_path)
        super().__init__(cfg, world, stage, robot)
        self.target_path = cfg.target_path
        # The receiver is not in obj_paths, but its live rigid-body state is
        # equally important to liquid reset/replay and must be synchronized.
        if liquid_enabled(cfg):
            root = stage.GetPrimAtPath(self.target_path)
            inverse = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            self._object_rigid_body_offsets[self.target_path] = [
                (
                    str(prim.GetPath()),
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * inverse,
                )
                for prim in Usd.PrimRange(root)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
        self.liquid = PourLiquid(stage, world, cfg) if liquid_enabled(cfg) else None

    def reset(self) -> None:
        if self.liquid is not None:
            self.liquid.pause_for_reset()
        super().reset()
        self.robot.initialize()
        self.current_obj_path = self.place_objects_with_visibility_management(self.current_obj_idx)
        self._episode_init_state["extra"]["current_obj_idx"] = self.current_obj_idx
        self.randomize_object_position(self.target_path, self.cfg.task.left_pos)
        self._apply_source_mass_override()
        if self.liquid is not None:
            # Initialize filled cups at rest, not as a 15 cm drop of glass+water.
            self._episode_init_state["extra"]["robot_rendering"] = self._liquid_robot_rendering
            # This is scene setup only; no per-action object/liquid pose writes.
            for _ in range(round(1.5 / self.world.get_physics_dt())):
                self.world.step(render=False)
            self._episode_init_state["extra"]["physics_pre_warmup"] = capture_pre_warmup(self)
            self._episode_init_state["extra"]["liquid"] = self.liquid.reset(self.current_obj_path)

    def reset_with_init_state(self, init_state: dict) -> None:
        if self.liquid is not None:
            self.liquid.pause_for_reset()
        super().reset_with_init_state(init_state)
        current_obj_idx = init_state.get("extra", {}).get("current_obj_idx", self.current_obj_idx)
        self.current_obj_path = self.obj_configs[current_obj_idx]["path"]
        for i, obj_cfg in enumerate(self.obj_configs):
            prim = self.stage.GetPrimAtPath(obj_cfg["path"])
            if prim.IsValid():
                set_prim_visibility(prim, i == current_obj_idx)
        self._apply_source_mass_override()
        if self.liquid is not None:
            extra = init_state.get("extra", {})
            if "liquid" not in extra or "physics_pre_warmup" not in extra:
                raise ValueError("Liquid replay requires freshly collected liquid + physics initial state")
            restore_pre_warmup(self, extra["physics_pre_warmup"])
            self.liquid.reset(self.current_obj_path, extra["liquid"])
            self._episode_init_state["extra"]["robot_rendering"] = self._liquid_robot_rendering

    def _apply_source_mass_override(self) -> None:
        """Lower the source object's mass for a wobble-free pour, when requested.

        Opt-in via the pour configs only (so the shared lab USD and the ~10 other
        tasks that grasp these beakers are untouched):

            task:
              source_mass_kg: 0.05      # absolute override, OR
              source_mass_scale: 0.33   # multiply the body's current PhysX mass
        """
        task = getattr(self.cfg, "task", None)
        mass_kg = getattr(task, "source_mass_kg", None) if task else None
        scale = getattr(task, "source_mass_scale", None) if task else None
        if mass_kg is None and scale is None:
            return
        self.object_utils.set_object_mass(self.current_obj_path + "/mesh", mass_kg=mass_kg, scale=scale)

    def step(self) -> dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        source_quaternion = self.object_utils.get_transform_quat(object_path=self.current_obj_path + "/mesh")
        liquid = self.liquid.observe() if self.liquid is not None else None
        if liquid is not None and self.frame_idx % 120 == 0:
            logger.info(
                "[liquid] frame={} target={:.1%} source={:.1%} outside/inflight={:.1%}",
                self.frame_idx,
                liquid["target_fraction"],
                liquid["source_fraction"],
                liquid["outside_fraction"],
            )
        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            target_path=self.target_path,
            additional_info={
                "object_quaternion": source_quaternion,
                "source_beaker": self.current_obj_path,
                **({"liquid": liquid} if liquid is not None else {}),
            },
        )

    def get_reserved_placement_requests(self) -> list[dict[str, Any]]:
        requests = super().get_reserved_placement_requests()
        requests.append(
            {
                "path": self.target_path,
                "position_range": self.cfg.task.left_pos,
            }
        )
        return requests
