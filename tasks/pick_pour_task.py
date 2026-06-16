from typing import Any

from isaacsim.core.utils.prims import set_prim_visibility

from .base_task import BaseTask


class PickPourTask(BaseTask):
    """Pick-and-pour task: pick source beaker, tilt to pour into target.

    Source object cycles across multiple beaker variants (visibility managed);
    target object position is randomised from ``cfg.task.left_pos``.
    """

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        super().__init__(cfg, world, stage, robot)
        self.target_path = cfg.target_path

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx
        )
        self._episode_init_state["extra"]["current_obj_idx"] = self.current_obj_idx
        self.randomize_object_position(self.target_path, self.cfg.task.left_pos)
        self._apply_source_mass_override()

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        current_obj_idx = init_state.get("extra", {}).get("current_obj_idx", self.current_obj_idx)
        self.current_obj_path = self.obj_configs[current_obj_idx]["path"]
        for i, obj_cfg in enumerate(self.obj_configs):
            prim = self.stage.GetPrimAtPath(obj_cfg["path"])
            if prim.IsValid():
                set_prim_visibility(prim, i == current_obj_idx)
        self._apply_source_mass_override()

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
        self.object_utils.set_object_mass(
            self.current_obj_path + "/mesh", mass_kg=mass_kg, scale=scale
        )

    def step(self) -> dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        source_quaternion = self.object_utils.get_transform_quat(
            object_path=self.current_obj_path + "/mesh"
        )
        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            target_path=self.target_path,
            additional_info={
                "object_quaternion": source_quaternion,
                "source_beaker":     self.current_obj_path,
            },
        )

    def get_reserved_placement_requests(self) -> list[dict[str, Any]]:
        requests = super().get_reserved_placement_requests()
        requests.append({
            "path": self.target_path,
            "position_range": self.cfg.task.left_pos,
        })
        return requests
