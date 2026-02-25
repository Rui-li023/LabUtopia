from isaacsim.core.utils.prims import set_prim_visibility
from .base_task import BaseTask


class SingleObjectTask(BaseTask):
    """Base class for tasks that operate on one target object at a time.

    Manages object cycling (multiple objects across episodes), visibility
    (only the active object is shown), and material switching.

    Suitable for: pick, open/close, shake, pour, etc.
    """

    def on_task_complete(self, success: bool) -> None:
        """Advance object/material indices without triggering a reset.

        Single-object tasks manage their own reset timing (e.g. via
        ``check_frame_limits``), so only the indices are updated here.
        """
        self.update_object_and_material_indices(success)

    def reset(self) -> None:
        """Reset the scene: re-apply materials, place active object, hide others."""
        super().reset()
        self.robot.initialize()
        self.current_obj_path = self.place_objects_with_visibility_management(
            self.current_obj_idx, far_distance=10.0
        )
        self._episode_init_state["extra"]["current_obj_idx"] = self.current_obj_idx

    def reset_with_init_state(self, init_state: dict) -> None:
        """Restore scene from a recorded initial state.

        Applies saved materials and poses, then restores object visibility so
        exactly the same object is shown as when data was collected.

        Args:
            init_state: Dict containing ``object_poses``, ``object_materials``,
                        and ``extra`` (with ``current_obj_idx``).
        """
        super().reset_with_init_state(init_state)
        current_obj_idx = init_state.get("extra", {}).get("current_obj_idx", self.current_obj_idx)
        self.current_obj_path = self.obj_configs[current_obj_idx]["path"]
        for i, obj_cfg in enumerate(self.obj_configs):
            prim = self.stage.GetPrimAtPath(obj_cfg["path"])
            if prim.IsValid():
                set_prim_visibility(prim, i == current_obj_idx)

    def step(self):
        """Return the current state centred on the active object."""
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        return self.get_basic_state_info(object_path=self.current_obj_path)
