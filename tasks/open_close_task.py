from isaacsim.core.utils.prims import set_prim_visibility
from .single_object_task import SingleObjectTask


class OpenCloseTask(SingleObjectTask):
    """Open-and-close task for doors and drawers.

    Extends :class:`SingleObjectTask` to track both the furniture prim
    (``current_obj_path``) and the handle sub-prim (``current_sub_obj_path``).
    """

    def reset(self) -> None:
        super().reset()
        self._set_sub_obj_path()

    def reset_with_init_state(self, init_state: dict) -> None:
        """Restore scene from recorded init state and reconstruct path aliases."""
        super().reset_with_init_state(init_state)
        self._set_sub_obj_path()

    def _set_sub_obj_path(self) -> None:
        """Derive the handle prim path from config or naming convention."""
        if self.cfg.get("handle_path"):
            self.current_sub_obj_path = self.cfg.get("handle_path")
        else:
            self.current_sub_obj_path = self.current_obj_path + "/handle"

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        object_position = self.object_utils.get_geometry_center(object_path=self.current_sub_obj_path)
        object_size     = self.object_utils.get_object_size(object_path=self.current_sub_obj_path)
        close_gripper_distance = self.obj_configs[self.current_obj_idx].get("close_gripper_distance", 0.023)

        additional = {
            "object_position":    object_position,
            "object_size":        object_size,
            "close_gripper_distance": close_gripper_distance,
        }
        if self.cfg.task.get("operate_type") == "door":
            additional["revolute_joint_position"] = self.object_utils.get_revolute_joint_positions(
                joint_path=self.current_obj_path + "/RevoluteJoint"
            )

        return self.get_basic_state_info(
            object_path=self.current_obj_path,
            additional_info=additional,
        )
