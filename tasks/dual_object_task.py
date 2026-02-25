from .base_task import BaseTask


class DualObjectTask(BaseTask):
    """Base class for tasks that operate on a source object and a target object.

    Reads both objects from ``cfg.task.obj_paths[0]`` and
    ``cfg.task.obj_paths[1]``, randomises their positions each episode, and
    exposes them as ``self.source_obj`` / ``self.target_obj``.

    Suitable for: place, pick-and-place, pick-and-pour, liquid mixing, etc.
    """

    def reset(self) -> None:
        """Randomise positions of source and target objects."""
        super().reset()
        self.robot.initialize()
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj  = self.cfg.task.obj_paths[1]["path"]
        self.randomize_object_position(self.source_obj, self.cfg.task.obj_paths[0]["position_range"])
        self.randomize_object_position(self.target_obj,  self.cfg.task.obj_paths[1]["position_range"])

    def reset_with_init_state(self, init_state: dict) -> None:
        """Restore scene and re-initialise path aliases."""
        super().reset_with_init_state(init_state)
        self.source_obj = self.cfg.task.obj_paths[0]["path"]
        self.target_obj  = self.cfg.task.obj_paths[1]["path"]

    def step(self):
        """Return state with source object and target object info."""
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        return self.get_basic_state_info(
            object_path=self.source_obj,
            target_path=self.target_obj,
        )
