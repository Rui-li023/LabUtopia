import numpy as np
from .base_task import BaseTask


class PlacePressTask(BaseTask):
    """Place-and-press task: place source beaker on target platform then press button.

    Uses three objects from ``cfg.task.obj_paths``:
    - index 0: source beaker (randomised position)
    - index 1: target platform (randomised position); ``sub_path`` points to
               the actual placement surface used for height offset computation.
    - index 2: button prim
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.source_beaker  = self.cfg.task.obj_paths[0]["path"]
        self.target_plat    = self.cfg.task.obj_paths[1]["path"]
        self.target_sub_plat = self.cfg.task.obj_paths[1]["sub_path"]
        self.button         = self.cfg.task.obj_paths[2]["path"]

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self.randomize_object_position(self.source_beaker, self.cfg.task.obj_paths[0]["position_range"])
        self.randomize_object_position(self.target_plat,   self.cfg.task.obj_paths[1]["position_range"])

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        target_position = self.object_utils.get_object_xform_position(self.target_sub_plat)
        button_position = self.object_utils.get_object_xform_position(self.button)
        target_position[2] += 0.045

        return self.get_basic_state_info(
            object_path=self.source_beaker,
            additional_info={
                "source_beaker":  self.source_beaker,
                "target_position": target_position,
                "target_name":    self.target_plat.split("/")[-1],
                "target_path":    self.target_plat,
                "button_position": button_position,
            },
        )
