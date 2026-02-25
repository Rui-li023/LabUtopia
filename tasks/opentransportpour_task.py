import numpy as np
from typing import Dict, Any
from .base_task import BaseTask


class OpenTransportPourTask(BaseTask):
    """Level-4 composite task: Open door → Transport beaker → Pour.

    Scene objects are placed at fixed randomised ranges defined here.
    Paths for the primary beaker and target platform come from
    ``cfg.task.obj_paths``.
    """

    # Hardcoded scene randomisation ranges (x_range, y_range, z_fixed)
    _SCENE_OBJECTS = [
        ("/World/beaker2",        (0.04, 0.05),  (0.32, 0.33),  0.86),
        ("/World/conical_bottle02", (0.11, 0.12), (-0.41, -0.40), 0.86),
        ("/World/beaker1",        (0.17, 0.18),  (-0.155, -0.145), 0.86),
        ("/World/target_plat2",   (0.10, 0.11),  (-0.52, -0.51), 0.775),
        ("/World/target_plat",    (0.03, 0.04),  (0.54, 0.55),  0.775),
        ("/World/MuffleFurnace",  (0.69, 0.70),  (0.09, 0.10),  0.78),
    ]

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.beaker_path      = cfg.task.obj_paths[0]["path"]
        self.target_plat_path = cfg.task.obj_paths[1]["path"]

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_path, x_range, y_range, z in self._SCENE_OBJECTS:
            pos = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range), z])
            self.object_utils.set_object_position(obj_path, pos)
            self._record_object_pose(obj_path)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)

    def step(self) -> Dict[str, Any]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=self.cfg.task.max_steps):
            return None
        return self.get_basic_state_info(
            object_path=self.beaker_path,
            target_path=self.target_plat_path,
        )
