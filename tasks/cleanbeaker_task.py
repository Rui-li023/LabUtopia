import numpy as np
from .base_task import BaseTask


class CleanBeakerTask(BaseTask):
    """Multi-object beaker cleaning task.

    Fixed scene objects (paths are hardcoded to match the lab USD):
    - ``target_beaker``: the beaker being cleaned (primary object)
    - ``beaker_1``, ``beaker_2``: dirty beakers placed randomly
    - ``plat_1``, ``plat_2``: target platforms

    4000-step episode budget.
    """

    TARGET_BEAKER = "/World/target_beaker"
    BEAKER_1      = "/World/beaker_hard_1"
    BEAKER_2      = "/World/beaker_hard_2"
    PLAT_1        = "/World/target_plat_1"
    PLAT_2        = "/World/target_plat_2"

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.world.reset()

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        b1_pos = np.array([
            np.random.uniform(0.20, 0.25),
            np.random.uniform(-0.10, -0.05),
            0.77,
        ])
        self.object_utils.set_object_position(self.BEAKER_1, b1_pos)
        self._record_object_pose(self.BEAKER_1)

        p1_pos = b1_pos + np.array([0.03, 0.0, -0.057])
        self.object_utils.set_object_position(self.PLAT_1, p1_pos)
        self._record_object_pose(self.PLAT_1)

        b2_pos = np.array([
            np.random.uniform(0.20, 0.25),
            np.random.uniform(0.20, 0.25),
            0.77,
        ])
        self.object_utils.set_object_position(self.BEAKER_2, b2_pos)
        self._record_object_pose(self.BEAKER_2)

        p2_pos = np.array([0.056, np.random.uniform(0.27, 0.32), 0.713])
        self.object_utils.set_object_position(self.PLAT_2, p2_pos)
        self._record_object_pose(self.PLAT_2)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=4000):
            return None

        return self.get_basic_state_info(
            object_path=self.TARGET_BEAKER,
            additional_info={
                "beaker_1_position": self.object_utils.get_geometry_center(self.BEAKER_1),
                "beaker_2_position": self.object_utils.get_geometry_center(self.BEAKER_2),
                "plat_1_position":   self.object_utils.get_geometry_center(self.PLAT_1),
                "plat_2_position":   self.object_utils.get_geometry_center(self.PLAT_2),
                "beaker_1":          self.BEAKER_1,
                "beaker_2":          self.BEAKER_2,
            },
        )
