import numpy as np
from .base_task import BaseTask


class StirTask(BaseTask):
    """Glass-rod stirring task.

    Reads object paths from config attributes (``obj_path``, ``target_path``,
    ``sub_obj_path``).  Episode budget: 2000 steps.
    """

    TEST_TUBE_RACK = "/World/test_tube_rack"

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.glass_rod      = self.cfg.obj_path
        self.target_beaker  = self.cfg.target_path
        self.glass_rod_mesh = self.cfg.sub_obj_path

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        beaker_pos = np.array([
            0.24125 + np.random.uniform(-0.075, 0.075),
            -0.31358 + np.random.uniform(-0.075, 0.075),
            0.77,
        ])
        self.object_utils.set_object_position(self.target_beaker, beaker_pos)
        self._record_object_pose(self.target_beaker)

        rack_pos = np.array([0.28421, 0.30755, 0.82291])
        self.object_utils.set_object_position(self.TEST_TUBE_RACK, rack_pos)
        self._record_object_pose(self.TEST_TUBE_RACK)

        rod_pos = rack_pos + np.array([-0.01152, -0.1125, 0.03197])
        self.object_utils.set_object_position(self.glass_rod, rod_pos)
        self._record_object_pose(self.glass_rod)

        mesh_pos = np.array([0.0, 0.0, 0.0])
        self.object_utils.set_object_position(self.glass_rod_mesh, mesh_pos)
        self._record_object_pose(self.glass_rod_mesh)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=2000):
            return None

        return self.get_basic_state_info(
            object_path=self.glass_rod,
            target_path=self.target_beaker,
            additional_info={
                "target_beaker":    self.target_beaker,
                "object_position":  self.object_utils.get_object_xform_position(self.glass_rod),
                "glass_rod_position": self.object_utils.get_object_xform_position(self.glass_rod_mesh),
            },
        )
