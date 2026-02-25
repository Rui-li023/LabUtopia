import random
import numpy as np
from .base_task import BaseTask


class PressTask(BaseTask):
    """Button-press task with two distractor buttons.

    One of the three buttons is designated as the target; all three are
    shuffled into random vertical positions each episode so the robot cannot
    rely on positional memory.  Episode budget: 1000 steps.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.object_utils.set_object_position(
            object_path=self.cfg.instrument_path,
            position=np.array([0.73, -0.1, 0.64]),
        )
        self.target_button_path      = self.cfg.target_button_path
        self.distractor_button1_path = self.cfg.distractor_button1_path
        self.distractor_button2_path = self.cfg.distractor_button2_path

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        base_pos = np.array([0.40, random.uniform(-0.06, 0.04), 1.1 + np.random.uniform(-0.1, 0.1)])
        positions = [
            base_pos,
            base_pos + np.array([0.0, random.uniform(-0.25, -0.15), 0.0]),
            base_pos + np.array([0.0, random.uniform(-0.40, -0.30), 0.0]),
        ]
        random.shuffle(positions)

        for path, pos in zip(
            [self.target_button_path, self.distractor_button1_path, self.distractor_button2_path],
            positions,
        ):
            self.object_utils.set_object_position(object_path=path, position=pos)
            self._record_object_pose(path)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)

    def step(self):
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=1000):
            return None

        return self.get_basic_state_info(
            object_path=self.target_button_path,
            additional_info={
                "object_position": self.object_utils.get_object_xform_position(self.target_button_path),
            },
        )
