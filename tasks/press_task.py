import random
import numpy as np
from typing import Any, Dict, Optional
from .base_task import BaseTask


class PressTask(BaseTask):
    """Button-press task with two distractor buttons.

    One of the three buttons is designated as the target; all three are
    shuffled into random vertical positions each episode so the robot cannot
    rely on positional memory.  Episode budget: 1000 steps.
    """

    _INSTRUMENT_POSITION = np.array([0.73, -0.1, 0.64])
    # Robot base sits at world Z≈0.71. Button at world Z=0.80 (≈9 cm above
    # base) forces Franka into an "elbow-below-base" pose with EE pitched
    # straight down — RMP cannot plan and returns null actions. Keep the
    # button at a height where the gripper can comfortably reach it while
    # pointing down (≈0.30 m above the robot base).
    _BUTTON_BASE_X = 0.30
    _BUTTON_BASE_Y_RANGE = (-0.06, 0.04)
    _BUTTON_BASE_Z = 1.05
    _BUTTON_Z_JITTER = (-0.02, 0.02)
    _DISTRACTOR1_Y_OFFSET = (-0.25, -0.15)
    _DISTRACTOR2_Y_OFFSET = (-0.40, -0.30)

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        super().__init__(cfg, world, stage, robot)
        self.object_utils.set_object_position(
            object_path=self.cfg.instrument_path,
            position=self._INSTRUMENT_POSITION,
        )
        self.target_button_path      = self.cfg.target_button_path
        self.distractor_button1_path = self.cfg.distractor_button1_path
        self.distractor_button2_path = self.cfg.distractor_button2_path

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        base_pos = np.array([
            self._BUTTON_BASE_X,
            random.uniform(*self._BUTTON_BASE_Y_RANGE),
            self._BUTTON_BASE_Z + np.random.uniform(*self._BUTTON_Z_JITTER),
        ])
        positions = [
            base_pos,
            base_pos + np.array([0.0, random.uniform(*self._DISTRACTOR1_Y_OFFSET), 0.0]),
            base_pos + np.array([0.0, random.uniform(*self._DISTRACTOR2_Y_OFFSET), 0.0]),
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

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=1000):
            return None

        return self.get_basic_state_info(
            object_path=self.target_button_path,
            additional_info={
                "object_position": self.object_utils.get_object_xform_position(self.target_button_path),
            },
        )
