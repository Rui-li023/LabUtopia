from typing import Any, Dict, Optional
from .base_task import BaseTask


class LiquidMixingTask(BaseTask):
    """Level-4 composite task: Open door → Transfer beaker → Stir.

    This task uses a fixed scene layout (paths hardcoded to the lab USD).
    Randomisation is handled at the device level, not by moving objects.

    Primary object: beaker placed on the heating device platform.
    """

    BEAKER_PATH      = "/World/beaker_4"
    TARGET_PLAT_PATH = "/World/heat_device/heat_device/heat_device/plat"

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=6000):
            return None
        return self.get_basic_state_info(
            object_path=self.BEAKER_PATH,
            target_path=self.TARGET_PLAT_PATH,
        )
