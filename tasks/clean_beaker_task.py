import numpy as np
from typing import Any, Dict, Optional
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

    _BEAKER_1_X_RANGE = (0.20, 0.25)
    _BEAKER_1_Y_RANGE = (-0.10, -0.05)
    _BEAKER_Z = 0.77
    _PLAT_1_OFFSET = np.array([0.03, 0.0, -0.057])
    _BEAKER_2_X_RANGE = (0.20, 0.25)
    _BEAKER_2_Y_RANGE = (0.20, 0.25)
    _PLAT_2_X = 0.056
    _PLAT_2_Y_RANGE = (0.27, 0.32)
    _PLAT_2_Z = 0.713

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        super().__init__(cfg, world, stage, robot)
        self.world.reset()

    def _apply_grasp_friction(self) -> None:
        """Raise gripper↔beaker contact friction so the near-zero-squeeze grasp
        (grip 0.024 — the least-bad point in a razor-thin width window) holds the
        beaker through the shake/pour instead of slipping out, and reproduces in
        open-loop replay. Realistic glass-on-gripper µ; a scene-physics
        correction, not gaming. Applied in BOTH collect and replay reset for
        parity. No-op unless task.grasp_friction is set in the config."""
        task = getattr(self.cfg, "task", None)
        mu = getattr(task, "grasp_friction", None) if task else None
        if mu is None:
            return
        for path in (self.BEAKER_1, self.BEAKER_2, self.TARGET_BEAKER):
            self.object_utils.set_physics_friction(
                path, static_friction=float(mu), dynamic_friction=float(mu)
            )

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()

        b1_pos = np.array([
            np.random.uniform(*self._BEAKER_1_X_RANGE),
            np.random.uniform(*self._BEAKER_1_Y_RANGE),
            self._BEAKER_Z,
        ])
        self.object_utils.set_object_position(self.BEAKER_1, b1_pos)
        self._record_object_pose(self.BEAKER_1)

        p1_pos = b1_pos + self._PLAT_1_OFFSET
        self.object_utils.set_object_position(self.PLAT_1, p1_pos)
        self._record_object_pose(self.PLAT_1)

        b2_pos = np.array([
            np.random.uniform(*self._BEAKER_2_X_RANGE),
            np.random.uniform(*self._BEAKER_2_Y_RANGE),
            self._BEAKER_Z,
        ])
        self.object_utils.set_object_position(self.BEAKER_2, b2_pos)
        self._record_object_pose(self.BEAKER_2)

        p2_pos = np.array([self._PLAT_2_X, np.random.uniform(*self._PLAT_2_Y_RANGE), self._PLAT_2_Z])
        self.object_utils.set_object_position(self.PLAT_2, p2_pos)
        self._record_object_pose(self.PLAT_2)

        self._apply_grasp_friction()

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self._apply_grasp_friction()

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=4000):
            return None

        return self.get_basic_state_info(
            object_path=self.TARGET_BEAKER,
            target_path=self.TARGET_BEAKER,
            additional_info={
                "beaker_1_position": self.object_utils.get_geometry_center(object_path=self.BEAKER_1),
                "beaker_2_position": self.object_utils.get_geometry_center(object_path=self.BEAKER_2),
                "beaker_1_size":     self.object_utils.get_object_size(object_path=self.BEAKER_1),
                "beaker_2_size":     self.object_utils.get_object_size(object_path=self.BEAKER_2),
                "plat_1_position":   self.object_utils.get_geometry_center(object_path=self.PLAT_1),
                "plat_2_position":   self.object_utils.get_geometry_center(object_path=self.PLAT_2),
                "beaker_1":          self.BEAKER_1,
                "beaker_2":          self.BEAKER_2,
                "target_beaker":     self.TARGET_BEAKER,
            },
        )
