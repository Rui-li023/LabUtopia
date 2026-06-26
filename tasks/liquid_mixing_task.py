from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from .base_task import BaseTask


class LiquidMixingTask(BaseTask):
    """Level-4 composite task: Open door → Transfer beaker → Stir.

    This task uses a fixed scene layout (paths hardcoded to the lab USD).
    Randomisation is handled at the device level, not by moving objects.

    Primary object: beaker placed on the heating device platform.
    """

    BEAKER_PATH      = "/World/beaker_4"
    TARGET_PLAT_PATH = "/World/heat_device/heat_device/heat_device/plat"
    _GRASP_OBJECTS   = ("/World/beaker_03", "/World/beaker_04", "/World/beaker_05")
    # Objects whose poses must be snapshotted into init_state so replay restores
    # the EXACT collect-time scene. Without this, the 3 grasped beakers were never
    # recorded (unlike clean_beaker / open_transport_pour), so replay let them sit
    # at the USD pose and settle freely — the recorded trajectory then aimed at
    # the collect-time beaker position and grasped empty air. Includes the central
    # mix beaker (beaker_4) for completeness.
    _RECORD_OBJECTS  = ("/World/beaker_03", "/World/beaker_04", "/World/beaker_05",
                        "/World/beaker_4")

    def _apply_grasp_friction(self) -> None:
        """Raise gripper<->beaker contact friction on the three poured beakers so
        each near-zero-squeeze grasp holds through the pour instead of slipping,
        and reproduces in open-loop replay. Realistic glass-on-gripper µ — a
        scene-physics correction. Applied in BOTH collect and replay reset for
        parity. No-op unless task.grasp_friction is set."""
        task = getattr(self.cfg, "task", None)
        mu = getattr(task, "grasp_friction", None) if task else None
        if mu is None:
            return
        for path in self._GRASP_OBJECTS:
            self.object_utils.set_physics_friction(
                path, static_friction=float(mu), dynamic_friction=float(mu)
            )

    def _apply_beaker_reposition(self) -> None:
        """Shift the 3 grasped beakers toward the robot (x) by cfg.task.beaker_x_shift
        so they sit in the Franka's comfortable workspace (~0.5 m reach) instead of
        the far edge (~0.58-0.64 m), where open-loop tracking error is largest and
        the grasp misses. Done BEFORE _record_object_pose so the moved poses are
        snapshotted into init_state and replay restores them. No-op unless set."""
        task = getattr(self.cfg, "task", None)
        dx = getattr(task, "beaker_x_shift", None) if task else None
        if dx is None:
            return
        for path in self._GRASP_OBJECTS:
            wp = self.object_utils.get_world_pose(path)
            if wp is None:
                continue
            p = np.asarray(wp["position"], dtype=float)
            newp = np.array([p[0] + float(dx), p[1], p[2]])
            self.object_utils.set_object_position(path, newp)
            logger.info(f"[reposition] {path}: x {p[0]:.3f} -> {newp[0]:.3f}")

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self._apply_grasp_friction()
        self._apply_beaker_reposition()
        # Snapshot the manipulation objects so replay restores the exact scene
        # (the missing piece vs clean_beaker/otp — see _RECORD_OBJECTS).
        for path in self._RECORD_OBJECTS:
            self._record_object_pose(path)

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self._apply_grasp_friction()

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits(max_steps=6000):
            return None
        return self.get_basic_state_info(
            object_path=self.BEAKER_PATH,
            target_path=self.TARGET_PLAT_PATH,
        )
