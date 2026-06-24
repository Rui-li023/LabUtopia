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

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self._apply_grasp_friction()
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
