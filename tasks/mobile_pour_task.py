import os
from typing import Any, Dict, Optional

from loguru import logger

from .mobile_pick_task import MobilePickTask


class MobilePourTask(MobilePickTask):
    """Navigate to the bench, pick the source beaker, and pour it into a nearby
    target container at the SAME dock (Level 5, no carry).

    Adds to :class:`MobilePickTask` a pour-target container referenced at runtime
    from another lab USD (same mechanism as the transport-place platform). The
    dock is still computed from the SOURCE beaker, so the target range must keep
    the container inside the arm's reach of that dock (~0.75 m).
    """

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        self.pour_target_path: Optional[str] = None
        super().__init__(cfg, world, stage, robot)

    # ── Setup ────────────────────────────────────────────────────────────

    def setup_objects(self) -> None:
        super().setup_objects()
        target = self.cfg.task.pour_target
        self.pour_target_path = str(target.prim_path)
        self.pour_target_position_range = target.position_range
        prim = self.stage.GetPrimAtPath(self.pour_target_path)
        if not prim.IsValid():
            prim = self.stage.DefinePrim(self.pour_target_path, "Xform")
            prim.GetReferences().AddReference(
                os.path.abspath(str(target.usd_path)), str(target.source_prim_path))
            logger.info(f"Referenced pour target {target.source_prim_path} "
                        f"from {target.usd_path} at {self.pour_target_path}")

    # ── Spawn / path generation ──────────────────────────────────────────

    def _generate_navigation_task(self) -> bool:
        # Place the target container before dock computation so it is fixed for
        # the whole episode (the dock itself is derived from the source beaker).
        self.randomize_object_position(self.pour_target_path, self.pour_target_position_range)
        return super()._generate_navigation_task()

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self) -> Optional[Dict[str, Any]]:
        state = super().step()
        if state is None:
            return None
        state.update({
            "target_position": self.object_utils.get_geometry_center(
                object_path=self.pour_target_path),
            "object_quaternion": self.object_utils.get_transform_quat(
                object_path=self.target_object_path, w_first=True),
            "pour_target_path": self.pour_target_path,
        })
        return state
