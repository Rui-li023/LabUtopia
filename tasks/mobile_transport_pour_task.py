import os
from typing import Any, Dict, Optional

from loguru import logger

from .mobile_pick_task import MobilePickTask


class MobileTransportPourTask(MobilePickTask):
    """Navigate, pick the source beaker, optionally carry it to a second
    bench, and pour it into a target container (Level 5).

    Adds to :class:`MobilePickTask`:
    - a pour-target container referenced at runtime from another lab USD,
    - a pour dock point in front of the target container,
    - an optional carry path (dock A -> pour dock) planned at reset when
      ``cfg.task.pour_target.carry_navigation`` is true.
    """

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        self.pour_target_path: Optional[str] = None
        self.pour_dock: Optional[list] = None
        self.carry_path: Optional[list] = None
        super().__init__(cfg, world, stage, robot)

    # ── Setup ────────────────────────────────────────────────────────────

    def setup_objects(self) -> None:
        super().setup_objects()
        target = self.cfg.task.pour_target
        self.pour_target_path = str(target.prim_path)
        self.pour_target_position_range = target.position_range
        self.carry_navigation = bool(getattr(target, "carry_navigation", False))
        prim = self.stage.GetPrimAtPath(self.pour_target_path)
        if not prim.IsValid():
            prim = self.stage.DefinePrim(self.pour_target_path, "Xform")
            prim.GetReferences().AddReference(
                os.path.abspath(str(target.usd_path)), str(target.source_prim_path))
            logger.info(f"Referenced pour target {target.source_prim_path} "
                        f"from {target.usd_path} at {self.pour_target_path}")

    # ── Lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self.randomize_object_position(self.pour_target_path, self.pour_target_position_range)
        self.pour_dock = self._compute_dock_point(self.pour_target_path)
        self.carry_path = None
        if self.carry_navigation and self.dock_point is not None:
            self.carry_path = self._try_plan_path(self.dock_point, self.pour_dock)
            if self.carry_path is None:
                logger.warning("Unable to plan the carry path — resetting")
                self.reset_needed = True

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.carry_path = None
        self.pour_dock = self._compute_dock_point(self.pour_target_path)

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self) -> Optional[Dict[str, Any]]:
        state = super().step()
        if state is None:
            return None
        source_quaternion = self.object_utils.get_transform_quat(
            object_path=self.target_object_path + "/mesh")
        state.update({
            "pour_target_position": self.object_utils.get_geometry_center(
                object_path=self.pour_target_path),
            "pour_target_path":     self.pour_target_path,
            "pour_dock":            self.pour_dock,
            "carry_waypoints":      self.carry_path,
            "carry_navigation":     self.carry_navigation,
            "object_quaternion":    source_quaternion,
        })
        return state
