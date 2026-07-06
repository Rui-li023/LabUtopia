import os
from typing import Any, Dict, Optional

from loguru import logger
from pxr import UsdPhysics

from .mobile_pick_task import MobilePickTask


class MobileTransportPlaceTask(MobilePickTask):
    """Navigate, pick the source beaker, optionally carry it to a second
    bench, and place it on a target platform (Level 5).

    Adds to :class:`MobilePickTask`:
    - a placement platform referenced at runtime from another lab USD,
    - a place dock point in front of the platform,
    - an optional carry path (dock A -> place dock) planned at reset when
      ``cfg.task.place_target.carry_navigation`` is true.

    Dock midpoint: for the close (same-bench) variant the base must dock
    BETWEEN the source beaker and the platform, so the plat is randomized
    before the parent computes the dock and ``_compute_dock_point`` averages
    the source dock with the plat position.
    """

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        self.place_target_path: Optional[str] = None
        self.place_dock: Optional[list] = None
        self.carry_path: Optional[list] = None
        super().__init__(cfg, world, stage, robot)

    # ── Setup ────────────────────────────────────────────────────────────

    def setup_objects(self) -> None:
        super().setup_objects()
        target = self.cfg.task.place_target
        self.place_target_path = str(target.prim_path)
        self.place_target_position_range = target.position_range
        self.carry_navigation = bool(getattr(target, "carry_navigation", False))
        prim = self.stage.GetPrimAtPath(self.place_target_path)
        if not prim.IsValid():
            prim = self.stage.DefinePrim(self.place_target_path, "Xform")
            prim.GetReferences().AddReference(
                os.path.abspath(str(target.usd_path)), str(target.source_prim_path))
            logger.info(f"Referenced place target {target.source_prim_path} "
                        f"from {target.usd_path} at {self.place_target_path}")
        # The source asset's plat is VISUAL-ONLY (no collision APIs) — in the
        # L4 scenes beakers actually rest on the bench top underneath it. On
        # bench B nothing physical exists at the release point, so the placed
        # beaker fell straight through (landed inside the cabinet, z~0.33).
        # Make the plat a static convex collider so the beaker rests ON it.
        mesh = self.stage.GetPrimAtPath(self.place_target_path + "/mesh")
        if mesh.IsValid() and not mesh.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(mesh)
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set("convexHull")
            logger.info(f"Applied static convex collision to {self.place_target_path}/mesh")

    # ── Lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        # The plat is randomized inside _generate_navigation_task (before the
        # dock is computed); here we only derive the place dock and carry path.
        self.place_dock = self._compute_dock_point(self.place_target_path)
        self.carry_path = None
        if self.carry_navigation and self.dock_point is not None:
            self.carry_path = self._try_plan_path(self.dock_point, self.place_dock)
            if self.carry_path is None:
                logger.warning("Unable to plan the carry path — resetting")
                self.reset_needed = True

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.carry_path = None
        self.place_dock = self._compute_dock_point(self.place_target_path)

    # ── Spawn / path generation ──────────────────────────────────────────

    def _generate_navigation_task(self) -> bool:
        # Place the plat before dock computation so its dock point can be
        # derived from the plat's randomized position in the same reset.
        self.randomize_object_position(self.place_target_path, self.place_target_position_range)
        return super()._generate_navigation_task()

    def _compute_dock_point(self, object_path: str) -> list:
        dock = super()._compute_dock_point(object_path)
        if object_path == self.place_target_path:
            # DOCK_X_OFFSET exists for the GRASP's left/right closing
            # asymmetry; the place is a release, so center the dock on the
            # plat to minimize lateral reach (an offset dock undershot the
            # plat by ~0.2 m at the far bench and dropped the beaker short).
            dock[0] -= self.DOCK_X_OFFSET
            return dock
        if self.carry_navigation:
            return dock
        # Fallback for a hypothetical same-bench (non-carry) variant: dock at
        # the source/plat midpoint. Both shipped place configs carry-navigate,
        # so this is not currently exercised.
        plat = self.object_utils.get_object_xform_position(object_path=self.place_target_path)
        return [(dock[0] + float(plat[0])) / 2.0, dock[1]]

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self) -> Optional[Dict[str, Any]]:
        state = super().step()
        if state is None:
            return None
        state.update({
            "place_target_position": self.object_utils.get_geometry_center(
                object_path=self.place_target_path),
            "place_target_path":     self.place_target_path,
            "place_dock":            self.place_dock,
            "carry_waypoints":       self.carry_path,
            "carry_navigation":      self.carry_navigation,
        })
        return state
