import os
import numpy as np
from typing import Any, Dict, Optional

from .navigation_base_task import NavigationBaseTask


class MobilePickTask(NavigationBaseTask):
    """Navigate to a fixed target position and then pick up an object.

    Extends :class:`NavigationBaseTask` by:
    - Loading an optional pick-target object into the stage.
    - Using a *fixed* navigation end-point (from config) instead of a random one.
    - Tracking the target object's position and size each step.
    """

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        self.target_object_path: Optional[str] = None
        self.initial_object_position: Optional[np.ndarray] = None
        self.navigation_done: bool = False
        self.target_position: Optional[list] = None
        super().__init__(cfg, world, stage, robot)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def setup_objects(self) -> None:
        """Load nav config, obstacle grid, and optional pick-target object."""
        super().setup_objects()

        task = self.cfg.task
        pick_object_usd = getattr(task, "pick_object_usd", None)
        pick_object_prim_path = getattr(task, "pick_object_prim_path", None)
        if pick_object_usd is not None and pick_object_prim_path is not None:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path=os.path.abspath(pick_object_usd),
                prim_path=pick_object_prim_path,
            )
            self.target_object_path = task.pick_object_path
            object_position = getattr(task, "object_position", None)
            if object_position is not None:
                self.object_utils.set_object_position(
                    task.pick_object_path, np.array(object_position)
                )
        else:
            pick_object_path = getattr(task, "pick_object_path", None)
            if pick_object_path is not None:
                self.target_object_path = pick_object_path

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self.navigation_done = False
        self.initial_object_position = None
        if self.navigation_assets and not self._generate_navigation_task():
            print("Warning: Unable to generate a valid navigation path")

    def _generate_navigation_task(self) -> bool:
        """Sample a free-space start, plan an A* path to the config target.

        Returns:
            ``True`` on success, ``False`` after ``MAX_SAMPLE_ATTEMPTS`` failed attempts.
        """
        nav_scene = self.navigation_assets[0]
        self.target_position = getattr(self.cfg.task, "target_position", [-3.0, -0.46])
        for _ in range(self.MAX_SAMPLE_ATTEMPTS):
            start = self._sample_free_point(nav_scene["x_bounds"], nav_scene["y_bounds"])
            if start is None:
                continue
            waypoints = self._try_plan_path(start, self.target_position)
            if waypoints is not None:
                self.current_start = start
                self.current_path  = waypoints
                self.robot.set_world_pose(position=np.array([start[0], start[1], 0.0]))
                return True
        return False

    def set_navigation_done(self, done: bool) -> None:
        """Allow the controller to signal that navigation is complete."""
        self.navigation_done = done

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        object_position = object_size = None
        if self.target_object_path:
            object_position = self.object_utils.get_geometry_center(object_path=self.target_object_path)
            object_size     = self.object_utils.get_object_size(object_path=self.target_object_path)
            if self.initial_object_position is None and object_position is not None:
                self.initial_object_position = object_position.copy()

        state = self.get_navigation_state()
        state.update({
            "start_point":            self.current_start,
            "target_position":        self.target_position,
            "navigation_done":        self.navigation_done,
            "object_position":        object_position,
            "object_size":            object_size,
            "object_path":            self.target_object_path,
            "object_name":            self.target_object_path.split("/")[-1] if self.target_object_path else "",
            "initial_object_position": self.initial_object_position,
        })
        return state
