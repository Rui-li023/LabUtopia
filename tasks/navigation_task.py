import numpy as np
from typing import Dict, Any, Optional

from .navigation_base_task import NavigationBaseTask


class NavigationTask(NavigationBaseTask):
    """Pure point-to-point navigation task.

    Generates a random start and end point within the free space of the
    navigation scene each episode, plans an A* path between them, and places
    the robot at the start.
    """

    def __init__(self, cfg, world, stage, robot):
        self.current_end: Optional[list] = None
        super().__init__(cfg, world, stage, robot)

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        if self.navigation_assets and not self._generate_navigation_task():
            print("Warning: Unable to generate a valid navigation path")

    def _generate_navigation_task(self) -> bool:
        """Sample start & end in free space and plan an A* path.

        Returns:
            ``True`` on success, ``False`` if no valid path was found within
            100 attempts.
        """
        nav_scene = self.navigation_assets[0]
        for _ in range(100):
            start = self._sample_free_point(nav_scene["x_bounds"], nav_scene["y_bounds"])
            end   = self._sample_free_point(nav_scene["x_bounds"], nav_scene["y_bounds"])
            if start is None or end is None:
                continue
            waypoints = self._try_plan_path(start, end)
            if waypoints is not None:
                self.current_start = start
                self.current_end   = end
                self.current_path  = waypoints
                self.robot.set_world_pose(position=np.array([start[0], start[1], 0.0]))
                return True
        return False

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        state = self.get_navigation_state()
        state.update({
            "start_point": self.current_start,
            "end_point":   self.current_end,
        })
        return state
