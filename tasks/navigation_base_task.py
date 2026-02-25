import yaml
import numpy as np
from typing import Dict, Any, List, Optional

from isaacsim.core.utils.rotations import quat_to_euler_angles

from .base_task import BaseTask
from utils.a_star import plan_navigation_path, real_to_grid, load_grid


class NavigationBaseTask(BaseTask):
    """Shared base for navigation tasks (pure navigation and mobile-pick).

    Handles:
    - Loading the nav-scene config and A* obstacle grid from YAML.
    - Sampling free-space start/end points.
    - Running A* and converting the raw path to waypoints with heading (theta).
    - Setting the robot's initial world pose.
    - Providing ``get_navigation_state()`` with common nav fields.
    """

    def __init__(self, cfg, world, stage, robot):
        self.navigation_assets: List[dict] = []
        self.grid = None
        self.current_start = None
        self.current_path: Optional[List] = None
        super().__init__(cfg, world, stage, robot)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def setup_objects(self) -> None:
        """Load the A* navigation config and obstacle grid in addition to normal objects."""
        super().setup_objects()
        if hasattr(self.cfg.task, "navigation_config_path"):
            with open(self.cfg.task.navigation_config_path, "r") as f:
                config = yaml.safe_load(f)
                self.navigation_assets = config.get("assets", [])
        if self.navigation_assets:
            nav_scene = self.navigation_assets[0]
            self.grid, self.W, self.H = load_grid(nav_scene["barrier_image_path"])

    # -------------------------------------------------------------------------
    # Path planning helpers (shared by both nav tasks)
    # -------------------------------------------------------------------------

    def _sample_free_point(self, x_bounds: list, y_bounds: list) -> Optional[List[float]]:
        """Sample a random 2-D point that lies in free space on the obstacle grid.

        Args:
            x_bounds: ``[min, max]`` range for the X axis.
            y_bounds: ``[min, max]`` range for the Y axis.

        Returns:
            ``[x, y]`` in world coordinates, or ``None`` if no free cell found
            after 100 attempts.
        """
        W = len(self.grid[0])
        H = len(self.grid)
        for _ in range(100):
            x = np.random.uniform(x_bounds[0], x_bounds[1])
            y = np.random.uniform(y_bounds[0], y_bounds[1])
            i, j = real_to_grid(x, y, x_bounds, y_bounds, (W, H))
            if self.grid[i][j] == 0:
                return [x, y]
        return None

    def _build_waypoints(self, merged_path_real: list) -> List[List[float]]:
        """Convert a raw A* path to ``[[x, y, theta], ...]`` waypoints.

        Each waypoint's heading (theta) points toward the next waypoint.
        The last waypoint reuses the previous heading.
        """
        waypoints = []
        for i, (x, y, _) in enumerate(merged_path_real):
            if i < len(merged_path_real) - 1:
                nx, ny, _ = merged_path_real[i + 1]
                theta = np.arctan2(ny - y, nx - x)
            else:
                theta = waypoints[-1][2] if waypoints else 0.0
            waypoints.append([x, y, theta])
        return waypoints

    def _try_plan_path(self, start: list, end: list) -> Optional[List[List[float]]]:
        """Attempt A* path planning from *start* to *end*.

        Returns:
            Waypoint list on success, ``None`` if no path exists or no nav
            scene is configured.
        """
        if not self.navigation_assets:
            return None
        path_result = plan_navigation_path({"asset": self.navigation_assets[0], "start": start, "end": end})
        if path_result is None:
            return None
        merged_path_real, _ = path_result
        return self._build_waypoints(merged_path_real)

    # -------------------------------------------------------------------------
    # Common step state
    # -------------------------------------------------------------------------

    def get_navigation_state(self) -> Dict[str, Any]:
        """Return state fields common to all navigation tasks.

        Includes current robot pose (x, y, heading), waypoints, camera data,
        and ``init_state`` for episode recording.
        """
        position, orientation = self.robot.get_world_pose()
        euler = quat_to_euler_angles(orientation, extrinsic=False)
        camera_data, display_data = self.get_camera_data()
        joint_positions = self.robot.get_joint_positions()
        return {
            "current_pose":        np.array([position[0], position[1], euler[2]]),
            "waypoints":           self.current_path,
            "camera_data":         camera_data,
            "camera_display":      display_data,
            "done":                self.reset_needed,
            "frame_idx":           self.frame_idx,
            "robot_world_position": position,
            "joint_positions":     joint_positions,
            "gripper_position":    self.robot.get_gripper_position(),
            "init_state":          self._episode_init_state,
        }

    # -------------------------------------------------------------------------
    # Task completion (no material/object cycling for nav tasks)
    # -------------------------------------------------------------------------

    def on_task_complete(self, success: bool) -> None:
        self.reset_needed = True
