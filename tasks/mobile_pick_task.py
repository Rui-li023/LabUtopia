import numpy as np
from typing import Any, Dict, Optional

from loguru import logger

from .navigation_base_task import NavigationBaseTask


class MobilePickTask(NavigationBaseTask):
    """Navigate to a bench and pick up an object (Level 5).

    Two spawn modes, selected by ``cfg.task.spawn.mode``:

    - ``near``: spawn 1-2 m (``spawn.distance_range``) from the dock point,
      in the aisle half-plane — the close-range end-to-end VLA variant.
    - ``far``: spawn anywhere in free space with an A* path of at least
      ``spawn.min_path_length`` metres — the navigation-model + VLA variant.

    The dock point is derived from the (randomized) pick object position:
    ``[obj_x, obj_y - dock_standoff]``, facing the bench (+y, theta = pi/2).
    """

    FINAL_NAV_ANGLE = np.pi / 2  # benches face -y; dock faces +y

    def __init__(self, cfg: Any, world: Any, stage: Any, robot: Any) -> None:
        self.target_object_path: Optional[str] = None
        self.initial_object_position: Optional[np.ndarray] = None
        self.dock_point: Optional[list] = None
        super().__init__(cfg, world, stage, robot)

    # ── Setup ────────────────────────────────────────────────────────────

    def setup_objects(self) -> None:
        super().setup_objects()
        task = self.cfg.task
        self.target_object_path = task.pick_object_path
        self.object_position_range = getattr(task, "object_position_range", None)
        self.dock_standoff = float(getattr(task, "dock_standoff", 0.85))
        spawn = getattr(task, "spawn", None)
        self.spawn_mode = str(getattr(spawn, "mode", "far")) if spawn else "far"
        self.spawn_distance_range = (
            [float(v) for v in spawn.distance_range]
            if spawn is not None and hasattr(spawn, "distance_range") else [1.0, 2.0])
        self.min_path_length = (
            float(getattr(spawn, "min_path_length", 4.0)) if spawn else 4.0)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self.initial_object_position = None
        if self.object_position_range is not None:
            self.randomize_object_position(self.target_object_path, self.object_position_range)
        self._record_object_pose(self.target_object_path)
        if self.navigation_assets and not self._generate_navigation_task():
            logger.warning("Unable to generate a valid navigation task")
            self.reset_needed = True

    def reset_with_init_state(self, init_state: dict) -> None:
        super().reset_with_init_state(init_state)
        self.initial_object_position = None

    def _record_object_pose(self, obj_path: str) -> None:
        """Record an object's pose into the episode init state for replay."""
        position = self.object_utils.get_object_xform_position(object_path=obj_path)
        orientation = self.object_utils.get_transform_quat(object_path=obj_path)
        if position is not None and orientation is not None:
            self._episode_init_state["object_poses"][obj_path] = {
                "position": np.asarray(position, dtype=np.float32),
                "orientation": np.asarray(orientation, dtype=np.float32),
            }

    # ── Spawn / path generation ──────────────────────────────────────────

    def _compute_dock_point(self, object_path: str) -> list:
        obj = self.object_utils.get_object_xform_position(object_path=object_path)
        return [float(obj[0]), float(obj[1]) - self.dock_standoff]

    def _sample_spawn(self, nav_scene: dict) -> Optional[list]:
        if self.spawn_mode == "near":
            d = np.random.uniform(*self.spawn_distance_range)
            # Aisle half-plane relative to the dock (y < dock_y): phi in (pi, 2pi)
            phi = np.random.uniform(np.pi + 0.3, 2 * np.pi - 0.3)
            x = self.dock_point[0] + d * np.cos(phi)
            y = self.dock_point[1] + d * np.sin(phi)
            return [x, y] if self._is_free_point(x, y) else None
        return self._sample_free_point(nav_scene["x_bounds"], nav_scene["y_bounds"])

    def _generate_navigation_task(self) -> bool:
        """Sample a spawn point and plan an A* path to the dock point.

        Returns:
            True on success, False after MAX_SAMPLE_ATTEMPTS failed attempts.
        """
        nav_scene = self.navigation_assets[0]
        self.dock_point = self._compute_dock_point(self.target_object_path)
        for _ in range(self.MAX_SAMPLE_ATTEMPTS):
            start = self._sample_spawn(nav_scene)
            if start is None:
                continue
            waypoints = self._try_plan_path(start, self.dock_point)
            if waypoints is None:
                continue
            if self.spawn_mode == "far" and self._path_length(waypoints) < self.min_path_length:
                continue
            self.current_start = start
            self.current_path = waypoints
            self.robot.set_world_pose(position=np.array([start[0], start[1], 0.0]))
            return True
        return False

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self) -> Optional[Dict[str, Any]]:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        object_position = self.object_utils.get_geometry_center(object_path=self.target_object_path)
        object_size = self.object_utils.get_object_size(object_path=self.target_object_path)
        if self.initial_object_position is None and object_position is not None:
            self.initial_object_position = object_position.copy()

        state = self.get_navigation_state()
        state.update({
            "start_point":             self.current_start,
            "dock_point":              self.dock_point,
            "final_nav_angle":         self.FINAL_NAV_ANGLE,
            "object_position":         object_position,
            "object_size":             object_size,
            "object_path":             self.target_object_path,
            "object_name":             self.target_object_path.split("/")[-1],
            "initial_object_position": self.initial_object_position,
        })
        return state
