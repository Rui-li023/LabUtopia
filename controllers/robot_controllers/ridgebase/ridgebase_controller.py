import numpy as np
from typing import List, Tuple, Optional
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.prims.impl import Articulation
from isaacsim.core.utils.types import ArticulationAction


class RidgebaseController:
    """Holonomic-base waypoint follower with pure-pursuit smoothing.

    The raw A* path has one waypoint per grid cell (8-connected, so segment
    bearings zigzag in 45-degree steps). Chasing each waypoint individually
    made the base alternate between rotating and translating at every cell.
    Instead the controller steers at a LOOKAHEAD-distance carrot point along
    the path, blends rotation and translation continuously (cos^2 of the
    heading error), and decelerates against the remaining path length rather
    than each intermediate waypoint — one smooth turn-while-driving motion.
    """

    # Pure-pursuit lookahead distance (m) along the remaining path. Large
    # enough to average out the 45-degree grid zigzag, small enough to track
    # the path through door-sized gaps (obstacles are inflated by the robot
    # radius, so a small corner cut is safe).
    LOOKAHEAD = 0.5

    # Heading error (rad) beyond which the base turns fully in place (initial
    # alignment / U-turns). Below it, translation blends in as cos^2(err), so
    # path-following corrections never stall the base the way the previous
    # hard 35-degree gate did.
    HARD_ALIGN = np.radians(60.0)

    # Per-step EMA weight for the heading target (0-1). Low = heavily filtered,
    # so the base yaw tracks the mean travel direction, not the A* zig-zag.
    HEADING_SMOOTH = 0.12

    def __init__(
        self,
        robot_articulation: Articulation,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.0,
        position_threshold: float = 0.1,
        angle_threshold: float = 0.1,
        dt: float = 0.01,
        final_angle: float = None
    ):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold
        self.dt = 0.02
        self.final_angle = final_angle

        self.k_p_linear = 1
        # Gentle heading P-gain. High gains (the old 4) overshoot the base yaw
        # against the drive delay and ring; the base is holonomic, so heading
        # only needs to ease toward the travel direction for the cameras.
        self.k_p_angular = 1.5

        self.waypoints = None
        self.current_waypoint_idx = 0
        self._remaining_from = None
        # Low-passed heading target. The raw travel bearing follows the A* grid
        # zig-zag (45-deg per-cell swings); commanding it directly wobbles the
        # base yaw and shakes every camera. Filtered here so the heading eases
        # toward the average travel direction instead of chasing each cell.
        self._heading_cmd = None
        # True only while the base is inside position_threshold of the final
        # waypoint (the rotate-to-final-angle stage). The done-check must gate
        # on this: with lookahead advancing, current_waypoint_idx reaches the
        # last index while the base is still ~LOOKAHEAD away.
        self._docked = False

        self._joints_subset = ArticulationSubset(
            robot_articulation,
            ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]
        )

    def set_waypoints(self, waypoints: List[Tuple[float, float, float]], final_angle: Optional[float] = None) -> None:
        self.waypoints = np.array(waypoints, dtype=float)
        self.current_waypoint_idx = 0
        self.final_angle = final_angle
        self._docked = False
        self._heading_cmd = None
        # remaining_from[i] = path length from waypoint i to the last waypoint,
        # used to decelerate against the DOCK instead of each grid cell.
        pts = self.waypoints[:, :2]
        if len(pts) > 1:
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            self._remaining_from = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        else:
            self._remaining_from = np.zeros(1)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap an angle to (-pi, pi] so any rotation is pursued the SHORT way.

        The base heading fed in here is ``spawn_euler + revolute_joint``; the
        revolute joint is an unwrapped accumulator that reaches several turns
        over a long A* path. Wrapping it before taking angle differences keeps
        every rotation target (travel bearing AND final dock angle) shortest-path.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _lookahead_point(self, base_xy: np.ndarray) -> np.ndarray:
        """Carrot point LOOKAHEAD meters ahead of the base along the path."""
        remaining = self.LOOKAHEAD
        prev = base_xy
        for i in range(self.current_waypoint_idx, len(self.waypoints)):
            wp = self.waypoints[i][:2]
            seg = wp - prev
            length = float(np.linalg.norm(seg))
            if length > 1e-9:
                if length >= remaining:
                    return prev + seg * (remaining / length)
                remaining -= length
            prev = wp
        return self.waypoints[-1][:2].copy()

    def compute_control(self, current_pose: np.ndarray) -> Tuple[float, float, float, float]:
        if self.waypoints is None or self.current_waypoint_idx >= len(self.waypoints):
            return 0.0, 0.0, 0.0, 0.0
        joint_positions = self._joints_subset.get_joint_positions()
        # Work on a copy: the caller's pose must not accumulate joint offsets.
        pose = np.asarray(current_pose, dtype=float).copy()
        pose[0] += joint_positions[0]
        pose[1] += joint_positions[1]
        pose[2] += joint_positions[2]
        heading = self._wrap_to_pi(pose[2])
        base_xy = pose[:2]

        # Consume every waypoint already inside the lookahead circle (raw A*
        # paths have one waypoint per grid cell; hopping them one position
        # threshold at a time caused per-cell stop-and-go).
        last = len(self.waypoints) - 1
        while (self.current_waypoint_idx < last
               and np.linalg.norm(self.waypoints[self.current_waypoint_idx][:2] - base_xy) < self.LOOKAHEAD):
            self.current_waypoint_idx += 1
        idx = self.current_waypoint_idx
        target = self.waypoints[idx]

        dx = target[0] - base_xy[0]
        dy = target[1] - base_xy[1]
        distance = float(np.hypot(dx, dy))

        # Docked: rotate in place to the final angle.
        self._docked = idx == last and distance < self.position_threshold
        if self._docked:
            final_target = self.final_angle if self.final_angle is not None else target[2]
            final_angle_diff = self._wrap_to_pi(final_target - heading)
            if abs(final_angle_diff) < self.angle_threshold:
                return 0.0, 0.0, 0.0, final_angle_diff
            theta_vel = np.clip(self.k_p_angular * final_angle_diff,
                                -self.max_angular_speed, self.max_angular_speed)
            return 0.0, 0.0, theta_vel, final_angle_diff

        # Steer toward the carrot. Near the dock the live bearing to the final
        # waypoint swings wildly as the base passes beside it, so steer by the
        # stored segment bearing there while translation keeps homing on the
        # dock point itself.
        carrot = self._lookahead_point(base_xy)
        to_carrot = carrot - base_xy
        carrot_dist = float(np.linalg.norm(to_carrot))
        travel_bearing = np.arctan2(to_carrot[1], to_carrot[0]) if carrot_dist > 1e-6 else heading
        near_dock = idx == last and distance < 1.5 * self.position_threshold
        raw_target = float(target[2]) if near_dock else travel_bearing
        # Low-pass the heading target so the base yaw eases toward the mean
        # travel direction instead of chasing the A* grid zig-zag frame-to-frame
        # (that chasing was what rocked the base and shook the cameras).
        if self._heading_cmd is None:
            self._heading_cmd = heading
        self._heading_cmd = self._wrap_to_pi(
            self._heading_cmd + self.HEADING_SMOOTH * self._wrap_to_pi(raw_target - self._heading_cmd))
        angle_diff = self._wrap_to_pi(self._heading_cmd - heading)

        # Translation is DECOUPLED from heading (holonomic base): drive toward
        # the carrot at full speed, decelerating only against the remaining
        # path length, and stop to turn in place ONLY when grossly misaligned
        # (> HARD_ALIGN). The old cos^2(heading-error) coupling stalled forward
        # motion whenever the yaw lagged, which — with the smooth (slow) yaw —
        # made navigation crawl. The EMA heading below keeps the cameras facing
        # forward without throttling translation.
        dist_remaining = distance + float(self._remaining_from[idx])
        speed = min(0.25 * dist_remaining, self.max_linear_speed)
        if abs(angle_diff) > self.HARD_ALIGN:
            speed = 0.0

        x_vel = speed * np.cos(travel_bearing)
        y_vel = speed * np.sin(travel_bearing)
        theta_vel = np.clip(self.k_p_angular * angle_diff,
                            -self.max_angular_speed, self.max_angular_speed)

        return x_vel, y_vel, theta_vel, angle_diff

    def get_action(self, current_pose: np.ndarray) -> Tuple[Optional[ArticulationAction], bool]:
        x_vel, y_vel, theta_vel, angle_diff = self.compute_control(current_pose)
        x_vel = np.clip(abs(x_vel), 0, self.max_linear_speed) * np.sign(x_vel)
        y_vel = np.clip(abs(y_vel), 0, self.max_linear_speed) * np.sign(y_vel)
        theta_vel = np.clip(theta_vel, -self.max_angular_speed, self.max_angular_speed)

        joint_positions = self._joints_subset.get_joint_positions()

        next_x = joint_positions[0] + x_vel
        next_y = joint_positions[1] + y_vel
        next_theta = joint_positions[2] + theta_vel

        position = np.array([next_x, next_y, next_theta])
        action = self._joints_subset.make_articulation_action(
            joint_positions=position,
            joint_velocities=None
        )

        if self.final_angle is None:
            done = self._docked and abs(theta_vel) < self.angle_threshold
        else:
            done = (self._docked and
                    abs(theta_vel) < self.angle_threshold and
                    abs(angle_diff) < self.angle_threshold)

        return action, done

    def is_path_complete(self) -> bool:
        return (self.waypoints is not None and
                self.current_waypoint_idx >= len(self.waypoints))
