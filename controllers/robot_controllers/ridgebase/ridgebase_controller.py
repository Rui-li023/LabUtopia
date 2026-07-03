import numpy as np
from typing import List, Tuple, Optional
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.prims.impl import Articulation
from isaacsim.core.utils.types import ArticulationAction

class RidgebaseController:
    # Heading error (rad) below which the base is allowed to translate. Beyond
    # this the base turns in place: plain cos-scaling still let it creep at
    # ~cos(85 deg)=0.09 of full speed while slowly rotating, and on short paths
    # that creep dominated the recorded nav motion (mean heading err ~45 deg).
    ALIGN_DEADBAND = np.radians(35.0)

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
        self.k_p_angular = 4
        
        self.waypoints = None
        self.current_waypoint_idx = 0
        
        self._joints_subset = ArticulationSubset(
            robot_articulation,
            ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]
        )

    def set_waypoints(self, waypoints: List[Tuple[float, float, float]], final_angle: Optional[float] = None) -> None:
        self.waypoints = np.array(waypoints)
        self.current_waypoint_idx = 0
        self.final_angle = final_angle

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap an angle to (-pi, pi] so any rotation is pursued the SHORT way.

        The base heading fed in here is ``spawn_euler + revolute_joint``; the
        revolute joint is an unwrapped accumulator that reaches several turns
        over a long A* path. Wrapping it before taking angle differences keeps
        every rotation target (travel bearing AND final dock angle) shortest-path.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_control(self, current_pose: np.ndarray) -> Tuple[float, float, float]:
        if self.waypoints is None or self.current_waypoint_idx >= len(self.waypoints):
            return 0.0, 0.0, 0.0, 0.0
        joint_positions = self._joints_subset.get_joint_positions()
        target = self.waypoints[self.current_waypoint_idx]
        current_pose[0] += joint_positions[0]
        current_pose[1] += joint_positions[1]
        current_pose[2] += joint_positions[2]
        heading = self._wrap_to_pi(current_pose[2])

        dx = target[0] - current_pose[0]
        dy = target[1] - current_pose[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < self.position_threshold:
            if self.current_waypoint_idx == len(self.waypoints) - 1:
                final_target = self.final_angle if self.final_angle is not None else target[2]
                final_angle_diff = self._wrap_to_pi(final_target - heading)
                if abs(final_angle_diff) < self.angle_threshold:
                    return 0.0, 0.0, 0.0, final_angle_diff
                # Clip here too (not only in get_action) so all branches return a
                # consistently bounded theta_vel.
                theta_vel = np.clip(self.k_p_angular * final_angle_diff,
                                    -self.max_angular_speed, self.max_angular_speed)
                return 0.0, 0.0, theta_vel, final_angle_diff
            else:
                self.current_waypoint_idx += 1
                return self.compute_control(current_pose)

        # Steering heading. The live bearing to the current waypoint swings wildly
        # as the base passes close beside it (tight A* spacing on short paths),
        # so the heading chases a spinning target and the base crabs. Within
        # ~1.5x the position threshold, steer by the waypoint's stored segment
        # bearing (points toward the NEXT waypoint) so the heading stays stable
        # through the corner and can actually align with the travel direction.
        live_bearing = np.arctan2(dy, dx)
        heading_target = float(target[2]) if distance < 1.5 * self.position_threshold else live_bearing
        angle_diff = self._wrap_to_pi(heading_target - heading)

        speed = min(distance * 0.2, self.max_linear_speed)
        # Face-forward driving: turn (nearly) in place until the heading is within
        # ALIGN_DEADBAND of the travel direction, then cos-scale. The hard gate
        # kills the sideways creep that plain cos-scaling left during the slow
        # in-place turn; cos-scaling then still floors translation at 0 beyond
        # 90 deg so the base never crabs sideways/backward.
        if abs(angle_diff) > self.ALIGN_DEADBAND:
            speed = 0.0
        else:
            speed *= max(0.0, np.cos(angle_diff))
        # Translation still homes on the actual waypoint (live bearing) so
        # position tracking/convergence is unchanged; only the heading target is
        # stabilized. Near the waypoint distance (hence speed) is tiny, so the
        # residual live/segment mismatch moves the base negligibly.
        x_vel = speed * np.cos(live_bearing)
        y_vel = speed * np.sin(live_bearing)
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
            done = (self.waypoints is not None and 
                    self.current_waypoint_idx == len(self.waypoints) - 1 and 
                    abs(theta_vel) < self.angle_threshold)
        else:
            done = (self.waypoints is not None and 
                    self.current_waypoint_idx == len(self.waypoints) - 1 and 
                    abs(theta_vel) < self.angle_threshold and 
                    abs(angle_diff) < self.angle_threshold)

        return action, done

    def is_path_complete(self) -> bool:
        return (self.waypoints is not None and 
                self.current_waypoint_idx >= len(self.waypoints))
