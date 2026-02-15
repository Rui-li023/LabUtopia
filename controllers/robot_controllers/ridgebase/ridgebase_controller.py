import numpy as np
from typing import List, Tuple, Optional
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.prims.impl import Articulation
from isaacsim.core.utils.types import ArticulationAction


def _normalize_angle(angle: float) -> float:
    """Normalize angle to the range [-π, π]."""
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


class RidgebaseController:
    def __init__(
        self,
        robot_articulation: Articulation,
        max_linear_speed: float = 1.0,  # It is recommended to set this to at least 0.5 or 1.0 in the YAML config
        max_angular_speed: float = 1.0,
        position_threshold: float = 0.15,
        angle_threshold: float = 0.15,
        dt: float = 0.2,
        final_angle: Optional[float] = None,
    ):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold
        self.dt = dt
        self.final_angle = final_angle

        # --- Angular control gains ---
        self.k_p_angular = 1.5
        self.k_d_angular = 0.3

        # --- Linear velocity control parameters ---
        self.k_p_linear = 2.0        # Distance gain: at 1 m distance we target 2 m/s (clamped by max_linear_speed).
        self.min_start_speed = 0.1   # Core change: minimum starting speed to avoid zero initial motion.

        # --- Velocity smoothing (low-pass filter) ---
        self.vel_filter_alpha = 0.6
        self._last_cmd_vel = np.zeros(3)

        self.waypoints = None
        self.current_waypoint_idx = 0

        self._joints_subset = ArticulationSubset(
            robot_articulation,
            ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"]
        )

    def set_waypoints(
        self, 
        waypoints: List[Tuple[float, float, float]], 
        final_angle: Optional[float] = None
    ) -> None:
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.current_waypoint_idx = 0
        if final_angle is not None:
            self.final_angle = final_angle
        self._last_cmd_vel = np.zeros(3)

    def compute_control(self, current_pose: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute control command.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Step 1: get current state.
        # Note: this keeps your original logic.
        # If current_pose is the robot base pose in world frame and joint_positions are dummy offsets,
        # then curr_x is computed as "base position + offset". Make sure this matches your intention.
        joint_positions = self._joints_subset.get_joint_positions()
        joint_velocities = self._joints_subset.get_joint_velocities()

        if joint_velocities is None or len(joint_velocities) < 3:
            joint_velocities = np.zeros(3)

        curr_x = current_pose[0] + joint_positions[0]
        curr_y = current_pose[1] + joint_positions[1]
        curr_theta = current_pose[2] + joint_positions[2]

        # Step 2: get current target waypoint.
        target = self.waypoints[self.current_waypoint_idx]
        dx = target[0] - curr_x
        dy = target[1] - curr_y
        distance = np.hypot(dx, dy)

        # Step 3: check whether the current waypoint has been reached.
        reached_position = distance < self.position_threshold
        if reached_position:
            if self.current_waypoint_idx == len(self.waypoints) - 1:
                # end_points
                target_heading = self.final_angle if self.final_angle is not None else target[2]
                angle_error = _normalize_angle(target_heading - curr_theta)

                if abs(angle_error) < self.angle_threshold:
                    return 0.0, 0.0, 0.0, angle_error

                # In-place rotation to correct heading.
                current_omega = joint_velocities[2]
                omega = self.k_p_angular * angle_error - self.k_d_angular * current_omega
                return 0.0, 0.0, omega, angle_error
            else:
                # Switch to the next waypoint.
                self.current_waypoint_idx += 1
                target = self.waypoints[self.current_waypoint_idx]
                dx = target[0] - curr_x
                dy = target[1] - curr_y
                distance = np.hypot(dx, dy)

        # Step 4: compute heading toward the target.
        target_angle = np.arctan2(dy, dx)
        angle_diff = _normalize_angle(target_angle - curr_theta)

        # Step 5: compute linear speed (core modification).

        # 1. Basic P control: speed proportional to distance.
        desired_speed = self.k_p_linear * distance

        # 2. Minimum starting speed: when not at the final goal, enforce at least min_start_speed.
        if distance > self.position_threshold:
            desired_speed = max(desired_speed, self.min_start_speed)

        # 3. Slow-down region: when very close to the target (< 0.2 m), allow smooth deceleration
        #    to avoid overshoot. We don't need very complex logic here, because k_p_linear = 2.0
        #    already yields 0.2 m/s at 0.1 m distance; combined with min_start_speed it might be
        #    a bit aggressive, so we add a smooth transition:
        if distance < 0.2:
             # At very short distance, remove the minimum speed limit and allow it to go to zero.
             desired_speed = self.k_p_linear * distance

        # 4. Clamp to maximum speed.
        speed = np.clip(desired_speed, 0.0, self.max_linear_speed)

        # 5. Decompose speed into x/y components.
        x_vel = speed * np.cos(target_angle)
        y_vel = speed * np.sin(target_angle)

        # Step 6: compute angular velocity (P + D).
        current_omega = joint_velocities[2]
        theta_vel = (
            self.k_p_angular * angle_diff 
            - self.k_d_angular * current_omega
        )

        return x_vel, y_vel, theta_vel, angle_diff

    def get_action(self, current_pose: np.ndarray) -> Tuple[Optional[ArticulationAction], bool]:
        x_vel, y_vel, theta_vel, angle_diff = self.compute_control(current_pose)

        # Clamp velocities.
        x_vel = np.clip(x_vel, -self.max_linear_speed, self.max_linear_speed)
        y_vel = np.clip(y_vel, -self.max_linear_speed, self.max_linear_speed)
        theta_vel = np.clip(theta_vel, -self.max_angular_speed, self.max_angular_speed)

        # Low-pass filter on velocities.
        raw_vel = np.array([x_vel, y_vel, theta_vel])
        filtered_vel = (
            self.vel_filter_alpha * raw_vel +
            (1 - self.vel_filter_alpha) * self._last_cmd_vel
        )
        self._last_cmd_vel = filtered_vel.copy()
        x_vel, y_vel, theta_vel = filtered_vel

        # Integrate velocities to obtain next joint positions.
        joint_positions = self._joints_subset.get_joint_positions()
        next_x = joint_positions[0] + x_vel * self.dt
        next_y = joint_positions[1] + y_vel * self.dt
        next_theta = joint_positions[2] + theta_vel * self.dt

        position = np.array([next_x, next_y, next_theta])
        velocity = np.array([x_vel, y_vel, theta_vel])  # Add velocity feed-forward.

        action = self._joints_subset.make_articulation_action(
            joint_positions=position,
            joint_velocities=velocity
        )

        # Check whether the path has been completed.
        done = False
        if self.waypoints is not None and self.current_waypoint_idx >= len(self.waypoints) - 1:
            joint_pos = self._joints_subset.get_joint_positions()
            # Keep world-frame computation consistent.
            world_x = current_pose[0] + joint_pos[0]
            world_y = current_pose[1] + joint_pos[1]
            world_theta = current_pose[2] + joint_pos[2]

            target = self.waypoints[-1]
            dist_err = np.hypot(target[0] - world_x, target[1] - world_y)

            if self.final_angle is not None:
                angle_err = abs(_normalize_angle(self.final_angle - world_theta))
            else:
                angle_err = abs(_normalize_angle(target[2] - world_theta))

            done = (dist_err < self.position_threshold) and (angle_err < self.angle_threshold)

        return action, done

    def is_path_complete(self) -> bool:
        return self.waypoints is not None and self.current_waypoint_idx >= len(self.waypoints)