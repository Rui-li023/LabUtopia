import os
import numpy as np
from typing import List, Optional

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.prims.impl import Articulation
from isaacsim.core.utils.types import ArticulationAction
from robots.base_robot import BaseRobot, GRIPPER_CLOSED
from robots.franka.rmpflow_controller import RMPFlowController


class FrankaTrajectoryController(RMPFlowController):
    """Franka robotic arm trajectory controller with support for continuous trajectory generation and execution"""

    def __init__(
        self, 
        name: str, 
        robot_articulation: Articulation, 
        physics_dt: float = 1.0/60.0,
        use_interpolation: bool = False
    ) -> None:
        super().__init__(name=name, robot_articulation=robot_articulation, physics_dt=physics_dt)
        
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        
        self._c_space_trajectory_generator = mg.LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        self._kinematics_solver = mg.LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf"
        )
        
        self._action_sequence = []
        self._action_sequence_index = 0
        self._end_effector_name = "panda_hand"
        self._physics_dt = physics_dt
        self._use_interpolation = use_interpolation

        # Resolve gripper open/close joint positions from robot
        self._gripper_open_joint_pos = np.array([0.04, 0.04])  # default Franka
        self._gripper_closed_joint_pos = np.array([0.0, 0.0])
        self._gripper_control_mode = "position"
        self._gripper_closing_force = 20.0
        self._gripper_closing_speed = 0.2
        self._gripper_dof_indices: list[int] = []
        robot = robot_articulation
        if isinstance(robot, BaseRobot):
            if hasattr(robot, 'gripper_open_positions'):
                open_pos = robot.gripper_open_positions
                close_pos = robot.gripper_closed_positions
                if len(open_pos) >= 2:
                    self._gripper_open_joint_pos = np.array(open_pos[:2], dtype=np.float64)
                    self._gripper_closed_joint_pos = np.array(close_pos[:2], dtype=np.float64)
            # Inherit gripper control mode from robot
            mode = getattr(robot, '_gripper_control_mode', 'position')
            if mode != "position":
                self._gripper_control_mode = mode
                self._gripper_closing_force = robot._gripper_closing_force
                self._gripper_closing_speed = robot._gripper_closing_speed
                try:
                    self._gripper_dof_indices = list(robot.gripper.joint_dof_indicies)
                except (AttributeError, TypeError):
                    self._gripper_dof_indices = []

    def _map_gripper_state_to_positions(self, state: float) -> np.ndarray:
        """Map binary gripper command (0=open, 1=closed) to finger positions.

        Training convention: action[7] ∈ {0, 1}. Threshold at 0.5 since
        model outputs cluster around 0 or 1 — intermediate values are noise.
        """
        if float(state) >= 0.5:
            return self._gripper_closed_joint_pos.copy()
        return self._gripper_open_joint_pos.copy()

    def _gripper_action_extras(self, gripper_state: float, n_dof: int):
        """Return (efforts, velocities) for the current gripper mode.

        Always returns (None, None).  In velocity/force modes the actual
        gripper command is applied per-step by ``robot.apply_gripper_effort()``
        using ``joint_indices`` to avoid touching arm DOFs.
        """
        return None, None

    def generate_trajectory(
        self, 
        waypoints: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> None:
        """Generate joint space trajectory using direct waypoints or interpolated trajectory
        
        Args:
            waypoints (np.ndarray): Array of shape (N, 8) containing N waypoints with joint angles and gripper positions
            timestamps (Optional[np.ndarray]): Array of timestamps for waypoints when using interpolation
        """
        joint_waypoints = waypoints[:, :7]
        self.gripper_positions = waypoints[:, 7]

        if np.allclose(joint_waypoints, joint_waypoints[0]):
            self._action_sequence = []
            for i in range(len(joint_waypoints)):
                gripper_joints = self._map_gripper_state_to_positions(self.gripper_positions[i])
                pos = np.concatenate([joint_waypoints[0], gripper_joints])
                efforts, vels = self._gripper_action_extras(self.gripper_positions[i], len(pos))
                action = ArticulationAction(
                    joint_positions=pos,
                    joint_velocities=vels,
                    joint_efforts=efforts,
                )
                self._action_sequence.append(action)
            total_actions = len(self._action_sequence)
            self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
            self._action_sequence_index = 0
            return
        
        if self._use_interpolation:
            joint_limits = self._c_space_trajectory_generator.get_c_space_position_limits()
            joint_min, joint_max = joint_limits[0], joint_limits[1]
            joint_waypoints = np.clip(joint_waypoints, joint_min, joint_max)
            if timestamps is not None:
                trajectory = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(
                    joint_waypoints, timestamps
                )
            else:
                trajectory = self._c_space_trajectory_generator.compute_c_space_trajectory(joint_waypoints)

            if trajectory is not None:
                articulation_trajectory = mg.ArticulationTrajectory(
                    self._articulation_motion_policy._robot_articulation,
                    trajectory,
                    self._physics_dt
                )
                self._action_sequence = articulation_trajectory.get_action_sequence()
                total_actions = len(self._action_sequence)
                self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
                self._action_sequence_index = 0
            else:
                print("Warning: Failed to generate trajectory")
                self._action_sequence = []
                self.gripper_positions = []
                self.gripper_indices = []
        else:
            # Direct waypoint output without interpolation
            joint_limits = self._c_space_trajectory_generator.get_c_space_position_limits()
            joint_min, joint_max = joint_limits[0], joint_limits[1]
            joint_waypoints = np.clip(joint_waypoints, joint_min, joint_max)
            
            self._action_sequence = []
            for i in range(len(joint_waypoints)):
                gripper_joints = self._map_gripper_state_to_positions(self.gripper_positions[i])
                pos = np.concatenate([joint_waypoints[i], gripper_joints])
                efforts, vels = self._gripper_action_extras(self.gripper_positions[i], len(pos))
                action = ArticulationAction(
                    joint_positions=pos,
                    joint_velocities=vels,
                    joint_efforts=efforts,
                )
                self._action_sequence.append(action)
                
            total_actions = len(self._action_sequence)
            self.gripper_indices = np.linspace(0, len(self.gripper_positions)-1, total_actions, dtype=int)
            self._action_sequence_index = 0

    def get_next_action(self) -> Optional[ArticulationAction]:
        """Get the next action in the sequence
        
        Returns:
            Optional[ArticulationAction]: Next action to execute, or None if sequence is complete
        """
        if not self._action_sequence or self._action_sequence_index >= len(self._action_sequence):
            return None
            
        action = self._action_sequence[self._action_sequence_index]
        
        if self._use_interpolation:
            # Add gripper position to interpolated trajectory actions
            if hasattr(self, 'gripper_positions') and len(self.gripper_positions) > 0:
                gripper_idx = self.gripper_indices[self._action_sequence_index]
                gripper_state = self.gripper_positions[gripper_idx]
                gripper_joints = self._map_gripper_state_to_positions(gripper_state)

                joint_positions = np.concatenate([
                    action.joint_positions,
                    gripper_joints.astype(np.float32)
                ])
                if action.joint_velocities is not None:
                    joint_velocities = np.concatenate([
                        action.joint_velocities,
                        np.array([0.0, 0.0], dtype=np.float32)
                    ])
                else:
                    joint_velocities = None

                efforts, extra_vels = self._gripper_action_extras(gripper_state, len(joint_positions))
                if efforts is None:
                    efforts = action.joint_efforts
                if extra_vels is not None:
                    joint_velocities = extra_vels  # velocity mode overrides

                action = ArticulationAction(
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities,
                    joint_efforts=efforts,
                )
        
        self._action_sequence_index += 1
        return action

    def is_trajectory_complete(self) -> bool:
        """Check if trajectory execution is complete
        
        Returns:
            bool: True if trajectory is complete, False otherwise
        """
        return len(self._action_sequence) == 0 or self._action_sequence_index >= len(self._action_sequence)

    def sync_gripper_mode(self, robot) -> None:
        """Sync gripper control mode from the robot."""
        if isinstance(robot, BaseRobot):
            mode = getattr(robot, '_gripper_control_mode', 'position')
            if mode != "position":
                self._gripper_control_mode = mode
                self._gripper_closing_force = robot._gripper_closing_force
                self._gripper_closing_speed = robot._gripper_closing_speed
                self._gripper_dof_indices = list(robot.gripper.joint_dof_indicies)

    def reset(self) -> None:
        """Reset controller state"""
        super().reset()
        self._action_sequence = []
        self._action_sequence_index = 0
        self.gripper_positions = []
        self.gripper_indices = []