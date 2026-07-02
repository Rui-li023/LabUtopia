# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ridgebase mobile platform robot with Franka arm.

A mobile manipulator combining a Ridgeback mobile base with a Franka Panda arm.
"""

from typing import List, Optional, Tuple

import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.sensors.physics import ContactSensor

from robots.base_robot import BaseRobot
from utils.object_utils import ObjectUtils


class Ridgebase(BaseRobot):
    """Ridgebase mobile platform robot with Franka arm.

    This class encapsulates a mobile manipulator that combines a Ridgeback
    mobile base with a Franka Panda robotic arm. It provides unified access
    to both base and arm joints.

    Attributes:
        prim_path_str (str): The path of the robot in the USD scene
        name (str): The name of the robot
        usd_path (str): The USD file path
        position (np.ndarray): The initial position of the robot
        orientation (np.ndarray): The initial orientation of the robot
    """

    # Ridgebase + Franka default joint positions
    # 3 base joints + 7 arm joints + 2 gripper joints
    DEFAULT_JOINT_POSITIONS = np.array([
        0.0, 0.0, 0.0,  # base joints (x, y, theta)
        0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398,  # arm joints
        0.04, 0.04  # gripper joints
    ])

    # Base joint names for mobile platform
    _BASE_JOINT_NAMES = [
        "dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
        "dummy_base_revolute_z_joint"
    ]

    # Franka arm joint names
    _ARM_JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]

    # Franka gripper joint names
    _GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

    def __init__(
        self,
        prim_path: str = "/World/Ridgebase",
        name: str = "ridgebase",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Ridgebase robot.

        Args:
            prim_path: The path of the robot in the USD scene
            name: The name of the robot
            usd_path: The USD file path, if None then use the default path
            position: The initial position [x, y, z]
            orientation: The initial orientation (quaternion)
        """
        prim = get_prim_at_path(prim_path)

        # If the prim does not exist, load the USD file
        if not prim.IsValid():
            if usd_path is None:
                usd_path = "assets/robots/ridgeback_franka.usd"
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # Initialize base robot
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
        )

        # ── Gripper setup ────────────────────────────────────────────────────
        # The Ridgebase reuses the Franka Panda hand, but (unlike robots/franka)
        # never instantiated a ParallelGripper, so self._gripper stayed None and
        # every gripper command (open/close/close_to_distance) silently no-op'd
        # at the `if self._gripper is None` guard — the arm reached the object
        # but the fingers never closed, so Level-5 pick could never lift it.
        # Mirror the Franka's gripper construction so grasps actually actuate.
        self._end_effector_prim_path = prim_path + "/panda_rightfinger"
        self._default_joint_positions = self.DEFAULT_JOINT_POSITIONS.copy()
        self._gripper_open_position = np.array([0.04, 0.04])
        self._gripper_closed_position = np.array([0.0, 0.0])
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=self._GRIPPER_JOINT_NAMES,
            joint_opened_positions=self._gripper_open_position,
            joint_closed_positions=self._gripper_closed_position,
            action_deltas=np.array([0.04, 0.04]) / get_stage_units(),
        )
        self._gripper_control_mode = "position"

    # ── Implement abstract properties from BaseRobot ─────────────────────────

    @property
    def arm_joint_names(self) -> List[str]:
        """Ordered list of arm joint names (Franka Panda)."""
        return self._ARM_JOINT_NAMES

    @property
    def gripper_joint_names(self) -> List[str]:
        """Ordered list of gripper joint names (Franka Panda)."""
        return self._GRIPPER_JOINT_NAMES

    @property
    def base_joint_names(self) -> List[str]:
        """Ordered list of mobile base joint names."""
        return self._BASE_JOINT_NAMES

    @property
    def end_effector_prim_path(self) -> str:
        """USD prim path of the end effector."""
        return self.prim_path_str + "/panda_rightfinger"

    @property
    def gripper_center_prim_path(self) -> str:
        """USD prim path of the gripper center (tool center point)."""
        return self.prim_path_str + "/panda_hand/tool_center"

    @property
    def has_mobile_base(self) -> bool:
        """Ridgebase has a mobile base."""
        return True

    # ── Ridgebase-specific methods ──────────────────────────────────────────

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the position and orientation of the robot's base.

        Returns:
            Tuple of (position, orientation) - position as [x, y, z],
            orientation as quaternion [w, x, y, z].
        """
        return self.get_world_pose()

    # ── Implement abstract methods from BaseRobot ───────────────────────────

    def get_gripper_position(self) -> np.ndarray:
        """Get the gripper position in world coordinates.

        Returns:
            np.ndarray: The gripper position [x, y, z].
        """
        return ObjectUtils.get_instance().get_object_xform_position(
            object_path=self.gripper_center_prim_path
        )

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize the physical properties of the robot.

        Args:
            physics_sim_view: The physics simulation view
        """
        super().initialize(physics_sim_view)
        # Bring the Franka hand's gripper online (see __init__ note). Without
        # this, ParallelGripper.forward() has no articulation hooks and the
        # gripper channel stays inert.
        self._end_effector = SingleRigidPrim(
            prim_path=self._end_effector_prim_path,
            name=self.name + "_end_effector",
        )
        self._end_effector.initialize(physics_sim_view)
        dof_names = self.dof_names if self.dof_names is not None else self.get_all_joint_names()
        gripper_default_state = np.array(
            self._default_joint_positions[-self.num_gripper_joints:], dtype=np.float64)
        self._gripper.set_default_state(gripper_default_state)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=dof_names,
        )

    def post_reset(self) -> None:
        """Post-reset operations."""
        super().post_reset()
        if self._gripper is not None:
            self._gripper.post_reset()
