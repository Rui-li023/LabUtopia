# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for all robots in LabUtopia.

Defines the common interface for arm robots (fixed-base) and mobile manipulators.
Subclasses must implement abstract properties and methods for robot-specific configurations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.sensors.physics import ContactSensor


class BaseRobot(Robot, ABC):
    """Abstract base class for all robots in LabUtopia.

    This class defines the common interface for both fixed-base manipulators
    (e.g., Franka, Piper) and mobile manipulators (e.g., Ridgebase).

    Subclasses must implement:
        - arm_joint_names: List of arm joint names
        - gripper_joint_names: List of gripper joint names
        - end_effector_prim_path: USD prim path of the end effector
        - gripper_center_prim_path: USD prim path of the gripper center (TCP)
        - get_gripper_position(): Get gripper position in world coordinates
        - initialize(): Initialize robot components

    Optional overrides:
        - base_joint_names: List of mobile base joint names (empty for fixed-base)
        - get_contact_sensor(): Return contact sensors (default: (None, None))
        - camera: Camera sensor (default: None)

    Attributes:
        prim_path_str (str): String representation of the robot's prim path.
        _end_effector (Optional[SingleRigidPrim]): End effector rigid body.
        _gripper (Optional[ParallelGripper]): Gripper controller.
    """

    # Subclasses should override this class-level constant
    DEFAULT_JOINT_POSITIONS: np.ndarray = np.array([])

    def __init__(
        self,
        prim_path: str,
        name: str,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the base robot.

        Args:
            prim_path: USD prim path for the robot.
            name: Robot name.
            position: Robot base position. Defaults to None.
            orientation: Robot base orientation. Defaults to None.
        """
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            articulation_controller=None,
        )
        self.prim_path_str = prim_path
        self._end_effector: Optional[SingleRigidPrim] = None
        self._gripper: Optional[ParallelGripper] = None

    # ── Abstract properties (must be implemented by subclasses) ─────────────

    @property
    @abstractmethod
    def arm_joint_names(self) -> List[str]:
        """Ordered list of arm joint names.

        Returns:
            List of arm joint names in the order they appear in the articulation.
        """
        ...

    @property
    @abstractmethod
    def gripper_joint_names(self) -> List[str]:
        """Ordered list of gripper joint names.

        Returns:
            List of gripper joint names in the order they appear in the articulation.
        """
        ...

    @property
    @abstractmethod
    def end_effector_prim_path(self) -> str:
        """USD prim path of the end effector.

        Returns:
            Full prim path string to the end effector link.
        """
        ...

    @property
    @abstractmethod
    def gripper_center_prim_path(self) -> str:
        """USD prim path of the gripper center (tool center point).

        This is typically a point between the gripper fingers, used for
        grasping and manipulation calculations.

        Returns:
            Full prim path string to the gripper center point.
        """
        ...

    # ── Optional overrides for mobile manipulators ───────────────────────────

    @property
    def base_joint_names(self) -> List[str]:
        """Ordered list of base/mobile joint names.

        Override this property for mobile manipulators. Default is empty list
        for fixed-base arms.

        Returns:
            List of mobile base joint names. Empty for fixed-base robots.
        """
        return []

    @property
    def has_mobile_base(self) -> bool:
        """Whether this robot has a mobile base.

        Returns:
            True if the robot has mobile base joints, False otherwise.
        """
        return len(self.base_joint_names) > 0

    # ── Derived properties for joint counts ──────────────────────────────────

    @property
    def num_arm_joints(self) -> int:
        """Number of arm joints.

        Returns:
            Count of arm joints.
        """
        return len(self.arm_joint_names)

    @property
    def num_gripper_joints(self) -> int:
        """Number of gripper joints.

        Returns:
            Count of gripper joints.
        """
        return len(self.gripper_joint_names)

    @property
    def num_base_joints(self) -> int:
        """Number of mobile base joints.

        Returns:
            Count of base joints.
        """
        return len(self.base_joint_names)

    @property
    def num_joints(self) -> int:
        """Total number of joints (arm + gripper + base).

        Returns:
            Total count of all joints in the robot.
        """
        return self.num_arm_joints + self.num_gripper_joints + self.num_base_joints

    def get_all_joint_names(self) -> List[str]:
        """Return all joint names in order: base + arm + gripper.

        This ordering matches the typical articulation structure where
        base joints come first, followed by arm joints, then gripper joints.

        Returns:
            Combined list of all joint names.
        """
        return self.base_joint_names + self.arm_joint_names + self.gripper_joint_names

    # ── Common accessors ─────────────────────────────────────────────────────

    @property
    def end_effector(self) -> Optional[SingleRigidPrim]:
        """End effector rigid body prim.

        Returns:
            SingleRigidPrim for the end effector, or None if not initialized.
        """
        return self._end_effector

    @property
    def gripper(self) -> Optional[ParallelGripper]:
        """Gripper controller.

        Returns:
            ParallelGripper instance, or None if not initialized.
        """
        return self._gripper

    @property
    def camera(self) -> Optional[object]:
        """Wrist-mounted camera.

        Override this property if the robot has a camera.

        Returns:
            Camera instance, or None if no camera.
        """
        return None

    def get_contact_sensor(self) -> Tuple[Optional[ContactSensor], Optional[ContactSensor]]:
        """Return contact sensors for gripper fingers.

        Override this method if the robot has contact sensors.

        Returns:
            Tuple of (left_contact_sensor, right_contact_sensor).
            Default returns (None, None).
        """
        return None, None

    # ── Abstract methods ─────────────────────────────────────────────────────

    @abstractmethod
    def get_gripper_position(self) -> np.ndarray:
        """Get gripper position in world coordinates.

        Returns:
            np.ndarray: Gripper position [x, y, z].
        """
        ...

    @abstractmethod
    def initialize(self, physics_sim_view=None) -> None:
        """Initialize robot components.

        This method is called after the simulation is created.
        Subclasses should:
        1. Call super().initialize(physics_sim_view)
        2. Initialize end effector (SingleRigidPrim)
        3. Initialize gripper controller
        4. Set default joint positions

        Args:
            physics_sim_view: Physics simulation view from Isaac Sim.
        """
        ...

    @abstractmethod
    def post_reset(self) -> None:
        """Post reset callback.

        This method is called after each simulation reset.
        Subclasses should:
        1. Call super().post_reset()
        2. Reset gripper state
        3. Set joint control modes
        4. Set default joint positions
        """
        ...

    # ── Utility methods ──────────────────────────────────────────────────────

    def get_arm_joint_indices(self) -> List[int]:
        """Get articulation indices for arm joints.

        Returns:
            List of joint indices corresponding to arm joints.
        """
        indices = []
        for name in self.arm_joint_names:
            if name in self.dof_names:
                indices.append(self.dof_names.index(name))
        return indices

    def get_gripper_joint_indices(self) -> List[int]:
        """Get articulation indices for gripper joints.

        Returns:
            List of joint indices corresponding to gripper joints.
        """
        indices = []
        for name in self.gripper_joint_names:
            if name in self.dof_names:
                indices.append(self.dof_names.index(name))
        return indices

    def get_base_joint_indices(self) -> List[int]:
        """Get articulation indices for base joints.

        Returns:
            List of joint indices corresponding to base joints.
        """
        indices = []
        for name in self.base_joint_names:
            if name in self.dof_names:
                indices.append(self.dof_names.index(name))
        return indices
