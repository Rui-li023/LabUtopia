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
from isaacsim.core.utils.types import ArticulationAction


# Gripper state constants (control signal semantics)
# 0 = deactivate grip = open, 1 = activate grip = close
GRIPPER_OPEN = 0
GRIPPER_CLOSED = 1


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
        # Remembered so post_reset() can re-apply it. A fix-base articulation is anchored
        # by a root fixed joint at its AUTHORED transform, so the constructor's position
        # silently does nothing -- both converted arms (piper, arx_x5) come up welded at
        # their asset origin. set_world_pose() after the world reset does move them.
        self._requested_position = None if position is None else np.asarray(position, dtype=float)
        self._requested_orientation = None if orientation is None else np.asarray(orientation, dtype=float)
        self._end_effector: Optional[SingleRigidPrim] = None
        self._gripper: Optional[ParallelGripper] = None
        self._gripper_state: int = GRIPPER_OPEN  # Track current gripper state

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
    def motion_config(self) -> dict:
        """Lula/RMPFlow configuration for this arm.

        Returns the dict ``mg.lula.motion_policies.RmpFlow`` expects:
        ``robot_description_path``, ``urdf_path``, ``rmpflow_config_path`` and
        ``end_effector_frame_name``. The same paths drive the Lula kinematics solver
        and c-space trajectory generator.

        Every motion component reads these from the robot instead of hardcoding one
        arm's files, which is what lets the shared controllers drive any arm. Robots
        without a Lula description (mobile bases used purely for navigation) may leave
        this unimplemented; only motion-planned arms need it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define motion_config; it cannot be used "
            "with RMPFlow-based controllers."
        )

    @property
    def tool_frame_correction_euler_deg(self) -> list[float]:
        """Rotation from the CANONICAL grasp frame to this arm's tool frame.

        Every ``grasp.ee_euler_deg`` in the configs is written against the Franka/Piper
        convention: tool local +Z is the approach direction, local +Y separates the
        fingers. Not all arms agree -- measured by forward kinematics, ARX X5/R5 and the
        WidowX put their approach on local +X with the fingers still on +Y. Feeding them
        a canonical euler rotates the gripper 90 degrees off, so the fingers end up
        lying across the top of a cup instead of around it.

        Those arms declare [0, -90, 0], which maps their local +X onto the canonical +Z
        and leaves +Y alone. Default is no correction.
        """
        return [0.0, 0.0, 0.0]

    @property
    def ik_end_effector_frame(self) -> str:
        """Lula frame name the kinematics solver and trajectory generator target.

        Deliberately separate from ``motion_config["end_effector_frame_name"]``: on
        Franka, RMPFlow steers ``right_gripper`` while IK and c-space trajectories are
        posed against ``panda_hand``. Collapsing the two would silently move Franka's
        grasp frame.
        """
        return self.motion_config["end_effector_frame_name"]

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

    @property
    def gripper_distance_multipliers(self) -> List[float]:
        """Per-joint multipliers for mapping a scalar gripper distance.

        By default all gripper joints use the same sign/magnitude (+1).
        Subclasses can override this for mirrored grippers (e.g., +d/-d).
        """
        return [1.0] * self.num_gripper_joints

    @property
    def default_pick_gripper_distance(self) -> float:
        """Default scalar distance used for opening the gripper in pick.

        The scalar is inferred from default joint positions and the
        ``gripper_distance_multipliers`` mapping.
        """
        if self.num_gripper_joints == 0:
            return 0.0

        default_positions = getattr(self, "_default_joint_positions", None)
        if default_positions is None or len(default_positions) < self.num_gripper_joints:
            default_positions = self.DEFAULT_JOINT_POSITIONS
        if default_positions is None or len(default_positions) < self.num_gripper_joints:
            return 0.0

        gripper_defaults = np.array(default_positions[-self.num_gripper_joints :], dtype=np.float64)
        multipliers = np.array(self.gripper_distance_multipliers, dtype=np.float64)
        if len(multipliers) != self.num_gripper_joints:
            raise ValueError(
                f"gripper_distance_multipliers length ({len(multipliers)}) does not match "
                f"num_gripper_joints ({self.num_gripper_joints}) for robot '{self.name}'."
            )

        nonzero = np.abs(multipliers) > 1e-8
        if not np.any(nonzero):
            return 0.0
        inferred = np.abs(gripper_defaults[nonzero] / multipliers[nonzero])
        return float(np.mean(inferred))

    def get_gripper_joint_targets_from_distance(self, distance: float) -> np.ndarray:
        """Map a scalar gripper distance to per-joint position targets."""
        if self.num_gripper_joints == 0:
            return np.array([], dtype=np.float64)
        multipliers = np.array(self.gripper_distance_multipliers, dtype=np.float64)
        if len(multipliers) != self.num_gripper_joints:
            raise ValueError(
                f"gripper_distance_multipliers length ({len(multipliers)}) does not match "
                f"num_gripper_joints ({self.num_gripper_joints}) for robot '{self.name}'."
            )
        return multipliers * float(distance)

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

    # ── Gripper control methods ───────────────────────────────────────────────

    def open_gripper(self) -> None:
        """Open the gripper.

        Uses the ParallelGripper's forward method with action="open".
        """
        if self._gripper is not None:
            action = self._gripper.forward(action="open")
            self.apply_action(action)
            self._gripper_state = GRIPPER_OPEN
            self._gripper_cmd_opening = float(self.gripper_open_positions[0])

    def close_gripper(self) -> None:
        """Close the gripper.

        Uses the ParallelGripper's forward method with action="close".
        """
        if self._gripper is not None:
            action = self._gripper.forward(action="close")
            self.apply_action(action)
            self._gripper_state = GRIPPER_CLOSED
            self._gripper_cmd_opening = float(self.gripper_closed_positions[0])

    def close_gripper_to_distance(self, distance: float) -> None:
        """Close the gripper to a target per-finger opening instead of slamming
        fully shut.

        A full (0/1) close drives the fingers to position 0 under the position
        controller; against a rigid object the controller keeps fighting and the
        resulting force can eject light/round objects (e.g. round_bottom_flask).
        Targeting the object's grasp width lets the fingers stop on contact with
        a gentle hold. Falls back silently if the robot has no gripper.
        """
        if self._gripper is None:
            return
        # Command only the gripper DOFs to the distance-derived targets, with
        # explicit joint_indices so positions/indices stay the same length
        # (avoids the full-DOF broadcast mismatch from ParallelGripper.forward).
        targets = np.asarray(
            self.get_gripper_joint_targets_from_distance(distance), dtype=np.float32
        )
        indices = np.asarray(self.get_gripper_joint_indices(), dtype=np.int32)
        if indices.size == 0:
            return
        self.apply_action(
            ArticulationAction(joint_positions=targets, joint_indices=indices)
        )
        self._gripper_state = GRIPPER_CLOSED
        self._gripper_cmd_opening = float(targets[0])

    def set_gripper_state(self, state: int) -> None:
        """Set gripper state using discrete signal.

        Args:
            state: 0 = open, 1 = closed
        """
        if state == GRIPPER_CLOSED:
            self.close_gripper()
        elif state == GRIPPER_OPEN:
            self.open_gripper()
        else:
            raise ValueError(f"Invalid gripper state: {state}. Must be 0 (open) or 1 (closed).")

    def get_gripper_state(self) -> int:
        """Get current gripper state.

        Returns:
            int: 0 = open, 1 = closed
        """
        return self._gripper_state

    def get_gripper_commanded_opening(self) -> float:
        """Last commanded per-finger opening (metres) — the gripper DRIVE target,
        not the measured finger position. Survives controller handoffs (tracked
        at robot level), so it can be recorded as the action's gripper channel:
        replaying it reproduces the exact same drive target (e.g. a binary close
        keeps squeezing toward 0 even when fingers rest on the object at >0).

        Returns:
            float: commanded opening; defaults to fully open before any command.
        """
        cmd = getattr(self, "_gripper_cmd_opening", None)
        if cmd is None:
            return float(self.gripper_open_positions[0])
        return float(cmd)

    def sync_gripper_from_action(self, action) -> None:
        """Sync gripper state from action (velocity/force mode). No-op by default."""
        pass

    def apply_gripper_effort(self) -> None:
        """Apply persistent gripper effort (velocity/force mode). No-op by default."""
        pass

    # ── Abstract methods ─────────────────────────────────────────────────────

    @abstractmethod
    def get_gripper_position(self) -> np.ndarray:
        """Get gripper position in world coordinates.

        Returns:
            np.ndarray: Gripper position [x, y, z].
        """
        ...

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
        super().initialize(physics_sim_view)
        self.ensure_drive_damping()

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

    # Fraction of drive stiffness used as damping when the USD ships none.
    #
    # NOT Franka's own 0.2 (22918/4584). Damping only has to be a few times critical to
    # kill the feedback blow-up; critical damping for these arms is 2*sqrt(K*I), i.e. a
    # ratio near 0.004 for UR5e and 0.001 for ARX, so 0.2 is ~50x over-damped. That is
    # not free: it makes the arm sluggish, and the pick controller's phases have fixed
    # durations, so at 0.2 the UR5e finished every lift short of the +0.1 m success gate
    # (0/38 episodes) even though its static tracking error looked excellent. 0.02 keeps
    # the arm responsive and still holds tracking to a few mm in both control regimes.
    DRIVE_DAMPING_RATIO: float = 0.02
    # Some vendor URDFs contain a tiny physical joint damping value (iiwa: 0.5).
    # The importer copies it into a high-stiffness position drive, where it is orders
    # of magnitude too small but still bypasses the zero-only repair below. Keep this
    # opt-in so existing tuned assets are unaffected.
    ENFORCE_DRIVE_DAMPING_FLOOR: bool = False

    def ensure_drive_damping(self) -> None:
        """Give position drives a damping term when the USD ships none.

        The URDF importer only writes drive damping when the URDF declares
        ``<dynamics damping="..."/>``, and none of the arm descriptions converted here do
        -- ARX X5/R5, WidowX and UR5e all come up with stiffness ~35810 and damping 0.

        That is not a cosmetic difference. RMPFlow reads the MEASURED joint velocity back
        every step (``ignore_robot_state_updates`` is off outside position-only
        collection), so an undamped position drive feeds its damping term velocity noise
        and the policy diverges: ARX ended up 0.33 m away from a target 0.05 m from its
        start, and UR5e 0.57 m from its grasp pose. Restoring Franka's ratio brings both
        to millimetres. Franka itself is untouched -- its drives already have damping.

        Call from ``initialize()``, after the articulation view exists.
        """
        view = getattr(self, "_articulation_view", None)
        if view is None:
            return
        try:
            stiffness, damping = view.get_gains()
        except Exception:  # not every articulation exposes gains
            return
        stiffness = np.atleast_2d(np.asarray(stiffness, dtype=float))
        damping = np.atleast_2d(np.asarray(damping, dtype=float))
        # Only joints that are actually position-driven AND undamped. Mimic followers
        # have stiffness 0 and are carried by their lead joint, so leave them alone.
        damping_floor = stiffness * self.DRIVE_DAMPING_RATIO
        needs_damping = (damping <= 0.0) & (stiffness > 0.0)
        if self.ENFORCE_DRIVE_DAMPING_FLOOR:
            needs_damping |= (stiffness > 0.0) & (damping < damping_floor)
        if not np.any(needs_damping):
            return
        damping = np.where(needs_damping, damping_floor, damping)
        view.set_gains(kds=damping)

    def enforce_requested_world_pose(self) -> None:
        """Re-apply the configured base pose after a reset.

        Call from ``post_reset()`` on any arm whose USD welds its base. Silently does
        nothing when no position was requested, so arms that already honour the
        constructor argument are unaffected.
        """
        if self._requested_position is None:
            return
        current, _ = self.get_world_pose()
        if np.allclose(current, self._requested_position, atol=1e-4):
            return
        self.set_world_pose(
            position=self._requested_position,
            orientation=self._requested_orientation if self._requested_orientation is not None else None,
        )

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

    @property
    def gripper_open_positions(self) -> np.ndarray:
        """Joint positions for fully open gripper."""
        return getattr(self, "_gripper_open_position", np.zeros(self.num_gripper_joints))

    @property
    def gripper_closed_positions(self) -> np.ndarray:
        """Joint positions for fully closed gripper."""
        return getattr(self, "_gripper_closed_position", np.zeros(self.num_gripper_joints))

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
