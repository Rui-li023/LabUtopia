from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import BaseRobot, GRIPPER_CLOSED, GRIPPER_OPEN


class PickController(AtomicBaseController):
    """A state machine controller for picking up objects.

    Manages the process of picking an object through multiple phases:
    - Phase 0: Move end effector above the object.
    - Phase 1: Lower end effector closer to the object.
    - Phase 2: Position end effector for grasping.
    - Phase 3: Wait for robot dynamics to settle.
    - Phase 4: Close gripper to grasp the object.
    - Phase 5: Lift the object.
    - Phase 6: Complete the sequence.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (typing.Any): Cartesian space controller that returns ArticulationAction.
        events_dt (List[float], optional): Duration for each phase. Defaults to [0.004, 0.002, 0.01, 0.2, 0.05, 0.004, 0.006].
        robot (BaseRobot, optional): Robot articulation. If not provided, it is inferred from ``cspace_controller``.

    Raises:
        Exception: If events_dt is not a list or numpy array, or if its length is not 7.
        ValueError: If robot cannot be inferred from ``cspace_controller``.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0

        if events_dt is None:
            self._events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._cspace_controller = cspace_controller
        self._start = True
        self.object_size = None
        self._position_threshold = position_threshold
        self._robot_position = None
        self._randomization_sampled = False
        self._pre_offset_z_noise = 0.0
        self._after_offset_z_noise = 0.0
        self._orientation_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._orientation_angle_deg = 0.0
        self._current_gripper_state = GRIPPER_OPEN

        self._robot = self._resolve_robot(robot=robot, cspace_controller=cspace_controller)
        if self._robot.num_gripper_joints <= 0:
            raise ValueError(
                f"PickController requires at least one gripper joint, got {self._robot.num_gripper_joints} "
                f"for robot '{self._robot.name}'."
            )

    def _resolve_robot(
        self,
        robot: typing.Optional[BaseRobot],
        cspace_controller: typing.Any,
    ) -> BaseRobot:
        if robot is not None:
            if not isinstance(robot, BaseRobot):
                raise TypeError(f"robot must be BaseRobot, got {type(robot)}")
            return robot

        candidates = []

        for attr in ("robot", "robot_articulation", "_robot", "_robot_articulation"):
            candidate = getattr(cspace_controller, attr, None)
            if candidate is not None:
                candidates.append(candidate)

        articulation_motion_policy = getattr(cspace_controller, "_articulation_motion_policy", None)
        if articulation_motion_policy is not None:
            for attr in ("_robot_articulation", "robot_articulation", "_robot", "robot"):
                candidate = getattr(articulation_motion_policy, attr, None)
                if candidate is not None:
                    candidates.append(candidate)

        articulation_rmp = getattr(cspace_controller, "articulation_rmp", None)
        if articulation_rmp is not None:
            for attr in ("_robot_articulation", "robot_articulation", "_robot", "robot"):
                candidate = getattr(articulation_rmp, attr, None)
                if candidate is not None:
                    candidates.append(candidate)

        for candidate in candidates:
            if isinstance(candidate, BaseRobot):
                return candidate

        raise ValueError(
            "PickController could not resolve a BaseRobot instance from cspace_controller. "
            "Please pass robot=... explicitly when constructing PickController."
        )

    def _calculate_approach_direction(self, picking_position: np.ndarray) -> np.ndarray:
        if self._robot_position is None:
            return np.array([-1, 0, 0])

        relative_pos = picking_position - self._robot_position

        horizontal_vec = relative_pos.copy()
        horizontal_vec[2] = 0

        if np.linalg.norm(horizontal_vec) > 0:
            horizontal_vec = -horizontal_vec / np.linalg.norm(horizontal_vec)
        else:
            horizontal_vec = np.array([-1, 0, 0])
        return horizontal_vec

    def _sample_episode_randomization(self) -> None:
        """Sample per-episode randomization values once."""
        self._pre_offset_z_noise = float(np.random.uniform(-0.04, 0.04))
        self._after_offset_z_noise = float(np.random.uniform(-0.04, 0.04))

        axis_choices = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        self._orientation_axis = axis_choices[int(np.random.randint(0, len(axis_choices)))]
        self._orientation_angle_deg = float(np.random.uniform(-15.0, 15.0))
        self._randomization_sampled = True

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication for [x, y, z, w] format."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )

    def _apply_axis_rotation_to_quat(
        self,
        quat: np.ndarray,
        axis: np.ndarray,
        angle_deg: float,
    ) -> np.ndarray:
        """Apply an axis-angle rotation to a quaternion in [x, y, z, w] format."""
        quat = np.asarray(quat, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 0:
            return quat
        axis = axis / axis_norm

        half_angle = np.deg2rad(angle_deg) / 2.0
        sin_half = np.sin(half_angle)
        delta_quat = np.array(
            [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half_angle)],
            dtype=np.float64,
        )
        rotated = self._quat_multiply(delta_quat, quat)
        norm = np.linalg.norm(rotated)
        if norm > 0:
            rotated = rotated / norm
        return rotated

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_name: str,
        object_size: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        pre_offset_z: float = 0.12,
        after_offset_z: float = 0.15,
        pre_offset_x: float = 0.1,
        gripper_distances: float = None
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        """Computes the joint positions for the current picking phase.

        Args:
            picking_position (np.ndarray): Target position for picking.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            object_name (str): Name of the object to pick.
            object_size (np.ndarray): Size of the object.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current position of the gripper.
            end_effector_orientation (np.ndarray, optional): Target orientation for the end effector. Defaults to [0, pi, 0] Euler angles.

        Returns:
            Tuple[ArticulationAction, np.ndarray]: Joint positions for the robot to execute and 8-dim record array.
        """
        self.object_size = object_size

        if self._start:
            self._start = False
            self._robot.open_gripper()
            self._current_gripper_state = GRIPPER_OPEN
            action = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            record_array = self._build_record_array(action, current_joint_positions, gripper_state=GRIPPER_OPEN)
            return action, record_array

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if not self._randomization_sampled:
            self._sample_episode_randomization()

        self.pre_offset_z = max(0.0, pre_offset_z + self._pre_offset_z_noise)
        self.after_offset_z = max(0.0, after_offset_z + self._after_offset_z_noise)
        self.pre_offset_x = pre_offset_x
        end_effector_orientation = self._apply_axis_rotation_to_quat(
            quat=end_effector_orientation,
            axis=self._orientation_axis,
            angle_deg=self._orientation_angle_deg,
        )

        target_joint_positions = self._execute_phase(
            picking_position,
            end_effector_orientation,
            current_joint_positions,
            object_name,
            gripper_control,
            gripper_position,
            gripper_distances
        )

        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        record_array = self._build_record_array(target_joint_positions, current_joint_positions, gripper_state=self._current_gripper_state)
        return target_joint_positions, record_array

    def _execute_phase(self, picking_position, end_effector_orientation, current_joint_positions, object_name, gripper_control, gripper_position, gripper_distances):
        """Executes the current phase of the picking sequence.

        Args:
            picking_position (np.ndarray): Target position for picking.
            end_effector_orientation (np.ndarray): Target orientation for end effector.
            current_joint_positions (np.ndarray): Current robot joint positions.
            object_name (str): Name of the target object.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current gripper position.

        Returns:
            ArticulationAction: Joint position targets for robot control.
        """

        approach_dir = self._calculate_approach_direction(picking_position)

        if self._event == 0:
            picking_position = picking_position + approach_dir * (self.pre_offset_x / get_stage_units())
            picking_position[2] += self.object_size[2] + self.pre_offset_z
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 1:
            picking_position = picking_position + approach_dir * (self.pre_offset_x / get_stage_units())
            picking_position[2] += self.get_pickprez_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 2:
            picking_position[2] += self.get_pickz_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - picking_position[:2])
            z_distance = abs(gripper_position[2] - picking_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 3:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:
            # Close gripper to grasp
            self._robot.close_gripper()
            self._current_gripper_state = GRIPPER_CLOSED
            self.target_position = picking_position.copy()
            self.target_position[2] += self.after_offset_z / get_stage_units()
            if "glass" in object_name:
                gripper_control.add_object_to_gripper("/World/glass_rod/Cylinder", self._robot.gripper_center_prim_path)
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 5:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - self.target_position[:2])
            z_distance = abs(gripper_position[2] - self.target_position[2])
            if xy_distance < self._position_threshold and z_distance < self._position_threshold:
                self._event += 1
                self._t = 0
            return target_joint_positions
        else:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the controller to the initial phase.

        Args:
            events_dt (List[float], optional): New phase durations. Defaults to None.

        Raises:
            Exception: If events_dt is not a list or numpy array, or if its length is not 7.
        """
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0

        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._start = True
        self.object_size = None
        self._robot_position = None
        self._reset_record_state()
        self._randomization_sampled = False
        self._pre_offset_z_noise = 0.0
        self._after_offset_z_noise = 0.0
        self._orientation_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._orientation_angle_deg = 0.0
        self._current_gripper_state = GRIPPER_OPEN

    def get_gripper_distance(self, item_name):
        """Determines the gripper opening distance for the specified object.

        Args:
            item_name (str): Name of the object to be gripped.

        Returns:
            float: Gripper finger distance in meters.
        """
        gripper_distances = {
            "rod": 0.003,
            "tube": 0.01,
            "beaker": 0.022,
            "beaker_l": 0.03,
            "beaker_04": 0.025,
            "beaker_05": 0.025,
            "beaker_03": 0.025,
            "Erlenmeyer flask": 0.018,
            "pipette": 0.008,
            "microscope slide": 0.002,
            "graduated_cylinder_01": 0.005,
            "graduated_cylinder_02": 0.018,
            "graduated_cylinder_04": 0.030,
        }

        for key in gripper_distances:
            if key == item_name.lower():
                return gripper_distances[key]

        return 0.0

    def get_pickz_offset(self, item_name):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.065,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker": 0.0,
            "beaker_04": 0.0,
            "beaker_05": 0.0,
            "beaker_03": 0.0,
            "beaker2": 0.0,
            "beaker_2": 0.0,
            "beaker_l": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "glass_rod": 0.02,
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5

    def get_pickprez_offset(self, item_name):
        """Calculates the vertical offset for the pre-grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "volume_flask": 0,
            "beaker2": 0.05,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "graduated_cylinder_01": 0.05,
            "graduated_cylinder_02": 0.03,
            "graduated_cylinder_03": 0.03,
            "graduated_cylinder_04": 0.03
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 3
