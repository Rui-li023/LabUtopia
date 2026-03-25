from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from .atomic_base_controller import AtomicBaseController

from robots.base_robot import BaseRobot, GRIPPER_CLOSED, GRIPPER_OPEN


class PlaceController(AtomicBaseController):
    """A state machine controller for placing objects.

    Manages the process of placing an object through multiple phases:
    - Phase 0: Move to pre-place position above target.
    - Phase 1: Lower to place position.
    - Phase 2: Wait for dynamics to settle.
    - Phase 3: Open gripper to release.
    - Phase 4: Retreat from placed object.
    - Phase 5: Complete the sequence.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (typing.Any): Cartesian space controller that returns ArticulationAction.
        gripper: Gripper controller instance (deprecated, use robot instead).
        events_dt (List[float], optional): Duration for each phase. Defaults to [0.005, 0.01, 0.08, 0.05, 0.01, 0.1].
        _position_threshold (float): Position threshold for phase transitions. Defaults to 0.01.
        robot (BaseRobot, optional): Robot articulation. If not provided, it is inferred from cspace_controller.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        gripper=None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01,
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0

        if events_dt is None:
            self._events_dt = [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        else:
            if not isinstance(events_dt, (np.ndarray, list)):
                raise Exception("events dt must be numpy or list")
            if isinstance(events_dt, np.ndarray):
                events_dt = events_dt.tolist()
            if len(events_dt) != 6:
                raise Exception("events dt must have length 6")
            self._events_dt = events_dt

        self._position_threshold = _position_threshold
        self._cspace_controller = cspace_controller
        self._gripper = gripper  # Deprecated, kept for backward compatibility
        self._start = True
        self.target_position = None
        self._current_gripper_state = GRIPPER_CLOSED  # Start with closed gripper (holding object)

        self._robot = self._resolve_robot(robot=robot, cspace_controller=cspace_controller)
        if self._robot.num_gripper_joints <= 0:
            raise ValueError(
                f"PlaceController requires at least one gripper joint, got {self._robot.num_gripper_joints} "
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
            "PlaceController could not resolve a BaseRobot instance from cspace_controller. "
            "Please pass robot=... explicitly when constructing PlaceController."
        )

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_position: np.ndarray = None,
        pre_place_z: float = 0.2,
        place_offset_z: float = 0.05,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        """Computes the joint positions for the current placing phase.

        Args:
            place_position (np.ndarray): Target position for placing.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            gripper_control: Gripper controller instance.
            end_effector_orientation (np.ndarray, optional): Target orientation for end effector. Defaults to [0, pi, 0] Euler angles.
            gripper_position (np.ndarray): Current position of the gripper.
            pre_place_z (float): Pre-place height offset. Defaults to 0.2.
            place_offset_z (float): Place height offset. Defaults to 0.05.

        Returns:
            Tuple[ArticulationAction, np.ndarray]: Joint positions for the robot to execute and 8-dim record array.
        """
        if self._start:
            action = self._handle_start_state(current_joint_positions)
            record_array = self._build_record_array(action, current_joint_positions, gripper_state=GRIPPER_CLOSED)
            return action, record_array

        if self.is_done():
            action = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            record_array = self._build_record_array(action, current_joint_positions, gripper_state=self._current_gripper_state)
            return action, record_array

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        target_joint_positions = self._execute_phase(
            place_position,
            end_effector_orientation,
            current_joint_positions,
            gripper_control,
            gripper_position,
            pre_place_z,
            place_offset_z,
        )

        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        record_array = self._build_record_array(target_joint_positions, current_joint_positions, gripper_state=self._current_gripper_state)
        return target_joint_positions, record_array

    def _handle_start_state(self, current_joint_positions: np.ndarray) -> ArticulationAction:
        """Handles the initial state.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.

        Returns:
            ArticulationAction: Joint positions for the initial state.
        """
        self._start = False
        return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def _execute_phase(
        self,
        place_position: np.ndarray,
        end_effector_orientation: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        pre_place_z: float,
        place_offset_z: float,
    ) -> ArticulationAction:
        """Executes the current phase of the placing sequence.

        Args:
            place_position (np.ndarray): Target position for placing.
            end_effector_orientation (np.ndarray): Target orientation for end effector.
            current_joint_positions (np.ndarray): Current robot joint positions.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current gripper position.
            pre_place_z (float): Pre-place height offset.
            place_offset_z (float): Place height offset.

        Returns:
            ArticulationAction: Joint position targets for robot control.
        """
        if self._event == 0:
            self.target_position = place_position.copy()
            self.target_position[2] += pre_place_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
                    self._event += 1
                    self._t = 0
            return target_joint_positions

        elif self._event == 1:
            self.target_position = place_position.copy()
            self.target_position[2] += place_offset_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                distance = np.linalg.norm(self.target_position - gripper_position)
                if distance < 0.02:
                    self._event += 1
                    self._t = 0
            return target_joint_positions

        elif self._event == 2:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 3:
            # Open gripper to release
            self._robot.open_gripper()
            self._current_gripper_state = GRIPPER_OPEN
            self.target_position = place_position.copy()
            self.target_position[2] += 0.15 / get_stage_units()
            self.target_position[0] -= 0.15 / get_stage_units()
            gripper_control.release_object()
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if gripper_position is not None:
                xy_distance = np.linalg.norm(self.target_position[:2] - gripper_position[:2])
                if xy_distance < self._position_threshold:
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
        """
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self.target_position = None
        self._current_gripper_state = GRIPPER_CLOSED
        self._reset_record_state()

        if events_dt is not None:
            if not isinstance(events_dt, (np.ndarray, list)):
                raise Exception("events dt must be numpy or list")
            if isinstance(events_dt, np.ndarray):
                events_dt = events_dt.tolist()
            if len(events_dt) != 6:
                raise Exception("events dt must have length 6")
            self._events_dt = events_dt
