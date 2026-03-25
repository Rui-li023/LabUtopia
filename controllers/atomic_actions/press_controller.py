from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from .atomic_base_controller import AtomicBaseController
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from robots.base_robot import BaseRobot, GRIPPER_OPEN, GRIPPER_CLOSED

class PressController(AtomicBaseController):
    """
    A pressing state machine controller.
    
    This controller handles the process of pressing a button, including the following phases:
    - Phase 0: Move the end effector in front of the target object (along X-axis).
    - Phase 1: Close the gripper.
    - Phase 2: Press forward to the target position.
    
    Args:
        name (str): Identifier for the controller.
        cspace_controller (typing.Any): Cartesian space controller that returns ArticulationAction.
        gripper (Gripper): Controller for opening/closing the gripper.
        initial_offset (float, optional): Initial offset distance (along X-axis), defaults to 0.1 meters.
        events_dt (list of float, optional): Duration for each phase, defaults to [0.01, 0.01, 0.01].
    """
    
    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        gripper: Gripper = None,
        end_effector_initial_height: typing.Optional[float] = None,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        # Initialize parent controller
        super().__init__(name=name)
        self._current_gripper_state = GRIPPER_OPEN

        # Resolve robot for gripper control
        self._robot = None
        if robot is not None:
            self._robot = robot
        else:
            for attr in ("robot", "robot_articulation", "_robot", "_robot_articulation"):
                candidate = getattr(cspace_controller, attr, None)
                if candidate is not None and isinstance(candidate, BaseRobot):
                    self._robot = candidate
                    break
            if self._robot is None:
                amp = getattr(cspace_controller, "_articulation_motion_policy", None)
                if amp is not None:
                    for attr in ("_robot_articulation", "robot_articulation"):
                        candidate = getattr(amp, attr, None)
                        if candidate is not None and isinstance(candidate, BaseRobot):
                            self._robot = candidate
                            break
        
        self._event = 0  # Current phase number
        self._t = 0  # Current phase time counter
        self._initial_offset = initial_offset if initial_offset is not None else 0.2 / get_stage_units()
        # Initial offset distance, default 0.1 meters (adjusted by stage units)
        
        if events_dt is None:
            self._events_dt = [0.005, 0.1, 0.01]  # Default phase durations
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events_dt length must be exactly 3")
        
        self._cspace_controller = cspace_controller
        self._start = True
        self._current_gripper_state = GRIPPER_OPEN
        self._reset_record_state()

    def get_current_event(self) -> int:
        """
        Get the current phase/event of the state machine.

        Returns:
            int: Current phase/event number.
        """
        return self._event
    
    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        press_distance: float = 0.04
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        """
        Execute one step of the pressing action.
        
        Args:
            target_position (np.ndarray): Target pressing position.
            current_joint_positions (np.ndarray): Current robot joint positions.
            gripper_control: Gripper controller.
            end_effector_orientation (np.ndarray, optional): End effector orientation.
        
        Returns:
            ArticulationAction: Robot control action.
        """
        
        if self._start:
            self._start = False
            self._current_gripper_state = GRIPPER_OPEN
            if self._robot is not None:
                self._robot.open_gripper()
            action = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            return action, self._build_record_array(action, current_joint_positions, gripper_state=GRIPPER_OPEN)

        if self.is_done():
            action = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            return action, self._build_record_array(action, current_joint_positions, gripper_state=self._current_gripper_state)
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        # Execute the current phase action
        if self._event == 0:
            # Phase 0: Move in front of the target object
            target_position[0] -= self._initial_offset  # Offset forward along X-axis
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        elif self._event == 1:
            # Phase 1: Close the gripper
            self._current_gripper_state = GRIPPER_CLOSED
            if self._robot is not None:
                self._robot.close_gripper()
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 2:
            # Phase 2: Press forward to the target position
            target_position[0]+= press_distance/ get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        record_array = self._build_record_array(target_joint_positions, current_joint_positions, gripper_state=self._current_gripper_state)
        return target_joint_positions, record_array

    
    def reset(
        self,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None
    ) -> None:
        """
        Reset the state machine to initial state.
        
        Args:
            initial_offset (float, optional): New initial offset distance.
            events_dt (list of float, optional): New list of phase durations.
        """
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if initial_offset is not None:
            self._initial_offset = initial_offset
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or NumPy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 3:
                raise Exception("events_dt length must be exactly 3")
        self._start = True
        self._current_gripper_state = GRIPPER_OPEN
        self._reset_record_state()
    
