from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction

import numpy as np
import typing
from scipy.spatial.transform import Rotation as R

# Control frequency 60 Hz: pour action uses velocity control, record positions are integrated with dt = 1/60
CONTROL_FREQUENCY = 60
PHYSICS_DT = 1.0 / CONTROL_FREQUENCY


class PourController(BaseController):
    """
    PourController implements a state machine for pouring liquid. The state transitions are as follows:

    State 0: Move above the target position (with random height offset). When XY distance is close, proceed to next state.
    State 1: Further adjust height and position (considering object size and offset). When XY distance is close, proceed to next state.
    State 2: Switch joint 7 to velocity mode, start pouring (positive velocity).
    State 3: Hold joint 7 velocity at 0, pause pouring.
    State 4: Switch joint 7 to velocity mode, pour in reverse (negative velocity).
    State 5: Hold joint 7 velocity at 0, finish pouring.

    Pour phase (states 2-5) uses velocity control at 60 Hz; record_array is computed as position + velocity * dt.
    Each state's duration is controlled by self._events_dt. State transitions are managed by self._event and self._t.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        speed: float = 1,
        position_threshold: float = 0.006,
        control_frequency: float = CONTROL_FREQUENCY,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._events_dt = events_dt
        self._physics_dt = 1.0 / control_frequency
        if self._events_dt is None:
            self._events_dt = [dt / speed for dt in [0.002, 0.01, 0.009, 0.005, 0.009, 0.5]]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            assert len(self._events_dt) == 6, "events dt need have length of 6 or less"
        self._cspace_controller = cspace_controller

        self._pour_default_speed = - 120.0 / 180.0 * np.pi
        self._position_threshold = position_threshold

        self._height_range_1 = (0.3, 0.4)
        self._height_range_2 = (0.1, 0.2)
        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        self._last_record_positions = None
        return

    def _build_record_array(
        self,
        action: ArticulationAction,
        current_joint_positions: typing.Optional[np.ndarray] = None,
    ) -> typing.Optional[np.ndarray]:
        """Build 9-dim record array. For position actions use/copy positions; for velocity-only actions integrate at 60 Hz."""
        jp = action.joint_positions
        jv = action.joint_velocities

        if jp is not None and current_joint_positions is not None:
            n = len(current_joint_positions)
            positions = current_joint_positions.copy().astype(np.float64)
            for i in range(min(len(jp), n)):
                if jp[i] is not None:
                    positions[i] = float(jp[i])
                else:
                    positions[i] = current_joint_positions[i]
            if len(jp) < n:
                positions[len(jp):] = current_joint_positions[len(jp):]
            self._last_record_positions = positions
            return positions
        if jp is not None:
            positions = np.array([float(p) for p in jp])
            if current_joint_positions is not None and len(positions) < len(current_joint_positions):
                full = current_joint_positions.copy().astype(np.float64)
                full[:len(positions)] = positions
                positions = full
            self._last_record_positions = positions
            return positions
        # Velocity-only action (pour phase): integrate position = base + velocity * dt at 60 Hz
        if jv is not None:
            base = self._last_record_positions if self._last_record_positions is not None else current_joint_positions
            if base is not None:
                base = np.asarray(base, dtype=np.float64)
                n_base = len(base)
                positions = base.copy()
                for i in range(min(len(jv), n_base)):
                    if jv[i] is not None:
                        positions[i] = positions[i] + float(jv[i]) * self._physics_dt
                self._last_record_positions = positions
                return positions
        if self._last_record_positions is not None:
            return self._last_record_positions.copy()
        if current_joint_positions is not None:
            return current_joint_positions.copy()
        return None

    def forward(
        self,
        articulation_controller: ArticulationController,
        source_size: np.ndarray,
        target_position: np.ndarray,
        current_joint_velocities: np.ndarray,
        gripper_position: np.ndarray,
        source_name: str = None,
        pour_speed: float = None,
        current_joint_positions: typing.Optional[np.ndarray] = None,
        target_end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat()
    ) -> typing.Tuple[ArticulationAction, typing.Optional[np.ndarray]]:
        """
        Execute one step of the controller. Control rate is 60 Hz; pour phase uses velocity, record_array uses position + velocity*dt.

        Args:
            articulation_controller: The articulation controller for the robot.
            source_size: Size of the source object being poured.
            current_joint_velocities: Current joint velocities of the robot.
            current_joint_positions: Optional current joint positions (9-dim); used to build 9-dim record_array and for velocity integration.
            pour_speed: Speed for the pouring action. Defaults to None.

        Returns:
            (ArticulationAction, record_array): Action to execute and 9-dim position array for recording.
        """
        self.object_size = source_size
        
        if pour_speed is None:
            self._pour_speed = self._pour_default_speed
        else:
            self._pour_speed = pour_speed
            
        if  self._event >= len(self._events_dt):
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            action = ArticulationAction(joint_velocities=target_joint_velocities)
            return action, self._build_record_array(action, current_joint_positions)
        
        if self._event == 0:
            target_position[2] += self._random_height_1
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            self._random_height_1 = np.random.uniform(*self._height_range_1)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < 0.08:
                self._event += 1
                self._t = 0
                return target_joints, self._build_record_array(target_joints, current_joint_positions)

        elif self._event == 1:
            target_position[2] += self._random_height_2 + self.object_size[2] / 2 + self.get_pickz_offset(source_name)
            target_position[1] -= self.object_size[2] / 2 - self.get_pickz_offset(source_name)
            target_joints = self._cspace_controller.forward(
                target_end_effector_position=target_position, 
                target_end_effector_orientation=target_end_effector_orientation
            )
            self._random_height_2 = np.random.uniform(*self._height_range_2)
            xy_distance = np.linalg.norm(gripper_position[:2] - target_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
                return target_joints, self._build_record_array(target_joints, current_joint_positions)
        elif self._event == 2:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 3:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 4:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = -self._pour_speed
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)
        elif self._event == 5:
            articulation_controller.switch_dof_control_mode(dof_index=6, mode="velocity")
            target_joint_velocities = [None] * current_joint_velocities.shape[0]
            target_joint_velocities[6] = 0
            target_joints = ArticulationAction(joint_velocities=target_joint_velocities)

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0

        record_array = self._build_record_array(target_joints, current_joint_positions)
        return target_joints, record_array

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.

        Raises:
            Exception: If 'events_dt' is not a list or numpy array.
            Exception: If 'events_dt' length is greater than 3.
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self.object_size = None
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 3:
                raise Exception("events dt need have length of 3 or less")

        self._random_height_1 = np.random.uniform(*self._height_range_1)
        self._random_height_2 = np.random.uniform(*self._height_range_2)
        self._last_record_positions = None
        return

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)
    
    def get_pickz_offset(self, item_name):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.03,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker2": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "beaker": 0.02,
            "beaker_l": 0.02,
            
        }

        for key in offsets:
            if key in item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5