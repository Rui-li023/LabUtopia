from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units

from pxr import Gf
import numpy as np
import typing
from .atomic_base_controller import AtomicBaseController

class ShakeController(AtomicBaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        events_dt: typing.List[float] = [0.02, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.015],
        shake_distance: float = 0.1   
    ) -> None:
        super().__init__(name=name)  
        self._forward_start = False  
        self._event = 0  
        self._t = 0
        self._events_dt = events_dt  
        if not isinstance(self._events_dt, (np.ndarray, list)):  
            raise Exception("events_dt is not a list or numpy array")
        elif isinstance(self._events_dt, np.ndarray):  
            self._events_dt = self._events_dt.tolist()  
        if len(self._events_dt) != 10:  
            raise Exception("events_dt length is not 10")
        self._cspace_controller = cspace_controller 
        self._shake_distance = shake_distance / get_stage_units() 
        self._initial_position = np.array([0.25, 0, 1.0])
        self._reset_record_state()
        return

    def forward(
        self,
        current_joint_positions: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        if self._event == 0:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 1:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 2:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 3:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 4:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 5:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 6:
            target_position = self._initial_position + np.array([0, -self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 7:
            target_position = self._initial_position + np.array([0, self._shake_distance, 0])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation
            )

        elif self._event == 8:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._initial_position,
                target_end_effector_orientation=end_effector_orientation
            )
        else:
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions = ArticulationAction(
                joint_positions=target_joint_positions,
                joint_velocities=None,
            )
        
        if self._event < len(self._events_dt):  
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        record_array = self._build_record_array(target_joint_positions, current_joint_positions)
        return target_joint_positions, record_array

    def reset(self, events_dt: typing.Optional[typing.List[float]] = None) -> None:
        super().reset()  
        self._cspace_controller.reset()  
        self._event = 0  
        self._t = 0
        self._reset_record_state()
        if events_dt is not None:  
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be NumPy or list")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 10:  
                raise Exception(" 10")
        return
