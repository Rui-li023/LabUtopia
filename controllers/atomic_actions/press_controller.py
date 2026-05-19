from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import BaseRobot, GRIPPER_OPEN, GRIPPER_CLOSED


class PressController(AtomicBaseController):
    """State machine for pressing buttons (3 phases).

    Phase 0: Move in front of target.  Phase 1: Close gripper.
    Phase 2: Press forward.

    Per-episode randomization:
      - initial offset noise (±0.03 m)
      - press distance noise (±0.01 m)
    """

    DEFAULT_DT = [0.005, 0.1, 0.01, 0.005]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        gripper=None,
        end_effector_initial_height: typing.Optional[float] = None,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=self.DEFAULT_DT,
            robot=robot,
        )
        self._initial_offset = (initial_offset if initial_offset is not None
                                else 0.2 / get_stage_units())

        # Per-episode noise
        self._offset_noise = 0.0
        self._press_noise = 0.0

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        self._offset_noise = self._noisy(0.0, 0.03)
        self._press_noise = self._noisy(0.0, 0.003)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        press_distance: float = 0.04,
        gripper_position: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        n = current_joint_positions.shape[0]

        if self._start:
            self._start = False
            self._open_gripper()
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=GRIPPER_OPEN)

        if self.is_done():
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._ensure_randomization()
        su = get_stage_units()

        if self._event == 0:
            offset = self._initial_offset + self._offset_noise / su
            target_position[0] -= offset
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation)
            if gripper_position is not None and self._xyz_reached(
                gripper_position, target_position, threshold=0.03
            ):
                self._next_event()
                return action, self._build_record_array(
                    action, current_joint_positions,
                    gripper_state=self._current_gripper_state)

        elif self._event == 1:
            self._close_gripper()
            action = self._null_action(n)

        elif self._event == 2:
            dist = press_distance + self._press_noise
            target_position[0] += dist / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation)
            if gripper_position is not None and self._xyz_reached(
                gripper_position, target_position, threshold=0.02
            ):
                self._next_event()
                return action, self._build_record_array(
                    action, current_joint_positions,
                    gripper_state=self._current_gripper_state)

        elif self._event == 3:
            # Open the gripper first so we release the button before pulling
            # away — otherwise the closed fingers drag the button back with
            # the arm and it never sits at its pressed-in position.
            self._open_gripper()
            # Retract noticeably further than the approach offset so the EE
            # ends up well clear of the button (the success criterion needs
            # wrist→button ≥ 10 cm).
            offset = 0.15 + self._offset_noise / su
            target_position[0] -= offset
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation)
            if gripper_position is not None and self._xyz_reached(
                gripper_position, target_position, threshold=0.03
            ):
                self._next_event()
                return action, self._build_record_array(
                    action, current_joint_positions,
                    gripper_state=self._current_gripper_state)

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions,
            gripper_state=self._current_gripper_state)

    def get_current_event(self) -> int:
        return self._event

    def force_done(self) -> None:
        """Fast-forward to the terminal phase. Used when the high-level
        controller detects success (button pressed enough) and wants to stop
        driving the EE further."""
        self._event = len(self._events_dt)

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, initial_offset=None, events_dt=None):
        super().reset(events_dt)
        if initial_offset is not None:
            self._initial_offset = initial_offset
        self._offset_noise = 0.0
        self._press_noise = 0.0
