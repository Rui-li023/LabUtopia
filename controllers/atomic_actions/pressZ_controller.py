from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import BaseRobot, GRIPPER_OPEN, GRIPPER_CLOSED


class PressZController(AtomicBaseController):
    """State machine for vertical pressing (3 phases).

    Phase 0: Move above target.  Phase 1: Close gripper.
    Phase 2: Press down.

    Per-episode randomization:
      - initial offset noise (±0.03 m)
      - press depth noise (±0.008 m)
    """

    DEFAULT_DT = [0.005, 0.01, 0.01]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
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
        self._position_threshold = 0.01 / get_stage_units()

        # Press-target height above the button (metres, +z = up). Lower = deeper
        # press. 0.005 (vs the old 0.025) sinks the position-only steady-state
        # below the 0.761 success threshold so the press completes promptly.
        self._press_depth = 0.005

        # Per-episode noise
        self._offset_noise = 0.0
        self._press_depth_noise = 0.0

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        self._offset_noise = self._noisy(0.0, 0.03)
        self._press_depth_noise = self._noisy(0.0, 0.008)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_position: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        n = current_joint_positions.shape[0]

        if self._start:
            self._start = False
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

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
            target_position[2] += offset
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation)
            if gripper_position is not None and self._xy_reached(gripper_position, target_position):
                self._next_event()

        elif self._event == 1:
            self._close_gripper()
            action = self._null_action(n)

        elif self._event == 2:
            # Press target height above the button (+z = up). A pure-position PD
            # (position-only collection) leaves a steady-state error against the
            # spring-loaded button: with the old +0.025 the button settled at
            # ~0.769, ~8 mm short of the 0.761 success threshold, so the arm
            # idled for thousands of frames while the button crept across.
            # Commanding a deeper press (lower EE target) moves the steady-state
            # below the threshold so success fires promptly — no long idle tail.
            depth = self._press_depth + self._press_depth_noise
            target_position[2] += depth / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=end_effector_orientation)

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions,
            gripper_state=self._current_gripper_state)

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, initial_offset=None, events_dt=None):
        super().reset(events_dt)
        if initial_offset is not None:
            self._initial_offset = initial_offset
        self._offset_noise = 0.0
        self._press_depth_noise = 0.0
