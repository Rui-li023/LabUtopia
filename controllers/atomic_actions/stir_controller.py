from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import GRIPPER_CLOSED


class StirController(AtomicBaseController):
    """Position-based controller for stirring (5 phases).

    Phase 0: Lift glass rod.  Phase 1: Move above beaker.
    Phase 2: Lower into beaker.  Phase 3: Circular stirring.
    Phase 4: Lift out.

    Per-episode randomization:
      - stir radius (0.006–0.012 m, default 0.009)
      - stir speed (2.0–4.0, default 3.0)
      - height offsets (±0.02 m per phase)
    """

    DEFAULT_DT = [0.004, 0.004, 0.005, 0.001, 0.004]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.005,
        stir_radius: float = 0.009,
        stir_speed: float = 3.0,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=self.DEFAULT_DT,
            position_threshold=position_threshold,
        )
        self._base_stir_radius = stir_radius
        self._base_stir_speed = stir_speed
        self._stir_radius = stir_radius / get_stage_units()
        self._stir_speed = stir_speed
        self._current_stir_angle = 0.0

        # Per-episode noise
        self._height_noise = np.zeros(4)  # for phases 0-2, 4

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        su = get_stage_units()
        self._stir_radius = self._uniform(0.006, 0.012) / su
        self._stir_speed = self._uniform(2.0, 4.0)
        self._height_noise = np.array([
            self._noisy(0.0, 0.02),  # lift
            self._noisy(0.0, 0.02),  # above beaker
            self._noisy(0.0, 0.02),  # lower into
            self._noisy(0.0, 0.02),  # lift out
        ])

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        center_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        if self._start:
            self._start = False
            self._event = 0
            self._t = 0

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._ensure_randomization()
        n = current_joint_positions.shape[0]

        if self._event >= len(self._events_dt):
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=GRIPPER_CLOSED)

        action = self._execute_phase(
            center_position, gripper_position, end_effector_orientation, n)

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions, gripper_state=GRIPPER_CLOSED)

    # ── Phase execution ──────────────────────────────────────────

    def _execute_phase(self, center, grip_pos, orient, n):
        su = get_stage_units()

        if self._event == 0:
            target = center.copy()
            target[2] += (0.3 + self._height_noise[0]) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if float(np.linalg.norm(grip_pos - target)) < self._position_threshold:
                self._next_event()
            return action

        elif self._event == 1:
            target = center.copy()
            target[2] += (0.3 + self._height_noise[1]) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if self._xy_reached(grip_pos, target):
                self._next_event()
            return action

        elif self._event == 2:
            target = center.copy()
            target[2] += (0.12 + self._height_noise[2]) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if abs(float(grip_pos[2] - target[2])) < self._position_threshold:
                self._next_event()
            return action

        elif self._event == 3:
            self._current_stir_angle += self._stir_speed * 0.01
            target = center.copy()
            target[0] += self._stir_radius * np.cos(self._current_stir_angle)
            target[1] += self._stir_radius * np.sin(self._current_stir_angle)
            target[2] += 0.1 / su
            # Early-exit after ~2 full revolutions so we don't record hundreds
            # of redundant stirring frames (was capped only by _t-driven advance
            # over event_dt[3]=0.001 ≈ 1000 steps; the policy would otherwise
            # learn to keep stirring indefinitely).
            if self._current_stir_angle >= 4 * np.pi:
                self._next_event()
            return self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)

        elif self._event == 4:
            target = center.copy()
            target[2] += (0.2 + self._height_noise[3]) / su
            action = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orient)
            if abs(float(grip_pos[2] - target[2])) < self._position_threshold:
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, events_dt=None):
        super().reset(events_dt)
        su = get_stage_units()
        self._stir_radius = self._base_stir_radius / su
        self._stir_speed = self._base_stir_speed
        self._current_stir_angle = 0.0
        self._height_noise = np.zeros(4)
