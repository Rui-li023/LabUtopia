from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import BaseRobot, GRIPPER_CLOSED, GRIPPER_OPEN


class PlaceController(AtomicBaseController):
    """State machine for placing objects (6 phases).

    Phase 0: Move above target.  Phase 1: Lower to place height.
    Phase 2: Wait for settle.  Phase 3: Open gripper / release.
    Phase 4: Retreat.  Phase 5: Done.

    Per-episode randomization:
      - pre_place_z noise (±0.03 m)
      - place_offset_z noise (±0.015 m)
      - retreat offset noise (±0.03 m)
      - end-effector orientation perturbation (±10 deg)
    """

    DEFAULT_DT = [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        gripper=None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01,
        robot: typing.Optional[BaseRobot] = None,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=self.DEFAULT_DT,
            robot=robot,
            position_threshold=_position_threshold,
            require_gripper=True,
        )
        self._gripper = gripper  # backward compat
        self._current_gripper_state = GRIPPER_CLOSED
        self.target_position = None

        # Per-episode noise
        self._pre_place_z_noise = 0.0
        self._place_offset_z_noise = 0.0
        self._retreat_x_noise = 0.0
        self._retreat_z_noise = 0.0
        self._orientation_noise_axis = np.array([0, 0, 1.0])
        self._orientation_noise_deg = 0.0

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        # Positive-only z noise: never place below the default target height
        # (a too-low release drops the object and bounces it off the table).
        self._pre_place_z_noise = self._uniform(0.0, 0.03)
        self._place_offset_z_noise = self._uniform(0.0, 0.015)
        self._retreat_x_noise = self._noisy(0.0, 0.03)
        self._retreat_z_noise = self._uniform(0.0, 0.03)
        # Rotate only around the tool Z axis to keep the held object upright at release.
        self._orientation_noise_axis = np.array([0, 0, 1.0])
        self._orientation_noise_deg = self._noisy(0.0, 10.0)

    # ── Forward ──────────────────────────────────────────────────

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
        n = current_joint_positions.shape[0]

        # Guard: bail out with a null action if the task could not resolve
        # the target position (avoids a TypeError deep in the C++ stack).
        if place_position is None:
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        if self._start:
            self._start = False
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=GRIPPER_CLOSED)

        if self.is_done():
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions,
                gripper_state=self._current_gripper_state)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._ensure_randomization()

        # Apply per-episode noise
        pre_place_z = max(0.0, pre_place_z + self._pre_place_z_noise)
        place_offset_z = max(0.0, place_offset_z + self._place_offset_z_noise)
        end_effector_orientation = self._apply_axis_rotation(
            end_effector_orientation, self._orientation_noise_axis,
            self._orientation_noise_deg)

        action = self._execute_phase(
            place_position, end_effector_orientation,
            current_joint_positions, gripper_control, gripper_position,
            pre_place_z, place_offset_z)

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions,
            gripper_state=self._current_gripper_state)

    # ── Phase execution ──────────────────────────────────────────

    def _execute_phase(self, place_pos, orient, jpos, grip_ctrl,
                       grip_pos, pre_z, offset_z):
        n = jpos.shape[0]
        su = get_stage_units()

        if self._event == 0:
            self.target_position = place_pos.copy()
            self.target_position[2] += pre_z / su
            action = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=orient)
            if grip_pos is not None and self._xy_reached(self.target_position, grip_pos):
                self._next_event()
            return action

        elif self._event == 1:
            self.target_position = place_pos.copy()
            self.target_position[2] += offset_z / su
            action = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=orient)
            if grip_pos is not None:
                if float(np.linalg.norm(self.target_position - grip_pos)) < 0.02:
                    self._next_event()
            return action

        elif self._event == 2:
            return self._null_action(n)

        elif self._event == 3:
            self._open_gripper()
            retreat_x = 0.15 + self._retreat_x_noise
            retreat_z = 0.15 + self._retreat_z_noise
            self.target_position = place_pos.copy()
            self.target_position[2] += max(0.05, retreat_z) / su
            self.target_position[0] -= max(0.05, retreat_x) / su
            grip_ctrl.release_object()
            return self._null_action(n)

        elif self._event == 4:
            action = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=orient)
            if grip_pos is not None and self._xy_reached(self.target_position, grip_pos):
                self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, events_dt=None):
        super().reset(events_dt)
        self._current_gripper_state = GRIPPER_CLOSED
        self.target_position = None
        self._pre_place_z_noise = 0.0
        self._place_offset_z_noise = 0.0
        self._retreat_x_noise = 0.0
        self._retreat_z_noise = 0.0
        self._orientation_noise_axis = np.array([0, 0, 1.0])
        self._orientation_noise_deg = 0.0
