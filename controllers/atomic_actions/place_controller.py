import typing

import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction

from robots.base_robot import GRIPPER_CLOSED, BaseRobot

from .atomic_base_controller import AtomicBaseController


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
        events_dt: list[float] | None = None,
        _position_threshold: float = 0.01,
        robot: BaseRobot | None = None,
        precision: bool = False,
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
        self.precision = precision
        self.precision_position_threshold = 0.005
        self.precision_height_threshold: float | None = None
        self.retreat_position_threshold = 0.010
        self.retreat_offset: np.ndarray | None = None
        self.post_retreat_position: np.ndarray | None = None
        self._post_retreat_started = False
        self.approach_offset: np.ndarray | None = None
        self._approach_complete = False
        self._phase_failure_reason = ""
        self._last_gripper_position = None
        self._transitioned_this_step = False
        self.placement_feature_position: np.ndarray | None = None
        self.placement_feature_target: np.ndarray | None = None
        self.placement_feature_aligned = True
        # A task may stream intermediate transport waypoints. Reaching one is
        # not permission to descend until the final hover waypoint is reached.
        self.approach_arrival_allowed = True
        if precision and events_dt is None:
            self._events_dt = [0.0025, 0.0025, 0.025, 0.025, 0.005, 0.05]

        # Per-episode noise
        self._pre_place_z_noise = 0.0
        self._place_offset_z_noise = 0.0
        self._retreat_x_noise = 0.0
        self._retreat_z_noise = 0.0
        self._orientation_noise_deg = 0.0

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        if self.precision:
            return
        # Positive-only z noise: never place below the default target height
        # (a too-low release drops the object and bounces it off the table).
        self._pre_place_z_noise = self._uniform(0.0, 0.03)
        self._place_offset_z_noise = self._uniform(0.0, 0.015)
        self._retreat_x_noise = self._noisy(0.0, 0.03)
        self._retreat_z_noise = self._uniform(0.0, 0.03)
        # World-vertical yaw noise: keeps the held object level for side grasps
        # (a tool-Z rotation only stays upright for top-down grasps).
        self._orientation_noise_deg = self._noisy(0.0, 10.0)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: np.ndarray | None = None,
        gripper_position: np.ndarray = None,
        pre_place_z: float = 0.2,
        place_offset_z: float = 0.05,
    ) -> tuple[ArticulationAction, np.ndarray]:
        n = current_joint_positions.shape[0]

        # Guard: bail out with a null action if the task could not resolve
        # the target position (avoids a TypeError deep in the C++ stack).
        if place_position is None:
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=self._current_gripper_state
            )

        if self._start:
            self._start = False
            action = self._null_action(n)
            return action, self._build_record_array(action, current_joint_positions, gripper_state=GRIPPER_CLOSED)

        if self.is_done():
            action = self._null_action(n)
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=self._current_gripper_state
            )

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._ensure_randomization()

        # Apply per-episode noise
        pre_place_z = max(0.0, pre_place_z + self._pre_place_z_noise)
        place_offset_z = max(0.0, place_offset_z + self._place_offset_z_noise)
        end_effector_orientation = self._apply_world_yaw(end_effector_orientation, self._orientation_noise_deg)

        self._transitioned_this_step = False
        self._last_gripper_position = gripper_position
        action = self._execute_phase(
            place_position,
            end_effector_orientation,
            current_joint_positions,
            gripper_control,
            gripper_position,
            pre_place_z,
            place_offset_z,
        )

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions, gripper_state=self._current_gripper_state
        )

    # ── Phase execution ──────────────────────────────────────────

    def _execute_phase(self, place_pos, orient, jpos, grip_ctrl, grip_pos, pre_z, offset_z):
        n = jpos.shape[0]
        su = get_stage_units()

        if self._event == 0:
            self.target_position = place_pos.copy()
            self.target_position[2] += pre_z / su
            approaching = self.precision and self.approach_offset is not None and not self._approach_complete
            if approaching:
                self.target_position += self.approach_offset / su
            action = self._cspace_controller.forward(
                target_end_effector_position=self.target_position, target_end_effector_orientation=orient
            )
            reached = self._xyz_reached if self.precision else self._xy_reached
            if grip_pos is not None and self.approach_arrival_allowed and reached(self.target_position, grip_pos):
                if approaching:
                    self._approach_complete = True
                    self._t = 0.0
                    self._transitioned_this_step = True
                else:
                    self._next_event()
            return action

        elif self._event == 1:
            self.target_position = place_pos.copy()
            self.target_position[2] += offset_z / su
            action = self._cspace_controller.forward(
                target_end_effector_position=self.target_position, target_end_effector_orientation=orient
            )
            if grip_pos is not None:
                threshold = self.precision_position_threshold if self.precision else 0.02
                delta = self.target_position - grip_pos
                if (
                    self.precision
                    and self.placement_feature_position is not None
                    and self.placement_feature_target is not None
                ):
                    delta = self.placement_feature_target - self.placement_feature_position
                reached = float(np.linalg.norm(delta)) < threshold
                if self.precision and self.precision_height_threshold is not None:
                    reached = (
                        float(np.linalg.norm(delta[:2])) < threshold
                        and abs(float(delta[2])) < self.precision_height_threshold
                    )
                if reached and self.placement_feature_aligned:
                    self._next_event()
            return action

        elif self._event == 2:
            return self._null_action(n)

        elif self._event == 3:
            self._open_gripper()
            if self.precision:
                self.target_position = place_pos.copy()
                self.target_position[2] += offset_z / su
                self.target_position += (
                    np.array([0.0, 0.0, 0.12]) if self.retreat_offset is None else self.retreat_offset
                ) / su
                grip_ctrl.release_object()
                return self._null_action(n)
            retreat_x = 0.15 + self._retreat_x_noise
            retreat_z = 0.15 + self._retreat_z_noise
            self.target_position = place_pos.copy()
            self.target_position[2] += max(0.05, retreat_z) / su
            self.target_position[0] -= max(0.05, retreat_x) / su
            grip_ctrl.release_object()
            return self._null_action(n)

        elif self._event == 4:
            action = self._cspace_controller.forward(
                target_end_effector_position=self.target_position, target_end_effector_orientation=orient
            )
            reached = (
                self._xyz_reached(self.target_position, grip_pos, threshold=self.retreat_position_threshold)
                if self.precision and grip_pos is not None
                else grip_pos is not None and self._xy_reached(self.target_position, grip_pos)
            )
            if reached:
                if self.post_retreat_position is not None and not self._post_retreat_started:
                    self.target_position = self.post_retreat_position.copy()
                    self._post_retreat_started = True
                    self._t = 0.0
                    self._transitioned_this_step = True
                else:
                    self._next_event()
            return action

        else:
            return self._null_action(n)

    # ── Reset ────────────────────────────────────────────────────

    def _next_event(self):
        super()._next_event()
        self._transitioned_this_step = True

    def _advance_state(self):
        if not self.precision:
            return super()._advance_state()
        if self._transitioned_this_step:
            return
        if self._event in (0, 1, 4):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._phase_failure_reason = (
                    f"Place phase {self._event} timed out before TCP arrival: "
                    f"target={self.target_position}, actual={self._last_gripper_position}"
                )
                if self._event == 1 and self.placement_feature_position is not None:
                    self._phase_failure_reason += (
                        f"; object_feature_target={self.placement_feature_target}, "
                        f"actual={self.placement_feature_position}, aligned={self.placement_feature_aligned}"
                    )
                self._event = len(self._events_dt)
            return
        super()._advance_state()

    def get_failure_reason(self) -> str:
        return self._phase_failure_reason

    def reset(self, events_dt=None):
        super().reset(events_dt)
        self._post_retreat_started = False
        self._current_gripper_state = GRIPPER_CLOSED
        self.target_position = None
        self._pre_place_z_noise = 0.0
        self._place_offset_z_noise = 0.0
        self._retreat_x_noise = 0.0
        self._retreat_z_noise = 0.0
        self._orientation_noise_deg = 0.0
        self._phase_failure_reason = ""
        self._last_gripper_position = None
        self._transitioned_this_step = False
        self._approach_complete = False
        self.placement_feature_position = None
        self.placement_feature_target = None
        self.placement_feature_aligned = True
        self.approach_arrival_allowed = True
