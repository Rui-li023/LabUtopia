from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import GRIPPER_CLOSED


class ShakeController(AtomicBaseController):
    """State machine for shaking objects (10 phases).

    Phase 0-1: Move to initial position and hold.
    Phase 2-7: Alternating shake motions.
    Phase 8: Return to center.
    Phase 9: Done.

    Per-episode randomization:
      - shake distance (0.06–0.14 m, default 0.1)
      - initial position XY offset (±0.03 m)
      - shake axis angle (random direction in XY plane)
    """

    DEFAULT_DT = [0.02, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.018, 0.015]

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        events_dt: typing.Optional[typing.List[float]] = None,
        shake_distance: float = 0.1,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            events_dt=events_dt,
            default_events_dt=self.DEFAULT_DT,
        )
        self._base_shake_distance = shake_distance
        self._shake_distance = shake_distance / get_stage_units()
        self._initial_position = np.array([0.25, 0, 1.0])

        # Per-episode noise (populated by _sample_randomization)
        self._shake_axis = np.array([0.0, 1.0])  # unit direction in XY
        self._pos_offset = np.zeros(2)

    # ── Randomization ────────────────────────────────────────────

    def _sample_randomization(self):
        su = get_stage_units()
        self._shake_distance = self._uniform(0.06, 0.14) / su
        self._pos_offset = np.array([self._noisy(0.0, 0.03),
                                     self._noisy(0.0, 0.03)])
        # Random shake axis in XY plane
        theta = self._uniform(0, 2 * np.pi)
        self._shake_axis = np.array([np.cos(theta), np.sin(theta)])

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        current_joint_positions: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._ensure_randomization()
        n = current_joint_positions.shape[0]

        center = self._initial_position.copy()
        center[0] += self._pos_offset[0]
        center[1] += self._pos_offset[1]

        # Determine target position for this phase
        if self._event in (0, 1, 8):
            target = center
        elif self._event in (2, 4, 6):
            target = center.copy()
            target[0] -= self._shake_axis[0] * self._shake_distance
            target[1] -= self._shake_axis[1] * self._shake_distance
        elif self._event in (3, 5, 7):
            target = center.copy()
            target[0] += self._shake_axis[0] * self._shake_distance
            target[1] += self._shake_axis[1] * self._shake_distance
        else:
            action = self._null_action(n)
            self._advance_state()
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=GRIPPER_CLOSED)

        action = self._cspace_controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=end_effector_orientation)

        self._advance_state()
        return action, self._build_record_array(
            action, current_joint_positions, gripper_state=GRIPPER_CLOSED)

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, events_dt=None):
        super().reset(events_dt)
        su = get_stage_units()
        self._shake_distance = self._base_shake_distance / su
        self._shake_axis = np.array([0.0, 1.0])
        self._pos_offset = np.zeros(2)
