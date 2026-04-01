from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

from .atomic_base_controller import AtomicBaseController
from robots.base_robot import GRIPPER_OPEN


class MoveController(AtomicBaseController):
    """Simple controller for moving end effector to a target pose.

    Supports single-target, multi-segment, and two-point movement.
    No state-machine phases — done when position threshold is reached.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any,
        position_threshold: float = 0.02,
        orientation_threshold: float = 0.1,
    ) -> None:
        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            position_threshold=position_threshold,
        )
        self._orientation_threshold = orientation_threshold
        self._target_position = None
        self._target_orientation = None
        self._is_done = False
        self._waypoints = []
        self._current_waypoint_index = 0
        self._multi_segment_mode = False

    # ── Single target ────────────────────────────────────────────

    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        target_orientation: typing.Optional[np.ndarray] = None,
        gripper_state: typing.Optional[int] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        if target_orientation is None:
            target_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self._target_position = target_position
        self._target_orientation = target_orientation

        action = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation)

        dist = float(np.linalg.norm(gripper_position - target_position))
        self._is_done = dist < self._position_threshold

        return action, self._build_record_array(
            action, current_joint_positions, gripper_state=gripper_state)

    # ── Multi-segment ────────────────────────────────────────────

    def forward_multi_segment(
        self,
        waypoints: typing.List[np.ndarray],
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        target_orientation: typing.Optional[np.ndarray] = None,
        gripper_state: typing.Optional[int] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        if target_orientation is None:
            target_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        if not self._multi_segment_mode:
            self._waypoints = waypoints.copy()
            self._current_waypoint_index = 0
            self._multi_segment_mode = True
            self._is_done = False

        if self._current_waypoint_index >= len(self._waypoints):
            self._is_done = True
            action = self._cspace_controller.forward(
                target_end_effector_position=self._waypoints[-1],
                target_end_effector_orientation=target_orientation)
            return action, self._build_record_array(
                action, current_joint_positions, gripper_state=gripper_state)

        current_target = self._waypoints[self._current_waypoint_index]
        self._target_position = current_target
        self._target_orientation = target_orientation

        if float(np.linalg.norm(gripper_position - current_target)) < 0.08:
            self._current_waypoint_index += 1
            if self._current_waypoint_index < len(self._waypoints):
                current_target = self._waypoints[self._current_waypoint_index]
                self._target_position = current_target

        action = self._cspace_controller.forward(
            target_end_effector_position=current_target,
            target_end_effector_orientation=target_orientation)

        if self._current_waypoint_index >= len(self._waypoints):
            self._is_done = True

        return action, self._build_record_array(
            action, current_joint_positions, gripper_state=gripper_state)

    # ── Two-point ────────────────────────────────────────────────

    def forward_two_points(
        self,
        first_position: np.ndarray,
        final_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        target_orientation: typing.Optional[np.ndarray] = None,
    ) -> typing.Tuple[ArticulationAction, np.ndarray]:
        return self.forward_multi_segment(
            waypoints=[first_position.copy(), final_position.copy()],
            current_joint_positions=current_joint_positions,
            gripper_position=gripper_position,
            target_orientation=target_orientation)

    # ── Accessors ────────────────────────────────────────────────

    def is_done(self) -> bool:
        return self._is_done

    def get_target_position(self):
        return self._target_position

    def get_target_orientation(self):
        return self._target_orientation

    def set_position_threshold(self, threshold: float):
        self._position_threshold = threshold

    def set_orientation_threshold(self, threshold: float):
        self._orientation_threshold = threshold

    # ── Reset ────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self._target_position = None
        self._target_orientation = None
        self._is_done = False
        self._waypoints = []
        self._current_waypoint_index = 0
        self._multi_segment_mode = False
