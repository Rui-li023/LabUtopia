from isaacsim.core.utils.types import ArticulationAction

import numpy as np

from robots.base_robot import GRIPPER_CLOSED, GRIPPER_OPEN


class AtomicBaseController:
    """Shared base for atomic action controllers.

    Provides a unified way to build dense joint-position records from sparse
    ArticulationAction outputs and to manage record cache lifecycle.

    The record array output is 8 dimensions:
        - indices 0-6: arm joint positions
        - index 7: gripper state (0 = open, 1 = closed)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._last_record_positions = None
        self._last_gripper_state = GRIPPER_OPEN

    def reset(self) -> None:
        self._reset_record_state()

    def is_done(self) -> bool:
        if hasattr(self, "_is_done"):
            return bool(self._is_done)
        if hasattr(self, "_event") and hasattr(self, "_events_dt"):
            return self._event >= len(self._events_dt)
        return False

    def _reset_record_state(self) -> None:
        self._last_record_positions = None
        self._last_gripper_state = GRIPPER_OPEN

    def _build_record_array(
        self,
        action: ArticulationAction,
        current_joint_positions: np.ndarray,
        gripper_state: int = None,
    ) -> np.ndarray:
        """Build an 8-dimensional record array.

        Args:
            action: ArticulationAction from the controller
            current_joint_positions: Current joint positions (9 dims for Franka)
            gripper_state: Optional gripper state (0=open, 1=closed).
                          If None, uses the last recorded state.

        Returns:
            np.ndarray: 8-dimensional array (7 arm joints + 1 gripper state)
        """
        n = len(current_joint_positions)
        jp = action.joint_positions

        # Get arm joint positions (first 7 joints)
        if jp is None:
            if self._last_record_positions is not None:
                arm_positions = self._last_record_positions[:7].copy()
            else:
                arm_positions = current_joint_positions[:7].copy()
        else:
            fallback = (
                self._last_record_positions[:7]
                if self._last_record_positions is not None
                else current_joint_positions[:7]
            )
            arm_positions = fallback.copy().astype(np.float64)
            for i in range(min(len(jp), 7)):
                if jp[i] is not None:
                    arm_positions[i] = float(jp[i])

        # Determine gripper state
        if gripper_state is not None:
            self._last_gripper_state = gripper_state

        # Build 8-dimensional output: 7 arm joints + 1 gripper state
        record = np.zeros(8, dtype=np.float64)
        record[:7] = arm_positions
        record[7] = float(self._last_gripper_state)

        self._last_record_positions = record.copy()
        return record
