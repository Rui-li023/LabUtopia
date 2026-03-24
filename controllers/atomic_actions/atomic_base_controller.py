from isaacsim.core.utils.types import ArticulationAction

import numpy as np


class AtomicBaseController:
    """Shared base for atomic action controllers.

    Provides a unified way to build dense joint-position records from sparse
    ArticulationAction outputs and to manage record cache lifecycle.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._last_record_positions = None

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

    def _build_record_array(self, action: ArticulationAction, current_joint_positions: np.ndarray) -> np.ndarray:
        n = len(current_joint_positions)
        jp = action.joint_positions
        if jp is None:
            if self._last_record_positions is not None:
                return self._last_record_positions.copy()
            return current_joint_positions.copy()

        fallback = self._last_record_positions if self._last_record_positions is not None else current_joint_positions
        positions = fallback.copy().astype(np.float64)
        for i in range(min(len(jp), n)):
            if jp[i] is not None:
                positions[i] = float(jp[i])

        self._last_record_positions = positions
        return positions
