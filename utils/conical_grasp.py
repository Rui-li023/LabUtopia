"""Measured straight-neck grasp for lab_001's original conical_bottle02."""

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class ConicalNeckGrasp:
    # Original mouth: 164.524 mm above base. Straight outer neck: roughly
    # 129.7-155.6 mm; the finger face spans about 21 mm along the bottle axis.
    # A 24 mm depth centers that face at 140.5 mm, clear of shoulder/rolled rim.
    depth: float = 0.024
    # This is a blocked, force-limited jaw target, NOT an achieved jaw width.
    # Leave enough position error to maintain the configured actuator force
    # even at the narrowest section of the original neck.
    closed_finger_position: float = 0.010

    def __post_init__(self) -> None:
        if not isfinite(self.depth) or not 0.020 <= self.depth <= 0.027:
            raise ValueError("Conical grasp depth must stay within the measured straight-neck band (20-27 mm)")
        if not isfinite(self.closed_finger_position) or not 0.005 <= self.closed_finger_position <= 0.016:
            raise ValueError("Conical jaw target must maintain force-limited closure (5-16 mm per finger)")

    @classmethod
    def from_config(cls, cfg: Any) -> "ConicalNeckGrasp":
        options = getattr(cfg.task, "conical_grasp", {})
        return cls(**dict(options))
