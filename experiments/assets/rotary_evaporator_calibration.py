"""Fixed calibration of the engraved 0..120 rpm dial, clockwise from its top index.

Operating limits in interaction_config.json restrict requested setpoints; they
must not stretch this scale or change the meaning of the engraved marks.
"""

import math

DIAL_MAX_RPM = 120.0
DIAL_SWEEP_DEG = 300.0


def dial_angle_to_rpm(angle: float) -> float:
    """Convert a measured +Y joint angle in radians to the engraved setting."""
    return min(DIAL_MAX_RPM, max(0.0, math.degrees(angle) * DIAL_MAX_RPM / DIAL_SWEEP_DEG))


def rpm_to_dial_angle(rpm: float) -> float:
    """Convert a setting to radians without remapping the printed scale."""
    return math.radians(min(DIAL_MAX_RPM, max(0.0, rpm)) * DIAL_SWEEP_DEG / DIAL_MAX_RPM)
