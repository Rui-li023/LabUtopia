"""Pick over a wide x/y spawn area, with a grasp frame that follows the object.

Why a separate controller instead of widening `pick`'s config:

`PickTaskController` defaults to a **fixed world-frame** grasp orientation
(`grasp.ee_euler_deg`). That is correct only for objects roughly straight ahead
of the base. `approach_from_base` already swings the *pre-grasp offset* around
to point from the object back to the base, but the *orientation* stays put, so
as the object moves laterally the two disagree: the hand keeps a fixed yaw while
approaching along a rotated radial direction. The wrist ends up folded, IK picks
awkward configurations, and the side grasp slides off the flask.

Measured on the 500-episode sim2real/pick collect (x limited to +-0.18 m):
success by planar radius from the shoulder was 9 % at r ~ 0.42 m rising to 84 %
at r ~ 0.6 m; lateral offset |x| showed no effect *within that narrow band*.
Widening x is exactly where the fixed orientation starts to bite.

This controller is `pick` with `bearing_gain` defaulted to 1.0, i.e. the grasp
frame rotates about world +Z by the object's bearing relative to the base so the
hand meets the object along the same radial line the approach already uses. The
mechanics live in `controllers/grasp_frame.py`, shared with the pour/place/press
controllers. Everything else - phases, success check, data recording - is
inherited unchanged.
"""

from controllers.pick_controller import PickTaskController


class PickWideTaskController(PickTaskController):
    """Pick with a bearing-following grasp frame + per-episode grasp jitter."""

    # The one difference from `pick`. Jitter (`grasp.yaw_jitter_deg` /
    # `tilt_jitter_deg`) is off unless the config asks for it, here as anywhere.
    DEFAULT_BEARING_GAIN = 1.0
