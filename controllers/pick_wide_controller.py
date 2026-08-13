"""Pick over a wide x/y spawn area, with a grasp frame that follows the object.

Why a separate controller instead of widening `pick`'s config:

`PickTaskController` commands a **fixed world-frame** grasp orientation
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

This controller rotates the whole grasp frame about world +Z by the object's
bearing relative to the base, so the hand meets the object along the same
radial line the approach already uses. Everything else - phases, success check,
data recording - is inherited unchanged.
"""

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from controllers.pick_controller import PickTaskController


class PickWideTaskController(PickTaskController):
    """Pick with a bearing-following grasp frame + per-episode grasp jitter."""

    def __init__(self, cfg, robot):
        grasp_cfg = getattr(cfg, "grasp", None)

        # Bearing the fixed `ee_euler_deg` was tuned for, measured in the world
        # XY plane. None => take it from the base yaw, i.e. "straight ahead of
        # the base", which is how every existing value was tuned.
        raw = getattr(grasp_cfg, "nominal_bearing_deg", None) if grasp_cfg else None
        self._nominal_bearing_deg: float | None = float(raw) if raw is not None else None

        # Per-episode jitter on the grasp frame, degrees. Yaw jitter rotates the
        # hand about world +Z (how "square" it meets the object); tilt jitter
        # perturbs the other two axes. Collected data with a single exact grasp
        # pose teaches the policy that one pose is the only correct one, which
        # transfers badly to a real rig whose object pose is never exact.
        self._yaw_jitter_deg = float(getattr(grasp_cfg, "yaw_jitter_deg", 0.0)) if grasp_cfg else 0.0
        self._tilt_jitter_deg = float(getattr(grasp_cfg, "tilt_jitter_deg", 0.0)) if grasp_cfg else 0.0
        # How much of the bearing to follow. 1.0 = fully radial. Lower values
        # blend toward the fixed pose, useful if a scene's reachability is
        # better off-radial.
        self._bearing_gain = float(getattr(grasp_cfg, "bearing_gain", 1.0)) if grasp_cfg else 1.0

        self._base_xy = np.zeros(2)
        self._base_yaw_deg = 0.0
        self._episode_jitter = np.zeros(3)

        super().__init__(cfg, robot)

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        position, orientation = robot.get_world_pose()
        self._base_xy = np.asarray(position, dtype=float)[:2]
        # Isaac returns (w, x, y, z); scipy wants (x, y, z, w).
        q = np.asarray(orientation, dtype=float)
        self._base_yaw_deg = float(
            R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)[2]
        )
        if self._nominal_bearing_deg is None:
            # Base +X points along the base yaw, and "straight ahead of the base"
            # is what the fixed ee_euler_deg was tuned against.
            self._nominal_bearing_deg = self._base_yaw_deg
        logger.info(
            f"[grasp-wide] base xy={np.round(self._base_xy, 3).tolist()} "
            f"yaw={self._base_yaw_deg:.1f}deg  nominal bearing={self._nominal_bearing_deg:.1f}deg  "
            f"bearing_gain={self._bearing_gain}  jitter yaw+-{self._yaw_jitter_deg} "
            f"tilt+-{self._tilt_jitter_deg} deg"
        )

    def reset(self):
        super().reset()
        # One jitter draw per episode, not per step: a pose that wobbles inside
        # a single grasp would just be noise on the trajectory, not a different
        # demonstration.
        j = np.array([self._tilt_jitter_deg, self._tilt_jitter_deg, self._yaw_jitter_deg])
        self._episode_jitter = np.random.uniform(-j, j)

    def _grasp_quat(self, state) -> np.ndarray:
        obj = np.asarray(state["object_position"], dtype=float)[:2]
        d = obj - self._base_xy
        if np.linalg.norm(d) < 1e-6:
            bearing_deg = self._nominal_bearing_deg
        else:
            bearing_deg = float(np.degrees(np.arctan2(d[1], d[0])))

        # Shortest signed difference, so an object that wraps past +-180 deg does
        # not spin the wrist the long way round.
        delta = (bearing_deg - self._nominal_bearing_deg + 180.0) % 360.0 - 180.0
        delta *= self._bearing_gain

        nominal = R.from_euler("xyz", np.radians(self._ee_euler_deg + self._episode_jitter))
        # Pre-multiply: the swing is about the WORLD +Z axis, not the hand's own.
        return (R.from_euler("z", delta, degrees=True) * nominal).as_quat()
