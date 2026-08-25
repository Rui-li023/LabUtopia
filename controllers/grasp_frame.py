"""Config-driven grasp frame: nominal pose + bearing following + per-episode jitter.

Every grasping task controller used to hard-code its end-effector orientation
(``R.from_euler('xyz', np.radians([0, 90, 30]))`` and friends). Two costs:

1. **It is only correct straight ahead of the base.** As the object moves
   laterally the approach direction rotates but the hand's yaw does not, the two
   disagree, IK picks a folded wrist, and a side grasp slides off the glassware.
   That is what caps how wide an object's spawn range can be.
2. **Every episode shows the policy the same exact wrist pose**, which teaches
   "this one pose is the only correct one" -- a lesson that transfers badly to
   any rig whose object pose is never exact.

``GraspFrame`` centralises both fixes so a config can opt in per task:

.. code-block:: yaml

    grasp:
      ee_euler_deg: [0, 90, 30]    # nominal, world frame, scipy extrinsic "xyz"
      bearing_gain: 1.0            # 1.0 = frame fully follows the object's bearing
      yaw_jitter_deg: 8.0          # per-episode jitter about world +Z
      tilt_jitter_deg: 4.0         # per-episode jitter on the other two axes
      nominal_bearing_deg: null    # bearing ee_euler_deg was tuned at; null = base yaw

``bearing_gain`` defaults to 0.0 -- the historical fixed pose -- so every
existing L1-L5 config keeps its exact behaviour until it opts in.

First validated on the 500-episode sim2real/pick collect: it is what let the
spawn area grow from 36x26 cm to 60x28 cm with no lateral success penalty
(success by |x| bin: 82.6 / 88.0 / 80.0 %, 69 episodes).
"""

from typing import Any, Sequence

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R


class GraspFrame:
    """One end-effector frame: nominal euler, bearing following, episode jitter.

    A task with two grasp-like phases (pick then pour/place) builds one instance
    per phase, each with its own nominal euler and its own jitter draw.
    """

    def __init__(
        self,
        grasp_cfg: Any,
        robot: Any,
        nominal_euler_deg: Sequence[float],
        *,
        prefix: str = "",
        default_bearing_gain: float = 0.0,
        label: str = "grasp",
    ) -> None:
        self._label = label
        self._grasp_cfg = grasp_cfg
        self._prefix = prefix
        # inherit=False: a prefixed frame must NOT fall back to the unprefixed
        # `ee_euler_deg`. That one belongs to the PICKING phase, and the two poses
        # are genuinely different ([0,90,30] to grasp vs [0,90,15] to pour) — a
        # silent fallback would hand the pour phase the grasp pose. Gains and
        # jitter below do inherit, which is the useful half of the sharing.
        self._ee_euler_deg = np.asarray(self._get("ee_euler_deg", nominal_euler_deg, inherit=False), dtype=float)
        # How much of the object's bearing to follow. 1.0 = fully radial; 0.0 =
        # the historical fixed world-frame pose. Values in between blend, which
        # is useful when a scene's reachability is better off-radial.
        self._bearing_gain = float(self._get("bearing_gain", default_bearing_gain))
        self._yaw_jitter_deg = float(self._get("yaw_jitter_deg", 0.0))
        self._tilt_jitter_deg = float(self._get("tilt_jitter_deg", 0.0))

        position, orientation = robot.get_world_pose()
        self._base_xy = np.asarray(position, dtype=float)[:2]
        # Isaac returns (w, x, y, z); scipy wants (x, y, z, w).
        q = np.asarray(orientation, dtype=float)
        base_yaw_deg = float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)[2])

        # The bearing the fixed euler was tuned at. None => the base yaw, i.e.
        # "straight ahead of the base", which is how every existing value was tuned.
        raw = self._get("nominal_bearing_deg", None)
        self._nominal_bearing_deg = float(raw) if raw is not None else base_yaw_deg

        # The configured euler is written in the canonical frame (tool +Z is the
        # approach axis, +Y separates the fingers); not every arm agrees.
        self._correction = R.from_euler("xyz", np.radians(robot.tool_frame_correction_euler_deg))

        self._episode_jitter = np.zeros(3)

        logger.info(
            f"[{self._label}] euler={self._ee_euler_deg.tolist()} bearing_gain={self._bearing_gain} "
            f"nominal_bearing={self._nominal_bearing_deg:.1f}deg base_xy={np.round(self._base_xy, 3).tolist()} "
            f"jitter yaw+-{self._yaw_jitter_deg} tilt+-{self._tilt_jitter_deg} deg"
        )

    def _get(self, key: str, default: Any, *, inherit: bool = True) -> Any:
        """``grasp.<prefix><key>``, else ``grasp.<key>`` (when *inherit*), else *default*.

        The prefixed form is how a multi-phase task overrides one phase only.
        It matters: level2/pour's success gate gives the *object* a 30 deg total
        rotation budget measured against its spawn orientation, so letting the
        POURING frame follow the receiving beaker's bearing (-14 to -38 deg over
        its spawn range) spends that whole budget before the tilt even starts —
        the wrist returns to a rotated "upright" and the gate never closes.
        Measured: 110 of 118 failure records were "Return rotation not complete".
        `pour_bearing_gain: 0.0` pins that one phase while PICKING still follows.
        """
        cfg = self._grasp_cfg
        if cfg is None:
            return default
        if self._prefix:
            val = getattr(cfg, self._prefix + key, None)
            if val is not None:
                return val
            if not inherit:
                return default
        val = getattr(cfg, key, None)
        return default if val is None else val

    def new_episode(self) -> None:
        """Draw this episode's jitter. One draw per episode, not per step: a pose
        that wobbled inside a single grasp would be noise on the trajectory, not a
        different demonstration."""
        j = np.array([self._tilt_jitter_deg, self._tilt_jitter_deg, self._yaw_jitter_deg])
        self._episode_jitter = np.random.uniform(-j, j)

    def quat(self, target_position: Sequence[float] | None = None) -> np.ndarray:
        """World-frame orientation as an xyzw quaternion, for this step.

        ``target_position`` is whatever the hand is reaching for this phase (the
        object when picking, the receiving vessel when pouring/placing). It may be
        omitted when ``bearing_gain`` is 0.
        """
        delta = 0.0
        if self._bearing_gain and target_position is not None:
            d = np.asarray(target_position, dtype=float)[:2] - self._base_xy
            if np.linalg.norm(d) >= 1e-6:
                bearing_deg = float(np.degrees(np.arctan2(d[1], d[0])))
                # Shortest signed difference, so a target that wraps past +-180 deg
                # does not spin the wrist the long way round.
                delta = (bearing_deg - self._nominal_bearing_deg + 180.0) % 360.0 - 180.0
                delta *= self._bearing_gain

        nominal = R.from_euler("xyz", np.radians(self._ee_euler_deg + self._episode_jitter))
        # Pre-multiply: the swing is about the WORLD +Z axis, not the hand's own.
        return (R.from_euler("z", delta, degrees=True) * nominal * self._correction).as_quat()


def resolve_pick_z_offset(grasp_cfg: Any) -> float | None:
    """``grasp.pick_z_offset`` as a float, or None to keep the shared per-object table.

    The table in ``atomic_actions/pick_controller.get_pickz_offset`` is shared by
    every L1-L5 task, so a scene that wants a different grasp height must set this
    override rather than edit the table.
    """
    raw = getattr(grasp_cfg, "pick_z_offset", None) if grasp_cfg else None
    return float(raw) if raw is not None else None
