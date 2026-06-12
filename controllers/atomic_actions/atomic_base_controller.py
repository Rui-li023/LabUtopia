from isaacsim.core.utils.types import ArticulationAction
import numpy as np
import typing

from robots.base_robot import BaseRobot, GRIPPER_CLOSED, GRIPPER_OPEN

# Franka per-finger opening at the fully-open pose (metres). Used to normalize
# the recorded gripper channel to [0,1]. Must match the open joint target the
# replay mapping interpolates from (franka gripper_open_position = 0.05) so a
# recorded grip round-trips to the same finger opening.
GRIPPER_MAX_OPEN = 0.05


class AtomicBaseController:
    """Base class for finite-state-machine atomic action controllers.

    Provides: state machine, per-episode randomization, 8-dim record array
    (7 arm joints + 1 gripper state), robot resolution, quaternion math,
    position checks, and gripper shortcuts.

    Subclasses override ``_sample_randomization()`` and implement ``forward()``.
    """

    # Class-wide flag set by main.py alongside cfg.collect_position_only. When
    # True, record[7] is the robot-level COMMANDED gripper opening (the drive
    # target) instead of the measured finger position. Replaying a commanded
    # channel re-issues the exact same drive target — e.g. a binary close keeps
    # squeezing toward 0 in replay just like collect did, instead of driving to
    # the measured contact width (which only touches, so the object slips).
    record_commanded_gripper: bool = False

    # Class-wide flag set by main.py from cfg.action_randomization. When False,
    # per-episode action noise is neutralized: _noisy() returns the base value,
    # _uniform() returns the range midpoint, _apply_quat_noise() is identity.
    # Sampling still runs, so parameters that are *defined* by _uniform (shake
    # amplitude, stir radius/speed) keep a sane deterministic mid-range value
    # instead of a zero default. Used for long-horizon (level4) tasks where
    # noise compounds across many chained atomic actions.
    randomization_enabled: bool = True

    def __init__(
        self,
        name: str,
        cspace_controller: typing.Any = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        default_events_dt: typing.Optional[typing.List[float]] = None,
        robot: typing.Optional[BaseRobot] = None,
        position_threshold: float = 0.01,
        require_gripper: bool = False,
    ) -> None:
        self.name = name
        self._cspace_controller = cspace_controller
        self._position_threshold = position_threshold

        # State machine
        self._event = 0
        self._t = 0.0
        self._start = True

        # Events timing
        self._default_n_phases = len(default_events_dt) if default_events_dt else None
        if events_dt is not None:
            dt = list(events_dt) if isinstance(events_dt, np.ndarray) else events_dt
            if not isinstance(dt, list):
                raise ValueError("events_dt must be a list or numpy array")
            if self._default_n_phases is not None and len(dt) != self._default_n_phases:
                raise ValueError(
                    f"events_dt length must be {self._default_n_phases}, got {len(dt)}"
                )
            self._events_dt = dt
        elif default_events_dt is not None:
            self._events_dt = list(default_events_dt)
        else:
            self._events_dt = []

        # Gripper / record state
        self._current_gripper_state = GRIPPER_OPEN
        self._last_record_positions = None
        self._last_gripper_state = GRIPPER_OPEN

        # Randomization
        self._randomization_sampled = False

        # Robot
        self._robot = self._resolve_robot(robot, cspace_controller)
        if (require_gripper and self._robot is not None
                and self._robot.num_gripper_joints <= 0):
            raise ValueError(
                f"'{name}' requires gripper, got 0 for robot '{self._robot.name}'"
            )

    # ── State Machine ────────────────────────────────────────────

    def is_done(self) -> bool:
        if hasattr(self, "_is_done"):
            return bool(self._is_done)
        return self._event >= len(self._events_dt)

    def _advance_state(self):
        """Tick the state machine. Call at end of forward()."""
        if self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0.0

    def _next_event(self):
        """Force immediate transition to next event."""
        self._event += 1
        self._t = 0.0

    def _null_action(self, n: int) -> ArticulationAction:
        """No-op action with *n* joints."""
        return ArticulationAction(joint_positions=[None] * n)

    # ── Position Checks ──────────────────────────────────────────

    def _xy_reached(self, a: np.ndarray, b: np.ndarray,
                    threshold: float = None) -> bool:
        th = threshold if threshold is not None else self._position_threshold
        return float(np.linalg.norm(a[:2] - b[:2])) < th

    def _xyz_reached(self, a: np.ndarray, b: np.ndarray,
                     threshold: float = None) -> bool:
        th = threshold if threshold is not None else self._position_threshold
        return (float(np.linalg.norm(a[:2] - b[:2])) < th
                and abs(float(a[2] - b[2])) < th)

    # ── Randomization ────────────────────────────────────────────

    def _ensure_randomization(self):
        """Call once per episode (idempotent)."""
        if not self._randomization_sampled:
            self._sample_randomization()
            self._randomization_sampled = True

    def _sample_randomization(self):
        """Override to sample per-episode random values."""
        pass

    @staticmethod
    def _noisy(value: float, noise: float) -> float:
        """Return *value* ± uniform(*noise*)."""
        if not AtomicBaseController.randomization_enabled:
            return float(value)
        return value + float(np.random.uniform(-noise, noise))

    @staticmethod
    def _uniform(lo: float, hi: float) -> float:
        if not AtomicBaseController.randomization_enabled:
            return float((lo + hi) / 2.0)
        return float(np.random.uniform(lo, hi))

    # ── Quaternion Helpers ───────────────────────────────────────

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product for [x, y, z, w] quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float64)

    @classmethod
    def _axis_angle_to_quat(cls, axis: np.ndarray,
                            angle_deg: float) -> np.ndarray:
        """Axis-angle → quaternion [x, y, z, w]."""
        axis = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(axis)
        if n <= 0:
            return np.array([0, 0, 0, 1], dtype=np.float64)
        axis = axis / n
        half = np.deg2rad(angle_deg) / 2.0
        s = np.sin(half)
        return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(half)],
                        dtype=np.float64)

    @classmethod
    def _apply_quat_noise(cls, quat: np.ndarray,
                          max_angle_deg: float = 15.0) -> np.ndarray:
        """Apply a random axis-angle perturbation to *quat*."""
        if not cls.randomization_enabled:
            return np.asarray(quat, dtype=np.float64)
        axes = [np.array([1,0,0.]), np.array([0,1,0.]), np.array([0,0,1.])]
        axis = axes[int(np.random.randint(0, 3))]
        angle = float(np.random.uniform(-max_angle_deg, max_angle_deg))
        delta = cls._axis_angle_to_quat(axis, angle)
        result = cls._quat_multiply(delta, np.asarray(quat, dtype=np.float64))
        norm = np.linalg.norm(result)
        return result / norm if norm > 0 else result

    @classmethod
    def _apply_axis_rotation(cls, quat: np.ndarray, axis: np.ndarray,
                             angle_deg: float) -> np.ndarray:
        """Apply a deterministic axis-angle rotation to *quat* in the
        end-effector's local frame.

        Local-frame rotation (q_existing * delta) means the *axis* refers to
        the gripper's own coordinate system, so axis=[0, 0, 1] always rotates
        around the tool spin axis regardless of how the gripper is oriented in
        the world. This preserves grasp alignment for both top-down and
        horizontal picks.

        NOTE: for orientation NOISE on grasps that carry containers, use
        ``_apply_world_yaw`` instead — a tool-Z rotation is an upright yaw only
        for TOP-DOWN grasps; for side grasps (ee pitched 90°) the tool Z axis
        is horizontal, so rotating around it ROLLS the held object (tilted
        beakers during carry/place).
        """
        delta = cls._axis_angle_to_quat(axis, angle_deg)
        result = cls._quat_multiply(np.asarray(quat, dtype=np.float64), delta)
        norm = np.linalg.norm(result)
        return result / norm if norm > 0 else result

    @classmethod
    def _apply_world_yaw(cls, quat: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate *quat* around the WORLD vertical axis (pre-multiply).

        A world-Z rotation varies the approach/wrist heading while keeping the
        held object level for ANY grasp style: identical to the old tool-Z
        noise for top-down grasps, and a pure (no-roll) heading change for side
        grasps.
        """
        delta = cls._axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), angle_deg)
        result = cls._quat_multiply(delta, np.asarray(quat, dtype=np.float64))
        norm = np.linalg.norm(result)
        return result / norm if norm > 0 else result

    # ── Geometry ─────────────────────────────────────────────────

    @staticmethod
    def _rotate_point_around_z(point: np.ndarray, center: np.ndarray,
                               angle_deg: float) -> np.ndarray:
        """Rotate *point* around *center* on the Z axis."""
        rad = np.deg2rad(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rel = point - center
        rotated = np.array([c*rel[0] - s*rel[1],
                            s*rel[0] + c*rel[1],
                            rel[2]])
        return rotated + center

    # ── Robot Resolution ─────────────────────────────────────────

    @staticmethod
    def _resolve_robot(robot, cspace_controller):
        if robot is not None:
            if not isinstance(robot, BaseRobot):
                raise TypeError(f"robot must be BaseRobot, got {type(robot)}")
            return robot
        if cspace_controller is None:
            return None
        search = [cspace_controller]
        for attr in ("_articulation_motion_policy", "articulation_rmp"):
            obj = getattr(cspace_controller, attr, None)
            if obj is not None:
                search.append(obj)
        for target in search:
            for attr in ("robot", "robot_articulation",
                         "_robot", "_robot_articulation"):
                candidate = getattr(target, attr, None)
                if isinstance(candidate, BaseRobot):
                    return candidate
        return None

    # ── Record Array ─────────────────────────────────────────────

    def _build_record_array(
        self,
        action: ArticulationAction,
        current_joint_positions: np.ndarray,
        gripper_state: int = None,
    ) -> np.ndarray:
        """Build 8-dim record: 7 arm joints + 1 gripper state."""
        jp = action.joint_positions
        if jp is None:
            arm = (self._last_record_positions[:7].copy()
                   if self._last_record_positions is not None
                   else current_joint_positions[:7].copy())
        else:
            fallback = (self._last_record_positions[:7]
                        if self._last_record_positions is not None
                        else current_joint_positions[:7])
            arm = fallback.copy().astype(np.float64)
            for i in range(min(len(jp), 7)):
                if jp[i] is not None:
                    arm[i] = float(jp[i])

        if gripper_state is not None:
            self._last_gripper_state = gripper_state

        record = np.zeros(8, dtype=np.float64)
        record[:7] = arm
        # Unified gripper channel: normalized "closedness" in [0,1] — 0 = fully
        # open, 1 = fully closed — keeping the legacy binary 0/1 polarity but now
        # CONTINUOUS, so delicate grips reproduce in replay (round-bottom-flask
        # neck grip ≈ 0.65) instead of a binary slam that ejects them. Uses the
        # MEASURED finger opening: it reflects the physical hold across the whole
        # episode (incl. the pick→place controller handoff), where a per-
        # controller commanded target would reset to "open" during placing.
        if AtomicBaseController.record_commanded_gripper and self._robot is not None:
            # Commanded channel (position-only collect): normalize the robot-level
            # drive target so replay's hybrid map inverts it exactly —
            # s = 1 - cmd/0.05  ↔  pos = 0.05*(1-s). Binary open/close land on the
            # exact endpoints {0,1}; distance grips round-trip to the same target.
            opening = float(np.clip(self._robot.get_gripper_commanded_opening(),
                                    0.0, GRIPPER_MAX_OPEN))
            record[7] = float(np.clip(1.0 - opening / GRIPPER_MAX_OPEN, 0.0, 1.0))
        elif current_joint_positions is not None and len(current_joint_positions) > 7:
            opening = float(np.clip(current_joint_positions[7], 0.0, GRIPPER_MAX_OPEN))
            record[7] = float(np.clip(1.0 - opening / GRIPPER_MAX_OPEN, 0.0, 1.0))
        else:
            record[7] = 1.0 if int(self._last_gripper_state) == GRIPPER_CLOSED else 0.0
        self._last_record_positions = record.copy()
        return record

    # ── Gripper Shortcuts ────────────────────────────────────────

    def _open_gripper(self):
        self._current_gripper_state = GRIPPER_OPEN
        self._last_gripper_target_m = GRIPPER_MAX_OPEN
        if self._robot is not None:
            self._robot.open_gripper()

    def _close_gripper(self):
        self._current_gripper_state = GRIPPER_CLOSED
        self._last_gripper_target_m = 0.0
        if self._robot is not None:
            self._robot.close_gripper()

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, events_dt=None) -> None:
        self._event = 0
        self._t = 0.0
        self._start = True
        self._randomization_sampled = False
        self._current_gripper_state = GRIPPER_OPEN
        self._last_record_positions = None
        self._last_gripper_state = GRIPPER_OPEN
        if self._cspace_controller is not None:
            self._cspace_controller.reset()
        if events_dt is not None:
            dt = list(events_dt) if isinstance(events_dt, np.ndarray) else events_dt
            if not isinstance(dt, list):
                raise ValueError("events_dt must be a list or numpy array")
            if self._default_n_phases is not None and len(dt) != self._default_n_phases:
                raise ValueError(
                    f"events_dt length must be {self._default_n_phases}, got {len(dt)}"
                )
            self._events_dt = dt
