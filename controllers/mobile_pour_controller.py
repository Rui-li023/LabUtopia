from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .mobile_base_controller import PHASE_PICK
from .mobile_pick_controller import MobilePickController

# Recorded phase id for the pour segment (navigate=0, pick=1 in the mobile layout).
PHASE_POUR = 5

# panda_joint7 (wrist roll) travel limit, rad — clamp the pour tilt to it.
_WRIST_LIMIT = 2.85


class MobilePourController(MobilePickController):
    """Level-5 mobile pour: navigate, pick the source beaker, carry it ABOVE the
    target container, then pour by rotating ONLY the wrist joint.

    Reuses :class:`MobilePickController`'s navigate + pick verbatim. The pour
    replicates the atomic ``PourController``'s geometry without its direct
    DOF-mode switching (which assumes a 7-DOF Franka and segfaults on the 12-DOF
    Ridgebase):

    - **Approach** — cspace-move the end-effector (rise / traverse at a safe
      height / descend) to a staging point computed so that at full tilt the
      beaker MOUTH lands on the target's near rim (``pour_spout_offset`` from
      the center, ``pour_height`` above it — a couple of cm above the rim, not
      a high hover). The wrist axis passes through the EE origin, so the mouth
      rotates about the EE point and the staging back-solves as
      ``hover_ee = mouth_goal - R_tilt @ r_mouth`` where ``r_mouth`` is the
      mouth-minus-EE vector measured right after the grasp. The result is a
      human-style lip-over-rim pour, matching the L1/L2 atomic pour geometry,
      instead of spinning the beaker high above the target center.
    - **Tilt/Hold/Return** — freeze the arm at that staging pose and ramp ONLY
      wrist joint 6 by ``pour_wrist_delta`` (position command, remappable).
      The mouth arcs from upright down onto the rim spout point and pours — the
      beaker never translates into the target (the earlier cspace-orientation
      ramp re-solved the whole arm and shoved the beaker into the container).

    Phases (recorded): 0 navigate, 1 pick, 5 pour.
    Success: source lifted (pick), the pour sequence completes with the source
    still held, AND at peak tilt the mouth was within ``pour_center_tolerance``
    of the target center with the beaker tilted at least ``pour_min_tilt_deg``
    (ground-truth pose check — "poured INTO the target", not just "moved").
    """

    # Sub-phase step budgets for the pour (at collect control rate).
    APPROACH_STEPS = 120   # carry the beaker to the pour staging point
    CENTER_STEPS = 60      # base micro-crab closing the residual mouth error
    TILT_STEPS = 120       # ramp wrist joint 6 to the pour tilt
    HOLD_STEPS = 50        # hold tilted (pour out)
    RETURN_STEPS = 120     # ramp the wrist back upright

    def __init__(self, cfg: Any, robot: Any) -> None:
        super().__init__(cfg, robot)
        self.pick_done = False
        grasp_euler = getattr(cfg.task, "grasp_ee_euler_deg", [-115, 90, 0])
        self._grasp_R = R.from_euler("xyz", np.radians([float(v) for v in grasp_euler]))
        # Pour = rotate wrist joint 6 by this angle (like the atomic pour). For a
        # side grasp this tips the beaker opening down. -110 empirically parks
        # the staging hand closest to the spout (~3 cm; the side grasp grips
        # just below the rim, so the mouth-to-EE arm is short); +110 landed it
        # 5 cm on the far side and lengthened the low crossing. A headroom
        # guard flips the sign if joint 6 lacks travel.
        self._pour_wrist_delta = float(np.radians(getattr(cfg.task, "pour_wrist_delta_deg", -110.0)))
        # Height (m) of the tilted MOUTH above the target's geometry center —
        # keep it a couple of cm above the RIM (center + half height), not a
        # high hover.
        self._pour_height = float(getattr(cfg.task, "pour_height", 0.08))
        # Horizontal offset (m) of the spout point from the target center toward
        # the beaker body — pour just inside the NEAR rim, like a human pour.
        self._pour_spout_offset = float(getattr(cfg.task, "pour_spout_offset", 0.01))
        # Transit clearance (m): the approach rises above the pour height by
        # this margin while traversing, then descends — a straight low carry
        # grazed the target rim and knocked the source out of the grip.
        self._pour_transit_clearance = float(getattr(cfg.task, "pour_transit_clearance", 0.07))
        # Success gate: at peak tilt the mouth must be within this radius of
        # the target center (the opening radius) and the beaker tilted at least
        # this much — "poured INTO the target", not just "did the motion".
        self._pour_center_tol = float(getattr(cfg.task, "pour_center_tolerance", 0.04))
        self._pour_min_tilt = float(getattr(cfg.task, "pour_min_tilt_deg", 80.0))
        self._pour_step = 0
        self._pour_ee_pos: Optional[np.ndarray] = None
        self._pour_target_pos: Optional[np.ndarray] = None
        self._pour_arm0: Optional[np.ndarray] = None
        self._pour_hover_ee: Optional[np.ndarray] = None
        self._pour_mouth_r: Optional[np.ndarray] = None
        self._pour_mouth_goal: Optional[np.ndarray] = None
        self._pour_tilt_R: Optional[R] = None
        self._pour_axis: Optional[np.ndarray] = None
        self._pour_center_from: Optional[np.ndarray] = None
        self._pour_center_delta: Optional[np.ndarray] = None
        self._pour_approach_extra = 0
        self._pour_half_h = 0.05
        self._pour_gate_met = False
        self._pour_peak_miss: Optional[float] = None
        self._pour_peak_tilt: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self.pick_done = False
        self._pour_step = 0
        self._pour_ee_pos = None
        self._pour_target_pos = None
        self._pour_arm0 = None
        self._pour_hover_ee = None
        self._pour_mouth_r = None
        self._pour_mouth_goal = None
        self._pour_tilt_R = None
        self._pour_axis = None
        self._pour_center_from = None
        self._pour_center_delta = None
        self._pour_approach_extra = 0
        self._pour_half_h = 0.05
        self._pour_gate_met = False
        self._pour_peak_miss = None
        self._pour_peak_tilt = None
        self._pour_wrist_delta = float(np.radians(
            getattr(self.cfg.task, "pour_wrist_delta_deg", -110.0)))

    # ── Success ──────────────────────────────────────────────────────────

    def _check_success(self) -> bool:
        lifted = super()._check_success()
        if self.mode != "collect":
            # Infer: latch "a pouring pose over the target was reached" (tilt +
            # mouth-over-opening from ground-truth poses), then require the
            # source to still be held for the base success counter.
            if lifted and not self._pour_gate_met and self.state is not None:
                self._update_pour_gate(self.state, log=False)
            return lifted and self._pour_gate_met
        total = (self.APPROACH_STEPS + self.CENTER_STEPS + self.TILT_STEPS
                 + self.HOLD_STEPS + self.RETURN_STEPS)
        return lifted and self._pour_step >= total and self._pour_gate_met

    # ── Collect ──────────────────────────────────────────────────────────

    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if not self.navigation_done:
            return self._navigation_phase(state)
        if not self.pick_done:
            return self._pick_then_pour(state)
        return self._pour_phase(state)

    def _pick_then_pour(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Parent pick, but on a successful lift transition to pour instead of
        ending the episode."""
        base_pos = self._sync_arm_base_pose()
        self.pick_controller.set_robot_position(base_pos)
        self._track_pick_ee()

        if not self.pick_controller.is_done():
            object_size = (state["object_size"] if state.get("object_size") is not None
                           else np.array([0.06, 0.06, 0.07]))
            action, record8 = self.pick_controller.forward(
                picking_position=state["object_position"],
                current_joint_positions=self.franka_subset.get_joint_positions(),
                object_name=state["object_name"],
                object_size=object_size,
                gripper_control=self.gripper_control,
                gripper_position=self.robot.get_gripper_position(),
                end_effector_orientation=self._grasp_orientation.copy(),
                gripper_distances=self._grip_distance(self.pick_controller, state["object_name"]),
            )
            self._record_step(state, self._arm_record_to_11(record8), PHASE_PICK)
            return self._remap_arm_action(action), False, False

        if not super()._check_success():
            self._log_pick_fail_diag(state)
            self._last_failure_reason = (
                f"Pick failed: object not lifted more than {self.LIFT_THRESHOLD} m")
            self.data_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False

        self._pour_ee_pos = np.asarray(self.robot.get_gripper_position(), dtype=np.float64)
        tgt = state.get("target_position")
        self._pour_target_pos = (np.asarray(tgt, dtype=np.float64) if tgt is not None
                                 else self._pour_ee_pos.copy())
        self._plan_pour_hover(state)
        logger.info(f"Pick OK — starting pour phase (ee={np.round(self._pour_ee_pos,3).tolist()} "
                    f"target={np.round(self._pour_target_pos,3).tolist()} "
                    f"hover_ee={np.round(self._pour_hover_ee,3).tolist()})")
        self.pick_done = True
        return None, False, False

    def _plan_pour_hover(self, state: Dict[str, Any]) -> None:
        """Solve the EE hover point that puts the tilted MOUTH over the target.

        Wrist joint 6 spins the hand (and the rigidly-held beaker) about the EE
        tool z-axis, and the EE origin lies ON that axis — so the mouth rotates
        about the EE point. Measure the mouth-minus-EE vector right after the
        grasp (the approach keeps the grasp orientation, so it stays valid) and
        back-solve: ``hover_ee = mouth_goal - R_tilt @ r_mouth``.
        """
        size = state.get("object_size")
        self._pour_half_h = float(size[2]) / 2.0 if size is not None else 0.05
        obj = np.asarray(state["object_position"], dtype=np.float64)
        mouth0 = obj + np.array([0.0, 0.0, self._pour_half_h])
        self._pour_mouth_r = mouth0 - self._pour_ee_pos
        # Decide the tilt sign NOW (the hover point depends on it): flip if
        # joint 6 lacks travel headroom for the configured direction.
        j6 = float(self.franka_subset.get_joint_positions()[6])
        if abs(j6 + self._pour_wrist_delta) > _WRIST_LIMIT:
            self._pour_wrist_delta = -self._pour_wrist_delta
        axis = self._grasp_R.apply(np.array([0.0, 0.0, 1.0]))
        self._pour_axis = axis
        self._pour_tilt_R = R.from_rotvec(axis * self._pour_wrist_delta)
        swung = self._pour_tilt_R.apply(self._pour_mouth_r)
        # Spout point: on the NEAR-side rim (the horizontal direction the tilted
        # body hangs from the mouth), so the lip pours over the rim edge with
        # the body beside the opening — not from high above the center.
        body_dir = -swung[:2]
        norm = float(np.linalg.norm(body_dir))
        body_dir = body_dir / norm if norm > 1e-6 else np.zeros(2)
        spout_xy = self._pour_target_pos[:2] + self._pour_spout_offset * body_dir
        self._pour_mouth_goal = np.array([spout_xy[0], spout_xy[1],
                                          self._pour_target_pos[2] + self._pour_height])
        self._pour_hover_ee = self._pour_mouth_goal - swung

    def _pour_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self._sync_arm_base_pose()
        total = (self.APPROACH_STEPS + self.CENTER_STEPS + self.TILT_STEPS
                 + self.HOLD_STEPS + self.RETURN_STEPS)
        s = self._pour_step

        if s >= total:
            held = super()._check_success()
            if held and self._pour_gate_met:
                logger.info("Pour complete — poured into the target and source still held; success")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data()
                self._last_success = True
            else:
                if not held:
                    self._last_failure_reason = "Pour failed: source dropped during pour"
                else:
                    self._last_failure_reason = (
                        f"Pour failed: mouth missed the target opening at peak tilt "
                        f"(xy_miss={self._pour_peak_miss} tol={self._pour_center_tol} "
                        f"tilt={self._pour_peak_tilt} min={self._pour_min_tilt})")
                self.data_collector.clear_cache()
                self._last_success = False
            self.reset_needed = True
            return None, True, self._last_success

        if s < self.APPROACH_STEPS:
            # Carry the beaker to the staging point solved by _plan_pour_hover
            # in three legs — rise to a safe height, traverse, descend — so the
            # held beaker clears the target rim instead of grazing it on a
            # straight low carry.
            t = s / max(self.APPROACH_STEPS, 1)
            start, hover = self._pour_ee_pos, self._pour_hover_ee
            safe_z = max(float(start[2]), float(hover[2]) + self._pour_transit_clearance)
            if t < 0.3:      # rise above the transit height
                f = t / 0.3
                pos = np.array([start[0], start[1], start[2] + f * (safe_z - start[2])])
            elif t < 0.7:    # traverse at the safe height
                f = (t - 0.3) / 0.4
                pos = np.array([start[0] + f * (hover[0] - start[0]),
                                start[1] + f * (hover[1] - start[1]), safe_z])
            else:            # descend onto the staging point
                f = (t - 0.7) / 0.3
                pos = np.array([hover[0], hover[1], safe_z + f * (hover[2] - safe_z)])
            # Convergence gate: RMPFlow lags the ramp, so at the nominal end of
            # the approach hold the hover target until the EE actually arrives
            # (else the tilt starts high/short and the base crab has to make up
            # 9+ cm — and nothing can make up the height).
            if s == self.APPROACH_STEPS - 1:
                ee_now = np.asarray(self.robot.get_gripper_position(), dtype=np.float64)
                err = float(np.linalg.norm(ee_now - hover))
                if err > 0.015 and self._pour_approach_extra < 90:
                    self._pour_approach_extra += 1
                    self._pour_step -= 1  # stay on this step; net zero after the shared +1
                elif self._pour_approach_extra:
                    logger.info(f"[POUR-APPROACH] converged err={err:.3f} "
                                f"after +{self._pour_approach_extra} extra steps")
                pos = hover
            action = self.rmp_controller.forward(
                target_end_effector_position=pos,
                target_end_effector_orientation=self._grasp_R.as_quat())
            arm = (np.asarray(action.joint_positions[:7], dtype=np.float32)
                   if action is not None and action.joint_positions is not None
                   else np.asarray(self.franka_subset.get_joint_positions()[:7], dtype=np.float32))
        elif self._tilt_a_end() <= s < self._tilt_a_end() + self.CENTER_STEPS:
            # Closed-loop centering at 60% tilt (below the spill angle): the
            # analytic swing model is off by 3-4 cm (the RMP-held orientation
            # differs from the commanded grasp at the reach edge), so measure
            # the mouth's GROUND-TRUTH position from the object's world pose,
            # extrapolate only the remaining 40% of the swing, and crab the
            # holonomic base by the horizontal error — the base carries the arm
            # rigidly, so the mouth shifts 1:1 (base prismatic joints are
            # world-aligned).
            if self._pour_center_from is None:
                ee = np.asarray(self.robot.get_gripper_position(), dtype=np.float64)
                mouth_now = self._measure_mouth(state)
                if mouth_now is None:
                    err = np.zeros(2)
                else:
                    rem = R.from_rotvec(self._pour_axis * (0.4 * self._pour_wrist_delta))
                    pred_full = ee + rem.apply(mouth_now - ee)
                    err = self._pour_mouth_goal[:2] - pred_full[:2]
                self._pour_center_from = np.asarray(
                    self.base_subset.get_joint_positions()[:3], dtype=np.float64).copy()
                self._pour_center_delta = np.array([err[0], err[1], 0.0])
                logger.info(f"[POUR-NUDGE] full-tilt mouth err={np.round(err, 3).tolist()} "
                            f"-> base crab {float(np.linalg.norm(err)):.3f} m (measured at 60% tilt)")
            frac = min((s - self._tilt_a_end() + 1) / max(self.CENTER_STEPS, 1), 1.0)
            base_tgt = self._pour_center_from + self._pour_center_delta * frac
            action = self.base_subset.make_articulation_action(
                joint_positions=base_tgt, joint_velocities=None)
            arm = np.asarray(self.franka_subset.get_joint_positions()[:7], dtype=np.float32)
            action11 = np.concatenate([base_tgt.astype(np.float32), arm,
                                       [self._gripper_closedness()]]).astype(np.float32)
            self._record_step(state, action11, PHASE_POUR)
            self._pour_step += 1
            return action, False, False
        else:
            # Freeze the arm at the staging pose and ramp ONLY wrist joint 6
            # -> the beaker tips its opening down over the target and pours.
            # Tilt runs in two legs (A: 0 -> 60%, B: 60% -> 100%) around the
            # centering crab.
            tilt_a = self._tilt_a_end() - self.APPROACH_STEPS
            tilt_b = self.TILT_STEPS - tilt_a
            b_start = self._tilt_a_end() + self.CENTER_STEPS
            hold_start = b_start + tilt_b
            if self._pour_arm0 is None:
                self._pour_arm0 = np.asarray(
                    self.franka_subset.get_joint_positions()[:7], dtype=np.float64).copy()
            if s < self._tilt_a_end():
                frac = 0.6 * (s - self.APPROACH_STEPS) / max(tilt_a, 1)
            elif s < hold_start:
                frac = 0.6 + 0.4 * (s - b_start) / max(tilt_b, 1)
            elif s < hold_start + self.HOLD_STEPS:
                frac = 1.0
                if s == hold_start:
                    self._update_pour_gate(state)
            else:
                frac = 1.0 - (s - hold_start - self.HOLD_STEPS) / max(self.RETURN_STEPS, 1)
            arm = self._pour_arm0.copy()
            arm[6] = float(np.clip(
                self._pour_arm0[6] + self._pour_wrist_delta * float(np.clip(frac, 0.0, 1.0)),
                -_WRIST_LIMIT, _WRIST_LIMIT))
            action = ArticulationAction(joint_positions=[float(x) for x in arm])
            arm = arm.astype(np.float32)

        record8 = np.concatenate([arm, [self._gripper_closedness()]]).astype(np.float32)
        self._record_step(state, self._arm_record_to_11(record8), PHASE_POUR)
        self._pour_step += 1
        return self._remap_arm_action(action), False, False

    def _tilt_a_end(self) -> int:
        """Step index where the first (pre-spill, 60%) tilt leg ends."""
        return self.APPROACH_STEPS + int(self.TILT_STEPS * 0.6)

    def _measure_mouth(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Ground-truth mouth position from the source's world pose (the task
        supplies the quaternion w-first)."""
        obj = state.get("object_position")
        quat = state.get("object_quaternion")
        if obj is None or quat is None:
            return None
        q = np.asarray(quat, dtype=np.float64)
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        return np.asarray(obj, dtype=np.float64) + rot.apply([0.0, 0.0, self._pour_half_h])

    def _pour_pose_metrics(self, state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """(mouth-to-target-center xy distance, beaker tilt in degrees) from
        the source's ground-truth world pose — 'would the liquid land in the
        target' without simulating liquid."""
        obj = state.get("object_position")
        quat = state.get("object_quaternion")
        tgt = (self._pour_target_pos if self._pour_target_pos is not None
               else state.get("target_position"))
        if obj is None or quat is None or tgt is None:
            return None
        size = state.get("object_size")
        half_h = float(size[2]) / 2.0 if size is not None else self._pour_half_h
        q = np.asarray(quat, dtype=np.float64)
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        tilt_deg = float(np.degrees(np.arccos(np.clip(rot.apply([0.0, 0.0, 1.0])[2], -1.0, 1.0))))
        mouth = np.asarray(obj, dtype=np.float64) + rot.apply([0.0, 0.0, half_h])
        miss = float(np.linalg.norm(mouth[:2] - np.asarray(tgt, dtype=np.float64)[:2]))
        return miss, tilt_deg

    def _update_pour_gate(self, state: Dict[str, Any], log: bool = True) -> None:
        """Evaluate the pour gate (mouth within the target opening AND tilted
        past the pour angle) and latch it once met."""
        metrics = self._pour_pose_metrics(state)
        if metrics is None:
            return
        miss, tilt_deg = metrics
        self._pour_peak_miss, self._pour_peak_tilt = miss, tilt_deg
        met = miss <= self._pour_center_tol and tilt_deg >= self._pour_min_tilt
        self._pour_gate_met = self._pour_gate_met or met
        if log:
            logger.info(f"[POUR-CENTER] xy_miss={miss:.3f} (tol {self._pour_center_tol}) "
                        f"tilt={tilt_deg:.0f}deg (min {self._pour_min_tilt:.0f}) "
                        f"gate={'OK' if met else 'FAIL'}")

    # ── Language ─────────────────────────────────────────────────────────

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state["object_name"]) if self.state else "beaker"
        if not self.navigation_done:
            return self._get_cached_instruction(
                "mobile_pour:navigate",
                self._build_instruction_templates(
                    f"Move to the bench and pick up the {object_name}",
                    f"Drive to the lab bench, stop in front of it, and pick up the {object_name}",
                ),
            )
        if not self.pick_done:
            return self._get_cached_instruction(
                "mobile_pour:pick",
                self._build_instruction_templates(
                    f"Pick up the {object_name}",
                    f"Pick up the {object_name} from the bench and lift it clear of the surface",
                ),
            )
        return self._get_cached_instruction(
            "mobile_pour:pour",
            self._build_instruction_templates(
                f"Pour the contents of the {object_name} into the target container",
                f"Hold the {object_name} over the target container and pour its contents",
            ),
        )
