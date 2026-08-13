from typing import Any, Dict, Tuple

import numpy as np
from loguru import logger

from .inference_engines.mobile_remote_inference_engine import MobileRemoteInferenceEngine
from .inference_engines.navdp_nav_inference_engine import NavDPNavInferenceEngine
from .inference_engines.vint_nav_inference_engine import ViNTNavInferenceEngine
from .mobile_base_controller import _NullTrajectory
from .mobile_pick_controller import MobilePickController


def _wrap_angle(a: float) -> float:
    return float(np.mod(a + np.pi, 2 * np.pi) - np.pi)


def _create_nav_engine(cfg: Any) -> Any:
    nav_type = str(getattr(cfg.infer.nav, "type", "vint"))
    if nav_type == "vint":
        return ViNTNavInferenceEngine(cfg)
    if nav_type == "navdp":
        return NavDPNavInferenceEngine(cfg)
    raise ValueError(
        f"Unsupported nav engine type: '{nav_type}'. Available: ['vint', 'navdp']"
    )


class MobileDecoupledPickController(MobilePickController):
    """Level-5 mobile pick with navigation and manipulation as SEPARATE models.

    Two-phase infer, unlike ``MobilePickController`` (single full-body VLA):

    - **Phase NAV** — an external navigation-only policy (``cfg.infer.nav.type``,
      e.g. ViNT) drives the base only; arm/gripper are held at their current
      pose in the recorded action so ``_apply_action11`` leaves them
      untouched. Ends when the nav engine reports ``is_docked()`` or its step
      budget is exhausted (safety net for a mis-calibrated distance head).
    - **Phase MANIP** — control hands to one of the existing close_pick
      full-body VLA checkpoints (any ``level5_close_pick_<model>`` config's
      ``infer:`` block, reused verbatim as ``cfg.infer``). It still predicts
      an 11-dim action including small base corrections — the same policy
      that was trained to move+pick from ~1 m out is now simply started
      already near the bench, not retrained.

    This lets the same 4 trained manipulation checkpoints combine with any
    number of navigation front-ends (2x4 combination matrix) without
    touching either model's weights.
    """

    def _init_infer_mode(self, cfg: Any, robot: Any = None) -> None:
        self.trajectory_controller = _NullTrajectory()
        self.nav_engine = _create_nav_engine(cfg)
        # Constructed lazily on handoff (see _step_infer) — RemoteInferenceEngine
        # blocks its __init__ for up to 15x10s connecting to the manip policy
        # server; doing that eagerly here would stall the nav phase before it
        # ever gets a frame.
        self.inference_engine = None
        self._infer_base_delta = bool(getattr(cfg.infer, "base_delta_actions", True))
        self._nav_handoff_done = False
        # Post-dock alignment (Path B): point-goal nav lands the correct dock
        # POSITION but an arbitrary heading; the close_pick VLA was trained on a
        # narrow ~87deg (final_nav_angle) approach and cannot grasp from a
        # rotated start. So after docking, rotate the base in place to the grasp
        # heading before handing off.
        nav_cfg = cfg.infer.nav
        self._align_enabled = bool(getattr(nav_cfg, "align_before_handoff", True))
        self._align_tol = float(getattr(nav_cfg, "align_tol", 0.05))
        self._align_max_steps = int(getattr(nav_cfg, "align_max_steps", 400))
        self._align_max_rot = float(getattr(cfg.task, "max_angular_speed", 0.12))
        self._nav_aligning = False
        self._align_steps = 0

    def reset(self) -> None:
        super().reset()
        self._nav_handoff_done = False
        self._nav_aligning = False
        self._align_steps = 0
        if hasattr(self, "nav_engine"):
            self.nav_engine.reset()

    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if not self._nav_handoff_done:
            return self._nav_phase_step(state)
        if self.inference_engine is None:
            logger.info("[Decoupled] nav handoff complete — connecting to the manipulation policy server")
            # Same attribute name MobilePickController._step_infer reads —
            # lets phase MANIP fall through to the parent implementation
            # unchanged once this is set.
            self.inference_engine = MobileRemoteInferenceEngine(self.cfg, self.trajectory_controller)
        return super()._step_infer(state)

    def _nav_phase_step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        state["agent_pose"] = self._state11()
        self._update_nav_progress(state)
        self._log_infer_diag(state)

        if self._nav_aligning:
            return self._align_step(state)

        delta = self.nav_engine.step_navigate(state)
        applied = None
        if delta is not None:
            arm = np.asarray(self.franka_subset.get_joint_positions()[:7], dtype=np.float32)
            action11 = np.concatenate(
                [np.asarray(delta, dtype=np.float32), arm, [self._gripper_closedness()]]
            ).astype(np.float32)
            applied = self._apply_action11(action11)

        docked = self.nav_engine.is_docked()
        budget_exceeded = self.nav_engine.nav_budget_exceeded()
        if docked or budget_exceeded:
            self._log_dock_diag(state, state["dock_point"], label="NAV-DOCKED")
            logger.info(
                f"[Decoupled] nav reached dock (docked={docked}, budget_exceeded={budget_exceeded})"
            )
            if self._align_enabled:
                self._nav_aligning = True  # rotate to grasp heading before handoff
                self._align_steps = 0
            else:
                self._log_dock_diag(state, state["dock_point"], label="NAV-HANDOFF")
                self._nav_handoff_done = True
        return applied, False, False

    def _align_step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Rotate the base in place to the grasp heading (``final_nav_angle``)
        before handing off, keeping the docked position. A pure in-place yaw:
        the body-frame delta translation is zero, so ``_apply_action11`` holds
        x/y and only advances the base revolute joint."""
        target = float(state.get("final_nav_angle", np.pi / 2 - 0.044))
        pose = np.asarray(state["current_pose"], dtype=float)
        base = np.asarray(state["agent_pose"], dtype=float)
        heading = _wrap_angle(float(pose[2] + base[2]))
        err = _wrap_angle(target - heading)
        self._align_steps += 1

        if abs(err) <= self._align_tol or self._align_steps >= self._align_max_steps:
            self._log_dock_diag(state, state["dock_point"], label="NAV-HANDOFF")
            logger.info(
                f"[Decoupled] aligned to grasp heading (err={np.degrees(err):.1f}deg, "
                f"steps={self._align_steps}) -> handoff to manipulation policy"
            )
            self._nav_handoff_done = True
            return None, False, False

        dtheta = float(np.clip(err, -self._align_max_rot, self._align_max_rot))
        arm = np.asarray(self.franka_subset.get_joint_positions()[:7], dtype=np.float32)
        action11 = np.concatenate(
            [np.array([0.0, 0.0, dtheta], dtype=np.float32), arm, [self._gripper_closedness()]]
        ).astype(np.float32)
        applied = self._apply_action11(action11)
        return applied, False, False
