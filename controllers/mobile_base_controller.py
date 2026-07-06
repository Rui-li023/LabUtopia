from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger

from utils.replay_data_loader import ReplayDataLoader

from .atomic_actions.atomic_base_controller import GRIPPER_MAX_OPEN
from .base_controller import BaseController
from .robot_controllers.ridgebase.ridgebase_controller import RidgebaseController

PHASE_NAVIGATE = 0
PHASE_PICK = 1
PHASE_CARRY_NAVIGATE = 2
PHASE_PLACE = 3

# Physical per-finger travel limit (m). The gripper channel is normalized
# against GRIPPER_MAX_OPEN (0.05) everywhere in this repo — the exact inverse
# is finger = GRIPPER_MAX_OPEN * (1 - s), clamped to the physical limit so a
# recorded fully-open value (s ~= 0.2) replays back to exactly 0.04.
_FINGER_LIMIT = 0.04

# Per-finger inward squeeze (m) applied to the atomic grip-distance table on
# the MOBILE robot. TESTED AT 0.004 AND IT HURT (close_pick 2/3 -> 1/5): the
# position drive pressing 4 mm under contact EJECTS the rigid beaker — the
# same failure the static replay margin test found (trajectory_controller
# _GRIP_MARGIN: "4 mm ... HURT (flask 81%->52%)"). Contact-width grips are
# correct; kept as a tuning knob at 0.
GRIP_SQUEEZE = 0.0


class _NullTrajectory:
    """Placeholder so BaseController.reset()'s replay branch stays a no-op."""

    def reset(self) -> None:
        pass


class MobileManipControllerBase(BaseController):
    """Shared base for Level-5 mobile manipulation controllers (Ridgebase).

    Provides the Ridgebase base-motion controller, joint subsets, the unified
    11-dim action/state layout ([base x, y, theta] + 7 arm joints + gripper),
    per-step recording with phase labels, and mobile replay.

    Contract note: unlike static-arm controllers, mobile controllers DO
    override ``_step_replay`` — recorded 11-dim actions are applied directly
    as per-frame joint-position commands on all 12 DOFs, because the Franka
    trajectory controller's 9-dim actions would land on the wrong DOFs here.
    """

    BASE_JOINT_NAMES = [
        "dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
        "dummy_base_revolute_z_joint",
    ]
    ARM_JOINT_NAMES = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

    def __init__(self, cfg: Any, robot: Any) -> None:
        super().__init__(cfg, robot, use_default_config=True)
        task = cfg.task
        self.ridgebase_controller = RidgebaseController(
            robot_articulation=robot,
            max_linear_speed=float(getattr(task, "max_linear_speed", 0.04)),
            max_angular_speed=float(getattr(task, "max_angular_speed", 1.5)),
            position_threshold=float(getattr(task, "position_threshold", 0.08)),
            angle_threshold=float(getattr(task, "angle_threshold", 0.1)),
        )
        self.base_subset = ArticulationSubset(robot, self.BASE_JOINT_NAMES)
        self.franka_subset = ArticulationSubset(
            robot, self.ARM_JOINT_NAMES + self.FINGER_JOINT_NAMES)
        self.all_subset = ArticulationSubset(
            robot, self.BASE_JOINT_NAMES + self.ARM_JOINT_NAMES + self.FINGER_JOINT_NAMES)
        self.waypoints_set = False

    # ── Mode init overrides ──────────────────────────────────────────────

    def _init_replay_mode(self, cfg: Any, robot: Any = None) -> None:
        """Replay without a Franka trajectory controller (wrong DOFs here)."""
        self.trajectory_controller = _NullTrajectory()
        episode_indices = (list(cfg.replay.episode_indices)
                           if hasattr(cfg.replay, "episode_indices") else None)
        self._replay_loader = ReplayDataLoader(
            dataset_path=cfg.replay.dataset_path, episode_indices=episode_indices)
        self._current_replay_idx = 0
        self._current_actions: Optional[Any] = None
        self._current_action_step = 0
        self._current_init_state: Optional[dict] = None
        if len(self._replay_loader) > 0:
            ep = self._replay_loader.get_episode(0)
            self._current_actions = ep.actions
            self._current_init_state = ep.init_state
            logger.info(f"Episode {ep.episode_idx}: {len(self._current_actions)} actions loaded.")
        self.reset_needed = True
        self._is_initial_replay_reset = True

    # ── Unified 11-dim helpers ───────────────────────────────────────────

    def _state11(self) -> np.ndarray:
        """Current 11-dim unified state: base(3) + arm(7) + finger1 position."""
        jp = self.all_subset.get_joint_positions()
        return np.asarray(jp[:11], dtype=np.float32)

    def _gripper_closedness(self) -> float:
        """Normalized closedness [0,1] from the measured finger opening."""
        width = float(np.clip(self.all_subset.get_joint_positions()[10],
                              0.0, GRIPPER_MAX_OPEN))
        return float(np.clip(1.0 - width / GRIPPER_MAX_OPEN, 0.0, 1.0))

    def _record_step(self, state: Dict[str, Any], action11: np.ndarray, phase: int) -> None:
        """Cache one collect-mode step in the unified layout."""
        if "camera_data" not in state:
            return
        self.data_collector.cache_step(
            camera_images=state["camera_data"],
            joint_angles=self._state11(),
            action=np.asarray(action11, dtype=np.float32),
            language_instruction=self.get_language_instruction(),
            phase=phase,
        )

    def _arm_record_to_11(self, record8: np.ndarray) -> np.ndarray:
        """Prepend the current base pose to an atomic 8-dim record array."""
        base = np.asarray(self.base_subset.get_joint_positions(), dtype=np.float32)
        rec = np.asarray(record8, dtype=np.float32)
        return np.concatenate([base, rec[:7], rec[7:8]]).astype(np.float32)

    # ── Navigation helpers ───────────────────────────────────────────────

    def _ensure_waypoints(self, state: Dict[str, Any], waypoints_key: str = "waypoints") -> None:
        """Feed the task's planned waypoints to the base controller once."""
        if self.waypoints_set or state.get(waypoints_key) is None:
            return
        final_angle = state.get("final_nav_angle", np.pi / 2)
        self.ridgebase_controller.set_waypoints(state[waypoints_key], final_angle)
        self.waypoints_set = True

    def _nav_step(self, state: Dict[str, Any]) -> Tuple[Optional[ArticulationAction], bool, np.ndarray]:
        """One base-motion step toward the current waypoints.

        Returns:
            (action, nav_done, action11) — action11 is the unified record
            vector: commanded base target + held arm pose + gripper channel.
        """
        action, done = self.ridgebase_controller.get_action(state["current_pose"])
        target_base = np.asarray(action.joint_positions, dtype=np.float32)
        arm = np.asarray(self.franka_subset.get_joint_positions()[:7], dtype=np.float32)
        action11 = np.concatenate(
            [target_base, arm, [self._gripper_closedness()]]).astype(np.float32)
        done = bool(done) or self.ridgebase_controller.is_path_complete()
        return action, done, action11

    def _grip_distance(self, pick_controller: Any, object_name: str) -> float:
        """Atomic grip-distance table value with the mobile squeeze margin."""
        return max(0.0, float(pick_controller.get_gripper_distance(object_name)) - GRIP_SQUEEZE)

    def _track_pick_ee(self) -> None:
        """Track the lowest EE height reached during the pick phase — the
        discriminator between 'closed in the air above the object' (min_ee_z
        high) and 'reached grasp depth but the object escaped' (min_ee_z low)."""
        ee_z = float(self.robot.get_gripper_position()[2])
        self._pick_min_ee_z = min(getattr(self, "_pick_min_ee_z", 1e9), ee_z)

    def _log_pick_fail_diag(self, state: Dict[str, Any]) -> None:
        """On a lift failure, log EE-vs-object geometry to discriminate the
        failure mode: EE far from the object = reach/IK undershoot; EE at the
        object with drift = knocked it over; EE at a still object = grip slip."""
        ee = np.asarray(self.robot.get_gripper_position(), dtype=float)
        obj = np.asarray(state["object_position"], dtype=float)
        init = np.asarray(state.get("initial_object_position", obj), dtype=float)
        logger.warning(f"[PICK-FAIL-DIAG] ee={np.round(ee, 3).tolist()} "
                       f"obj={np.round(obj, 3).tolist()} "
                       f"ee_obj_xy={float(np.linalg.norm(ee[:2] - obj[:2])):.3f} "
                       f"obj_drift={np.round(obj - init, 3).tolist()} "
                       f"closedness={self._gripper_closedness():.2f} "
                       f"min_ee_z={getattr(self, '_pick_min_ee_z', float('nan')):.3f}")

    def _log_dock_diag(self, state: Dict[str, Any], dock: Any, label: str = "DOCK-DIAG") -> None:
        """Log the parked base pose vs its dock target at a nav-phase end.

        Docking error correlates directly with grasp/place success (the arm
        works at the edge of its reach envelope), so every nav completion
        reports it.
        """
        base = self._state11()
        pose = np.asarray(state["current_pose"], dtype=float)
        world_xy = pose[:2] + base[:2]
        heading = float((pose[2] + base[2] + np.pi) % (2 * np.pi) - np.pi)
        dock = np.asarray(dock, dtype=float)
        obj = np.asarray(state["object_position"], dtype=float)
        logger.info(f"[{label}] parked at {np.round(world_xy, 3).tolist()} "
                    f"heading={np.degrees(heading):.1f}deg "
                    f"dock_err={np.round(world_xy - dock[:2], 3).tolist()} "
                    f"obj={np.round(obj, 3).tolist()} "
                    f"reach_xy={float(np.linalg.norm(obj[:2] - world_xy)):.3f}")

    # ── Arm-phase helpers ────────────────────────────────────────────────

    def _sync_arm_base_pose(self) -> np.ndarray:
        """Update RMPFlow with the (moved) arm base pose; returns its position."""
        base_link = self.robot.prim_path_str + "/panda_link0"
        pose = self.object_utils.get_object_xform_position(object_path=base_link)
        quat = self.object_utils.get_transform_quat(object_path=base_link, w_first=True)
        self.rmp_controller.rmp_flow.set_robot_base_pose(pose, quat)
        return pose

    def _remap_arm_action(self, action: Optional[ArticulationAction]) -> Optional[ArticulationAction]:
        """Re-index an arm-subset action onto the full articulation."""
        if action is None or action.joint_positions is None:
            return None
        return ArticulationAction(
            joint_positions=action.joint_positions,
            joint_velocities=action.joint_velocities,
            joint_indices=self.franka_subset.joint_indices[:len(action.joint_positions)],
        )

    # ── Replay (mobile override) ─────────────────────────────────────────

    def _apply_action11(self, act: np.ndarray) -> ArticulationAction:
        """Convert a recorded 11-dim action to a 12-DOF position command."""
        act = np.asarray(act, dtype=np.float64)
        s = float(np.clip(act[10], 0.0, 1.0))
        finger = float(np.clip(GRIPPER_MAX_OPEN * (1.0 - s), 0.0, _FINGER_LIMIT))
        positions = np.concatenate([act[:10], [finger, finger]])
        return self.all_subset.make_articulation_action(
            joint_positions=positions, joint_velocities=None)

    def _log_replay_divergence(self, k: int) -> None:
        """Diagnostic: compare the live 11-dim state against the collect-time
        measured state (``agent_pose[k]``) so replay failures can be localized
        to a frame/channel instead of only observing the end result."""
        ep = self._replay_loader.get_episode(self._current_replay_idx)
        if ep.agent_pose is None or k >= len(ep.agent_pose):
            return
        diff = np.abs(self._state11() - np.asarray(ep.agent_pose[k], dtype=np.float32))
        base_err, arm_err = float(diff[:3].max()), float(diff[3:10].max())
        first_exceed = (not getattr(self, "_replay_div_warned", False)
                        and (base_err > 0.03 or arm_err > 0.10))
        if first_exceed:
            self._replay_div_warned = True
        if first_exceed or k % 240 == 0:
            log = logger.warning if first_exceed else logger.info
            log(f"[Replay-DIV] frame {k}: base_err={np.round(diff[:3], 4).tolist()} "
                f"arm_err_max={arm_err:.4f} grip_meas={self._state11()[10]:.4f} "
                f"grip_rec={float(ep.agent_pose[k][10]):.4f}")

    def _step_replay(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Play back recorded 11-dim actions one per frame on all 12 DOFs."""
        action = None
        if (self._current_actions is not None
                and self._current_action_step < len(self._current_actions)):
            self._log_replay_divergence(self._current_action_step)
            action = self._apply_action11(self._current_actions[self._current_action_step])
            self._current_action_step += 1

        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        if self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS:
            self._last_failure_reason = ""
            self._last_success = True
            self.reset_needed = True
            self._advance_replay_episode()
            logger.success("[Replay] Task success!")
            return None, True, True

        actions_exhausted = (self._current_actions is None
                             or self._current_action_step >= len(self._current_actions))
        if actions_exhausted and action is None:
            settle_budget = max(self.REQUIRED_SUCCESS_STEPS * 4, 240)
            self._replay_settle_used = getattr(self, "_replay_settle_used", 0) + 1
            if self._replay_settle_used <= settle_budget:
                return None, False, False
            logger.warning("[Replay] Task failed — all actions exhausted.")
            self._replay_settle_used = 0
            self._advance_replay_episode()
            self.reset_needed = True
            return None, True, False

        return action, False, False

    def reset(self) -> None:
        super().reset()
        self.waypoints_set = False
        self._replay_div_warned = False
        self._pick_min_ee_z = 1e9
