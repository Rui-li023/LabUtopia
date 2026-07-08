from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .mobile_base_controller import (
    PHASE_CARRY_NAVIGATE,
    PHASE_NAVIGATE,
    PHASE_PICK,
    PHASE_PLACE,
    MobileManipControllerBase,
)


class Phase(Enum):
    NAV_A = "nav_a"
    PICKING = "picking"
    CARRY_NAV = "carry_nav"
    PLACING = "placing"
    FINISHED = "finished"


class MobileTransportPlaceController(MobileManipControllerBase):
    """Level-5 transport-place: navigate, pick the source beaker, optionally
    carry-navigate to a second bench, and place it on the target platform.

    Place gate (position-based, mirrors PickPlaceTaskController PLACING
    branch): the beaker's xy is within PLACE_XY_THRESHOLD of the platform
    geometry center and it has settled back within SETTLE_Z of its original
    resting height. Evaluated statefully so it also validates in replay.
    """

    LIFT_THRESHOLD = 0.10
    DROP_MARGIN = 0.04          # carry-phase drop guard above the initial height
    PLACE_XY_THRESHOLD = 0.05   # planar tolerance to the platform center
    SETTLE_Z = 0.06             # beaker z back near its resting height
    RELEASE_CLOSEDNESS = 0.45   # measured closedness below this = gripper released
                                # (beaker-grip ~0.56, fully open ~0.30)
    CARRY_HOLD_MAX_DIST = 2.0   # carries shorter than this crab (hold heading);
                                # longer ones (far cross-lab) face travel dir
    PLACE_SETTLE_BUDGET = 120   # frames the beaker may settle after the place
                                # motion completes before the episode fails

    def __init__(self, cfg: Any, robot: Any) -> None:
        super().__init__(cfg, robot)
        self.current_phase = Phase.NAV_A
        self.initial_object_z: Optional[float] = None
        self.carry_waypoints_set = False
        self._place_settle_frames = 0
        self._plat_logged = False
        grasp_euler = getattr(cfg.task, "grasp_ee_euler_deg", [-90, 90, 30])
        self._grasp_orientation = R.from_euler(
            "xyz", np.radians([float(v) for v in grasp_euler])).as_quat()
        place_euler = getattr(cfg.task, "place_ee_euler_deg", [0, 90, 20])
        self._place_orientation = R.from_euler(
            "xyz", np.radians([float(v) for v in place_euler])).as_quat()

    def _init_collect_mode(self, cfg: Any, robot: Any = None) -> None:
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
        )
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )

    def reset(self) -> None:
        super().reset()
        self.current_phase = Phase.NAV_A
        self.initial_object_z = None
        self.carry_waypoints_set = False
        self._place_settle_frames = 0
        if self.mode == "collect":
            self.pick_controller.reset()
            self.place_controller.reset()

    # ── Success (used by replay & infer via the base counter) ───────────

    def _check_success(self) -> bool:
        if self.state is None:
            return False
        return self._place_gate_satisfied()

    def _place_gate_satisfied(self) -> bool:
        """Position-based place gate, evaluated statefully across steps.

        Requires the gripper to have RELEASED the beaker: without this the
        gate fires while the beaker is still held mid-descent, ending collect
        episodes before the release/retreat motion is recorded (and making
        replay flaky on the borderline hover height).
        """
        obj = self.state.get("object_position")
        init = self.state.get("initial_object_position")
        target = self.state.get("place_target_position")
        if obj is None or init is None or target is None:
            return False
        if self._gripper_closedness() > self.RELEASE_CLOSEDNESS:
            return False
        xy_dist = float(np.linalg.norm(np.asarray(obj[:2]) - np.asarray(target[:2])))
        if xy_dist > self.PLACE_XY_THRESHOLD:
            return False
        return abs(float(obj[2]) - float(init[2])) < self.SETTLE_Z

    # ── Collect ──────────────────────────────────────────────────────────

    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.initial_object_z is None and state.get("object_position") is not None:
            self.initial_object_z = float(state["object_position"][2])

        if self.current_phase == Phase.NAV_A:
            return self._nav_a_phase(state)
        if self.current_phase == Phase.PICKING:
            return self._pick_phase(state)
        if self.current_phase == Phase.CARRY_NAV:
            return self._carry_phase(state)
        if self.current_phase == Phase.PLACING:
            return self._place_phase(state)
        self.reset_needed = True
        return None, True, self._last_success

    def _fail(self, reason: str) -> Tuple[Any, bool, bool]:
        self._last_failure_reason = reason
        logger.warning(reason)
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        self.reset_needed = True
        return None, True, False

    def _nav_a_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if not self.waypoints_set:
            self._ensure_waypoints(state)
            if self.waypoints_set:
                self.data_collector.set_task_properties({
                    "start_position": [float(v) for v in state["current_pose"]],
                    "dock_point": [float(v) for v in state["dock_point"]],
                    "place_dock": [float(v) for v in state["place_dock"]],
                    "object_name": state.get("object_name", "unknown"),
                    "carry_navigation": bool(state.get("carry_navigation", False)),
                })
        action, nav_done, action11 = self._nav_step(state)
        self._record_step(state, action11, PHASE_NAVIGATE)
        if nav_done:
            self._log_dock_diag(state, state["dock_point"], label="DOCK-A")
            logger.info("Navigation to bench A complete — starting pick")
            self.current_phase = Phase.PICKING
        return action, False, False

    def _pick_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
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

        lifted = (self.initial_object_z is not None
                  and float(state["object_position"][2]) - self.initial_object_z > self.LIFT_THRESHOLD)
        if not lifted:
            self._log_pick_fail_diag(state)
            return self._fail("TransportPlace pick failed: object not lifted")
        logger.info(f"[PICK-OK] min_ee_z={getattr(self, '_pick_min_ee_z', float('nan')):.3f}")
        if state.get("carry_navigation", False):
            logger.info("Pick complete — carry-navigating to bench B")
            self.current_phase = Phase.CARRY_NAV
        else:
            logger.info("Pick complete — placing on the same bench")
            self.current_phase = Phase.PLACING
        return None, False, False

    def _carry_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        # Drop guard: the beaker must stay lifted while the base moves.
        if (self.initial_object_z is not None
                and float(state["object_position"][2]) < self.initial_object_z + self.DROP_MARGIN):
            return self._fail("TransportPlace carry failed: beaker dropped during navigation")
        if not self.carry_waypoints_set:
            if state.get("carry_waypoints") is None:
                return self._fail("TransportPlace carry failed: no carry path available")
            final_angle = state.get("final_nav_angle", np.pi / 2)
            # Short same-bench carry -> crab sideways holding the bench heading
            # (no reorientation: kills the overshoot and the carry-time drop).
            # Long cross-lab carry -> face the travel direction as before.
            wp = np.asarray(state["carry_waypoints"], dtype=float)
            carry_len = float(np.linalg.norm(np.diff(wp[:, :2], axis=0), axis=1).sum()) if len(wp) > 1 else 0.0
            hold = carry_len < self.CARRY_HOLD_MAX_DIST
            self.ridgebase_controller.set_waypoints(
                state["carry_waypoints"], final_angle, hold_heading=hold)
            logger.info(f"[carry] len={carry_len:.2f}m hold_heading={hold}")
            self.carry_waypoints_set = True
        action, nav_done, action11 = self._nav_step(state)
        self._record_step(state, action11, PHASE_CARRY_NAVIGATE)
        if nav_done:
            self._log_dock_diag(state, state["place_dock"], label="DOCK-B")
            logger.info("Carry navigation complete — starting place")
            self.current_phase = Phase.PLACING
        return action, False, False

    def _place_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self._sync_arm_base_pose()
        if not self._plat_logged and state.get("place_target_position") is not None:
            logger.info(f"[place] target platform center = "
                        f"{np.asarray(state['place_target_position'])}, "
                        f"beaker rest z = {self.initial_object_z}")
            self._plat_logged = True

        if not self.place_controller.is_done():
            action, record8 = self.place_controller.forward(
                place_position=np.asarray(state["place_target_position"], dtype=float).copy(),
                current_joint_positions=self.franka_subset.get_joint_positions(),
                gripper_control=self.gripper_control,
                end_effector_orientation=self._place_orientation.copy(),
                gripper_position=self.robot.get_gripper_position(),
            )
            self._record_step(state, self._arm_record_to_11(record8), PHASE_PLACE)
            return self._remap_arm_action(action), False, False

        # The full place motion (lower -> release -> retreat) is recorded; only
        # now may the episode succeed, once the freed beaker settles on the plat.
        if self._place_gate_satisfied():
            self._last_failure_reason = ""
            logger.success("Place gate satisfied — episode success")
            self.data_collector.write_cached_data()
            self._last_success = True
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            return None, True, True

        self._place_settle_frames += 1
        if self._place_settle_frames <= self.PLACE_SETTLE_BUDGET:
            return None, False, False
        return self._fail("TransportPlace place failed: place controller finished without meeting the gate")

    # ── Infer ────────────────────────────────────────────────────────────

    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Scripted navigation phases + VLA manipulation, with oracle
        phase-advance so evals report per-phase progress."""
        if self.initial_object_z is None and state.get("object_position") is not None:
            self.initial_object_z = float(state["object_position"][2])

        if self.current_phase == Phase.NAV_A:
            self._ensure_waypoints(state)
            action, nav_done, _ = self._nav_step(state)
            if nav_done:
                self.current_phase = Phase.PICKING
            return action, False, False

        if self.current_phase == Phase.CARRY_NAV:
            if not self.carry_waypoints_set and state.get("carry_waypoints") is not None:
                self.ridgebase_controller.set_waypoints(
                    state["carry_waypoints"], state.get("final_nav_angle", np.pi / 2))
                self.carry_waypoints_set = True
            action, nav_done, _ = self._nav_step(state)
            if nav_done:
                self.current_phase = Phase.PLACING
            return action, False, False

        # Manipulation phases: policy-driven.
        self._sync_arm_base_pose()
        state["language_instruction"] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        if isinstance(action, ArticulationAction):
            action = self._remap_arm_action(action)

        # Oracle phase-advance: report per-phase progress.
        if self.current_phase == Phase.PICKING:
            lifted = (self.initial_object_z is not None
                      and state.get("object_position") is not None
                      and float(state["object_position"][2]) - self.initial_object_z > self.LIFT_THRESHOLD)
            if lifted:
                logger.info("[infer] pick phase passed")
                self.current_phase = (Phase.CARRY_NAV if state.get("carry_navigation", False)
                                      else Phase.PLACING)
            return action, False, False

        if self._place_gate_satisfied():
            self._last_success = True
            self.reset_needed = True
            return None, True, True
        return action, False, False

    # ── Language ─────────────────────────────────────────────────────────

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state["object_name"]) if self.state else "beaker"
        if self.current_phase == Phase.NAV_A:
            return self._get_cached_instruction(
                "mobile_tp:nav_a",
                self._build_instruction_templates(
                    f"Move to the bench and pick up the {object_name}",
                    f"Drive to the lab bench and stop in front of the {object_name}",
                ),
            )
        if self.current_phase == Phase.PICKING:
            return self._get_cached_instruction(
                "mobile_tp:pick",
                self._build_instruction_templates(
                    f"Pick up the {object_name}",
                    f"Pick up the {object_name} from the bench and lift it up",
                ),
            )
        if self.current_phase == Phase.CARRY_NAV:
            return self._get_cached_instruction(
                "mobile_tp:carry",
                self._build_instruction_templates(
                    f"Carry the {object_name} to the other bench",
                    f"Hold the {object_name} steady and drive to the second bench",
                ),
            )
        return self._get_cached_instruction(
            "mobile_tp:place",
            self._build_instruction_templates(
                f"Place the {object_name} on the target platform",
                f"Lower the {object_name} onto the target platform and release it",
            ),
        )
