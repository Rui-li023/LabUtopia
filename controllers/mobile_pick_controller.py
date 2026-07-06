from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .mobile_base_controller import (
    PHASE_NAVIGATE,
    PHASE_PICK,
    MobileManipControllerBase,
)


class MobilePickController(MobileManipControllerBase):
    """Level-5 mobile pick: navigate to the bench dock, then pick the object.

    Phases (recorded per step): 0 = navigate, 1 = pick.
    Success: object lifted more than LIFT_THRESHOLD above its initial height.
    """

    LIFT_THRESHOLD = 0.10

    def __init__(self, cfg: Any, robot: Any) -> None:
        super().__init__(cfg, robot)
        self.navigation_done = False
        grasp_euler = getattr(cfg.task, "grasp_ee_euler_deg", [-90, 90, 30])
        self._grasp_orientation = R.from_euler(
            "xyz", np.radians([float(v) for v in grasp_euler])).as_quat()

    def _init_collect_mode(self, cfg: Any, robot: Any = None) -> None:
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
        )

    def reset(self) -> None:
        super().reset()
        self.navigation_done = False
        if self.mode == "collect":
            self.pick_controller.reset()

    # ── Success ──────────────────────────────────────────────────────────

    def _check_success(self) -> bool:
        if self.state is None:
            return False
        obj = self.state.get("object_position")
        init = self.state.get("initial_object_position")
        if obj is None or init is None:
            return False
        return float(obj[2]) - float(init[2]) > self.LIFT_THRESHOLD

    # ── Collect ──────────────────────────────────────────────────────────

    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if not self.navigation_done:
            return self._navigation_phase(state)
        return self._pick_phase(state)

    def _navigation_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if not self.waypoints_set:
            self._ensure_waypoints(state)
            if self.waypoints_set:
                spawn = getattr(self.cfg.task, "spawn", None)
                self.data_collector.set_task_properties({
                    "start_position": [float(v) for v in state["current_pose"]],
                    "dock_point": [float(v) for v in state["dock_point"]],
                    "object_name": state.get("object_name", "unknown"),
                    "spawn_mode": str(getattr(spawn, "mode", "far")) if spawn is not None else "far",
                })
        action, nav_done, action11 = self._nav_step(state)
        self._record_step(state, action11, PHASE_NAVIGATE)
        if nav_done:
            self._log_dock_diag(state, state["dock_point"])
            logger.info("Navigation complete — starting pick phase")
            self.navigation_done = True
        return action, False, False

    def _pick_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        base_pos = self._sync_arm_base_pose()
        self.pick_controller.set_robot_position(base_pos)

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
                gripper_distances=self.pick_controller.get_gripper_distance(state["object_name"]),
            )
            self._record_step(state, self._arm_record_to_11(record8), PHASE_PICK)
            return self._remap_arm_action(action), False, False

        # Atomic pick finished: evaluate the lift.
        lifted = self._check_success()
        if lifted:
            self._last_failure_reason = ""
            self.data_collector.write_cached_data()
            self._last_success = True
        else:
            self._log_pick_fail_diag(state)
            self._last_failure_reason = (
                f"Pick failed: object not lifted more than {self.LIFT_THRESHOLD} m")
            self.data_collector.clear_cache()
            self._last_success = False
        self.reset_needed = True
        return None, True, self._last_success

    # ── Infer ────────────────────────────────────────────────────────────

    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Scripted navigation (stand-in for an external nav model) + VLA arm."""
        if not self.navigation_done:
            self._ensure_waypoints(state)
            action, nav_done, _ = self._nav_step(state)
            if nav_done:
                logger.info("[infer] Navigation complete — handing over to policy")
                self.navigation_done = True
            return action, False, False

        self._sync_arm_base_pose()
        state["language_instruction"] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        if isinstance(action, ArticulationAction):
            action = self._remap_arm_action(action)

        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        if self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS:
            self._last_success = True
            self.reset_needed = True
            return None, True, True
        return action, False, False

    # ── Language ─────────────────────────────────────────────────────────

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state["object_name"]) if self.state else "object"
        if not self.navigation_done:
            return self._get_cached_instruction(
                "mobile_pick:navigate",
                self._build_instruction_templates(
                    f"Move to the bench and pick up the {object_name}",
                    f"Drive to the lab bench, stop in front of it, and pick up the {object_name}",
                ),
            )
        return self._get_cached_instruction(
            "mobile_pick:pick",
            self._build_instruction_templates(
                f"Pick up the {object_name}",
                f"Pick up the {object_name} from the bench and lift it clear of the surface",
            ),
        )
