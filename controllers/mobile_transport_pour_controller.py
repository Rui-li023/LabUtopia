from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from loguru import logger
from scipy.spatial.transform import Rotation as R

from utils.task_utils import TaskUtils

from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .mobile_base_controller import (
    PHASE_CARRY_NAVIGATE,
    PHASE_NAVIGATE,
    PHASE_PICK,
    PHASE_POUR,
    MobileManipControllerBase,
)


class Phase(Enum):
    NAV_A = "nav_a"
    PICKING = "picking"
    CARRY_NAV = "carry_nav"
    POURING = "pouring"
    FINISHED = "finished"


class MobileTransportPourController(MobileManipControllerBase):
    """Level-5 transport-pour: navigate, pick the source beaker, optionally
    carry-navigate to a second bench, and pour into the target container.

    Pour gate (rotation-based, mirrors PickPourTaskController): source within
    the pour threshold of the target (xy), tilted >= 50 degrees, returned to
    within 30 degrees of upright, then held lifted for the success timer.
    """

    LIFT_THRESHOLD = 0.10
    DROP_MARGIN = 0.04  # carry-phase drop guard above the initial height

    def __init__(self, cfg: Any, robot: Any) -> None:
        super().__init__(cfg, robot)
        self.task_utils = TaskUtils.get_instance()
        self.current_phase = Phase.NAV_A
        self.initial_object_z: Optional[float] = None
        self.initial_object_size: Optional[np.ndarray] = None
        self.initial_quaternion: Optional[np.ndarray] = None
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0.0
        self.carry_waypoints_set = False
        grasp_euler = getattr(cfg.task, "grasp_ee_euler_deg", [-90, 90, 30])
        self._grasp_orientation = R.from_euler(
            "xyz", np.radians([float(v) for v in grasp_euler])).as_quat()
        pour_euler = getattr(cfg.task, "pour_ee_euler_deg", [0, 90, 15])
        self._pour_orientation = R.from_euler(
            "xyz", np.radians([float(v) for v in pour_euler])).as_quat()

    def _init_collect_mode(self, cfg: Any, robot: Any = None) -> None:
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
        )
        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.003, 0.01, 0.01, 0.01, 0.02],
            position_pour=True,
        )

    def reset(self) -> None:
        super().reset()
        self.current_phase = Phase.NAV_A
        self.initial_object_z = None
        self.initial_object_size = None
        self.initial_quaternion = None
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0.0
        self.carry_waypoints_set = False
        if self.mode == "collect":
            self.pick_controller.reset()
            self.pour_controller.reset()

    # ── Success (used by replay & infer via the base counter) ───────────

    def _check_success(self) -> bool:
        if self.state is None:
            return False
        return self._pour_gate_satisfied()

    def _pour_gate_satisfied(self) -> bool:
        """Rotation-based pour gate, evaluated statefully across steps."""
        obj = self.state.get("object_position")
        init = self.state.get("initial_object_position")
        target = self.state.get("pour_target_position")
        quat = self.state.get("object_quaternion")
        if obj is None or init is None or target is None or quat is None:
            return False
        if self.initial_quaternion is None:
            self.initial_quaternion = np.asarray(quat, dtype=float).copy()
            return False

        xy_dist = float(np.linalg.norm(np.asarray(obj[:2]) - np.asarray(target[:2])))
        threshold = self.task_utils.get_pour_threshold(
            self.state["object_name"], self.state["object_size"]) + 0.05
        if xy_dist > threshold:
            return False
        if not self.pour_complete:
            self.pour_complete = self.task_utils.check_rotation_angle(
                self.initial_quaternion, quat, threshold_degrees=50)
            return False
        if not self.return_complete:
            still_tilted = self.task_utils.check_rotation_angle(
                self.initial_quaternion, quat, threshold_degrees=30)
            if not still_tilted:
                self.return_complete = True
                self.return_timer = 0.0
            return False
        if float(obj[2]) > float(init[2]) + 0.05:
            self.return_timer += 0.012
            return self.return_timer >= 1.0
        return False

    # ── Collect ──────────────────────────────────────────────────────────

    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.initial_object_z is None and state.get("object_position") is not None:
            self.initial_object_z = float(state["object_position"][2])
        if self.initial_object_size is None and state.get("object_size") is not None:
            self.initial_object_size = np.asarray(state["object_size"], dtype=float).copy()

        if self.current_phase == Phase.NAV_A:
            return self._nav_a_phase(state)
        if self.current_phase == Phase.PICKING:
            return self._pick_phase(state)
        if self.current_phase == Phase.CARRY_NAV:
            return self._carry_phase(state)
        if self.current_phase == Phase.POURING:
            return self._pour_phase(state)
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
                    "pour_dock": [float(v) for v in state["pour_dock"]],
                    "object_name": state.get("object_name", "unknown"),
                    "carry_navigation": bool(state.get("carry_navigation", False)),
                })
        action, nav_done, action11 = self._nav_step(state)
        self._record_step(state, action11, PHASE_NAVIGATE)
        if nav_done:
            logger.info("Navigation to bench A complete — starting pick")
            self.current_phase = Phase.PICKING
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

        lifted = (self.initial_object_z is not None
                  and float(state["object_position"][2]) - self.initial_object_z > self.LIFT_THRESHOLD)
        if not lifted:
            return self._fail("TransportPour pick failed: object not lifted")
        if state.get("carry_navigation", False):
            logger.info("Pick complete — carry-navigating to bench B")
            self.current_phase = Phase.CARRY_NAV
        else:
            logger.info("Pick complete — pouring on the same bench")
            self.current_phase = Phase.POURING
        return None, False, False

    def _carry_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        # Drop guard: the beaker must stay lifted while the base moves.
        if (self.initial_object_z is not None
                and float(state["object_position"][2]) < self.initial_object_z + self.DROP_MARGIN):
            return self._fail("TransportPour carry failed: beaker dropped during navigation")
        if not self.carry_waypoints_set:
            if state.get("carry_waypoints") is None:
                return self._fail("TransportPour carry failed: no carry path available")
            final_angle = state.get("final_nav_angle", np.pi / 2)
            self.ridgebase_controller.set_waypoints(state["carry_waypoints"], final_angle)
            self.carry_waypoints_set = True
        action, nav_done, action11 = self._nav_step(state)
        self._record_step(state, action11, PHASE_CARRY_NAVIGATE)
        if nav_done:
            logger.info("Carry navigation complete — starting pour")
            self.current_phase = Phase.POURING
        return action, False, False

    def _pour_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self._sync_arm_base_pose()
        # Resolve the wrist articulation index once (subset indices need an
        # initialized articulation, so this can't happen in __init__).
        self.pour_controller.wrist_dof_index = int(self.franka_subset.joint_indices[6])

        if self._pour_gate_satisfied():
            self._last_failure_reason = ""
            logger.success("Pour gate satisfied — episode success")
            self.data_collector.write_cached_data()
            self._last_success = True
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            return None, True, True

        if not self.pour_controller.is_done():
            action, record8 = self.pour_controller.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                source_size=self.initial_object_size,
                target_position=np.asarray(state["pour_target_position"], dtype=float).copy(),
                current_joint_velocities=self.franka_subset.get_joint_velocities(),
                pour_speed=-1,
                source_name=state["object_name"],
                gripper_position=self.robot.get_gripper_position(),
                current_joint_positions=self.franka_subset.get_joint_positions(),
                target_end_effector_orientation=self._pour_orientation.copy(),
            )
            if record8 is not None:
                self._record_step(state, self._arm_record_to_11(record8), PHASE_POUR)
            return self._remap_arm_action(action), False, False

        return self._fail("TransportPour pour failed: pour controller finished without meeting the gate")

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
                self.current_phase = Phase.POURING
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
                                      else Phase.POURING)
            return action, False, False

        if self._pour_gate_satisfied():
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
            "mobile_tp:pour",
            self._build_instruction_templates(
                f"Pour the {object_name} into the large beaker",
                f"Move the {object_name} over the large beaker and pour its contents into it",
            ),
        )
