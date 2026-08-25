from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .atomic_actions.shake_controller import ShakeController
from .mobile_base_controller import PHASE_PICK
from .mobile_pick_controller import MobilePickController

# Recorded phase id for the shake segment (navigate=0, pick=1 in the mobile
# layout). Kept distinct so the collector can slice the shake sub-trajectory.
PHASE_SHAKE = 4


class MobileShakeController(MobilePickController):
    """Level-5 mobile shake: navigate to the bench, pick the object, then shake it.

    Reuses :class:`MobilePickController`'s navigate + pick verbatim and appends a
    shake phase (:class:`ShakeController`) that oscillates the grasped object
    about its lifted world position. No new scene object — the same
    ``/World/beaker`` as ``mobile_pick`` (task_type stays ``mobile_pick``; only
    the controller differs).

    Phases (recorded): 0 navigate, 1 pick, 4 shake.
    Success: the object is lifted (pick) AND the shake sequence completes without
    the object being dropped.
    """

    def __init__(self, cfg: Any, robot: Any) -> None:
        super().__init__(cfg, robot)
        self.pick_done = False
        shake_euler = getattr(cfg.task, "shake_ee_euler_deg", None)
        # Hold the grasp orientation through the shake by default — the beaker is
        # side-grasped, so ShakeController's top-down default would wrench it.
        self._shake_orientation = (
            R.from_euler("xyz", np.radians([float(v) for v in shake_euler])).as_quat()
            if shake_euler is not None else self._grasp_orientation.copy())

    def _init_collect_mode(self, cfg: Any, robot: Any = None) -> None:
        super()._init_collect_mode(cfg, robot)
        self.shake_controller = ShakeController(
            name="shake_controller",
            cspace_controller=self.rmp_controller,
            shake_distance=float(getattr(cfg.task, "shake_distance", 0.1)),
        )

    def reset(self) -> None:
        super().reset()
        self.pick_done = False
        if self.mode == "collect":
            self.shake_controller.reset()

    # ── Success ──────────────────────────────────────────────────────────

    def _check_success(self) -> bool:
        lifted = super()._check_success()
        if self.mode != "collect":
            return lifted  # infer/replay success = still lifted (no phase machine)
        return lifted and self.shake_controller.is_done()

    # ── Collect ──────────────────────────────────────────────────────────

    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if not self.navigation_done:
            return self._navigation_phase(state)
        if not self.pick_done:
            return self._pick_then_shake(state)
        return self._shake_phase(state)

    def _pick_then_shake(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Parent pick, but on a successful lift transition to shake instead of
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

        # Atomic pick finished — require a successful lift before shaking.
        if not super()._check_success():
            self._log_pick_fail_diag(state)
            self._last_failure_reason = (
                f"Pick failed: object not lifted more than {self.LIFT_THRESHOLD} m")
            self.data_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False

        # Anchor the shake about the lifted object's WORLD position: ShakeController
        # targets are world-frame and its default [0.25,0,1.0] assumes a
        # base-at-origin arm, which the docked mobile base is not.
        self.shake_controller._initial_position = np.asarray(
            self.robot.get_gripper_position(), dtype=np.float64)
        logger.info("Pick OK — starting shake phase")
        self.pick_done = True
        return None, False, False

    def _shake_phase(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self._sync_arm_base_pose()
        if not self.shake_controller.is_done():
            action, record8 = self.shake_controller.forward(
                current_joint_positions=self.franka_subset.get_joint_positions(),
                end_effector_orientation=self._shake_orientation.copy(),
            )
            self._record_step(state, self._arm_record_to_11(record8), PHASE_SHAKE)
            return self._remap_arm_action(action), False, False

        # Shake done — success if the object is still held aloft (not dropped).
        held = super()._check_success()
        if held:
            logger.info("Shake complete — object still held; success")
            self._last_failure_reason = ""
            self.data_collector.write_cached_data()
            self._last_success = True
        else:
            self._last_failure_reason = "Shake failed: object dropped during shake"
            self.data_collector.clear_cache()
            self._last_success = False
        self.reset_needed = True
        return None, True, self._last_success

    # ── Language ─────────────────────────────────────────────────────────

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state["object_name"]) if self.state else "object"
        if not self.navigation_done:
            return self._get_cached_instruction(
                "mobile_shake:navigate",
                self._build_instruction_templates(
                    f"Move to the bench and pick up the {object_name}",
                    f"Drive to the lab bench, stop in front of it, and pick up the {object_name}",
                ),
            )
        if not self.pick_done:
            return self._get_cached_instruction(
                "mobile_shake:pick",
                self._build_instruction_templates(
                    f"Pick up the {object_name}",
                    f"Pick up the {object_name} from the bench and lift it clear of the surface",
                ),
            )
        return self._get_cached_instruction(
            "mobile_shake:shake",
            self._build_instruction_templates(
                f"Shake the {object_name}",
                f"Shake the {object_name} back and forth to mix its contents",
            ),
        )
