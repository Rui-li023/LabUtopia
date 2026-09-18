import random
from enum import Enum

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController


class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"


class FlaskToCorkTaskController(BaseController):
    """Controller for the Level-2 flask-to-cork-ring pick-and-place task.

    Phase 1: grasp the round-bottom flask at its neck.
    Phase 2: lower the flask onto its target support.
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.current_phase = Phase.PICKING
        self.last_error_info = None
        self._phase_instructions: dict[Phase, str] = {}
        self._phase_task_indices: dict[Phase, int] = {}

        # Per-finger contact target, matched to the configured neck/stopper.
        task_cfg = getattr(cfg, "task", None)
        self._gripper_distance = float(getattr(task_cfg, "gripper_distance", 0.010))
        self._grasp_offset = None
        self._previous_object_position = None
        self._settle_steps = 0
        self._placement_steps = 0
        self._placement_rotation = R.identity()
        self._placement_hand_orientation = None

    # ------------------------------------------------------------------ setup

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.003, 0.02, 0.025, 0.003, 0.02],
        )
        self.pick_controller.orientation_noise_deg = 0.0
        self.pick_controller.lock_final_approach_target = True
        self.pick_controller.adaptive_phase_completion = True
        self.pick_controller.grasp_position_threshold = 0.006
        if bool(getattr(cfg.task, "original_flask_neck_grasp", False)):
            # The original mesh has a 31-33 mm straight neck. Keep contact below
            # its flared mouth, and wait for a settled, nonempty physical grasp.
            self.pick_controller.pick_z_offset_override = 0.0
            self.pick_controller.hold_pose_during_close = True
            self.pick_controller.arrival_stable_steps = 5
            self.pick_controller.close_contact_gap_range = (0.028, 0.038)
            self.pick_controller.close_stable_steps = 15
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
            precision=True,
        )
        self.active_controller = self.pick_controller

    # ------------------------------------------------------------------ reset

    def reset(self):
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.last_error_info = None
        self._phase_instructions = {}
        self._phase_task_indices = {}
        self._grasp_offset = None
        self._previous_object_position = None
        self._settle_steps = 0
        self._placement_steps = 0
        self._placement_rotation = R.identity()
        self._placement_hand_orientation = None

        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.place_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()

    # ---------------------------------------------------------- success checks

    def _check_success(self) -> bool:
        # In replay the phase machine (which only advances in _step_collect) is
        # frozen at PICKING, so a phase-based check would score the recorded
        # pick+place trajectory against "is the flask lifted" — which is false
        # once the flask has been set down at the end of the episode. Score
        # replay against the final task goal instead: flask resting on target,
        # using the same thresholds as the PLACING phase check.
        return self._check_placed_on_target()

    def _check_placed_on_target(self) -> bool:
        object_pos = self.state["object_position"]
        target_pos = self._goal_position(self.state)
        if target_pos is None:
            self._last_failure_reason = "Flask asset is missing the target FlaskSeat marker"
            return False
        xy_dist = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
        z_error = float(abs(object_pos[2] - target_pos[2]))
        previous = self._previous_object_position
        motion = float(np.linalg.norm(object_pos - previous)) if previous is not None else float("inf")
        self._previous_object_position = np.array(object_pos, copy=True)
        upright = self._placement_orientation_ok(self.state)
        released = np.linalg.norm(self.state["gripper_position"] - object_pos) > 0.07
        z_tolerance = float(getattr(self.cfg.task, "placement_z_tolerance", 0.012))
        success = bool(
            self._placement_xy_ok(object_pos, target_pos)
            and z_error < z_tolerance
            and upright
            and released
            and motion < 0.001
        )
        self._last_failure_reason = (
            ""
            if success
            else f"placement: xy={xy_dist:.4f}, z={z_error:.4f}, delta={np.round(object_pos - target_pos, 4)}, "
            f"upright={upright}, released={released}, motion={motion:.4f}"
        )
        return success

    def _goal_position(self, state):
        return state.get("target_seat_position")

    def _placement_xy_ok(self, position, target) -> bool:
        tolerance = float(getattr(self.cfg.task, "placement_xy_tolerance", 0.015))
        return bool(np.linalg.norm(position[:2] - target[:2]) < tolerance)

    def _placement_orientation_ok(self, state) -> bool:
        up = np.asarray(state.get("object_up", [0, 0, 1]))
        tolerance = float(getattr(self.cfg.task, "placement_tilt_tolerance_deg", 25.0))
        return bool(up[2] / max(float(np.linalg.norm(up)), 1e-9) > np.cos(np.radians(tolerance)))

    def _capture_grasp_offset(self, state):
        up = np.asarray(state.get("object_up", [0, 0, 1]), dtype=float)
        up /= max(float(np.linalg.norm(up)), 1e-9)
        axis = np.cross(up, [0, 0, 1])
        sine = float(np.linalg.norm(axis))
        if sine > 1e-8:
            self._placement_rotation = R.from_rotvec(axis / sine * np.arctan2(sine, up[2]))
        else:
            self._placement_rotation = R.identity()
        offset = np.array(state["gripper_position"]) - state["object_position"]
        return self._placement_rotation.apply(offset)

    def _check_phase_success(self) -> bool:
        object_pos = self.state["object_position"]

        if self.current_phase == Phase.PICKING:
            # 5 cm lift is enough to confirm the flask cleared its support.
            required_height = self.initial_position[2] + 0.05
            success = object_pos[2] > required_height
            if not success:
                self.last_error_info = {
                    "phase": "PICKING",
                    "current_height": float(object_pos[2]),
                    "required_height": float(required_height),
                    "height_diff": float(object_pos[2] - required_height),
                }
            return success

        if self.current_phase == Phase.PLACING:
            return self._check_placed_on_target()

        return False

    # ---------------------------------------------------------- language utils

    def _object_display_name(self, path: str) -> str:
        return path.split("/")[-1].replace("_", " ")

    def _sample_phase_instruction(self, phase: Phase) -> str:
        source = self._object_display_name(self.cfg.task.obj_paths[0]["path"])
        target = self._object_display_name(self.cfg.task.obj_paths[1]["path"])
        support = self._object_display_name(self.cfg.task.obj_paths[2]["path"])
        if phase == Phase.PICKING:
            templates = self._build_instruction_templates(
                f"Pick up the {source} from the {support}",
                f"Pick up the {source} from the {support} and lift it clear of the surface",
            )
        else:
            templates = self._build_instruction_templates(
                f"Place the {source} on the {target}",
                f"Move the {source} to the {target} and set it down carefully",
            )
        return random.choice(templates)

    def get_language_instruction(self) -> str:
        if self.current_phase not in self._phase_instructions:
            self._phase_instructions[self.current_phase] = self._sample_phase_instruction(self.current_phase)
        self._language_instruction = self._phase_instructions[self.current_phase]
        return self._language_instruction

    def get_task_index(self) -> int | None:
        if self.mode != "collect" or self.current_phase == Phase.FINISHED:
            return None
        if self.current_phase not in self._phase_task_indices:
            instruction = self.get_language_instruction()
            self._phase_task_indices[self.current_phase] = self.data_collector.register_task_instruction(instruction)
        return self._phase_task_indices[self.current_phase]

    # -------------------------------------------------------------------- step

    def step(self, state):
        self.state = state
        if self.initial_position is None:
            self.initial_position = np.array(state["object_position"])
        if self.initial_size is None:
            self.initial_size = state["object_size"]

        if self.mode == "collect":
            return self._step_collect(state)
        if self.mode == "replay":
            # Replay is handled by BaseController (trajectory playback + success
            # check); this override only exists to seed initial_position/size.
            return self._step_replay(state)
        return self._step_infer(state)

    # ------------------------------------------------------- collect-mode step

    def _step_collect(self, state):
        success = self._check_phase_success()

        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action, record_array = self._forward_active(state)

            if "camera_data" in state:
                self.data_collector.cache_step(
                    camera_images=state["camera_data"],
                    joint_angles=state["joint_positions"][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction(),
                    task_index=self.get_task_index(),
                )
            return action, False, False

        failure = self.active_controller.get_failure_reason()
        if self.current_phase == Phase.PLACING and not failure:
            self._settle_steps += 1
            self._placement_steps = self._placement_steps + 1 if success else 0
            settle_timeout = int(getattr(self.cfg.task, "settle_timeout_steps", 180))
            if self._placement_steps < self.REQUIRED_SUCCESS_STEPS and self._settle_steps < settle_timeout:
                action, record_array = self._forward_active(state)
                if "camera_data" in state:
                    self.data_collector.cache_step(
                        camera_images=state["camera_data"],
                        joint_angles=state["joint_positions"][:-1],
                        action=record_array,
                        language_instruction=self.get_language_instruction(),
                        task_index=self.get_task_index(),
                    )
                return action, False, False
            success = self._placement_steps >= self.REQUIRED_SUCCESS_STEPS
            if not success and not self._last_failure_reason:
                self._last_failure_reason = f"Placement remained stable for {self._placement_steps}/{self.REQUIRED_SUCCESS_STEPS} required frames"

        success = success and not failure
        # active controller finished: evaluate success
        if success:
            if self.current_phase == Phase.PICKING:
                logger.info("Object picked; switching to placement")
                self._cache_hold_step(state)
                held_targets = self.pick_controller._last_record_positions
                if held_targets is not None:
                    self.place_controller._last_record_positions = held_targets.copy()
                self._grasp_offset = self._capture_grasp_offset(state)
                self.last_error_info = None
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False

            if self.current_phase == Phase.PLACING:
                return self._on_placement_complete(state)

        # controller done but success not achieved
        detail = failure or self._last_failure_reason or str(self.last_error_info or "")
        self._last_failure_reason = f"{type(self).__name__} {self.current_phase.value} phase failed: {detail}"
        logger.warning(self._last_failure_reason)
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _on_placement_complete(self, state):
        """Commit one complete task; sequences override the intermediate handoff."""
        logger.info("Object stable on target support")
        self._last_failure_reason = ""
        self.data_collector.write_cached_data(state["joint_positions"][:-1])
        self._last_success = True
        self.current_phase = Phase.FINISHED
        return None, True, True

    def _cache_hold_step(self, state) -> None:
        """Record the real physics frame spent switching atomic controllers.

        A None action holds the previous target, but omitting that frame from
        the trajectory advances replay one timestep early at each handoff.
        """
        if "camera_data" not in state:
            return
        atomic = self.active_controller
        action = atomic._null_action(len(state["joint_positions"]))
        record = atomic._build_record_array(action, state["joint_positions"])
        self.data_collector.cache_step(
            camera_images=state["camera_data"], joint_angles=state["joint_positions"][:-1],
            action=record, language_instruction=self.get_language_instruction(), task_index=self.get_task_index(),
        )

    def _forward_active(self, state):
        """Dispatch forward() call to the currently active atomic controller."""
        # Approach orientation: wrist vertical, gripper aligned for neck grasp
        ee_orient = R.from_euler("xyz", np.radians([0, 90, 30])).as_quat()

        if self.current_phase == Phase.PICKING:
            picking_position = state["object_position"] + np.array([0, 0, 0.02])
            if bool(getattr(self.cfg.task, "original_flask_neck_grasp", False)):
                picking_position = state.get("object_neck_position")
                if picking_position is None:
                    raise ValueError("Original flask neck grasp requires the USD NeckGrasp marker")
            return self.pick_controller.forward(
                picking_position=picking_position,
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=ee_orient,
                pre_offset_x=0.04,
                pre_offset_z=0.02,
                after_offset_z=0.14 if bool(getattr(self.cfg.task, "original_flask_neck_grasp", False)) else 0.25,
                # Round-bottom flask: close to the neck width (per-finger target)
                # instead of a full 0/1 close, which ejected the flask. Tuned via
                # cfg.task.gripper_distance (see __init__); default 0.010.
                gripper_distances=self._gripper_distance,
            )

        # PLACING
        # Upright a flask that tilted in its source support. These legacy task
        # quaternions are handed directly to Isaac as wxyz.
        actual_orientation = R.from_quat(ee_orient[[1, 2, 3, 0]])
        ee_orient = (self._placement_rotation * actual_orientation).as_quat()[[3, 0, 1, 2]]
        place_position = self._goal_position(state) + self._grasp_offset
        original_neck = bool(getattr(self.cfg.task, "original_flask_neck_grasp", False))
        if original_neck:
            place_position, ee_orient = self._original_flask_place_pose(state)
        return self.place_controller.forward(
            place_position=place_position,
            current_joint_positions=state["joint_positions"],
            gripper_control=self.gripper_control,
            end_effector_orientation=ee_orient,
            gripper_position=state["gripper_position"],
            place_offset_z=0.003,
            pre_place_z=0.10 if original_neck else 0.15,
        )

    def _original_flask_place_pose(self, state):
        """Align the measured flask before release; never track a released body."""
        if self.place_controller._event <= 1:
            offset = np.asarray(state["gripper_position"]) - state["object_position"]
            if not np.all(np.isfinite(offset)) or np.linalg.norm(offset) > 0.07:
                self.place_controller._phase_failure_reason = "Flask left the gripper before release; stopping pose feedback"
                self.place_controller._event = len(self.place_controller._events_dt)
                orientation = self._placement_hand_orientation
                if orientation is None:
                    orientation = np.array([1.0, 0.0, 0.0, 0.0])
                return self._goal_position(state) + self._grasp_offset, orientation
            up = np.asarray(state["object_up"], dtype=float)
            up /= max(float(np.linalg.norm(up)), 1e-9)
            axis = np.cross(up, [0, 0, 1])
            sine = float(np.linalg.norm(axis))
            angle = float(np.arctan2(sine, up[2]))
            correction = R.identity() if sine < 1e-8 else R.from_rotvec(axis / sine * min(angle, np.radians(3)))
            joints = self.rmp_controller.articulation_rmp.get_active_joints_subset().get_joint_positions()
            _, hand_matrix = self.rmp_controller.rmp_flow.get_end_effector_pose(joints)
            self._placement_hand_orientation = (correction * R.from_matrix(hand_matrix)).as_quat()[[3, 0, 1, 2]]
            self._grasp_offset = correction.apply(offset)
            self.place_controller.placement_feature_position = np.asarray(state["object_position"])
            self.place_controller.placement_feature_target = self._goal_position(state) + np.array([0, 0, 0.003])
            self.place_controller.placement_feature_aligned = bool(angle < np.radians(8))
        return self._goal_position(state) + self._grasp_offset, self._placement_hand_orientation

    # ------------------------------------------------------- infer-mode step

    def _step_infer(self, state):
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        state["language_instruction"] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()

    def is_success(self) -> bool:
        # Collect and replay latch their result in their respective step
        # methods. Reading it (e.g. when naming a video) must not run another
        # placement check or count the same observation twice.
        if self.mode != "infer":
            return self._last_success
        if self.current_phase == Phase.PICKING and self._check_phase_success():
            self.current_phase = Phase.PLACING
        success = self._check_placed_on_target()
        self._placement_steps = self._placement_steps + 1 if success else 0
        if self._placement_steps >= self.REQUIRED_SUCCESS_STEPS:
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False
