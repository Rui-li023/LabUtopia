import os
import random

import numpy as np
from loguru import logger

from .atomic_actions.pick_controller import PickController
from .base_controller import BaseController
from .grasp_frame import GraspFrame, resolve_pick_z_offset


class PickTaskController(BaseController):
    """
    Controller for pick tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking
    """

    PICK_TEMPLATES = [
        "Pick up the {object_name}.",
        "Please help me pick up the {object_name}.",
        "Pick up the {object_name} from the table and lift it clear of the surface.",
    ]

    # World-frame grasp orientation, scipy extrinsic "xyz" degrees. The default
    # [0, 90, 25] is what every level1-4 config has always used; it assumes the
    # base faces world +X. A base yawed by theta needs the whole grasp frame
    # rotated with it: Rz(theta) * Rz(25) * Ry(90).
    DEFAULT_EE_EULER_DEG = (0.0, 90.0, 25.0)

    # How much of the object's bearing the grasp frame follows. 0.0 = the fixed
    # world-frame pose every L1-L4 config has always used. ``PickWideTaskController``
    # raises it to 1.0; see controllers/grasp_frame.py.
    DEFAULT_BEARING_GAIN = 0.0

    def __init__(self, cfg, robot):
        # Read the grasp block BEFORE super().__init__: BaseController.__init__
        # dispatches straight into this class's _init_collect_mode, which needs
        # these attributes to already exist.
        grasp_cfg = getattr(cfg, "grasp", None)
        self._grasp_cfg = grasp_cfg
        # Built in _init_collect_mode (it needs the robot's world pose); the
        # infer/replay paths never ask for a grasp orientation.
        self._grasp_frame: GraspFrame | None = None
        # Off by default: feeding the base position makes the pre-grasp approach
        # point from the object back toward the base instead of the hard-coded
        # world -X, which shifts the lab tasks' approach by up to 0.13 lateral.
        self._approach_from_base = bool(getattr(grasp_cfg, "approach_from_base", False)) if grasp_cfg else False
        # Height the gripper closes at, measured from the object origin. None keeps
        # the atomic controller's shared per-object table (which every other level
        # depends on); set it to grasp lower or higher on the object.
        self._pick_z_offset = resolve_pick_z_offset(grasp_cfg)
        # Standoff above the object's TOP that the pre-grasp waypoint sits at. The 0.12
        # default suits the Franka, whose base is well below its working volume, but it
        # is measured from the object top rather than the bench, so on a short arm it
        # can land outside the workspace: the ARX X5 cannot hold a pose 0.28 m above
        # and 0.44 m in front of its own base, so RMPFlow thrashes at event 0 and never
        # recovers, failing every episode without ever approaching the beaker.
        raw_pre_z = getattr(grasp_cfg, "pre_offset_z", None) if grasp_cfg else None
        self._pre_offset_z = float(raw_pre_z) if raw_pre_z is not None else 0.12
        raw_pre_x = getattr(grasp_cfg, "pre_offset_x", None) if grasp_cfg else None
        self._pre_offset_x = float(raw_pre_x) if raw_pre_x is not None else 0.05
        if self._pre_offset_x < 0.0:
            raise ValueError("grasp.pre_offset_x must be non-negative")
        raw_after_z = getattr(grasp_cfg, "after_offset_z", None) if grasp_cfg else None
        self._after_offset_z = float(raw_after_z) if raw_after_z is not None else 0.25
        if self._after_offset_z <= 0.0:
            raise ValueError("grasp.after_offset_z must be positive")
        # Optional target gripper opening in metres. Keeping None preserves the
        # existing binary full-close behaviour for every current configuration.
        raw_gripper_distance = getattr(grasp_cfg, "gripper_distance", None) if grasp_cfg else None
        self._gripper_distance = float(raw_gripper_distance) if raw_gripper_distance is not None else None
        if self._gripper_distance is not None and self._gripper_distance < 0.0:
            raise ValueError("grasp.gripper_distance must be non-negative")

        super().__init__(cfg, robot)
        self.initial_position = None
        self._pick_instruction: str | None = None
        self._pick_task_index: int | None = None

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        # Phase durations in legacy mode, or per-phase timeouts when adaptive phase
        # completion is enabled. A slower arm needs a longer window either way: Piper
        # only lifts ~3 cm inside the default 250-step window, short of the +10 cm score.
        default_events_dt = [0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
        events_dt = list(getattr(self._grasp_cfg, "events_dt", None) or default_events_dt)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=events_dt,
        )
        self.pick_controller.adaptive_phase_completion = bool(
            getattr(self._grasp_cfg, "adaptive_phase_completion", False)
        )
        raw_arrival_stable_steps = getattr(self._grasp_cfg, "arrival_stable_steps", 1)
        self.pick_controller.arrival_stable_steps = int(raw_arrival_stable_steps)
        if self.pick_controller.arrival_stable_steps <= 0:
            raise ValueError("grasp.arrival_stable_steps must be a positive integer")
        if self.pick_controller.adaptive_phase_completion:
            logger.info(
                "[grasp] adaptive phase completion enabled: motion phases require "
                f"{self.pick_controller.arrival_stable_steps} stable arrival steps; "
                "events_dt is used as the timeout"
            )
        raw_orientation_noise = getattr(self._grasp_cfg, "orientation_noise_deg", None)
        if raw_orientation_noise is not None:
            self.pick_controller.orientation_noise_deg = float(raw_orientation_noise)
            if self.pick_controller.orientation_noise_deg < 0.0:
                raise ValueError("grasp.orientation_noise_deg must be non-negative")
            logger.info(f"[grasp] orientation_noise_deg = {self.pick_controller.orientation_noise_deg:.1f}")
        self.pick_controller.lift_along_tool = bool(getattr(self._grasp_cfg, "lift_along_tool", False))
        if self.pick_controller.lift_along_tool:
            logger.info("[grasp] lift retracts along the tool axis instead of straight up")
        raw_lift_offset = getattr(self._grasp_cfg, "lift_offset_xyz", None)
        if raw_lift_offset is not None:
            lift_offset = np.asarray(raw_lift_offset, dtype=float)
            if lift_offset.shape != (3,):
                raise ValueError("grasp.lift_offset_xyz must contain exactly 3 values")
            self.pick_controller.lift_offset_xyz = lift_offset
            logger.info(f"[grasp] lift_offset_xyz = {lift_offset.tolist()} m")
        self.pick_controller.approach_along_tool = bool(getattr(self._grasp_cfg, "approach_along_tool", False))
        if self.pick_controller.approach_along_tool:
            logger.info("[grasp] pre-grasp backs off along the tool approach axis")
        self.pick_controller.require_pregrasp_xyz = bool(getattr(self._grasp_cfg, "require_pregrasp_xyz", False))
        if self.pick_controller.require_pregrasp_xyz:
            logger.info("[grasp] final approach waits for the full XYZ pre-grasp pose")
        self.pick_controller.lock_final_approach_target = bool(
            getattr(self._grasp_cfg, "lock_final_approach_target", False)
        )
        if self.pick_controller.lock_final_approach_target:
            logger.info("[grasp] object target locks when the final descent starts")
        raw_pregrasp_threshold = (
            getattr(self._grasp_cfg, "pregrasp_position_threshold", None) if self._grasp_cfg else None
        )
        if raw_pregrasp_threshold is not None:
            self.pick_controller.pregrasp_position_threshold = float(raw_pregrasp_threshold)
            if self.pick_controller.pregrasp_position_threshold <= 0.0:
                raise ValueError("grasp.pregrasp_position_threshold must be positive")
            logger.info(
                f"[grasp] pregrasp_position_threshold = {self.pick_controller.pregrasp_position_threshold:.3f} m"
            )
        raw_grasp_threshold = getattr(self._grasp_cfg, "grasp_position_threshold", None) if self._grasp_cfg else None
        if raw_grasp_threshold is not None:
            self.pick_controller.grasp_position_threshold = float(raw_grasp_threshold)
            if self.pick_controller.grasp_position_threshold <= 0.0:
                raise ValueError("grasp.grasp_position_threshold must be positive")
            logger.info(f"[grasp] grasp_position_threshold = {self.pick_controller.grasp_position_threshold:.3f} m")
        if events_dt != default_events_dt:
            logger.info(f"[grasp] events_dt override = {events_dt}")
        if self._approach_from_base:
            base_position, _ = robot.get_world_pose()
            self.pick_controller.set_robot_position(np.asarray(base_position))
            logger.info(f"[grasp] approach direction taken from base at {np.round(base_position, 3)}")
        if self._pick_z_offset is not None:
            self.pick_controller.pick_z_offset_override = self._pick_z_offset
            logger.info(f"[grasp] pick_z_offset override = {self._pick_z_offset:.3f} m above the object origin")
        if self._gripper_distance is not None:
            logger.info(f"[grasp] gripper_distance override = {self._gripper_distance:.3f} m")
        self._grasp_frame = GraspFrame(
            self._grasp_cfg,
            robot,
            self.DEFAULT_EE_EULER_DEG,
            default_bearing_gain=self.DEFAULT_BEARING_GAIN,
            label="grasp",
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
            self._grasp_frame.new_episode()
        elif self.mode == "replay":
            self.trajectory_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()
        self.initial_position = None
        self._pick_instruction = None
        self._pick_task_index = None

    def step(self, state):
        if self.initial_position is None:
            self.initial_position = np.asarray(state["object_position"], dtype=float).copy()
        elif self.mode == "collect" and getattr(self.pick_controller, "_event", 0) < 4:
            # The first controller observation can precede full physics settling.
            # Keep the success baseline at the object's lowest pre-close height so
            # "lift by 10 cm" is measured from the support surface, not from a pose
            # captured while the object was still falling.
            self.initial_position[2] = min(
                self.initial_position[2],
                float(state["object_position"][2]),
            )
        return super().step(state)

    def _check_success(self):
        # Replay applies recorded waypoints under PD with some lag, so the
        # peak lift achieved during collect may slip a couple of cm below
        # the +0.10 m threshold. Use a slightly looser cutoff in replay.
        threshold = 0.08 if self.mode == "replay" else 0.10
        return self.state["object_position"][2] > self.initial_position[2] + threshold

    def _sample_pick_instruction(self) -> str:
        object_name = self.clean_object_name(self.state["object_name"])
        template = random.choice(self.PICK_TEMPLATES)
        return template.format(object_name=object_name)

    def get_language_instruction(self) -> str | None:
        if self._pick_instruction is None:
            self._pick_instruction = self._sample_pick_instruction()
        self._language_instruction = self._pick_instruction
        return self._language_instruction

    def get_task_index(self) -> int | None:
        instruction = self.get_language_instruction()
        if instruction is None or self.mode != "collect":
            return None
        if self._pick_task_index is None:
            self._pick_task_index = self.data_collector.register_task_instruction(instruction)
        return self._pick_task_index

    def _grasp_quat(self, state) -> np.ndarray:
        """World-frame grasp orientation as a quaternion, per step.

        With the default ``bearing_gain`` of 0 this is the fixed
        ``grasp.ee_euler_deg`` every step (unchanged behaviour for L1-L5);
        ``PickWideTaskController`` and any config that sets ``bearing_gain``
        rotate the frame with the object's bearing so a much wider spawn area
        stays reachable.
        """
        return self._grasp_frame.quat(state["object_position"])

    def _debug_trace(self, state) -> None:
        """Log gripper-vs-object geometry per phase, for bringing up a new arm.

        Env-gated (LABUTOPIA_PICK_DEBUG=1). A success rate alone cannot separate
        "never reached the object" from "reached it and closed on nothing", which is
        the first thing you need to know when a different arm fails this task.
        """
        if not os.environ.get("LABUTOPIA_PICK_DEBUG"):
            return
        event = getattr(self.pick_controller, "_event", -1)
        if event < getattr(self, "_debug_last_event", 0):
            self._debug_step = 0  # new episode
        self._debug_last_event = event
        self._debug_step = getattr(self, "_debug_step", 0) + 1
        interval = int(os.environ.get("LABUTOPIA_PICK_DEBUG_INTERVAL", "60"))
        if interval <= 0:
            raise ValueError("LABUTOPIA_PICK_DEBUG_INTERVAL must be positive")
        if self._debug_step % interval:
            return
        grip = np.asarray(state.get("gripper_position", [np.nan] * 3), dtype=float)
        obj = np.asarray(state["object_position"], dtype=float)
        size = np.asarray(state.get("object_size", [np.nan] * 3), dtype=float)
        logger.info(
            f"[pick-debug] step={self._debug_step:>5} event={event} "
            f"tcp={np.round(grip, 3)} obj={np.round(obj, 3)} "
            f"size={np.round(size, 3)} dist={np.linalg.norm(grip - obj):.3f} "
            f"obj_z_rise={obj[2] - self.initial_position[2]:+.3f}"
        )
        if "robotiq" not in self.robot.name.lower():
            return

        joint_positions = np.asarray(state["joint_positions"], dtype=float)
        dof_names = list(self.robot.dof_names or [])
        gripper_indices = [
            index
            for index, name in enumerate(dof_names)
            if any(token in name for token in ("gripper", "finger", "knuckle"))
        ]
        gripper_joints = {dof_names[index]: round(float(joint_positions[index]), 4) for index in gripper_indices}
        root = self.robot.prim_path_str.rstrip("/")
        link_positions = {}
        for link_name in (
            "left_inner_finger",
            "left_inner_finger_pad",
            "right_inner_finger",
            "right_inner_finger_pad",
        ):
            link_positions[link_name] = np.round(
                self.object_utils.get_object_xform_position(object_path=f"{root}/{link_name}"),
                4,
            ).tolist()
        logger.info(f"[pick-debug-gripper] q={gripper_joints} links={link_positions}")

    def _step_collect(self, state):
        self._debug_trace(state)
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        # Early termination: as soon as the bottle is lifted and held for the
        # required window, stop. Otherwise the atomic pick controller's final
        # null-action phase (event 6) records ~167 frames of "frozen lift_target
        # + closed gripper", teaching the policy a strong "do nothing" attractor.
        if self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS:
            self._last_success = True
            self._last_failure_reason = ""
            self.data_collector.write_cached_data(state["joint_positions"][:-1])
            self.reset_needed = True
            return None, True, True

        if not self.pick_controller.is_done():
            action, record_array = self.pick_controller.forward(
                picking_position=state["object_position"],
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                end_effector_orientation=self._grasp_quat(state),
                gripper_position=state["gripper_position"],
                pre_offset_x=self._pre_offset_x,
                pre_offset_z=self._pre_offset_z,
                after_offset_z=self._after_offset_z,
                gripper_distances=self._gripper_distance,
            )

            if "camera_data" in state:
                instruction = self.get_language_instruction()
                # joint_angles[:-1] = 7 arm joints + panda_finger_joint1.
                # cache_step expands finger_joint1 to total gripper width.
                self.data_collector.cache_step(
                    camera_images=state["camera_data"],
                    joint_angles=state["joint_positions"][:-1],
                    action=record_array,
                    language_instruction=instruction,
                    task_index=self.get_task_index(),
                )

            return action, False, False

        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self._last_failure_reason = ""
            self.data_collector.write_cached_data(state["joint_positions"][:-1])
            self.reset_needed = True
            return None, True, True

        phase_failure_reason = self.pick_controller.get_failure_reason()
        self._last_failure_reason = phase_failure_reason or (
            "Pick task failed: object height did not reach required (initial_z + 0.1) for REQUIRED_SUCCESS_STEPS"
        )
        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False

    def _step_infer(self, state):
        language_instruction = self.get_language_instruction()
        state["language_instruction"] = language_instruction

        action = self.inference_engine.step_inference(state)

        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self._last_failure_reason = ""
            self.reset_needed = True
            return action, True, True
        return action, False, False
