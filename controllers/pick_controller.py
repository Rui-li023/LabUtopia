import random
from typing import Optional

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .base_controller import BaseController


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

    def __init__(self, cfg, robot):
        # Read the grasp block BEFORE super().__init__: BaseController.__init__
        # dispatches straight into this class's _init_collect_mode, which needs
        # these attributes to already exist.
        grasp_cfg = getattr(cfg, "grasp", None)
        self._ee_euler_deg = np.array(
            getattr(grasp_cfg, "ee_euler_deg", self.DEFAULT_EE_EULER_DEG) if grasp_cfg else self.DEFAULT_EE_EULER_DEG,
            dtype=float,
        )
        # Off by default: feeding the base position makes the pre-grasp approach
        # point from the object back toward the base instead of the hard-coded
        # world -X, which shifts the lab tasks' approach by up to 0.13 lateral.
        self._approach_from_base = bool(getattr(grasp_cfg, "approach_from_base", False)) if grasp_cfg else False
        # Height the gripper closes at, measured from the object origin. None keeps
        # the atomic controller's shared per-object table (which every other level
        # depends on); set it to grasp lower or higher on the object.
        raw_z = getattr(grasp_cfg, "pick_z_offset", None) if grasp_cfg else None
        self._pick_z_offset = float(raw_z) if raw_z is not None else None

        super().__init__(cfg, robot)
        self.initial_position = None
        self._pick_instruction: Optional[str] = None
        self._pick_task_index: Optional[int] = None

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008],
        )
        if self._approach_from_base:
            base_position, _ = robot.get_world_pose()
            self.pick_controller.set_robot_position(np.asarray(base_position))
            logger.info(f"[grasp] approach direction taken from base at {np.round(base_position, 3)}")
        if self._pick_z_offset is not None:
            self.pick_controller.pick_z_offset_override = self._pick_z_offset
            logger.info(f"[grasp] pick_z_offset override = {self._pick_z_offset:.3f} m above the object origin")
        logger.info(f"[grasp] world-frame ee_euler_deg = {self._ee_euler_deg.tolist()}")

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        elif self.mode == "replay":
            self.trajectory_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()
        self.initial_position = None
        self._pick_instruction = None
        self._pick_task_index = None

    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state["object_position"]
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

    def get_language_instruction(self) -> Optional[str]:
        if self._pick_instruction is None:
            self._pick_instruction = self._sample_pick_instruction()
        self._language_instruction = self._pick_instruction
        return self._language_instruction

    def get_task_index(self) -> Optional[int]:
        instruction = self.get_language_instruction()
        if instruction is None or self.mode != "collect":
            return None
        if self._pick_task_index is None:
            self._pick_task_index = self.data_collector.register_task_instruction(instruction)
        return self._pick_task_index

    def _grasp_quat(self, state) -> np.ndarray:
        """World-frame grasp orientation as a quaternion, per step.

        A hook, not a constant: the base class returns the fixed
        ``grasp.ee_euler_deg`` every step (unchanged behaviour for L1-L5), while
        ``PickWideTaskController`` overrides it to rotate the grasp frame with
        the object's bearing so a much wider spawn area stays reachable.
        """
        return R.from_euler("xyz", np.radians(self._ee_euler_deg)).as_quat()

    def _step_collect(self, state):
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
                pre_offset_x=0.05,
                after_offset_z=0.25,
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

        self._last_failure_reason = "Pick task failed: object height did not reach required (initial_z + 0.1) for REQUIRED_SUCCESS_STEPS"
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
