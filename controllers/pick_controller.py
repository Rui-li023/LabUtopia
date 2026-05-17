import random
from typing import Optional

import numpy as np
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

    def __init__(self, cfg, robot):
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

    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        if not self.pick_controller.is_done():
            action, record_array = self.pick_controller.forward(
                picking_position=state["object_position"],
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler("xyz", np.radians([0, 90, 25])).as_quat(),
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
