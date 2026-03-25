import random
from enum import Enum
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController


class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"


class PickPlaceTaskController(BaseController):
    PICK_TEMPLATES = [
        "Pick up the {object_name}.",
        "Please help me pick up the {object_name}.",
        "Pick up the {object_name} from the table and lift it clear of the surface.",
    ]
    PLACE_TEMPLATES = [
        "Place the {object_name} at the target.",
        "Please help me place the {object_name} at the target.",
        "Move the {object_name} to the target position and release it there.",
    ]

    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.current_phase = Phase.PICKING
        self._phase_instructions: dict[Phase, str] = {}
        self._phase_task_indices: dict[Phase, int] = {}

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        self.active_controller = self.pick_controller

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self._phase_instructions = {}
        self._phase_task_indices = {}

        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.place_controller.reset()
        else:
            self.inference_engine.reset()

    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion."""
        return self._check_phase_success()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']
        
        if self.current_phase == Phase.PICKING:
            return object_pos[2] > self.initial_position[2] + 0.1
        elif self.current_phase == Phase.PLACING:
            success = (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and 
                        abs(object_pos[2] - self.initial_position[2]) < 0.05)
            return success


    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state

        if self.initial_position is None:
            self.initial_position = self.state['object_position']
        if self.initial_size is None:
            self.initial_size = self.state['object_size']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _sample_phase_instruction(self, phase: Phase) -> str:
        object_name = self.clean_object_name(self.state["object_name"])
        templates = self.PICK_TEMPLATES if phase == Phase.PICKING else self.PLACE_TEMPLATES
        return random.choice(templates).format(object_name=object_name)

    def get_language_instruction(self) -> str:
        if self.current_phase not in self._phase_instructions:
            self._phase_instructions[self.current_phase] = self._sample_phase_instruction(self.current_phase)
        self._language_instruction = self._phase_instructions[self.current_phase]
        return self._language_instruction

    def get_task_index(self) -> Optional[int]:
        if self.mode != "collect" or self.current_phase == Phase.FINISHED:
            return None
        if self.current_phase not in self._phase_task_indices:
            instruction = self.get_language_instruction()
            self._phase_task_indices[self.current_phase] = self.data_collector.register_task_instruction(instruction)
        return self._phase_task_indices[self.current_phase]

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action, record_array = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    pre_offset_x=0.05,
                    pre_offset_z=0.05
                )
            else:
                action, record_array = self.place_controller.forward(
                    place_position = state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 20])).as_quat(),
                    gripper_position=state['gripper_position']
                )

            if "camera_data" in state:
                instruction = self.get_language_instruction()
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=instruction,
                    task_index=self.get_task_index(),
                )
            
            return action, False, False

        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to place...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Pour task success!")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
            else:
                self._last_failure_reason = f"PickPlace {self.current_phase.value} phase failed: phase success check did not pass after controller done"
                print(f"{self.current_phase.value} task failed!")
                self.data_collector.clear_cache()
                self._last_success = False
                self.current_phase = Phase.FINISHED
                return None, True, False
        
        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        state['language_instruction'] = self.get_language_instruction()

        action = self.inference_engine.step_inference(state)

        return action, False, self.is_success()

    def is_success(self):
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        if (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and abs(object_pos[2] - self.initial_position[2]) < 0.02
            and np.linalg.norm(self.state["gripper_position"] - object_pos) > 0.05):
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False
