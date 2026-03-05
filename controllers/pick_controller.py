from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController

class PickTaskController(BaseController):
    """
    Controller for pick-and-place tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        REQUIRED_SUCCESS_STEPS (int): Number of consecutive steps needed for success
        success_counter (int): Counter for tracking successful steps
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
            
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.05, 0.004, 0.008]
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
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        return super().step(state)
            
    def _check_success(self):
        return self.state['object_position'][2] > self.initial_position[2] + 0.1
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
        
        if not self.pick_controller.is_done():
            action, record_array = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_offset_x=0.05,
                after_offset_z=0.25
            )
            
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False
        
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self._last_failure_reason = ""
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self.reset_needed = True
            return None, True, True

        self._last_failure_reason = "Pick task failed: object height did not reach required (initial_z + 0.1) for REQUIRED_SUCCESS_STEPS"
        self.data_collector.clear_cache()
        self._last_success = False
        self.reset_needed = True
        return None, True, False
        
    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses inference engine to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
            
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

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = self.clean_object_name(self.state['object_name'])
        self._language_instruction = f"Pick up the {object_name} from the table"
        return self._language_instruction
