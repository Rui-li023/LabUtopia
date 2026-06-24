import queue
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.shake_controller import ShakeController

class ShakeBeakerTaskController(BaseController):
    """
    Controller for shake beaker tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous shaking

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        frame_count (int): Counter for tracking frame count
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._shake_positions = []
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=60)
        self._hold_step = 0
        self._shake_success = False
        self._initial_position = None
        self._task_started = False
            
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, shake controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
        )
        self.shake_controller = ShakeController(
            name="shake_controller",
            cspace_controller=self.rmp_controller,
        )

    def reset(self):
        super().reset()
        self._shake_positions = []
        self._shake_count = 0
        self._hold_positions = queue.Queue(maxsize=60)
        self._hold_step = 0
        self._shake_success = False
        self._initial_position = None
        self._task_started = False

        if self.mode == "collect":
            self.pick_controller.reset()
            self.shake_controller.reset()
            self.data_collector.clear_cache()
        elif self.mode == "infer":
            self.inference_engine.reset()
        
    def step(self, state):
        self.state = state

        if self._initial_position is None:
            self._initial_position = state['object_position']
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)
            
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if not self.pick_controller.is_done():
            action, record_array = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="beaker",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05
            )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                )
            return action, False, False
            
        if not self.shake_controller.is_done():
            action, record_array = self.shake_controller.forward(
                current_joint_positions=self.robot.get_joint_positions(),
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
            )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                )
            return action, False, self.is_success()
        elif self.is_success():
            self._last_failure_reason = ""
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
            self.reset_needed = True
            return None, True, True
        else:
            self._last_failure_reason = "ShakeBeaker task failed: shake success check (height, shake count, hold stability) did not pass"
            self.data_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Processes observations and generates control actions using learned policy.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)

        # is_success() carries its own debounce (5 shake cycles + 60-frame hold);
        # latch + early-terminate so the episode is counted instead of running to
        # the 2000-frame cap (self._last_success is otherwise only set in collect).
        if self.is_success():
            self._last_success = True
            self.reset_needed = True
            return action, True, True
        return action, False, False
        
    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion."""
        return self.is_success()

    def is_success(self):
        if self._initial_position is None:
            return False
        
        height_diff = self.state['object_position'][2] - self._initial_position[2]
        if height_diff < 0.05:
            return False
        
        if self._shake_count < 5:
            if self.state['object_position'] is not None:
                xy = self.state['object_position'][:2]
                self._shake_positions.append(xy)
                if len(self._shake_positions) > 1:
                    start_xy = np.array(self._shake_positions[0])
                    end_xy = np.array(self._shake_positions[-1])
                    dist = np.linalg.norm(end_xy - start_xy)
                    if dist >= 0.05:  # 5cm
                        self._shake_count += 1
                        self._shake_positions = []
        elif not self._shake_success:
            self._hold_positions.put(self.state['object_position'][:2])
            self._hold_step += 1
            if self._hold_step >= 60:
                arr = np.array(list(self._hold_positions.queue))
                max_xy = arr.max(axis=0)
                min_xy = arr.min(axis=0)
                delta = max_xy - min_xy

                if np.all(delta <= 0.01):  # 1cm
                    self._shake_success = True
                    return True
                else:
                    self._hold_positions.get()
                    self._hold_step -= 1
            return False
        else:
            return True

    def get_language_instruction(self) -> str:
        return self._get_cached_instruction(
            'shake_beaker',
            self._build_instruction_templates(
                'Shake the beaker',
                'Pick up the beaker and shake it to mix the contents thoroughly',
            ),
        )
