from controllers.atomic_actions.move_controller import MoveController
from robots.franka.rmpflow_controller import RMPFlowController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from typing import Optional
from .atomic_actions.open_controller import OpenController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.close_controller import CloseController
from .atomic_actions.press_controller import PressController
from .base_controller import BaseController

class Phase(Enum):
    OPEN_DOOR = "opening_door"
    MOVE_HIGHER = "move_higher"
    PICK_BEAKER = "picking_beaker" 
    PLACE_BEAKER = "placing_beaker"
    PICK_BEAKER3 = "picking_beaker3"
    PLACE_BEAKER3 = "placing_beaker3"
    PRESS_BUTTON = "pressing_button"
    FINISHED = "finished"

class DeviceOperateController(BaseController):
    """Controller for operating a laboratory device with multiple steps.
    
    Steps:
    1. Open device door
    2. Move higher
    3. Pick up beaker
    4. Place beaker inside device
    5. Close device door
    6. Press device button
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.success_steps = set() 
        self.current_phase = Phase.OPEN_DOOR
        self.last_phase = None  
        self.initial_handle_position = None
        self.initial_beaker_position = None
        self.initial_beaker3_position = None
        self.initial_button_position = None
            
    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        rmp_controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot
        )

        self.open_controller = OpenController(
            name="open_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            furniture_type="door",
            robot=robot,
        )
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=rmp_controller,
            gripper=robot.gripper,
            robot=robot,
        )
        
        self.close_controller = CloseController(
            name="close_controller",
            cspace_controller=rmp_controller,
            furniture_type="door",
            robot=robot,
        )
        
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=rmp_controller,
            events_dt=[0.004, 0.05, 0.02],
            initial_offset=0.05,
            robot=robot,
        )
        
        self.move_controller = MoveController(
            name="move_controller",
            cspace_controller=rmp_controller,
        )
        
        self.active_controller = self.open_controller

        
    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.OPEN_DOOR
        self.last_phase = None
        self.success_steps.clear()
        self.initial_handle_position = None  
        self.initial_beaker_position = None
        self.initial_beaker3_position = None
        self.end_handle_position = None
        
        if self.mode == "collect":
            self.active_controller = self.open_controller
            self.open_controller.reset()
            self.pick_controller.reset()
            self.place_controller.reset()
            self.close_controller.reset()
            self.press_controller.reset()
            self.move_controller.reset()
        else:
            pass

    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion."""
        return self._check_phase_success(self.state)

    def _step_collect(self, state):
        """Execute one step in collect mode (dispatched from base step)."""
        # Collection logic is handled in step() override; this satisfies the abstract contract.
        raise NotImplementedError("DeviceOperateController uses step() override for collect mode")

    def _check_phase_success(self, state):
        """Check success criteria for current phase."""
        if self.current_phase == Phase.OPEN_DOOR:
            current_pos = state['door_handle_position']
            gripper_position = state['gripper_position']
            if current_pos is None or self.initial_handle_position is None or gripper_position is None:
                return False
            end_effector_distance = abs(np.linalg.norm(np.array(gripper_position) - np.array(current_pos)))
            distance = abs(np.linalg.norm(np.array(current_pos) - self.initial_handle_position))
            return distance > 0.13 and end_effector_distance > 0.04
        elif self.current_phase == Phase.MOVE_HIGHER:
            return state['gripper_position'][2] - self.end_handle_position[2] > 0.15
        elif self.current_phase == Phase.PICK_BEAKER:
            object_pos = state['beaker_position']
            return object_pos[2] > self.initial_beaker_position[2] + 0.05  
        elif self.current_phase == Phase.PLACE_BEAKER:
            object_pos = state['beaker_position']
            target_pos = state['device_interior_position']
            dist = np.linalg.norm(object_pos[:2] - target_pos[:2])
            return dist < 0.2 and abs(object_pos[2] - target_pos[2]) < 0.1  
        elif self.current_phase == Phase.PICK_BEAKER3:
            object_pos = state['beaker3_position']
            return object_pos[2] > self.initial_beaker3_position[2] + 0.05  
        elif self.current_phase == Phase.PLACE_BEAKER3:
            object_pos = state['beaker3_position']
            target_pos = state['beaker3_target_position']
            dist = np.linalg.norm(object_pos[:2] - target_pos[:2])
            return dist < 0.2 and abs(object_pos[2] - target_pos[2]) < 0.1  
        elif self.current_phase == Phase.PRESS_BUTTON:
            return state['button_position'][0] - self.initial_button_position[0] > 0.005
        return False

    def _advance_to_next_phase(self):
        """Advance to the next phase of the task."""
        phase_sequence = {
            Phase.OPEN_DOOR: Phase.MOVE_HIGHER,
            Phase.MOVE_HIGHER: Phase.PICK_BEAKER,
            Phase.PICK_BEAKER: Phase.PLACE_BEAKER,
            Phase.PLACE_BEAKER: Phase.PICK_BEAKER3,
            Phase.PICK_BEAKER3: Phase.PLACE_BEAKER3,
            Phase.PLACE_BEAKER3: Phase.PRESS_BUTTON,
            Phase.PRESS_BUTTON: Phase.FINISHED
        }
        self.success_steps.add(self.current_phase)

        self.last_phase = self.current_phase
        self.current_phase = phase_sequence.get(self.current_phase, Phase.FINISHED)
        
        if self.mode == "collect":
            controller_map = {
                Phase.OPEN_DOOR: self.open_controller,
                Phase.MOVE_HIGHER: self.move_controller,
                Phase.PICK_BEAKER: self.pick_controller,
                Phase.PLACE_BEAKER: self.place_controller,
                Phase.PICK_BEAKER3: self.pick_controller,
                Phase.PLACE_BEAKER3: self.place_controller,
                Phase.PRESS_BUTTON: self.press_controller
            }
            self.active_controller = controller_map.get(self.current_phase)
            if self.active_controller:
                self.active_controller.reset()

    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        if self.initial_beaker_position is None:
            self.initial_beaker_position = state['beaker_position']
        if self.initial_beaker3_position is None:
            self.initial_beaker3_position = state['beaker3_position']
        if self.initial_button_position is None:
            self.initial_button_position = state['button_position']
            
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            print(self.success_steps)
            self._last_success = len(self.success_steps) == 7
            if self._last_success:
                self._last_failure_reason = ""
            return None, True, self._last_success

        if self.mode == "collect":
            # Collection mode using atomic controllers
            action = None
            record_array = None
            if self.current_phase == Phase.OPEN_DOOR:
                if self.initial_handle_position is None:
                    self.initial_handle_position = state['door_handle_position']
                    
                action, record_array = self.open_controller.forward(
                    handle_position=state['door_handle_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    revolute_joint_position=state['revolute_joint_position'],
                    end_effector_orientation=euler_angles_to_quat([0, 100, 0], degrees=True, extrinsic=False),
                    angle=100,
                    close_gripper_distance=0.015
                )
                self.end_handle_position = state['door_handle_position']
                
            elif self.current_phase == Phase.MOVE_HIGHER:
                target_position = self.end_handle_position.copy()
                target_position[0] -= 0.4
                target_position[2] += 0.22
                action, record_array = self.move_controller.forward(
                    target_position=target_position,
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    target_orientation=R.from_euler('xyz', np.radians([0, 90, 0])).as_quat(),
                )
            elif self.current_phase == Phase.PICK_BEAKER:
                action, record_array = self.pick_controller.forward(
                    picking_position=state['beaker_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['beaker_size'],
                    object_name="beaker",
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
            elif self.current_phase == Phase.PLACE_BEAKER:
                action, record_array = self.place_controller.forward(
                    place_position=np.array(state['device_interior_position']),
                    current_joint_positions=state['joint_positions'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    gripper_position=state['gripper_position'],
                    gripper_control=self.gripper_control,
                    pre_place_z=0.1,
                )
            elif self.current_phase == Phase.PICK_BEAKER3:
                action, record_array = self.pick_controller.forward(
                    picking_position=state['beaker3_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['beaker3_size'],
                    object_name="beaker",
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                )
            elif self.current_phase == Phase.PLACE_BEAKER3:
                action, record_array = self.place_controller.forward(
                    place_position=np.array(state['beaker3_target_position']),
                    current_joint_positions=state['joint_positions'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                    gripper_position=state['gripper_position'],
                    gripper_control=self.gripper_control,
                    pre_place_z=0.15,
                )
            elif self.current_phase == Phase.PRESS_BUTTON:
                action, record_array = self.press_controller.forward(
                    target_position=state['button_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    press_distance = 0.005
                )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                )
            
            if self.active_controller.is_done():
                success = self._check_phase_success(state)
                if success:
                    self._last_failure_reason = ""
                    print(f"{self.current_phase.value} success!")
                    self._advance_to_next_phase()
                    if self.current_phase == Phase.FINISHED:
                        # All phases succeeded — write collected data
                        self.data_collector.write_cached_data(state['joint_positions'][:-1])
                    return None, False, False
                else:
                    self._last_failure_reason = f"Phase {self.current_phase.value} failed: phase success check did not pass after controller done"
                    print(f"{self.current_phase.value} failed!")
                    self.data_collector.clear_cache()
                    self.current_phase = Phase.FINISHED
                    return None, True, False
            
            return action, False, False

        else:
            return self._step_infer(state)

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses policy to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if self.initial_beaker_position is None:
            self.initial_beaker_position = state['beaker_position']
        if self.initial_beaker3_position is None:
            self.initial_beaker3_position = state['beaker3_position']
        if self.initial_button_position is None:
            self.initial_button_position = state['button_position']
        if self.initial_handle_position is None:
            self.initial_handle_position = state['door_handle_position']

        if self.current_phase != Phase.FINISHED:
            success = self._check_phase_success(state)
            if success:
                print(f"Inference: {self.current_phase.value} success!")
                self.success_steps.add(self.current_phase)
                self._advance_to_next_phase()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            self._last_success = len(self.success_steps) == 7
            if self._last_success:
                self._last_failure_reason = ""
            return None, True, self._last_success

        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        else:
            state['language_instruction'] = "Operate the laboratory device by opening the door, placing beakers, and pressing the button"

        action = self.inference_engine.step_inference(state)
        
        return action, False, False

    def is_success(self):
        """Check if task was successful."""
        return len(self.success_steps) == 7

    def get_language_instruction(self) -> Optional[str]:
        phase_instructions = {
            Phase.OPEN_DOOR: ('Open the device door', 'Open the device door by pulling the handle until it swings open'),
            Phase.MOVE_HIGHER: ('Move the robot arm higher', 'Raise the robot arm to a higher position to avoid obstacles'),
            Phase.PICK_BEAKER: ('Pick up the first beaker', 'Pick up the first beaker from the table and lift it clear of the surface'),
            Phase.PLACE_BEAKER: ('Place the first beaker inside the device', 'Move the first beaker into the device and set it down carefully'),
            Phase.PICK_BEAKER3: ('Pick up the second beaker', 'Pick up the second beaker from the table and lift it clear of the surface'),
            Phase.PLACE_BEAKER3: ('Place the second beaker inside the device', 'Move the second beaker into its designated position inside the device'),
            Phase.PRESS_BUTTON: ('Press the device button', 'Press the device button to start the operation'),
        }
        direct, detailed = phase_instructions.get(
            self.current_phase,
            ('Operate the laboratory device', 'Complete the current step of the laboratory device operation carefully'),
        )
        return self._get_cached_instruction(
            f"device_operate:{self.current_phase.value}",
            self._build_instruction_templates(direct, detailed),
        )
