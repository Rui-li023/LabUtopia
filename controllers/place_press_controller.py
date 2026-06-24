from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.pressZ_controller import PressZController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    PRESSINGZ = "pressingz"
    FINISHED = "finished"

class PlacePressTaskController(BaseController):
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
        self.last_error_info = None
        self._debug_last_phase = None
        self._debug_atomic_done_logged = False
            
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

        self.press_controller = PressZController(
            name="press_controller",
            cspace_controller=RMPFlowController(
                name="press_controller",
                robot_articulation=robot
            ),
            events_dt=[0.004, 0.02, 0.01],
            robot=robot,
        )
        
        self.active_controller = self.pick_controller
        

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]
        )
        super()._init_infer_mode(cfg, robot)

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.last_error_info = None
        self._debug_last_phase = None
        self._debug_atomic_done_logged = False

        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pick_controller.reset()
            self.place_controller.reset()
            self.press_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()

    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion."""
        return self._check_phase_success()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']
        end_button_position = self.state['button_position']

        if self.current_phase == Phase.PICKING:
            success = object_pos[2] > 0.82
            if not success:
                self.last_error_info = {
                    'phase': 'PICKING',
                    'object_z': float(object_pos[2]),
                    'required_z': 0.82,
                }
            return success
        elif self.current_phase == Phase.PLACING:
            dx = float(abs(object_pos[0] - target_position[0]))
            dy = float(abs(object_pos[1] - target_position[1]))
            # Relative z check: beaker should rest on the plat surface, i.e.
            # its centre sits 0–8 cm above the (offset-adjusted) plat ref.
            # The previous absolute "<= 0.86" was geometrically unsatisfiable
            # — a normally-placed beaker centre is at plat_top + half_height
            # ≈ 0.87, so it was always judged a failure.
            dz = float(object_pos[2] - target_position[2])
            z_ok = 0.0 < dz < 0.08
            success = object_pos is not None and z_ok and dx < 0.05 and dy < 0.05
            if not success:
                self.last_error_info = {
                    'phase': 'PLACING',
                    'object_pos': object_pos.tolist() if object_pos is not None else None,
                    'target_pos': target_position.tolist() if target_position is not None else None,
                    'dx': dx, 'dx_threshold': 0.05,
                    'dy': dy, 'dy_threshold': 0.05,
                    'dz_above_target': dz,
                    'dz_min': 0.0, 'dz_max': 0.08,
                    'z_ok': bool(z_ok),
                }
            else:
                self.last_error_info = None
            return success
        elif self.current_phase == Phase.PRESSINGZ:
            dx = float(abs(object_pos[0] - target_position[0]))
            dy = float(abs(object_pos[1] - target_position[1]))
            dz = float(object_pos[2] - target_position[2])
            z_ok = 0.0 < dz < 0.08
            button_ok = end_button_position[2] < 0.761
            success = (
                object_pos is not None
                and z_ok and dx < 0.05 and dy < 0.05 and button_ok
            )
            if not success:
                self.last_error_info = {
                    'phase': 'PRESSINGZ',
                    'object_pos': object_pos.tolist() if object_pos is not None else None,
                    'target_pos': target_position.tolist() if target_position is not None else None,
                    'button_z': float(end_button_position[2]) if end_button_position is not None else None,
                    'button_z_required': 0.761,
                    'button_ok': bool(button_ok),
                    'dx': dx, 'dy': dy,
                    'dz_above_target': dz, 'z_ok': bool(z_ok),
                }
            else:
                self.last_error_info = None
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
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        # Debug: log first frame of each phase so we can see what state the
        # robot starts each phase with.
        if self.current_phase != self._debug_last_phase:
            print(
                f"[placepress debug] entered phase={self.current_phase.value} "
                f"object_pos={state['object_position']} "
                f"target_pos={state['target_position']} "
                f"button_pos={state['button_position']}"
            )
            self._debug_last_phase = self.current_phase
            self._debug_atomic_done_logged = False

        success = self._check_phase_success()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        # Debug: log the moment atomic action becomes done; print success-check
        # state at that exact moment so we can see why we DO or DON'T transition.
        atomic_done = self.active_controller.is_done()
        if atomic_done and not self._debug_atomic_done_logged:
            print(
                f"[placepress debug] phase={self.current_phase.value} atomic done | "
                f"success={success} | last_error_info={self.last_error_info}"
            )
            self._debug_atomic_done_logged = True

        if not atomic_done:
            action = None
            record_array = None
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
            elif self.current_phase == Phase.PLACING:
                action, record_array = self.place_controller.forward(
                    place_position=state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    gripper_position=state['gripper_position']
                )
            elif self.current_phase == Phase.PRESSINGZ:
                 action, record_array = self.press_controller.forward(
                    target_position=state['button_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                    gripper_position=state['gripper_position']
                )
                 
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
            
            return action, False, False

        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to place...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Place task success!")
                self.current_phase = Phase.PRESSINGZ
                self.active_controller = self.press_controller
                return None, False, False
            elif self.current_phase == Phase.PRESSINGZ:
                print("PressZ task success!")
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
            else:
                self._last_failure_reason = f"PlacePress {self.current_phase.value} phase failed: phase success check did not pass after controller done"
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

        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
        action = self.inference_engine.step_inference(state)
               
        return action, False, self.is_success()

    def is_success(self):
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        end_button_position = self.state['button_position']
        if (object_pos is not None and
            0.0 < (object_pos[2] - target_position[2]) < 0.08 and
            abs(object_pos[0] - target_position[0]) < 0.05 and
            abs(object_pos[1] - target_position[1]) < 0.05 and
            end_button_position[2] < 0.761 ):
            # Infer path: latch success so the terminal frame (current_phase ==
            # FINISHED -> returns (None, True, self._last_success)) is counted by
            # main.py. Without this, geometric successes are logged as failures.
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False

    def get_language_instruction(self):
        object_name = self.clean_object_name(self.state['object_name'])
        if self.current_phase == Phase.PICKING:
            return self._get_cached_instruction(
                'placepress:picking',
                self._build_instruction_templates(
                    f"Pick up the {object_name}",
                    f"Pick up the {object_name} from the table and lift it clear of the surface",
                ),
            )
        if self.current_phase == Phase.PLACING:
            return self._get_cached_instruction(
                'placepress:placing',
                self._build_instruction_templates(
                    f"Place the {object_name} at the target",
                    f"Move the {object_name} to the target position and set it down carefully",
                ),
            )
        return self._get_cached_instruction(
            'placepress:pressing',
            self._build_instruction_templates(
                'Press the button',
                'Press the button after placing the object at the target position',
            ),
        )
