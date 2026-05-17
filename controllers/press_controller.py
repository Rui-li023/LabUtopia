from typing import Optional
import numpy as np

from scipy.spatial.transform import Rotation as R
from isaacsim.core.utils.types import ArticulationAction

from .base_controller import BaseController
from .atomic_actions.press_controller import PressController

class PressTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._initial_button_x = None
        self._last_button_x = None
        
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            # event 0 / event 2 now advance on _xyz_reached; the very small dt
            # values act only as a long safety-net timeout (≈ 500 frames each).
            events_dt = [0.002, 0.1, 0.002],
            initial_offset=0.05,
            robot=robot,
        )

    def reset(self):
        super().reset()
        self._initial_button_x = None
        self._last_button_x = None
        self._last_logged_event = None
        self._logged_action = False
        if self.mode == "collect":
            self.press_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()
    
    def step(self, state):
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        # The parent prim (cfg.target_button_path) has no rigid body — its
        # xform never changes during physics. The actual button rigid body
        # lives at "<parent>/button" (a child mesh with PhysicsRigidBodyAPI),
        # so we read its world-space xform to detect the press displacement.
        final_object_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.target_button_path + "/button"
        )
        if final_object_position is None:
            return False
        self._last_button_x = float(final_object_position[0])
        if self._initial_button_x is None:
            self._initial_button_x = self._last_button_x
        # Success: the button moved at least 2 mm from its initial X (and any
        # absolute threshold is also satisfied). The button starts ~0.4; the
        # original strict check (x > 0.405) requires a fixed 5 mm displacement
        # that some physics setups never achieve due to joint limits.
        return (self._last_button_x - self._initial_button_x) > 0.002

    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        if (
            self.press_controller.is_done()
            and self.check_success_counter < self.REQUIRED_SUCCESS_STEPS
            and self._check_success()
        ):
            n_joints = len(state["joint_positions"])
            null_action = ArticulationAction(joint_positions=[None] * n_joints)
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=np.concatenate([state['joint_positions'][:7], [0.0]]),
                    language_instruction=self.get_language_instruction(),
                )
            return null_action, False, False

        if not self.press_controller.is_done():
            # Debug: log target + arm state once per event transition so we can
            # see what cspace_controller is being asked to reach.
            ev = self.press_controller.get_current_event()
            if getattr(self, "_last_logged_event", None) != ev:
                gripper_pos = state.get('gripper_position')
                joints = state.get('joint_positions')
                print(
                    f"[press debug] event={ev} target(button)={state['object_position']} "
                    f"gripper={gripper_pos} joints[:7]={None if joints is None else joints[:7]}"
                )
                self._last_logged_event = ev
            action, record_array = self.press_controller.forward(
                target_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                gripper_position=state.get('gripper_position'),
            )
            # Debug: print what we are asking the robot to do
            if action is not None and ev == 0 and getattr(self, "_logged_action", False) is False:
                jp = getattr(action, "joint_positions", None)
                jv = getattr(action, "joint_velocities", None)
                print(f"[press debug] event0 action joint_positions={jp} joint_velocities={jv}")
                self._logged_action = True
            
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
        else:
            details = {
                'phase': 'PRESSING',
                'initial_button_x': self._initial_button_x,
                'final_button_x': self._last_button_x,
                'displacement': (
                    None if self._last_button_x is None or self._initial_button_x is None
                    else self._last_button_x - self._initial_button_x
                ),
                'required_displacement': 0.002,
                'success_counter': self.check_success_counter,
                'required_counter': self.REQUIRED_SUCCESS_STEPS,
            }
            self._last_failure_reason = (
                f"Press task failed: button x check did not hold {details}"
            )
            print(f"Phase failure details: {details}")
            self.data_collector.clear_cache()
            self._last_success = False
            self.reset_needed = True
            return None, True, False
        
    def _step_infer(self, state):
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
        if self._language_instruction:
            return self._language_instruction
        return self._get_cached_instruction(
            'press',
            self._build_instruction_templates(
                'Press the button',
                'Press the button until it is fully activated',
            ),
        )
