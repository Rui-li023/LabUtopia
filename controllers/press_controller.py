from typing import Optional
import numpy as np

from isaacsim.core.utils.types import ArticulationAction

from .base_controller import BaseController
from .atomic_actions.press_controller import PressController
from .grasp_frame import GraspFrame

class PressTaskController(BaseController):
    # Button must be driven in by at least this much for a press to count.
    # The button has no joint drive, so it stays at its pushed position; the
    # arm must then retract.
    PRESS_DEPTH_THRESHOLD = 0.012   # m
    # EE (gripper) must be at least this far from the button along the press
    # axis (world X) for the press to be considered complete. ``gripper_x``
    # is wrist origin (fingertips are ~7 cm further forward), so a wrist→
    # button distance of 10 cm corresponds to fingertips ~3 cm clear.
    RETRACT_DISTANCE = 0.10   # m
    # ``gripper_position`` is the wrist origin, but the fingertips (the
    # actual contact point) are ~7 cm further forward along the press axis.
    # So "wrist within ~7 cm of button" already means the fingers are
    # touching the button. Using a too-tight threshold (e.g. 2 cm) only
    # triggers contact when fingers have already pushed the button several
    # cm in — by then the press is too deep.
    EE_CONTACT_DISTANCE = 0.07   # m

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._initial_button_x = None
        self._last_button_x = None
        self._peak_button_x = None
        self._has_been_pressed = False
        self._ee_has_contacted = False
        self._min_ee_to_button = float('inf')

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        # Joint travel is only 2 cm, so we don't need a long success hold —
        # otherwise the controller keeps issuing the (now-unreachable) press
        # target and physics keeps pushing the EE into the button.
        self.REQUIRED_SUCCESS_STEPS = 5
        self.press_controller = PressController(
            name="press_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            # event 0 / 2 / 3 advance on _xyz_reached; the small dt values
            # act only as a long safety-net timeout (≈ 500 frames each).
            events_dt = [0.002, 0.1, 0.002, 0.002],
            initial_offset=0.05,
            robot=robot,
        )
        # The wrist pose used to be a fixed [0, 90, 10]. With
        # `grasp.bearing_gain: 1.0` it follows the button's bearing instead, so
        # the instrument can be spawned off-axis; jitter varies the pose per
        # episode. Defaults reproduce the historical fixed pose exactly.
        grasp_cfg = getattr(cfg, "grasp", None)
        self._press_frame = GraspFrame(grasp_cfg, robot, (0.0, 90.0, 10.0), label="grasp/press")
        # Contact height relative to the button origin. Kept at 0 by default:
        # the button face is small, and pressing off-centre misses it entirely.
        self._press_z_offset = float(getattr(grasp_cfg, "press_z_offset", 0.0)) if grasp_cfg else 0.0
        # How far past the button face to drive. 0.018 clears the 0.012 depth
        # threshold with margin.
        self._press_distance = float(getattr(grasp_cfg, "press_distance", 0.018)) if grasp_cfg else 0.018

    def reset(self):
        super().reset()
        self._initial_button_x = None
        self._last_button_x = None
        self._peak_button_x = None
        self._has_been_pressed = False
        self._ee_has_contacted = False
        self._min_ee_to_button = float('inf')
        self._last_logged_event = None
        self._logged_action = False
        if self.mode == "collect":
            self.press_controller.reset()
            self._press_frame.new_episode()
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
            self._peak_button_x = self._last_button_x

        displacement = self._last_button_x - self._initial_button_x
        if displacement > (self._peak_button_x - self._initial_button_x):
            self._peak_button_x = self._last_button_x
        if displacement > self.PRESS_DEPTH_THRESHOLD:
            self._has_been_pressed = True

        # Success requires: button was driven in past the depth threshold,
        # AND the gripper has retracted far enough along the press axis (X).
        # The button has no spring, so it stays pressed.
        gripper_pos = None
        if self.state is not None:
            gripper_pos = self.state.get('gripper_position')
        if gripper_pos is None:
            return False
        # Use the rigid button mesh (slides under prismatic joint) as the
        # reference — the parent xform is constant and would trivially
        # satisfy the retract check from frame 0.
        ee_to_button = self._last_button_x - float(gripper_pos[0])
        if ee_to_button < self._min_ee_to_button:
            self._min_ee_to_button = ee_to_button
        if ee_to_button < self.EE_CONTACT_DISTANCE:
            self._ee_has_contacted = True
        return (self._has_been_pressed
                and self._ee_has_contacted
                and ee_to_button >= self.RETRACT_DISTANCE)

    def _step_collect(self, state):
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        # Advance to retract only when the EE has actually reached the button
        # AND the button has been pressed deep enough. Without the contact
        # check, the button can drift on its own under gravity and trip
        # ``_has_been_pressed`` before the EE has even arrived.
        if (
            self._has_been_pressed
            and self._ee_has_contacted
            and not self.press_controller.is_done()
            and self.press_controller.get_current_event() == 2
        ):
            self.press_controller._next_event()

        if (
            self.press_controller.is_done()
            and self.check_success_counter < self.REQUIRED_SUCCESS_STEPS
            and self._check_success()
        ):
            hold_action = ArticulationAction(
                joint_positions=np.asarray(state['joint_positions'], dtype=np.float32)
            )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=np.concatenate([state['joint_positions'][:7], [0.0]]),
                    language_instruction=self.get_language_instruction(),
                )
            return hold_action, False, False

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
            # Copy: the atomic controller shifts the target in place along the
            # press axis, and the caller's state dict should not carry that.
            button_position = np.asarray(state['object_position'], dtype=float).copy()
            button_position[2] += self._press_z_offset
            action, record_array = self.press_controller.forward(
                target_position=button_position,
                current_joint_positions=state['joint_positions'],
                gripper_control=self.gripper_control,
                end_effector_orientation=self._press_frame.quat(state['object_position']),
                press_distance=self._press_distance,
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
                'required_press_depth': self.PRESS_DEPTH_THRESHOLD,
                'required_retract_distance': self.RETRACT_DISTANCE,
                'ee_contact_distance': self.EE_CONTACT_DISTANCE,
                'has_been_pressed': self._has_been_pressed,
                'ee_has_contacted': self._ee_has_contacted,
                'min_ee_to_button': self._min_ee_to_button,
                'atomic_done': self.press_controller.is_done(),
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
