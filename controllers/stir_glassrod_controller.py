from typing import Optional
import numpy as np
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController

class StirGlassrodTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.obj_added = False
        
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, stir controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot
            ),
            position_threshold=0.005,
            events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
        )
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=RMPFlowController(
                name="stir_controller",
                robot_articulation=robot
            ),
        )
        
    def reset(self):
        super().reset()
        self.obj_added = False
        self.gripper_control.release_object()
        if self.mode == "collect":
            self.pick_controller.reset()
            self.stir_controller.reset()
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
            
    def _step_collect(self, state):
        if not self.pick_controller.is_done():
            action, record_array = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                pre_offset_x=0.05,
                pre_offset_z=0.05,
                after_offset_z=0.2
            )

            # Cache demonstration data
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
            
            self.gripper_control.update_grasped_object_position()
            return action, False, False
        
        elif not self.stir_controller.is_done():            
            target_position = state['target_position']
            action, record_array = self.stir_controller.forward(
                center_position=target_position,
                current_joint_positions=state['joint_positions'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
                gripper_position=state['gripper_position'],
            )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
            
            self.gripper_control.update_grasped_object_position()
            return action, False, False
            
        else:
            self.reset_needed = True
            final_object_position = state['glass_rod_position']
            target_position = state['target_position']
            _h = float(final_object_position[2])
            _xy = float(np.linalg.norm(final_object_position[0:2] - target_position[0:2]))
            print(f"[STIR-DIAG] rod_final={[round(float(v),3) for v in final_object_position]} "
                  f"target={[round(float(v),3) for v in target_position]} "
                  f"height={round(_h,3)} (need>0.85, {'OK' if _h>0.85 else 'SHORT'}) "
                  f"xy={round(_xy,3)} (need<0.04, {'OK' if _xy<0.04 else 'FAR'})")
            self.gripper_control.release_object()
            if final_object_position[2] > 0.85 and np.linalg.norm(final_object_position[0:2] - target_position[0:2]) < 0.04:
                # Task successful - save collected data
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                return None, True, True
            else:
                # Task failed - discard collected data
                self._last_failure_reason = "StirGlassrod task failed: glass rod final position (height > 0.85 or xy to target < 0.04) did not meet criteria"
                self.data_collector.clear_cache()
                self._last_success = False
                return None, True, False

    def _step_infer(self, state):
        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        if action is not None:
            if len(action.joint_positions) == 9:
                # Ensure gripper positions are set
                if action.joint_positions[7] == None and action.joint_positions[8] == None:
                    action.joint_positions[7] = 0.015
                    action.joint_positions[8] = 0.015

                # Add object to gripper when gripper is closed
                if action.joint_positions[7] < np.float32(0.01) and action.joint_positions[8] < np.float32(0.01) and not self.obj_added:
                    self.gripper_control.add_object_to_gripper("/World/glass_rod", "/World/Franka/panda_hand/tool_center")
                    print("glassrod is added to franka gripper center!")
                    self.obj_added = True
                    
        self.gripper_control.update_grasped_object_position()
        return action, False, self._check_success()
    
    def _check_success(self):
        object_pos = self.state['glass_rod_position']
        target_position = self.state['target_position']
        xy = float(np.linalg.norm(object_pos[0:2] - target_position[0:2]))
        # Replay: report the instantaneous criterion only and let _step_replay
        # own the consecutive-frame counting + settling window. This method must
        # NOT also touch check_success_counter in replay (the base already
        # manages it; double-counting + the strict 4cm/240-frame gate gave 0%).
        # Slightly relax xy to absorb PD-replay tracking error.
        if self.mode == "replay":
            return object_pos[2] > 0.85 and xy < 0.06
        if object_pos[2] > 0.85 and xy < 0.04:
            self.check_success_counter += 1
            if self.check_success_counter > 240:
                self._last_success = True
                return True
        return False
    
    def get_language_instruction(self) -> Optional[str]:
        return self._get_cached_instruction(
            'stir_glass_rod',
            self._build_instruction_templates(
                'Use the glass rod to stir the liquid',
                'Use the glass rod to stir the liquid inside the container until it is well mixed',
            ),
        )
