from typing import Optional
import numpy as np

from isaacsim.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController
class StirTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            position_threshold=0.005,
            events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02]
        )
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=self.rmp_controller,
        )

        self.gripper_control.release_object()
        self._last_pick_joint_data = None

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)

        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt = [0.004, 0.002, 0.005, 1, 0.05, 0.004, 1]
        )
        self.use_stir_model = False
        self.frame_count = 0

    def _init_replay_mode(self, cfg, robot):
        """Replay needs the scripted pick to bring the glass rod into the
        gripper before the recorded stir actions are fed.
        Disable randomization so the post-pick pose is reproducible."""
        super()._init_replay_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller_replay",
            cspace_controller=self.rmp_controller,
            position_threshold=0.005,
            events_dt=[0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02],
        )
        self.pick_controller._sample_randomization = lambda: None

    def reset(self):
        super().reset()
        self.gripper_control.release_object()
        self.pick_controller.reset()
        if self.mode == "collect":
            self.stir_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()
        self.initial_position = None
        self.use_stir_model = False
        self.frame_count = 0

    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        else:
            return self._step_infer(state)

    def _step_replay(self, state):
        """Run scripted pick first; then dispatch to BaseController._step_replay
        which feeds the recorded stir trajectory. The grasped glass rod's mesh
        needs its world pose synced to the gripper every frame."""
        if not self.pick_controller.is_done():
            action, _ = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                after_offset_z=0.15,
                gripper_distances=0.005,
            )
            self.gripper_control.update_grasped_object_position()
            return action, False, False
        result = super()._step_replay(state)
        self.gripper_control.update_grasped_object_position()
        return result
        
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
            action, _ = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                after_offset_z= 0.15,
                gripper_distances=0.005
            )
            
            self.gripper_control.update_grasped_object_position()

            return action, False, False
            
        elif not self.stir_controller.is_done():
            target_position = self.object_utils.get_object_xform_position(
                object_path=state['target_beaker']
            )
            if target_position is None:
                target_position = state['target_position']
            
            action, record_array = self.stir_controller.forward(
                center_position=target_position,
                current_joint_positions=state['joint_positions'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
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
            self.gripper_control.release_object()
            if (final_object_position is not None and 
                final_object_position[2] > 0.85 and
                np.linalg.norm(final_object_position[0:2] - target_position[0:2]) < 0.04):
                self._last_failure_reason = ""
                self.data_collector.write_cached_data(state['joint_positions'][:-1])
                self._last_success = True
                return None, True, True
            else:
                self._last_failure_reason = "Stir task failed: glass rod final position (height > 0.85 or xy to target < 0.04) did not meet criteria"
                self.data_collector.clear_cache()
                self._last_success = False
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
        if not self.pick_controller.is_done():
            action, _ = self.pick_controller.forward(
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 10])).as_quat(),
                after_offset_z=0.15,
            )
            
            final_object_position = state['glass_rod_position']
            if final_object_position is not None and final_object_position[2] > 0.82:
                self.use_stir_model = True

            self.gripper_control.update_grasped_object_position()

            return action, False, False
        
        state['language_instruction'] = self.get_language_instruction()
        
        if self.use_stir_model:
            action = self.inference_engine.step_inference(state)
            self.gripper_control.update_grasped_object_position()

            return action, False, self._check_success()
        
        return ArticulationAction(), False, False
    
    def _check_success(self):
        object_pos = self.state['glass_rod_position']
        target_position = self.state['target_position']
        criterion_met = (object_pos[2] > 0.85
                         and np.linalg.norm(object_pos[0:2] - target_position[0:2]) < 0.04)
        # In replay the BaseController._step_replay loop manages the
        # success-counter via self.check_success_counter, so don't fight it.
        if self.mode == "replay":
            return criterion_met
        if criterion_met:
            self.check_success_counter += 1
            if self.check_success_counter > 240:
                self._last_success = True
                return True
        return False

    def get_language_instruction(self) -> Optional[str]:
        return self._get_cached_instruction(
            'stir',
            self._build_instruction_templates(
                'Use the glass rod to stir the liquid',
                'Use the glass rod to stir the liquid until it is well mixed',
            ),
        )
