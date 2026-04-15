from typing import Optional
from robots.franka.rmpflow_controller import RMPFlowController
import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers.atomic_actions.close_controller import CloseController
from .base_controller import BaseController
from .robot_controllers.trajectory_controller import FrankaTrajectoryController
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from .inference_engines.inference_engine_factory import InferenceEngineFactory

class CloseTaskController(BaseController):
    """Controller for managing the task of closing a door in collect or infer mode.

    Args:
        cfg: Configuration object containing mode and other parameters.
        robot: Robot articulation instance.
    """

    def __init__(self, cfg, robot):
        self.operate_type = cfg.task.get("operate_type", "door")
        print(self.operate_type)
        super().__init__(cfg, robot)
        self.initial_handle_position = None
            
    def _init_collect_mode(self, cfg, robot):
        """Initializes the controller for data collection mode.

        Args:
            cfg: Configuration object for collect mode.
            robot: Robot articulation instance.
        """
        super()._init_collect_mode(cfg, robot)
        
        self.close_controller = CloseController(
            name="close_controller",
            cspace_controller=RMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot,
            ),
            gripper=robot.gripper,
            robot=robot,
            furniture_type=self.operate_type,
            door_open_direction="clockwise",
        )

    def reset(self):
        """Resets the controller to its initial state."""
        super().reset()
        self.initial_handle_position = None
        if self.mode == "collect":
            self.close_controller.reset()
        elif self.mode == "infer":
            self.inference_engine.reset()

    def step(self, state):
        """Executes one step of the task based on the current state.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        self.state = state
        if self.initial_handle_position is None:
            self.initial_handle_position = np.array(state["object_position"], dtype=np.float32)

        if self.mode == "collect":
            return self._step_collect(state)
        elif self.mode == "infer":
            return self._step_infer(state)
        else:
            return self._step_replay(state)

    def _step_collect(self, state):
        """Executes a step in collect mode using the close controller.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        if not self.close_controller.is_done():
            if self.operate_type == "door":
                action, record_array = self.close_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    revolute_joint_position=state['revolute_joint_position'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([350, 90, 25])).as_quat(),
                    after_move_distance=0.25
                )
            elif self.operate_type == "lid":
                action, record_array = self.close_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=euler_angles_to_quats([0, 130, 0], degrees=True, extrinsic=False),
                    push_distance=0.05,
                    after_move_distance=0.15,
                )
            else:
                action, record_array = self.close_controller.forward(
                    handle_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=euler_angles_to_quats([90, 90, 0], degrees=True, extrinsic=False),
                    push_distance=0.15
                )
            if 'camera_data' in state:
                self.data_collector.cache_step(
                    camera_images=state['camera_data'],
                    joint_angles=state['joint_positions'][:-1],
                    action=record_array,
                    language_instruction=self.get_language_instruction()
                )
            
            if self._check_success():
                self.check_success_counter += 1
            else:
                self.check_success_counter = 0
                
            return action, False, False

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            self._last_failure_reason = ""
            print("Task success!")
            self.data_collector.write_cached_data(state['joint_positions'][:-1])
            self._last_success = True
        else:
            print("Task failed!")
            self.data_collector.clear_cache()
            self._last_success = False
            
        self.reset_needed = True
        return None, True, success

    def _step_infer(self, state):
        """Executes a step in infer mode using the trained policy.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple containing the action, done flag, and success flag.
        """
        language_instruction = self.get_language_instruction()
        if language_instruction is not None:
            state['language_instruction'] = language_instruction
        elif self.operate_type == "lid":
            state['language_instruction'] = "Close the lid of the centrifuge"
        else:
            state['language_instruction'] = "Close the drawer of the object"
        
        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            self._last_failure_reason = ""
            print("Task success!")
            self._last_success = True
            self.reset_needed = True
            return None, True, True
            
        return action, False, False
        
    def _check_success(self):
        """Checks if the task has been successfully completed.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        current_pos = self.state['object_position']
        gripper_position = self.state['gripper_position']
        if self.operate_type == "drawer":
            handle_moved_enough = np.linalg.norm(np.array(current_pos) - self.initial_handle_position)
            gripper_far_enough = np.linalg.norm(np.array(gripper_position) - np.array(current_pos))
            success = handle_moved_enough  > 0.13 and gripper_far_enough > 0.04
            if not success:
                self._last_failure_reason = f"Close task failed: handle moved distance too short ({handle_moved_enough:.4f}<0.13) or gripper too close to object ({gripper_far_enough:.4f}<0.04)"
            else:
                self._last_failure_reason = ""
            return success
        elif self.operate_type == "lid":
            # Lid closes top-to-bottom: check Z decrease
            z_moved = self.initial_handle_position[2] - np.array(current_pos)[2]
            gripper_far_enough = np.linalg.norm(np.array(gripper_position) - np.array(current_pos))
            success = z_moved > 0.03 and gripper_far_enough > 0.04
            if not success:
                self._last_failure_reason = f"Close lid failed: lid Z moved too little ({z_moved:.4f}<0.03) or gripper too close ({gripper_far_enough:.4f}<0.04)"
            else:
                self._last_failure_reason = ""
            return success
        else:
            handle_moved_enough = np.array(current_pos)[0] - self.initial_handle_position[0]
            gripper_far_enough = np.linalg.norm(np.array(gripper_position) - np.array(current_pos))
            success = handle_moved_enough > 0.08 and gripper_far_enough > 0.08
            if not success:
                self._last_failure_reason = f"Close task failed: handle moved distance too short ({handle_moved_enough:.4f}<0.08) or gripper too close to object ({gripper_far_enough:.4f}<0.08)"
            else:
                self._last_failure_reason = ""
            return success

    def get_language_instruction(self) -> Optional[str]:
        object_name = self.clean_object_name(self.state['object_name'])
        if self.operate_type == "lid":
            return self._get_cached_instruction(
                f"close:{self.operate_type}",
                self._build_instruction_templates(
                    f"Close the lid of the {object_name}",
                    f"Close the lid of the {object_name} by pushing it down from above",
                ),
            )
        return self._get_cached_instruction(
            f"close:{self.operate_type}",
            self._build_instruction_templates(
                f"Close the {self.operate_type} of the {object_name}",
                f"Close the {self.operate_type} of the {object_name} by pushing it shut",
            ),
        )
