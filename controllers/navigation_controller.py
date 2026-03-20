import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_controller import BaseController
from .robot_controllers.ridgebase.ridgebase_controller import RidgebaseController


class NavigationController(BaseController):
    """
    Navigation controller for controlling the Ridgebase robot to navigate along path points.
    
    Supports two modes:
    - collect mode：Collect navigation trajectory data
    - infer mode：Use learned policies to navigate (reserved interface)
    
    Attributes:
        ridgebase_controller: Ridgebase low-level motion controller
        waypoints_set: Whether the path points have been set
    """
    
    def __init__(self, cfg, robot):
        """
        Initialize the navigation controller.
        
        Args:
            cfg: Configuration object
            robot: Robot instance
        """
        try:
            super().__init__(cfg, robot, use_default_config=False)
        except Exception as e:
            self.cfg = cfg
            self.robot = robot
            self.object_utils = None
            self.reset_needed = False
            self._last_success = False
            self._episode_num = 0
            self.success_count = 0
            self._language_instruction = ""
            self.REQUIRED_SUCCESS_STEPS = 60
            self.check_success_counter = 0
            self.rmp_controller = None
            self.gripper_control = None
            self._last_failure_reason = str(e) if e else "Navigation controller init failed"
            
            if hasattr(cfg, "mode"):
                self.mode = cfg.mode
                if self.mode == "collect":
                    self._init_collect_mode(cfg, robot)
                elif self.mode == "infer":
                    self._init_infer_mode(cfg, robot)
        
        self.ridgebase_controller = RidgebaseController(
            robot_articulation=robot,
            max_linear_speed=cfg.task.max_linear_speed if hasattr(cfg.task, 'max_linear_speed') else 0.02,
            max_angular_speed=cfg.task.max_angular_speed if hasattr(cfg.task, 'max_angular_speed') else 1.5,
            position_threshold=cfg.task.position_threshold if hasattr(cfg.task, 'position_threshold') else 0.08,
            angle_threshold=cfg.task.angle_threshold if hasattr(cfg.task, 'angle_threshold') else 0.1
        )
        
        self.waypoints_set = False
        
    def reset(self) -> None:
        """Reset the controller state"""
        super().reset()
        self.waypoints_set = False
    
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Perform one control step.
        
        Args:
            state: The current state dictionary, containing current_pose, waypoints etc.
            
        Returns:
            tuple: (action, done, is_success)
                - action: The control action
                - done: Whether the task is complete
                - is_success: Whether the task is successful
        """
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
    
    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Control step in collect mode.
        
        Args:
            state: The state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        if not self.waypoints_set and state.get('waypoints') is not None:
            self.ridgebase_controller.set_waypoints(state['waypoints'])
            self.waypoints_set = True
            
            # record the start and end positions of the navigation task
            if hasattr(self, 'data_collector') and hasattr(self.data_collector, 'set_task_properties'):
                waypoints = state['waypoints']
                task_properties = {
                    "start_position": waypoints[0].tolist() if hasattr(waypoints[0], 'tolist') else list(waypoints[0]),
                    "end_position": waypoints[-1].tolist() if hasattr(waypoints[-1], 'tolist') else list(waypoints[-1]),
                    "num_waypoints": len(waypoints)
                }
                self.data_collector.set_task_properties(task_properties)
        
        current_pose = state['current_pose']

        action, done = self.ridgebase_controller.get_action(current_pose)

        if 'camera_data' in state and not done:
            joint_positions = np.array([
                current_pose[0],
                current_pose[1],
                current_pose[2]
            ])
            
            # convert action to numpy array
            action_array = np.array(action) if not isinstance(action, np.ndarray) else action
            
            self.data_collector.cache_step(
                camera_images=state['camera_data'],
                joint_angles=joint_positions,
                action=action_array,
                language_instruction=self.get_language_instruction()
            )
        
        if done or self.ridgebase_controller.is_path_complete():
            self._last_failure_reason = ""
            self._last_success = True
            self.reset_needed = True
            
            if hasattr(self, 'data_collector'):
                # cache the last step data
                if 'camera_data' in state:
                    joint_positions = np.array([
                        current_pose[0],
                        current_pose[1],
                        current_pose[2]
                    ])
                    action_array = np.array(action) if not isinstance(action, np.ndarray) else action
                    
                    self.data_collector.cache_step(
                        camera_images=state['camera_data'],
                        joint_angles=joint_positions,
                        action=action_array,
                        language_instruction=self.get_language_instruction()
                    )
                
                # write cached data (no final_joint_positions parameter)
                self.data_collector.write_cached_data()
            
            return action, True, True
        
        return action, False, False
    
    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion."""
        if not self.waypoints_set:
            return False
        return self.ridgebase_controller.is_path_complete()

    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """
        Control step in inference mode (reserved interface).
        
        Args:
            state: The state dictionary
            
        Returns:
            tuple: (action, done, is_success)
        """
        if not self.waypoints_set and state.get('waypoints') is not None:
            self.ridgebase_controller.set_waypoints(state['waypoints'])
            self.waypoints_set = True
        
        # Get the current pose
        current_pose = state['current_pose']
        
        # Calculate the control action
        action, done = self.ridgebase_controller.get_action(current_pose)
        
        # If the navigation is complete
        if done or self.ridgebase_controller.is_path_complete():
            self._last_failure_reason = ""
            self._last_success = True
            self.reset_needed = True
            return action, True, True
        
        return action, False, False
    
    def _init_collect_mode(self, cfg, robot=None):
        """Initialize the collect mode"""
        from factories.collector_factory import create_collector
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )
    
    def _init_infer_mode(self, cfg, robot=None):
        """Initialize the inference mode (reserved interface)"""
        pass
    
    def get_language_instruction(self) -> Optional[str]:
        return self._get_cached_instruction(
            'navigate',
            self._build_instruction_templates(
                'Navigate to the target position',
                'Move the robot to the target position and stop there accurately',
            ),
        )
