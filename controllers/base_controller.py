from abc import ABC, abstractmethod 
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from omegaconf import OmegaConf
from controllers.inference_engines.inference_engine_factory import InferenceEngineFactory
from controllers.robot_controllers.grapper_manager import Gripper
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector
from utils.object_utils import ObjectUtils
from utils.replay_data_loader import ReplayDataLoader
from robots.franka.rmpflow_controller import RMPFlowController as FrankaRMPFlowController

class BaseController(ABC):
    """Base class for all controllers in the chemistry lab simulator.
    
    Provides common functionality for robot control, state management,
    and episode tracking.
    """
    
    def __init__(self, cfg, robot, use_default_config=True):
        """Initialize the base controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
            object_utils: Utility class for object manipulation
        """
        self.cfg = cfg
        self.robot = robot
        self.object_utils = ObjectUtils.get_instance()
        self.reset_needed = False
        self._last_success = False
        self._episode_num = 0
        self.success_count = 0 
        self._language_instruction = ""
        self.gripper_control = Gripper()
        self.REQUIRED_SUCCESS_STEPS = 60
        self.check_success_counter = 0
        self.rmp_controller = None
        self._last_failure_reason = ""
        
        self.rmp_controller = FrankaRMPFlowController(
            name="target_follower_controller",
            robot_articulation=robot,
            use_default_config=use_default_config
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        if hasattr(cfg, "mode"):
            self.mode = cfg.mode  # "collect", "infer", or "replay"
            if self.mode == "collect":
                self._init_collect_mode(cfg, robot)
            elif self.mode == "infer":
                self._init_infer_mode(cfg, robot)
            elif self.mode == "replay":
                self._init_replay_mode(cfg, robot)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Expected 'collect', 'infer', or 'replay'.")

    @property
    def language_instruction(self) -> Optional[str]:
        """Get the current language instruction for the task.
        
        Returns:
            Optional[str]: The language instruction or None if not set
        """
        return self._language_instruction
    
    @language_instruction.setter
    def language_instruction(self, instruction: Optional[str]):
        """Set the language instruction for the task.
        
        Args:
            instruction: The language instruction to set, or None to clear
        """
        self._language_instruction = instruction
    
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        This method can be overridden by subclasses to provide dynamic instructions.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        return self._language_instruction
    
    @abstractmethod
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing:
            - action: Control action to execute
            - done: Whether the episode is complete
            - is_success: Whether the task was completed successfully
        """
        pass
    
    def _init_collect_mode(self, cfg, robot=None):
        """Initialize the controller for collect mode."""
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression
        )
    
    def _init_replay_mode(self, cfg, robot=None):
        """Initialize the controller for replay mode.

        Creates a trajectory controller (no interpolation) and a ReplayDataLoader,
        then immediately loads the first episode so callers can access
        get_current_init_state() before the first reset().

        Args:
            cfg: Configuration object.  Must contain cfg.replay.dataset_path.
            robot: Robot articulation instance.
        """
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller_replay",
            robot_articulation=robot,
            use_interpolation=False,
        )

        episode_indices = list(cfg.replay.episode_indices) if hasattr(cfg.replay, "episode_indices") else None
        self._replay_loader = ReplayDataLoader(
            dataset_path=cfg.replay.dataset_path,
            episode_indices=episode_indices,
        )
        self._current_replay_idx = 0
        self._current_actions: Optional[Any] = None
        self._current_action_step = 0
        self._current_init_state: Optional[dict] = None

        # Load the first episode immediately so init_state is available right away.
        if len(self._replay_loader) > 0:
            ep = self._replay_loader.get_episode(0)
            self._current_actions = ep.actions
            self._current_init_state = ep.init_state
            print(f"[Replay] Episode {ep.episode_idx}: {len(self._current_actions)} actions loaded.")

    def _init_infer_mode(self, cfg, robot=None):
        """Initialize the controller for infer mode."""
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot,
            use_interpolation=False
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
    
    def episode_num(self) -> int:
        """Get the current episode number.

        Returns:
            int: Current episode number
        """
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num
    
    def print_failure_reason(self) -> None:
        """Print the last failure reason if it exists."""
        if self._last_failure_reason:
            print(f"Failure Reason: {self._last_failure_reason}")
    
    def reset(self) -> None:
        """Reset the controller state between episodes."""
        if self._last_success:
            self.success_count += 1
        self._episode_num += 1
        print(f"Episode Stats: Success Rate = {self.success_count}/{self._episode_num} ({(self.success_count/self._episode_num)*100:.2f}%)")
        self.check_success_counter = 0
        self.reset_needed = False
        self._last_success = False
        self._last_failure_reason = ""
        if self.mode == "collect":
            self.data_collector.clear_cache()
        elif self.mode == "replay":
            self._current_replay_idx += 1
            if self._current_replay_idx < len(self._replay_loader):
                ep = self._replay_loader.get_episode(self._current_replay_idx)
                self._current_actions = ep.actions
                self._current_init_state = ep.init_state
                self._current_action_step = 0
                self.trajectory_controller.reset()
                print(f"[Replay] Episode {ep.episode_idx}: {len(self._current_actions)} actions loaded.")

        
    def get_current_init_state(self) -> Optional[dict]:
        """Return the init_state dict for the current replay episode.

        Called by main.py so it can pass the recorded state to
        task.reset_with_init_state() instead of task.reset().

        Returns:
            Dict with 'object_poses', 'robot_init_joint_positions',
            'robot_world_position'; or None if unavailable.
        """
        if not hasattr(self, "_current_init_state") or self._current_init_state is None:
            return None
        raw = self._current_init_state
        out = {
            "robot_init_joint_positions": raw.get("robot_init_joint_positions"),
            "robot_world_position": raw.get("robot_world_position"),
        }
        if "object_pose_paths" in raw and "object_pose_positions" in raw and "object_pose_orientations" in raw:
            paths = raw["object_pose_paths"]
            if hasattr(paths, "tolist"):
                paths = paths.tolist()
            if np.isscalar(paths) or (isinstance(paths, np.ndarray) and paths.ndim == 0):
                paths = [paths]
            if isinstance(paths, np.ndarray):
                paths = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in paths]
            positions = np.atleast_2d(raw["object_pose_positions"])
            orientations = np.atleast_2d(raw["object_pose_orientations"])
            out["object_poses"] = {
                path: {"position": np.array(positions[i]), "orientation": np.array(orientations[i])}
                for i, path in enumerate(paths)
            }
        return out

    def _step_replay(self, state) -> Tuple[Any, bool, bool]:
        """Execute one step in replay mode.

        Feeds recorded waypoints through the trajectory controller one at a
        time and evaluates success via _check_success().

        Args:
            state: Current state of the environment.

        Returns:
            Tuple of (action, done, success).
        """
        if (
            self._current_actions is not None
            and self.trajectory_controller.is_trajectory_complete()
            and self._current_action_step < len(self._current_actions)
        ):
            chunk = self._current_actions[self._current_action_step : self._current_action_step + 1]
            self.trajectory_controller.generate_trajectory(chunk)
            self._current_action_step += 1

        action = self.trajectory_controller.get_next_action()

        if self._check_success(state):
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            self._last_failure_reason = None
            self._last_success = True
            self.reset_needed = True
            print("[Replay] Task success!")
            return None, True, True

        all_done = self._current_actions is None or (
            self._current_action_step >= len(self._current_actions)
            and self.trajectory_controller.is_trajectory_complete()
        )
        if all_done:
            print("[Replay] Task failed — all actions exhausted.")
            self.reset_needed = True
            return None, True, False

        return action, False, False

    def _check_success(self, state) -> bool:
        """Evaluate whether the current state meets the task success criterion.

        Subclasses must override this method with task-specific logic.

        Args:
            state: Current state of the environment.

        Returns:
            bool: True if the task is currently successful.
        """
        raise NotImplementedError("Subclasses must implement _check_success()")

    def close(self) -> None:
        """Clean up resources used by the controller."""
        if self.mode == "collect" and hasattr(self, "data_collector"):
            self.data_collector.close()
        
    def need_reset(self) -> bool:
        """Check if the controller needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed

    def is_success(self):
        return self._last_success