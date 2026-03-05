import json
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from controllers.inference_engines.inference_engine_factory import InferenceEngineFactory
from controllers.robot_controllers.grapper_manager import Gripper
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController
from factories.collector_factory import create_collector
from robots.franka.rmpflow_controller import RMPFlowController as FrankaRMPFlowController
from utils.object_utils import ObjectUtils
from utils.replay_data_loader import ReplayDataLoader


class BaseController(ABC):
    """Base class for all controllers in the chemistry lab simulator.

    Provides common functionality for robot control, state management,
    and episode tracking.
    """

    def __init__(self, cfg, robot, use_default_config: bool = True):
        """Initialize the base controller.

        Args:
            cfg: Configuration object containing controller settings.
            robot: Robot instance to control.
            use_default_config: Whether to use the default RMPFlow config.
        """
        self.cfg = cfg
        self.robot = robot
        self.object_utils = ObjectUtils.get_instance()
        self.reset_needed = False
        self._last_success = False
        self._episode_num = 0
        self.success_count = 0
        self._language_instruction = ""
        self._init_state_captured = False
        self.gripper_control = Gripper()
        self.REQUIRED_SUCCESS_STEPS = 60
        self.check_success_counter = 0
        self._last_failure_reason = ""

        self.rmp_controller = FrankaRMPFlowController(
            name="target_follower_controller", robot_articulation=robot, use_default_config=use_default_config
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not OmegaConf.has_resolver("eval"):
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
    def language_instruction(self) -> str | None:
        """Get the current language instruction for the task.

        Returns:
            Optional[str]: The language instruction or None if not set
        """
        return self._language_instruction

    @language_instruction.setter
    def language_instruction(self, instruction: str | None):
        """Set the language instruction for the task.

        Args:
            instruction: The language instruction to set, or None to clear
        """
        self._language_instruction = instruction

    def get_language_instruction(self) -> str | None:
        """Get the language instruction for the current task.
        This method can be overridden by subclasses to provide dynamic instructions.

        Returns:
            Optional[str]: The language instruction or None if not available
        """
        return self._language_instruction

    def step(self, state: dict[str, Any]) -> tuple[Any, bool, bool]:
        """Execute one step of control.

        Dispatches to the mode-specific step method. Stores ``state`` on
        ``self.state`` so that subclass helpers (e.g. ``_check_success``)
        can access the latest observation without extra arguments.

        Args:
            state: Current state dictionary containing sensor data and robot state.

        Returns:
            Tuple of (action, done, is_success).
        """
        self.state = state
        if self.mode == "collect":
            if not self._init_state_captured:
                init = dict(state.get("init_state", {}))
                init["robot_init_joint_positions"] = state["joint_positions"]
                init["robot_world_position"] = np.array(self.robot.get_world_pose()[0], dtype=np.float32)
                self.data_collector.set_init_state(init)
                self._init_state_captured = True
                logger.info(f"Set init state for episode {init}")
            return self._step_collect(state)
        elif self.mode == "replay":
            return self._step_replay(state)
        elif self.mode == "infer":
            return self._step_infer(state)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _init_collect_mode(self, cfg, robot=None):
        """Initialize the controller for collect mode."""
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression,
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
        self._current_actions: Any | None = None
        self._current_action_step = 0
        self._current_init_state: dict | None = None

        # Load the first episode immediately so init_state is available right away.
        if len(self._replay_loader) > 0:
            ep = self._replay_loader.get_episode(0)
            self._current_actions = ep.actions
            self._current_init_state = ep.init_state
            logger.info(f"Episode {ep.episode_idx}: {len(self._current_actions)} actions loaded.")

        # Set reset_needed so the first episode triggers reset_with_init_state()
        # before any actions are consumed.
        self.reset_needed = True
        self._is_initial_replay_reset = True

    def _init_infer_mode(self, cfg, robot=None):
        """Initialize the controller for infer mode."""
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller", robot_articulation=robot, use_interpolation=False
        )

        self.inference_engine = InferenceEngineFactory.create_inference_engine(cfg, self.trajectory_controller)

    @property
    def episode_num(self) -> int:
        """The current episode number."""
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num

    @staticmethod
    def clean_object_name(raw_name: str) -> str:
        """Remove trailing digits and underscores from an object name.

        Example: 'conical_bottle02' -> 'conical bottle'
        """
        return re.sub(r'\d+', '', raw_name).replace('_', ' ').replace('  ', ' ').strip().lower()

    def print_failure_reason(self) -> None:
        """Print the last failure reason if it exists."""
        if self._last_failure_reason:
            logger.warning(f"Failure Reason: {self._last_failure_reason}")

    def reset(self) -> None:
        """Reset the controller state between episodes."""
        # The very first replay reset is just scene setup — no episode has
        # finished yet, so skip the bookkeeping that counts a completed episode.
        if self.mode == "replay" and getattr(self, "_is_initial_replay_reset", False):
            self._is_initial_replay_reset = False
            self.reset_needed = False
            self.check_success_counter = 0
            self._current_action_step = 0
            self.trajectory_controller.reset()
            return

        if self._last_success:
            self.success_count += 1

        self._episode_num += 1

        logger.info(
            f"Episode Stats: Success Rate = {self.success_count}/{self._episode_num} ({(self.success_count / max(self._episode_num, 1)) * 100:.2f}%)"
        )
        self.check_success_counter = 0
        self.reset_needed = False
        self._last_success = False
        self._last_failure_reason = ""
        if self.mode == "collect":
            self._init_state_captured = False
            self.data_collector.clear_cache()
        elif self.mode == "replay":
            # Only reset the action step and trajectory controller here.
            # Episode advancement is handled in _step_replay() when the
            # episode finishes, so that _current_init_state and
            # _current_actions always refer to the same episode when
            # get_current_init_state() is called in the next reset cycle.
            self._current_action_step = 0
            self.trajectory_controller.reset()

    def get_current_init_state(self) -> dict | None:
        """Return the init_state dict for the current replay episode.

        Reconstructs the nested format expected by ``task.reset_with_init_state()``
        from the flat HDF5 representation stored during collection::

            {
                "object_poses":                  {usd_path: {"position": ..., "orientation": ...}},
                "object_materials":              {usd_path: material_path},
                "extra":                         {...},
                "robot_init_joint_positions":    ndarray,
                "robot_world_position":          ndarray,
            }

        Returns:
            Reconstructed dict, or ``None`` if no init state is available.
        """
        if not hasattr(self, "_current_init_state") or self._current_init_state is None:
            return None
        raw = self._current_init_state

        out: dict = {
            "object_poses": {},
            "object_materials": {},
            "extra": {},
            "robot_init_joint_positions": raw.get("robot_init_joint_positions"),
            "robot_world_position": raw.get("robot_world_position"),
        }

        # ---- Reconstruct object poses ----------------------------------------
        if "object_pose_paths" in raw and "object_pose_positions" in raw and "object_pose_orientations" in raw:
            paths = raw["object_pose_paths"]
            if hasattr(paths, "tolist"):
                paths = paths.tolist()
            if np.isscalar(paths) or (isinstance(paths, np.ndarray) and paths.ndim == 0):
                paths = [paths]
            paths = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in paths]
            positions = np.atleast_2d(raw["object_pose_positions"])
            orientations = np.atleast_2d(raw["object_pose_orientations"])
            out["object_poses"] = {
                path: {"position": np.array(positions[i]), "orientation": np.array(orientations[i])}
                for i, path in enumerate(paths)
            }

        # ---- Reconstruct object materials ------------------------------------
        if "object_material_paths" in raw and "object_material_values" in raw:
            mat_paths = raw["object_material_paths"]
            mat_values = raw["object_material_values"]

            def _decode(arr):
                if isinstance(arr, np.ndarray):
                    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr]
                return list(arr)

            out["object_materials"] = dict(zip(_decode(mat_paths), _decode(mat_values)))

        # ---- Reconstruct extra (task-specific) data --------------------------
        if "init_extra_json" in raw:
            try:
                extra_json = raw["init_extra_json"]
                if isinstance(extra_json, np.ndarray):
                    extra_json = extra_json.item()
                if isinstance(extra_json, bytes):
                    extra_json = extra_json.decode("utf-8")
                out["extra"] = json.loads(extra_json)
            except Exception:
                pass

        return out

    @abstractmethod
    def _step_collect(self, state) -> tuple[Any, bool, bool]:
        """Execute one step in collect mode.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple of (action, done, success).
        """
        raise NotImplementedError("Subclasses must implement _step_collect()")

    @abstractmethod
    def _step_infer(self, state) -> tuple[Any, bool, bool]:
        """Execute one step in infer mode.

        Args:
            state: Current state of the environment.

        Returns:
            Tuple of (action, done, success).
        """
        raise NotImplementedError("Subclasses must implement _step_infer()")

    def _advance_replay_episode(self) -> None:
        """Pre-load the next replay episode so that ``get_current_init_state()``
        returns the correct init state on the next reset cycle."""
        if self._current_replay_idx + 1 < len(self._replay_loader):
            self._current_replay_idx += 1
            ep = self._replay_loader.get_episode(self._current_replay_idx)
            self._current_actions = ep.actions
            self._current_init_state = ep.init_state
            logger.info(f"[Replay] Next episode {ep.episode_idx}: {len(self._current_actions)} actions preloaded.")

    def _step_replay(self, state) -> tuple[Any, bool, bool]:
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

        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0

        success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if success:
            self._last_failure_reason = ""
            self._last_success = True
            self.reset_needed = True
            self._advance_replay_episode()
            logger.success("[Replay] Task success!")
            return None, True, True

        all_done = self._current_actions is None or (
            self._current_action_step >= len(self._current_actions)
            and self.trajectory_controller.is_trajectory_complete()
        )
        if all_done:
            logger.warning("[Replay] Task failed — all actions exhausted.")
            self._advance_replay_episode()
            self.reset_needed = True
            return None, True, False

        return action, False, False

    @abstractmethod
    def _check_success(self) -> bool:
        """Evaluate whether the current state meets the task success criterion.

        Subclasses must override this method with task-specific logic.

        Returns:
            bool: True if the task is currently successful.
        """
        ...

    def close(self) -> None:
        """Clean up resources used by the controller."""
        if self.mode == "collect" and hasattr(self, "data_collector"):
            self.data_collector.close()

    def need_reset(self) -> bool:
        """Check if the controller needs to be reset."""
        return self.reset_needed

    def is_success(self) -> bool:
        """Check if the last episode was successful."""
        return self._last_success
