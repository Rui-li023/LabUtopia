import numpy as np
from typing import List, Optional

from .data_collector import DataCollector


class MockCollector(DataCollector):
    """Mock data collector for testing — same interface as DataCollector but does nothing."""

    def __init__(self, camera_configs: List[dict], save_dir="output",
                 max_episodes=10, max_workers=4, compression=None):
        # Minimal init: only set attributes that controllers read, skip I/O and process pool.
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.session_dir = ""
        self.episode_count = 0
        self.camera_configs = camera_configs
        self.task_instructions = None
        self.temp_cameras = {}
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self.temp_init_state: Optional[dict] = None
        self.pending_futures = []

    def cache_step(self, camera_images: dict, joint_angles: np.ndarray,
                   action: Optional[np.ndarray] = None,
                   language_instruction: Optional[str] = None) -> None:
        pass

    def set_init_state(self, init_state: dict) -> None:
        pass

    def set_init_state_from_step(self, state: dict) -> None:
        pass

    def set_task_properties(self, properties: dict) -> None:
        pass

    def write_cached_data(self, final_joint_positions=None) -> None:
        self.episode_count += 1

    def clear_cache(self) -> None:
        pass

    def close(self, merge=False) -> None:
        pass
