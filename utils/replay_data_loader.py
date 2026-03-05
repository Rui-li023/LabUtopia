import os
import glob
import numpy as np
import h5py
from typing import List, Dict, Optional
from loguru import logger


class EpisodeData:
    """Container for a single replayed episode's data."""

    def __init__(self, episode_idx: int, actions: np.ndarray, init_state: Dict[str, np.ndarray]):
        self.episode_idx = episode_idx
        self.actions = actions          # [T, n_joints]
        self.init_state = init_state    # keys: object_init_position, robot_init_joint_positions, robot_world_position


class ReplayDataLoader:
    """
    Unified utility for loading replay episodes from a dataset directory.

    Each episode lives in  <dataset_path>/episode_XXXX/episode_XXXX.h5
    Expected H5 layout per episode:
        /actions                           float32 [T, n_joints]
        /agent_pose                        float32 [T, n_joints]
        /init_state/object_init_position   float32 [3]
        /init_state/robot_init_joint_positions float32 [n_joints]
        /init_state/robot_world_position   float32 [3]
    """

    def __init__(self, dataset_path: str, episode_indices: Optional[List[int]] = None):
        """
        Args:
            dataset_path: Root dataset directory (contains episode_XXXX sub-dirs).
            episode_indices: If given, only load the specified episode indices.
                             If None, load all episodes found in the directory.
        """
        self.dataset_path = dataset_path
        self._episodes: List[EpisodeData] = []
        self._load(episode_indices)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> EpisodeData:
        return self._episodes[idx]

    def get_episode(self, idx: int) -> EpisodeData:
        return self._episodes[idx]

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self, episode_indices: Optional[List[int]]):
        episode_dirs = sorted(
            [e for e in os.scandir(self.dataset_path) if e.is_dir()],
            key=lambda e: e.name,
        )

        loaded = 0
        for ep_dir in episode_dirs:
            h5_files = glob.glob(os.path.join(ep_dir.path, "*.h5"))
            if not h5_files:
                continue
            h5_path = h5_files[0]

            # Parse episode index from directory name (episode_XXXX)
            try:
                ep_idx = int(ep_dir.name.split("_")[-1])
            except ValueError:
                ep_idx = loaded

            if episode_indices is not None and ep_idx not in episode_indices:
                continue

            episode = self._load_single(ep_idx, h5_path)
            if episode is not None:
                self._episodes.append(episode)
                loaded += 1

        logger.info(f"Loaded {loaded} episodes from {self.dataset_path}")

    def _load_single(self, ep_idx: int, h5_path: str) -> Optional[EpisodeData]:
        try:
            with h5py.File(h5_path, "r") as f:
                if "actions" not in f:
                    logger.warning(f"{h5_path} has no 'actions' dataset, skipping.")
                    return None

                actions = f["actions"][:]

                init_state: Dict[str, np.ndarray] = {}
                if "init_state" in f:
                    for key in f["init_state"]:
                        dataset = f[f"init_state/{key}"]
                        # Handle scalar datasets (shape=()) correctly
                        if dataset.shape == ():
                            init_state[key] = dataset[()]
                        else:
                            init_state[key] = dataset[:]
                else:
                    logger.warning(f"{h5_path} has no 'init_state' group. "
                                   "Scene will be reset randomly for this episode.")

            return EpisodeData(ep_idx, actions, init_state)

        except Exception as e:
            logger.error(f"Error loading {h5_path}: {e}")
            return None
