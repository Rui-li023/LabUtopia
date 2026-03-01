import os
import numpy as np
import h5py
import json
from typing import List, Optional
from loguru import logger

from .data_collector import DataCollector


class ActionStateDataCollector(DataCollector):
    """Data collector that stores images in HDF5 (compressed) instead of MP4.

    Suitable for scenarios where action and state are recorded separately,
    such as navigation tasks.  Supports per-episode task properties.
    """

    # ------------------------------------------------------------------
    # cache_step: action is REQUIRED (not optional like in DataCollector)
    # ------------------------------------------------------------------

    def cache_step(self, camera_images: dict, joint_angles: np.ndarray,
                   action: np.ndarray, language_instruction: Optional[str] = None) -> None:
        """Cache one step's data.

        Same as :meth:`DataCollector.cache_step` except ``action`` is required.
        """
        super().cache_step(camera_images, joint_angles, action=action,
                           language_instruction=language_instruction)

    # ------------------------------------------------------------------
    # write_cached_data: images go into HDF5 (not MP4) — override fully
    # ------------------------------------------------------------------

    def write_cached_data(self, final_joint_positions=None) -> None:
        """Write cached data to an HDF5 file with images stored inside."""
        if self.episode_count >= self.max_episodes:
            self.close()
            return

        # Convert cached lists to arrays
        camera_data = {
            name: np.array(images)
            for name, images in self.temp_cameras.items()
        }
        agent_pose_data = np.array(self.temp_agent_pose)
        actions_data = np.array(self.temp_actions)

        episode_name = f"episode_{self.episode_count:04d}"
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        logger.info(f"Writing episode {episode_name} to {episode_path}")

        future = self.process_pool.submit(
            _write_episode_hdf5,
            episode_path,
            episode_name,
            camera_data,
            agent_pose_data,
            actions_data,
            self.temp_task_properties,
            self.temp_language_instruction,
            self.compression,
            self.temp_init_state,
        )
        self.pending_futures.append(future)

        # Episode metadata
        info = {
            "episode_index": self.episode_count,
            "tasks": [self.task_instructions] if self.task_instructions else [],
            "length": len(self.temp_agent_pose),
            "task_properties": self.temp_task_properties,
        }
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

        # Clear cache (reuse parent's clear_cache then bump count)
        self.clear_cache()
        # clear_cache resets task_instructions; restore episode_count
        # (parent's clear_cache does NOT bump episode_count)
        self.episode_count += 1


# ---------------------------------------------------------------------------
# Free function executed in worker process
# ---------------------------------------------------------------------------

def _write_episode_hdf5(episode_path: str, episode_name: str,
                        camera_data: dict, agent_pose_data: np.ndarray,
                        actions_data: np.ndarray, task_properties: dict,
                        language_instruction: Optional[str] = None,
                        compression=None,
                        init_state: Optional[dict] = None) -> None:
    """Write one episode to HDF5 with images stored as datasets."""
    os.makedirs(os.path.dirname(episode_path), exist_ok=True)

    with h5py.File(episode_path, "w") as h5:
        # Camera image data — stored compressed inside HDF5
        for camera_name, image_data in camera_data.items():
            chunk_size = (min(64, image_data.shape[0]),) + image_data.shape[1:]
            kwargs = {"data": image_data, "dtype": "uint8", "chunks": chunk_size}
            if compression == "gzip":
                kwargs["compression"] = "gzip"
                kwargs["compression_opts"] = 5
            h5.create_dataset(camera_name, **kwargs)

        # Pose & action
        h5.create_dataset("agent_pose", data=agent_pose_data, dtype="float32", chunks=True)
        h5.create_dataset("actions", data=actions_data, dtype="float32", chunks=True)

        # Language instruction
        if language_instruction is not None:
            dt = h5py.special_dtype(vlen=str)
            h5.create_dataset("language_instruction", data=language_instruction, dtype=dt, shape=())

        # Task properties
        if task_properties:
            dt = h5py.special_dtype(vlen=str)
            h5.create_dataset(
                "task_properties",
                data=json.dumps(task_properties, ensure_ascii=False),
                dtype=dt, shape=(),
            )

        # Init state (reuse DataCollector's flattening logic)
        if init_state:
            from .data_collector import _flatten_init_state
            flat = (_flatten_init_state(init_state)
                    if "object_poses" in init_state or "object_materials" in init_state
                    else init_state)
            grp = h5.create_group("init_state")
            _STR_KEYS = {"object_pose_paths", "object_material_paths", "object_material_values"}
            _STR_SCALAR_KEYS = {"init_extra_json"}
            _FLOAT2D_KEYS = {"object_pose_positions", "object_pose_orientations"}
            for key, val in flat.items():
                if key in _STR_KEYS:
                    grp.create_dataset(key, data=np.array(val, dtype=object),
                                       dtype=h5py.special_dtype(vlen=str))
                elif key in _STR_SCALAR_KEYS:
                    grp.create_dataset(key, data=val,
                                       dtype=h5py.special_dtype(vlen=str), shape=())
                elif key in _FLOAT2D_KEYS:
                    grp.create_dataset(key, data=np.asarray(val, dtype="float32"))
                else:
                    try:
                        grp.create_dataset(key, data=np.array(val, dtype="float32"))
                    except Exception:
                        logger.error(f"Failed to serialize init_state key: {key}")

    logger.info(f"Finished writing episode {episode_name}")
