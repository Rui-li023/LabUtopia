import json
import os
from typing import Optional

import h5py
import numpy as np
from loguru import logger

from .data_collector import DataCollector


class ActionStateDataCollector(DataCollector):
    """Data collector that stores images in HDF5 (compressed) instead of MP4."""

    def cache_step(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        action: np.ndarray,
        language_instruction: Optional[str] = None,
        task_index: Optional[int] = None,
    ) -> None:
        super().cache_step(
            camera_images,
            joint_angles,
            action=action,
            language_instruction=language_instruction,
            task_index=task_index,
        )

    def write_cached_data(self, final_joint_positions=None) -> None:
        if self.episode_count >= self.max_episodes:
            self.close()
            return

        camera_data = {name: np.array(images) for name, images in self.temp_cameras.items()}
        agent_pose_data = np.array(self.temp_agent_pose)
        actions_data = np.array(self.temp_actions)
        task_index_data = np.array(self.temp_task_indices, dtype=np.int32) if self.temp_task_indices else None

        episode_name = f"episode_{self.episode_count:04d}"
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        logger.info(f"Writing episode {episode_name} to {episode_path}")

        future = self.process_pool.submit(
            _write_episode_hdf5,
            episode_path,
            camera_data,
            agent_pose_data,
            actions_data,
            self.temp_task_properties,
            None if task_index_data is not None else self.temp_language_instruction,
            task_index_data,
            self.compression,
            self.temp_init_state,
        )
        self.pending_futures.append(future)

        info = {
            "episode_index": self.episode_count,
            "tasks": self.task_instructions if self.task_instructions else [],
            "length": len(self.temp_agent_pose),
            "task_properties": self.temp_task_properties,
        }
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")

        self.clear_cache()
        self.episode_count += 1


def _write_episode_hdf5(
    episode_path: str,
    camera_data: dict,
    agent_pose_data: np.ndarray,
    actions_data: np.ndarray,
    task_properties: dict,
    language_instruction: Optional[str] = None,
    task_indices: Optional[np.ndarray] = None,
    compression=None,
    init_state: Optional[dict] = None,
) -> None:
    """Write one episode to HDF5 with images stored as datasets."""
    os.makedirs(os.path.dirname(episode_path), exist_ok=True)

    with h5py.File(episode_path, "w") as h5:
        for camera_name, image_data in camera_data.items():
            chunk_size = (min(64, image_data.shape[0]),) + image_data.shape[1:]
            kwargs = {"data": image_data, "dtype": "uint8", "chunks": chunk_size}
            if compression == "gzip":
                kwargs["compression"] = "gzip"
                kwargs["compression_opts"] = 5
            h5.create_dataset(camera_name, **kwargs)

        h5.create_dataset("agent_pose", data=agent_pose_data, dtype="float32", chunks=True)
        h5.create_dataset("actions", data=actions_data, dtype="float32", chunks=True)

        if language_instruction is not None:
            dt = h5py.special_dtype(vlen=str)
            h5.create_dataset("language_instruction", data=language_instruction, dtype=dt, shape=())

        if task_indices is not None:
            h5.create_dataset("task_index", data=task_indices, dtype="int32", chunks=True)

        if task_properties:
            dt = h5py.special_dtype(vlen=str)
            h5.create_dataset("task_properties", data=json.dumps(task_properties, ensure_ascii=False), dtype=dt, shape=())

        if init_state:
            from .data_collector import _flatten_init_state

            flat = _flatten_init_state(init_state) if "object_poses" in init_state or "object_materials" in init_state else init_state
            grp = h5.create_group("init_state")
            str_keys = {"object_pose_paths", "object_material_paths", "object_material_values"}
            str_scalar_keys = {"init_extra_json"}
            float2d_keys = {"object_pose_positions", "object_pose_orientations"}
            for key, value in flat.items():
                if key in str_keys:
                    grp.create_dataset(key, data=np.array(value, dtype=object), dtype=h5py.special_dtype(vlen=str))
                elif key in str_scalar_keys:
                    grp.create_dataset(key, data=value, dtype=h5py.special_dtype(vlen=str), shape=())
                elif key in float2d_keys:
                    grp.create_dataset(key, data=np.asarray(value, dtype="float32"))
                else:
                    try:
                        grp.create_dataset(key, data=np.asarray(value, dtype="float32"))
                    except Exception as exc:
                        logger.error(f"Failed to serialize init_state key '{key}': {exc}")
