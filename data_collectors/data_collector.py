import os
import json
from concurrent.futures import Future, ProcessPoolExecutor
from glob import glob
from typing import List, Optional

import cv2
import h5py
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Init-state serialisation helpers
# ---------------------------------------------------------------------------

def _flatten_init_state(init_state: dict) -> dict:
    """Convert the nested init_state dict produced by BaseTask into flat
    numpy/string arrays suitable for HDF5 storage.

    Input schema::

        {
            "object_poses":     {usd_path: {"position": [x,y,z], "orientation": [x,y,z,w]}},
            "object_materials": {usd_path: material_usd_path},
            "extra":            {arbitrary key-value pairs},
            # legacy keys also accepted as-is
        }

    Output schema (flat, for HDF5)::

        object_pose_paths         – vlen-str array of USD paths
        object_pose_positions     – float32 [N, 3]
        object_pose_orientations  – float32 [N, 4]
        object_material_paths     – vlen-str array of object USD paths
        object_material_values    – vlen-str array of material USD paths
        init_extra_json           – JSON string for arbitrary extra data
        (any other legacy float keys are stored as float32)
    """
    flat: dict = {}

    poses = init_state.get("object_poses", {})
    if poses:
        paths = list(poses.keys())
        flat["object_pose_paths"] = paths
        flat["object_pose_positions"] = np.array([poses[p]["position"] for p in paths], dtype="float32")
        flat["object_pose_orientations"] = np.array([poses[p]["orientation"] for p in paths], dtype="float32")

    materials = init_state.get("object_materials", {})
    if materials:
        flat["object_material_paths"] = list(materials.keys())
        flat["object_material_values"] = list(materials.values())

    extra = init_state.get("extra", {})
    if extra:
        flat["init_extra_json"] = json.dumps(extra)

    legacy_skip = {"object_poses", "object_materials", "extra"}
    for key, value in init_state.items():
        if key not in legacy_skip and key not in flat:
            flat[key] = value

    return flat


def _write_episode_data(
    episode_dir: str,
    episode_name: str,
    camera_data: dict,
    agent_pose_data: np.ndarray,
    actions_data: np.ndarray,
    task_properties: dict = None,
    language_instruction: Optional[str] = None,
    task_indices: Optional[np.ndarray] = None,
    compression=None,
    init_state: Optional[dict] = None,
):
    """Write one episode's data to an HDF5 file and camera videos."""
    os.makedirs(episode_dir, exist_ok=True)
    episode_path = os.path.join(episode_dir, f"{episode_name}.h5")
    logger.info(f"Writing episode {episode_name} to {episode_dir}")

    with h5py.File(episode_path, "w") as h5_file:
        h5_file.create_dataset("agent_pose", data=agent_pose_data, dtype="float32", chunks=True)
        h5_file.create_dataset("actions", data=actions_data, dtype="float32", chunks=True)

        # Keep scalar language only for backward compatibility.
        if language_instruction is not None:
            dt = h5py.special_dtype(vlen=str)
            h5_file.create_dataset("language_instruction", data=language_instruction, dtype=dt, shape=())

        if task_indices is not None:
            h5_file.create_dataset("task_index", data=task_indices, dtype="int32", chunks=True)

        if task_properties:
            task_properties_json = json.dumps(task_properties, ensure_ascii=False)
            dt = h5py.special_dtype(vlen=str)
            h5_file.create_dataset("task_properties", data=task_properties_json, dtype=dt, shape=())

        if init_state:
            flat = _flatten_init_state(init_state) if "object_poses" in init_state or "object_materials" in init_state else init_state
            grp = h5_file.create_group("init_state")
            str_keys = {"object_pose_paths", "object_material_paths", "object_material_values"}
            str_scalar_keys = {"init_extra_json"}
            float2d_keys = {"object_pose_positions", "object_pose_orientations"}
            for key, value in flat.items():
                if key in str_keys:
                    grp.create_dataset(key, data=np.array(value, dtype=object), dtype=h5py.special_dtype(vlen=str))
                elif key in str_scalar_keys:
                    dt = h5py.special_dtype(vlen=str)
                    grp.create_dataset(key, data=value, dtype=dt, shape=())
                elif key in float2d_keys:
                    grp.create_dataset(key, data=np.asarray(value, dtype="float32"))
                else:
                    try:
                        grp.create_dataset(key, data=np.array(value, dtype="float32"))
                    except Exception as exc:
                        logger.error(f"Failed to serialize init_state key '{key}': {exc}")

    for camera_name, image_data in camera_data.items():
        if image_data.ndim != 4:
            logger.warning(f"camera_data {camera_name} has wrong shape: {image_data.shape}")
            continue
        if image_data.shape[-1] != 3 and image_data.shape[1] == 3:
            image_data = image_data.transpose(0, 2, 3, 1)
        _, height, width, _ = image_data.shape
        video_path = os.path.join(episode_dir, f"{camera_name}.mp4")
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            if not writer.isOpened():
                logger.error(f"Failed to open VideoWriter for {video_path}")
                continue
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
            for frame in image_data:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception as exc:
            logger.error(f"Error saving video {video_path}: {exc}")

    logger.info(f"Finished writing episode {episode_name}")


class DataCollector:
    def __init__(self, camera_configs: List[dict], save_dir="output", max_episodes=10, max_workers=4, compression=None):
        """Initialize the data collector with multiprocessing support."""
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.session_dir = os.path.join(save_dir, "dataset")
        self.mate_dir = os.path.join(self.session_dir, "meta")
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")
        self.task_instruction_map_path = os.path.join(self.mate_dir, "task_instruction_map.json")
        self.episode_count = 0
        self.camera_configs = camera_configs
        self.task_instructions = None
        self.task_instruction_map: dict[int, str] = {}
        self.instruction_to_index: dict[str, int] = {}

        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.mate_dir, exist_ok=True)

        if os.path.exists(self.task_instruction_map_path):
            with open(self.task_instruction_map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            self.task_instruction_map = {int(k): v for k, v in raw_map.items()}
            self.instruction_to_index = {v: k for k, v in self.task_instruction_map.items()}
        else:
            with open(self.task_instruction_map_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

        self.temp_cameras = {}
        for config in camera_configs:
            if "+" in config["image_type"]:
                for image_type in config["image_type"].split("+"):
                    self.temp_cameras[f"{config['name']}_{image_type}"] = []
            else:
                self.temp_cameras[f"{config['name']}_{config['image_type']}"] = []

        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_indices = []
        self.temp_task_properties = {}
        self.temp_init_state: Optional[dict] = None

        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []

    def set_init_state(self, init_state: dict) -> None:
        """Store the per-episode initial state for deterministic replay."""
        self.temp_init_state = dict(init_state)

    def set_init_state_from_step(self, state: dict) -> None:
        """Convenience helper: extract and store ``init_state`` from a step dict."""
        if self.temp_init_state is None and "init_state" in state:
            self.set_init_state(state["init_state"])

    def set_task_properties(self, properties: dict):
        """Set the unique task properties for the current episode."""
        self.temp_task_properties = properties

    def register_task_instruction(self, instruction: str) -> int:
        """Register an instruction string and return its stable integer id."""
        if instruction in self.instruction_to_index:
            return self.instruction_to_index[instruction]

        next_index = max(self.task_instruction_map.keys(), default=-1) + 1
        self.task_instruction_map[next_index] = instruction
        self.instruction_to_index[instruction] = next_index
        with open(self.task_instruction_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in sorted(self.task_instruction_map.items())}, f, ensure_ascii=False, indent=2)
        return next_index

    def cache_step(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        action: Optional[np.ndarray] = None,
        language_instruction: Optional[str] = None,
        task_index: Optional[int] = None,
    ):
        """Cache each step's data in temporary lists.

        joint_angles is 8-dim:
          - indices 0-6: arm joint positions (rad)
          - index 7:     panda_finger_joint1 position (m, range [0, 0.04])
        We expand the finger joint to total gripper opening width (0–0.08 m),
        since the two Franka fingers are symmetric mimic joints. The recorded
        agent_pose therefore carries the physical gripper distance, not a
        discrete 0/1 control signal.
        """
        if task_index is None and language_instruction is not None:
            task_index = self.register_task_instruction(language_instruction)
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = []

        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(image)

        joint_angles_state = np.asarray(joint_angles, dtype=np.float32).copy()
        if len(joint_angles_state) >= 8:
            joint_angles_state[7] = float(joint_angles_state[7]) * 2.0

        self.temp_agent_pose.append(joint_angles_state)

        if action is not None:
            # Action already contains discrete gripper state (0=open, 1=closed)
            # No discretization needed - action comes from AtomicBaseController
            # which already provides discrete values
            self.temp_actions.append(np.asarray(action, dtype=np.float32))

        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
            if isinstance(self.task_instructions, list) and language_instruction not in self.task_instructions:
                self.task_instructions.append(language_instruction)
        self.temp_task_indices.append(-1 if task_index is None else int(task_index))

    def write_cached_data(self, final_joint_positions=None):
        """Write cached data asynchronously using process pool."""
        if self.episode_count >= self.max_episodes:
            self.close()
            return

        if len(self.temp_actions) == len(self.temp_agent_pose):
            actions_data = np.array(self.temp_actions)
        else:
            if final_joint_positions is None:
                final_joint_positions = self.temp_agent_pose[-1]
            else:
                # Match the finger→width scaling applied in cache_step so the
                # appended tail entry is consistent with the agent_pose stream.
                final_joint_positions = np.asarray(final_joint_positions, dtype=np.float32).copy()
                if len(final_joint_positions) >= 8:
                    final_joint_positions[7] = float(final_joint_positions[7]) * 2.0
            derived_actions = self.temp_agent_pose[1:] + [final_joint_positions]
            actions_data = np.array(derived_actions)

        camera_data = {name: np.array(images) for name, images in self.temp_cameras.items()}
        agent_pose_data = np.array(self.temp_agent_pose)
        task_index_data = np.array(self.temp_task_indices, dtype=np.int32) if self.temp_task_indices else None

        episode_name = f"episode_{self.episode_count:04d}"
        episode_dir = os.path.abspath(os.path.join(self.session_dir, episode_name))

        future = self.process_pool.submit(
            _write_episode_data,
            episode_dir,
            episode_name,
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

        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_indices = []
        self.temp_task_properties = {}
        self.temp_init_state = None
        self.task_instructions = None

        self.episode_count += 1

    def clear_cache(self):
        """Clear the cached data without writing to disk."""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_indices = []
        self.temp_task_properties = {}
        self.temp_init_state = None
        self.task_instructions = None

    def close(self, merge=False):
        """Close the data collector and merge all episode files."""
        for future in self.pending_futures:
            future.result()

        self.process_pool.shutdown(wait=True)

        if merge:
            merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
            episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*", "episode_*.h5")))

            if not episode_files:
                logger.warning("No episodes to merge")
                return

            with h5py.File(merged_path, "w") as merged_file:
                for episode_path in episode_files:
                    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                    with h5py.File(episode_path, "r") as episode_file:
                        episode_group = merged_file.create_group(episode_name)
                        for key in episode_file.keys():
                            episode_file.copy(key, episode_group)

                    os.remove(episode_path)
            os.rename(merged_path, os.path.join(self.session_dir, "episode_data.hdf5"))
            logger.success(f"Successfully merged {len(episode_files)} episodes into {merged_path}")
