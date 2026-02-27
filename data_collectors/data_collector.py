import os
import numpy as np
import cv2
import json
import h5py
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Optional
from glob import glob
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

    # Object poses
    poses = init_state.get("object_poses", {})
    if poses:
        paths = list(poses.keys())
        flat["object_pose_paths"]        = paths
        flat["object_pose_positions"]    = np.array([poses[p]["position"]    for p in paths], dtype="float32")
        flat["object_pose_orientations"] = np.array([poses[p]["orientation"] for p in paths], dtype="float32")

    # Object materials
    materials = init_state.get("object_materials", {})
    if materials:
        flat["object_material_paths"]  = list(materials.keys())
        flat["object_material_values"] = list(materials.values())

    # Extra (task-specific) data as JSON
    extra = init_state.get("extra", {})
    if extra:
        flat["init_extra_json"] = json.dumps(extra)

    # Pass through any legacy float keys (robot_init_joint_positions, etc.)
    legacy_skip = {"object_poses", "object_materials", "extra"}
    for k, v in init_state.items():
        if k not in legacy_skip and k not in flat:
            flat[k] = v

    return flat


def _write_episode_data(episode_dir: str, episode_name: str,
                        camera_data: dict, agent_pose_data: np.ndarray,
                        actions_data: np.ndarray, task_properties: dict = None,
                        language_instruction: Optional[str] = None, compression=None,
                        init_state: Optional[dict] = None):
    """Write one episode's data to an HDF5 file and camera videos.

    Args:
        episode_dir: Path to the individual episode directory.
        episode_name: Name of the episode (used as HDF5 file name).
        camera_data: ``{name: ndarray [T, H, W, 3]}`` image sequences.
        agent_pose_data: Robot joint angles ``[T, n_joints]``.
        actions_data: Robot actions ``[T, n_joints]``.
        task_properties: Arbitrary task-property dict (stored as JSON).
        language_instruction: Natural-language instruction string.
        compression: Image compression method (``None`` = no compression).
        init_state: Episode initial state dict.  Accepts the nested format
                    produced by :class:`BaseTask` or the legacy flat format.
    """
    os.makedirs(episode_dir, exist_ok=True)
    episode_path = os.path.join(episode_dir, f"{episode_name}.h5")
    logger.info(f"Writing episode {episode_name} to {episode_dir}")
    
    with h5py.File(episode_path, 'w') as h5_file:
        
        # Image data is saved as high-quality MP4 only; not stored in HDF5
        # Store pose and action data without compression (small size, frequent access)
        h5_file.create_dataset(
            "agent_pose", 
            data=agent_pose_data, 
            dtype='float32', 
            chunks=True
        )
        h5_file.create_dataset(
            "actions", 
            data=actions_data, 
            dtype='float32', 
            chunks=True
        )
        
        # Store language instruction if provided (as scalar string)
        if language_instruction is not None:
            dt = h5py.special_dtype(vlen=str)
            h5_file.create_dataset(
                "language_instruction",
                data=language_instruction,
                dtype=dt,
                shape=()
            )
        
        # Store task properties (as JSON string)
        if task_properties:
            task_properties_json = json.dumps(task_properties, ensure_ascii=False)
            dt = h5py.special_dtype(vlen=str)
            h5_file.create_dataset(
                "task_properties",
                data=task_properties_json,
                dtype=dt,
                shape=()
            )

        # Store per-episode initial state for deterministic replay.
        # Normalise nested format → flat before writing.
        if init_state:
            flat = _flatten_init_state(init_state) if "object_poses" in init_state or "object_materials" in init_state else init_state
            grp = h5_file.create_group("init_state")
            _STR_KEYS = {
                "object_pose_paths",
                "object_material_paths",
                "object_material_values",
            }
            _STR_SCALAR_KEYS = {"init_extra_json"}  # Scalar string datasets
            _FLOAT2D_KEYS = {"object_pose_positions", "object_pose_orientations"}
            for key, val in flat.items():
                if key in _STR_KEYS:
                    arr = np.array(val, dtype=object)
                    grp.create_dataset(key, data=arr, dtype=h5py.special_dtype(vlen=str))
                elif key in _STR_SCALAR_KEYS:
                    # Handle scalar string datasets (e.g., JSON strings)
                    dt = h5py.special_dtype(vlen=str)
                    grp.create_dataset(key, data=val, dtype=dt, shape=())
                elif key in _FLOAT2D_KEYS:
                    grp.create_dataset(key, data=np.asarray(val, dtype="float32"))
                else:
                    try:
                        grp.create_dataset(key, data=np.array(val, dtype="float32"))
                    except Exception:
                        logger.error("")
                        pass  # skip non-serialisable legacy keys

    # Save each camera stream as an MP4 video
    for camera_name, image_data in camera_data.items():
        if image_data.ndim != 4:
            logger.warning(f"camera_data {camera_name} has wrong shape: {image_data.shape}")
            continue
        elif image_data.shape[-1] != 3 and image_data.shape[1] == 3:
            image_data = image_data.transpose(0, 2, 3, 1)
        T, H, W, _ = image_data.shape
        video_path = os.path.join(episode_dir, f"{camera_name}.mp4")
        try:
            # Use mp4v (MPEG-4); H.264 often unavailable in OpenCV/FFmpeg builds
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30, (W, H))
            if not writer.isOpened():
                logger.error(f"Failed to open VideoWriter for {video_path}")
                continue
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
            for frame in image_data:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception as e:
            logger.error(f"Error saving video {video_path}: {e}")
    
    logger.info(f"Finished writing episode {episode_name}")

class DataCollector:
    def __init__(self, camera_configs: List[dict], save_dir="output", 
                 max_episodes=10, max_workers=4, compression=None):
        """Initialize the data collector with multiprocessing support
        
        Args:
            camera_configs: List of camera configuration dicts, each containing 'name' key
            save_dir (str): Root directory for saving data
            max_episodes (int): Maximum number of episodes to record
            max_workers (int): Maximum number of parallel processes
            compression: Compression method for image data, None for no compression
        """
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.session_dir = os.path.join(save_dir, "dataset")
        self.mate_dir = os.path.join(self.session_dir, "meta")
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")
        self.episode_count = 0
        self.camera_configs = camera_configs
        self.task_instructions = None
        # Create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.mate_dir, exist_ok=True)
        # Initialize temporary storage dictionaries with combined camera name and type
        self.temp_cameras = {}
        for config in camera_configs:
            if '+' in config['image_type']:
                types = config['image_type'].split('+')
                for t in types:
                    self.temp_cameras[f"{config['name']}_{t}"] = []
            else:
                self.temp_cameras[f"{config['name']}_{config['image_type']}"] = []
        
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self.temp_init_state: Optional[dict] = None

        # Initialize process pool and tracking variables
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []
    
    def set_init_state(self, init_state: dict) -> None:
        """Store the per-episode initial state for deterministic replay.

        Accepts either the **nested** format produced by :class:`BaseTask`::

            {
                "object_poses":     {usd_path: {"position": ..., "orientation": ...}},
                "object_materials": {usd_path: material_path},
                "extra":            {...},
            }

        or the legacy flat format used by older controllers.  Should be called
        once per episode (typically on the first step).
        """
        self.temp_init_state = dict(init_state)

    def set_init_state_from_step(self, state: dict) -> None:
        """Convenience helper: extract and store ``init_state`` from a step dict.

        Call this at the top of each controller's ``cache_step`` loop.  It is
        idempotent – only the *first* call per episode takes effect.

        Args:
            state: Step state dict; must contain an ``'init_state'`` key.
        """
        if self.temp_init_state is None and "init_state" in state:
            self.set_init_state(state["init_state"])

    def set_task_properties(self, properties: dict):
        """Set the unique task properties for the current episode
        
        Args:
            properties: Task properties dictionary, content depends on the specific task
                       For example, navigation task: {"start_position": [x, y, z], "end_position": [x, y, z]}
        """
        self.temp_task_properties = properties
        
    def cache_step(self, camera_images: dict, joint_angles: np.ndarray, 
                   action: Optional[np.ndarray] = None,
                   language_instruction: Optional[str] = None):
        """Cache each step's data in temporary lists
        
        Args:
            camera_images: Dict of camera name to RGB image {name: np.ndarray}
            joint_angles: Robot joint angles
            action: Action taken at this step; if None, action is derived from
                    next step's joint angles when writing the episode
            language_instruction: Language instruction for the task
        """
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = language_instruction
        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(image)
        self.temp_agent_pose.append(joint_angles)
        if action is not None:
            self.temp_actions.append(action)
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
        
    def write_cached_data(self, final_joint_positions=None):
        """Write cached data asynchronously using process pool
        
        Args:
            final_joint_positions: Final joint positions used to derive the last
                action when actions were not supplied via cache_step. Ignored
                when actions were already provided through cache_step.
        """
        if self.episode_count >= self.max_episodes:
            self.close()
            return

        # Determine actions_data
        if len(self.temp_actions) == len(self.temp_agent_pose):
            # Actions were provided explicitly at every step
            actions_data = np.array(self.temp_actions)
        else:
            # Fall back to deriving actions from next pose
            if final_joint_positions is None:
                final_joint_positions = self.temp_agent_pose[-1]
            derived_actions = self.temp_agent_pose[1:] + [final_joint_positions]
            actions_data = np.array(derived_actions)

        # Convert lists to numpy arrays
        camera_data = {
            name: np.array(images) 
            for name, images in self.temp_cameras.items()
        }
        agent_pose_data = np.array(self.temp_agent_pose)
        
        # Create per-episode directory (use absolute path so worker process writes to correct location)
        episode_name = f"episode_{self.episode_count:04d}"
        episode_dir = os.path.abspath(os.path.join(self.session_dir, episode_name))
        
        # Submit writing task to process pool
        future = self.process_pool.submit(
            _write_episode_data,
            episode_dir,
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

        info = {
            "episode_index": self.episode_count,
            "tasks": [self.task_instructions] if self.task_instructions else [],
            "length": len(self.temp_agent_pose),
            "task_properties": self.temp_task_properties
        }
        
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        
        # Clear cache
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self.temp_init_state = None
        
        # Increment episode count
        self.episode_count += 1

    def clear_cache(self):
        """Clear the cached data without writing to disk"""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.temp_task_properties = {}
        self.temp_init_state = None
        self.task_instructions = None
        
    def close(self, merge=False):
        """Close the data collector and merge all episode files"""
        # Wait for all pending writing operations to complete
        for future in self.pending_futures:
            future.result()
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        if merge:
            merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
            episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*", "episode_*.h5")))
            
            if not episode_files:
                logger.warning("No episodes to merge")
                return
                
            with h5py.File(merged_path, 'w') as merged_file:
                # Copy each episode file into the merged file
                for episode_path in episode_files:
                    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                    with h5py.File(episode_path, 'r') as episode_file:
                        # Create episode group in merged file
                        episode_group = merged_file.create_group(episode_name)
                        
                        # Copy all datasets with their original compression settings
                        for key in episode_file.keys():
                            episode_file.copy(key, episode_group)
                    
                    # Remove individual episode file after merging
                    os.remove(episode_path)
            os.rename(merged_path, os.path.join(self.session_dir, "episode_data.hdf5"))
            logger.success(f"Successfully merged {len(episode_files)} episodes into {merged_path}")