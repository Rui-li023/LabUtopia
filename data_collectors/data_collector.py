import os
import numpy as np
import cv2
from datetime import datetime
import json
import h5py
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Optional
from glob import glob

def _write_episode_data(episode_path: str, episode_name: str, 
                       camera_data: dict, agent_pose_data: np.ndarray, 
                       actions_data: np.ndarray, language_instruction: Optional[str] = None, compression=None):
    """Helper function to write episode data in a separate process
    
    Args:
        episode_path: Path to the individual episode HDF5 file
        episode_name: Name of the episode
        camera_data: Dict of camera name to image data {name: [T, H, W, 3]}
        agent_pose_data: Robot joint angles [T, num_joints]
        actions_data: Robot actions [T, num_joints]
        language_instruction: Language instruction for the task
        compression: Compression method for image data, None for no compression
    """
    
    with h5py.File(episode_path, 'w') as h5_file:
        print(f"Writing episode {episode_name} to {episode_path}")
        
        # Store camera data with Blosc compression and dynamic chunking
        for camera_name, image_data in camera_data.items():
            chunk_size = (min(64, image_data.shape[0]),) + image_data.shape[1:]
            kwargs = {
                'data': image_data,
                'dtype': 'uint8',
                'chunks': chunk_size
            }
            if compression == "gzip":
                kwargs.update({
                    'compression': "gzip",
                    'compression_opts': 5
                })
            h5_file.create_dataset(camera_name, **kwargs)
        
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
        
        # Store language instruction if provided
        if language_instruction is not None:
            h5_file.create_dataset(
                "language_instruction",
                data=language_instruction,
                dtype=h5py.special_dtype(vlen=str)
            )
        
        print(f"Finished writing episode {episode_name}")

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
        
        # Initialize process pool and tracking variables
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []
        
    def cache_step(self, camera_images: dict, joint_angles: np.ndarray, language_instruction: Optional[str] = None):
        """Cache each step's data in temporary lists
        
        Args:
            camera_images: Dict of camera name to RGB image {name: np.ndarray}
            joint_angles: Robot joint angles
            language_instruction: Language instruction for the task
        """
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = language_instruction
        for camera_name, image in camera_images.items():
            self.temp_cameras[camera_name].append(image)
        self.temp_agent_pose.append(joint_angles)
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
        
    def write_cached_data(self, final_joint_positions):
        """Write cached data asynchronously using process pool"""
        if self.episode_count >= self.max_episodes:
            self.close()
            return
            
        # Add the final action
        self.temp_actions = self.temp_agent_pose[1:] + [final_joint_positions]
        
        # Convert lists to numpy arrays
        camera_data = {
            name: np.array(images) 
            for name, images in self.temp_cameras.items()
        }
        agent_pose_data = np.array(self.temp_agent_pose)
        actions_data = np.array(self.temp_actions)
        
        # Create individual episode file path
        episode_name = f"episode_{self.episode_count:04d}"
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        
        # Submit writing task to process pool
        future = self.process_pool.submit(
            _write_episode_data,
            episode_path,
            episode_name,
            camera_data,
            agent_pose_data,
            actions_data,
            self.temp_language_instruction,
            self.compression
        )
        self.pending_futures.append(future)

        info = {
            "episode_index": self.episode_count,
            "tasks": [self.task_instructions] if self.task_instructions else [],
            "length": len(self.temp_agent_pose)
        }
        
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        
        # Clear cache
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        
        # Increment episode count
        self.episode_count += 1

    def clear_cache(self):
        """Clear the cached data without writing to disk"""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
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
            episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*.h5")))
            
            if not episode_files:
                print("No episodes to merge")
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
            print(f"Successfully merged {len(episode_files)} episodes into {merged_path}")