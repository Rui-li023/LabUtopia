"""
Convert LabUtopia dataset to LeRobot format.

Structure:
  dataset/episode_XXXX/episode_XXXX.h5   - state, action, language/task_index
  dataset/episode_XXXX/camera_N_rgb.mp4  - camera videos
  dataset/meta/episode.jsonl             - episode metadata
  dataset/meta/task_instruction_map.json - task_index to instruction mapping

Usage:
  python scripts/convert_labutopia_to_lerobot.py --data_dir /path/to/run_dir --repo_name my/dataset
"""

import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed, please run: pip install lerobot")


def read_video_frames(video_path: Path) -> np.ndarray:
    """Read all frames from a video, returns [T, H, W, C] uint8 RGB."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def _load_task_map(dataset_path: Path) -> dict[int, str]:
    map_path = dataset_path / "meta" / "task_instruction_map.json"
    if not map_path.exists():
        return {}
    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _decode_scalar_string(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main(data_dir: str, repo_name: str, *, push_to_hub: bool = False, fps: int = 60, robot_type: str = "franka"):
    if not LEROBOT_AVAILABLE:
        print("Error: LeRobot not installed")
        return

    data_path = Path(data_dir)
    dataset_path = data_path / "dataset"
    if not dataset_path.exists():
        print(f"Error: dataset directory not found: {dataset_path}")
        return

    episode_dirs = sorted([d for d in dataset_path.glob("episode_*") if d.is_dir() and d.name != "meta"])
    if not episode_dirs:
        print("Error: No episode directories found")
        return

    print(f"Found {len(episode_dirs)} episodes")

    first_ep = episode_dirs[0]
    camera_names = sorted(p.stem for p in first_ep.glob("*.mp4"))
    print(f"Cameras: {camera_names}")

    with h5py.File(first_ep / f"{first_ep.name}.h5", "r") as f:
        state_shape = (f["agent_pose"].shape[1],)
        action_shape = (f["actions"].shape[1],)

    cap = cv2.VideoCapture(str(first_ep / f"{camera_names[0]}.mp4"))
    ret, frame = cap.read()
    cap.release()
    image_shape = (frame.shape[0], frame.shape[1], 3) if ret else (256, 256, 3)

    print(f"State: {state_shape}, Action: {action_shape}, Image: {image_shape}")

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {cam: {"dtype": "video", "shape": image_shape, "names": ["height", "width", "channel"]} for cam in camera_names}
    features["state"] = {"dtype": "float32", "shape": state_shape, "names": ["state"]}
    features["actions"] = {"dtype": "float32", "shape": action_shape, "names": ["actions"]}

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    task_map = _load_task_map(dataset_path)

    for ep_dir in episode_dirs:
        h5_file = ep_dir / f"{ep_dir.name}.h5"
        if not h5_file.exists():
            print(f"Skipping {ep_dir.name}: no h5 file")
            continue

        with h5py.File(h5_file, "r") as f:
            agent_pose = f["agent_pose"][:]
            actions = f["actions"][:]
            if "task_index" in f:
                task_indices = f["task_index"][:].astype(np.int32)
                tasks = [task_map.get(int(idx), f"unknown task index {int(idx)}") for idx in task_indices]
            elif "language_instruction" in f:
                task = _decode_scalar_string(f["language_instruction"][()])
                tasks = [task] * len(agent_pose)
            else:
                tasks = [""] * len(agent_pose)

        total_steps = len(agent_pose)
        cam_frames = {}
        for cam in camera_names:
            video_path = ep_dir / f"{cam}.mp4"
            cam_frames[cam] = read_video_frames(video_path) if video_path.exists() else np.zeros((total_steps, *image_shape), dtype=np.uint8)

        preview_task = tasks[0] if tasks else ""
        print(f"Processing {ep_dir.name}: {total_steps} frames, first_task='{preview_task}'")

        for t in range(total_steps):
            frame_data = {
                "state": agent_pose[t].astype(np.float32),
                "actions": actions[t].astype(np.float32),
                "task": tasks[t],
            }
            for cam in camera_names:
                frames = cam_frames[cam]
                frame_data[cam] = frames[t] if t < len(frames) else np.zeros(image_shape, dtype=np.uint8)
            dataset.add_frame(frame_data)

        dataset.save_episode()

    dataset.finalize()
    print(f"Converted {len(episode_dirs)} episodes to: {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["labutopia", robot_type], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
