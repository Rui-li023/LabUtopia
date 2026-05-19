"""LabUtopia → in-memory Episode iterator.

Reads the LabUtopia collect output (HDF5 + per-camera mp4s + meta files)
and yields a normalized Episode dataclass that both v2.1 and v3.0 writers
can consume.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass
class Episode:
    index: int                              # 0-based episode index
    state: np.ndarray                       # [T, S] float32
    action: np.ndarray                      # [T, A] float32
    task: str                               # language instruction
    task_index: int                         # index into tasks table
    video_paths: dict[str, Path]            # camera_key → existing .mp4 path
    length: int


def discover_run(data_dir: Path) -> dict:
    """Inspect a LabUtopia run dir and return camera names + shapes."""
    dataset_root = data_dir / "dataset"
    episode_dirs = sorted(
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* dirs in {dataset_root}")
    first = episode_dirs[0]
    cameras = sorted(p.stem for p in first.glob("*.mp4"))
    with h5py.File(first / f"{first.name}.h5", "r") as f:
        state_dim = int(f["agent_pose"].shape[1])
        action_dim = int(f["actions"].shape[1])
    # Image shape from first frame of first camera
    if cameras:
        cap = cv2.VideoCapture(str(first / f"{cameras[0]}.mp4"))
        ok, frame = cap.read()
        cap.release()
        image_shape = (frame.shape[0], frame.shape[1], 3) if ok else (256, 256, 3)
    else:
        image_shape = (256, 256, 3)
    return {
        "episode_dirs": episode_dirs,
        "cameras": cameras,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": image_shape,
    }


def load_task_map(data_dir: Path) -> dict[int, str]:
    mp = data_dir / "dataset" / "meta" / "task_instruction_map.json"
    if not mp.exists():
        return {}
    return {int(k): v for k, v in json.loads(mp.read_text()).items()}


def iter_episodes(data_dir: Path):
    """Yield Episode for every episode_*/episode_*.h5 in deterministic order."""
    info = discover_run(data_dir)
    task_map = load_task_map(data_dir)
    for new_idx, ep_dir in enumerate(info["episode_dirs"]):
        h5_path = ep_dir / f"{ep_dir.name}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            state = np.asarray(f["agent_pose"][()], dtype=np.float32)
            action = np.asarray(f["actions"][()], dtype=np.float32)
            if "task_index" in f:
                ti = f["task_index"][()]
                ti_val = int(np.asarray(ti).flat[0])
                task = task_map.get(ti_val, "")
            elif "language_instruction" in f:
                v = f["language_instruction"][()]
                task = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                ti_val = 0
            else:
                task = ""
                ti_val = 0
        videos = {cam: ep_dir / f"{cam}.mp4" for cam in info["cameras"]}
        yield Episode(
            index=new_idx,
            state=state,
            action=action,
            task=task,
            task_index=ti_val,
            video_paths=videos,
            length=int(state.shape[0]),
        )


def collect_tasks(data_dir: Path) -> list[str]:
    """Distinct tasks in insertion order across episodes (after re-indexing)."""
    seen = []
    for ep in iter_episodes(data_dir):
        if ep.task not in seen:
            seen.append(ep.task)
    return seen
