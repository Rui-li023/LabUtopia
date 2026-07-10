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
    spawn_yaw: float = 0.0                  # base world yaw at spawn (mobile tasks)
    video_trim: int | None = None           # keep only the first N video frames


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
    # Image shape + fps from first frame of first camera
    fps = None
    if cameras:
        cap = cv2.VideoCapture(str(first / f"{cameras[0]}.mp4"))
        ok, frame = cap.read()
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        image_shape = (frame.shape[0], frame.shape[1], 3) if ok else (256, 256, 3)
        if src_fps and src_fps > 0:
            fps = int(round(src_fps))
    else:
        image_shape = (256, 256, 3)
    return {
        "episode_dirs": episode_dirs,
        "cameras": cameras,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": image_shape,
        "fps": fps,
    }


def load_task_map(data_dir: Path) -> dict[int, str]:
    mp = data_dir / "dataset" / "meta" / "task_instruction_map.json"
    if not mp.exists():
        return {}
    return {int(k): v for k, v in json.loads(mp.read_text()).items()}


def iter_episodes(data_dir: Path, nav_only: bool = False):
    """Yield Episode for every episode_*/episode_*.h5 in deterministic order.

    nav_only: trim each episode to its leading navigation segment (the
    contiguous ``phase == 0`` prefix recorded by the mobile collectors) —
    state/action are sliced and ``video_trim`` tells the writer to cut the
    videos to the same frame count. Episodes without a phase array or with a
    trivially short nav prefix (< 30 frames) are skipped.
    """
    info = discover_run(data_dir)
    task_map = load_task_map(data_dir)
    new_idx = -1
    for ep_dir in info["episode_dirs"]:
        h5_path = ep_dir / f"{ep_dir.name}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            state = np.asarray(f["agent_pose"][()], dtype=np.float32)
            action = np.asarray(f["actions"][()], dtype=np.float32)
            trim = None
            if nav_only:
                if "phase" not in f:
                    continue
                phase = np.asarray(f["phase"][()])
                nonnav = np.nonzero(phase != 0)[0]
                n0 = int(nonnav[0]) if nonnav.size else int(phase.shape[0])
                if n0 < 30:
                    continue
                state, action, trim = state[:n0], action[:n0], n0
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
            # Mobile (Ridgebase) episodes: the base joints are spawn-frame; the
            # spawn world yaw lives in task_properties.start_position[2] and is
            # needed to rotate world deltas into the body frame.
            spawn_yaw = 0.0
            if "task_properties" in f:
                try:
                    tp = json.loads(np.asarray(f["task_properties"][()]).item().decode("utf-8"))
                    sp = tp.get("start_position")
                    if sp is not None and len(sp) >= 3:
                        spawn_yaw = float(sp[2])
                except Exception:
                    pass
        videos = {cam: ep_dir / f"{cam}.mp4" for cam in info["cameras"]}
        new_idx += 1
        yield Episode(
            index=new_idx,
            state=state,
            action=action,
            task=task,
            task_index=ti_val,
            video_paths=videos,
            length=int(state.shape[0]),
            spawn_yaw=spawn_yaw,
            video_trim=trim,
        )


def base_action_to_body_delta(ep: Episode) -> np.ndarray:
    """Return a copy of ep.action with the base dims (0:3) converted from
    spawn-frame absolute position targets to per-step BODY-frame deltas:
    [forward, lateral, dtheta].

    The collect control law is ``action = current + v`` (position targets one
    velocity step ahead), so ``action - state`` recovers the commanded per-step
    velocity exactly. Rotating it by the base's world heading (spawn yaw +
    theta joint) yields an observation-consistent action the policy can learn
    without depending on the episode's world/spawn origin. Arm joints (3:10)
    stay absolute (body-frame already); gripper (10) unchanged.
    """
    act = ep.action.copy()
    d = ep.action[:, :3].astype(np.float64) - ep.state[:, :3].astype(np.float64)
    heading = ep.spawn_yaw + ep.state[:, 2].astype(np.float64)
    c, s = np.cos(heading), np.sin(heading)
    act[:, 0] = (c * d[:, 0] + s * d[:, 1]).astype(np.float32)      # forward
    act[:, 1] = (-s * d[:, 0] + c * d[:, 1]).astype(np.float32)     # lateral
    act[:, 2] = ((d[:, 2] + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)  # dtheta
    return act


def collect_tasks(data_dir: Path) -> list[str]:
    """Distinct tasks in insertion order across episodes (after re-indexing)."""
    seen = []
    for ep in iter_episodes(data_dir):
        if ep.task not in seen:
            seen.append(ep.task)
    return seen
