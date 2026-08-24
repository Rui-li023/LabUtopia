"""Write LabUtopia episodes to LeRobot v2.1 on-disk layout.

Layout produced (under <dst>):
  data/chunk-000/episode_NNNNNN.parquet
  videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4
  meta/info.json          (codebase_version="v2.1")
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/stats.json         (aggregate stats)

Parquet columns per episode:
  observation.state  list[float32]   length = state_dim
  action             list[float32]   length = action_dim
  timestamp          float32         (frame_index / fps)
  frame_index        int64
  episode_index      int64
  index              int64           (global running)
  task_index         int64
  next.done          bool
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .reader import base_action_to_body_delta, discover_run, iter_episodes
from .stats import Accum

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000  # LeRobot convention: 1000 episodes per chunk-{NNN}/ folder


def _chunk_of(ep_idx: int) -> int:
    return ep_idx // CHUNK_SIZE


def _features(state_dim: int, action_dim: int, image_shape: tuple, cameras: list[str], fps: int) -> dict:
    out = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for cam in cameras:
        out[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.height": int(image_shape[0]),
                "video.width": int(image_shape[1]),
                "video.channels": int(image_shape[2]),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return out


def _probe_codec(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    # avc1 / h264 → h264;  mp4v / FMP4 / xvid → mpeg4
    return "h264" if fourcc.lower() in ("avc1", "h264") else "mpeg4"


def _copy_video(src: Path, dst: Path, fps: int, n_frames: int | None = None):
    """Copy source mp4 into the LeRobot layout, transcoding to H264/yuv420p
    when the source uses a different codec (LabUtopia historically wrote
    mpeg4 via OpenCV mp4v). LeRobot loaders expect H264 for reliable random
    seeking. n_frames: keep only the first N frames (frame-accurate re-encode;
    used by --nav-only episode trimming)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if n_frames is None and _probe_codec(src) == "h264":
        shutil.copy2(src, dst)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if n_frames is not None:
        cmd += ["-frames:v", str(n_frames)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), "-an", str(dst)]
    subprocess.run(cmd, check=True)


def _scan_pixel_stats(video_paths: list[Path], frame_stride: int = 10) -> dict | None:
    """Compute per-channel pixel mean/std/min/max over sampled video frames.
    Sampling every Nth frame keeps stats representative while being ~10× faster."""
    if not video_paths:
        return None
    acc = Accum(shape=(3,))
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                acc.update(rgb.reshape(-1, 3))
            idx += 1
        cap.release()
    out = acc.finalize()
    # LeRobot expects pixel stats with shape (3, 1, 1) for channel-first images
    def _r(v):
        arr = np.asarray(v, dtype=np.float32)
        return arr.reshape(3, 1, 1).tolist()
    return {"mean": _r(out["mean"]), "std": _r(out["std"]),
            "min": _r(out["min"]), "max": _r(out["max"]),
            "count": out["count"]}


def write_v21(src: Path, dst: Path, fps: int | None = None, robot_type: str = "franka",
              base_action: str = "abs", nav_only: bool = False) -> dict:
    info_disc = discover_run(src)
    state_dim = info_disc["state_dim"]
    action_dim = info_disc["action_dim"]
    cameras = info_disc["cameras"]
    image_shape = info_disc["image_shape"]
    # Prefer the fps actually encoded in the source mp4 over any caller-supplied
    # default — LabUtopia writes 30 fps, but earlier defaults were 60.
    if fps is None:
        fps = info_disc.get("fps") or 30

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    # chunk-{NNN}/ subdirs are created on demand per episode below.

    # Tasks table (collect in pass 1)
    tasks: list[str] = []

    # Stats accumulators
    state_acc = Accum(shape=(state_dim,))
    action_acc = Accum(shape=(action_dim,))
    cam_video_paths: dict[str, list[Path]] = {cam: [] for cam in cameras}

    episode_rows = []   # for episodes.jsonl
    episode_stats_rows = []  # for episodes_stats.jsonl (v2.1 per-episode stats)
    global_index = 0
    total_frames = 0
    total_episodes = 0

    def _stat_dict(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return {"mean": [], "std": [], "min": [], "max": [], "count": [0]}
        mean = arr.mean(axis=0).astype(np.float32).tolist()
        std = arr.std(axis=0, ddof=0).astype(np.float32).tolist()
        return {
            "mean": mean,
            "std": std,
            "min": arr.min(axis=0).astype(np.float32).tolist(),
            "max": arr.max(axis=0).astype(np.float32).tolist(),
            "count": [int(arr.shape[0])],
        }

    def _scalar_stat_dict(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, dtype=np.float32)
        return {
            "mean": [float(arr.mean())],
            "std": [float(arr.std(ddof=0))],
            "min": [float(arr.min())],
            "max": [float(arr.max())],
            "count": [int(arr.size)],
        }

    def _video_pixel_stats(video_path: Path, stride: int = 10) -> dict:
        acc = Accum(shape=(3,))
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                acc.update(rgb.reshape(-1, 3))
            idx += 1
        cap.release()
        out = acc.finalize()
        def _r(v):
            arr = np.asarray(v, dtype=np.float32)
            return arr.reshape(3, 1, 1).tolist()
        return {"mean": _r(out["mean"]), "std": _r(out["std"]),
                "min": _r(out["min"]), "max": _r(out["max"]),
                "count": out["count"]}

    for ep in iter_episodes(src, nav_only=nav_only):
        if base_action == "body_delta":
            ep.action = base_action_to_body_delta(ep)
        T = ep.length
        # Per-frame instructions when the episode carries them (multi-phase tasks),
        # else the single episode-level label repeated.
        ftasks = ep.frame_tasks if ep.frame_tasks and len(ep.frame_tasks) == T else [ep.task] * T
        for s in ftasks:
            if s not in tasks:
                tasks.append(s)
        frame_task_idx = np.asarray([tasks.index(s) for s in ftasks], dtype=np.int64)

        df = pd.DataFrame({
            "observation.state": [row.tolist() for row in ep.state],
            "action":            [row.tolist() for row in ep.action],
            "timestamp":         (np.arange(T, dtype=np.float32) / float(fps)),
            "frame_index":       np.arange(T, dtype=np.int64),
            "episode_index":     np.full(T, ep.index, dtype=np.int64),
            "index":             np.arange(global_index, global_index + T, dtype=np.int64),
            "task_index":        frame_task_idx,
            "next.done":         np.array([False] * (T - 1) + [True], dtype=bool),
        })
        chunk_idx = _chunk_of(ep.index)
        (dst / "data" / f"chunk-{chunk_idx:03d}").mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep.index:06d}.parquet",
                      index=False)

        # Per-episode stats (scalar/feature)
        ep_stats = {
            "observation.state": _stat_dict(ep.state),
            "action": _stat_dict(ep.action),
            "timestamp": _scalar_stat_dict(np.arange(T, dtype=np.float32) / float(fps)),
            "frame_index": _scalar_stat_dict(np.arange(T, dtype=np.float32)),
            "episode_index": _scalar_stat_dict(np.full(T, ep.index, dtype=np.float32)),
            "index": _scalar_stat_dict(np.arange(global_index, global_index + T, dtype=np.float32)),
            "task_index": _scalar_stat_dict(frame_task_idx.astype(np.float32)),
            "next.done": _scalar_stat_dict(np.array([0.0] * (T - 1) + [1.0], dtype=np.float32)),
        }

        for cam, src_vid in ep.video_paths.items():
            if not src_vid.exists():
                continue
            cam_dir = dst / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{cam}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            out_vid = cam_dir / f"episode_{ep.index:06d}.mp4"
            _copy_video(src_vid, out_vid, fps=fps, n_frames=ep.video_trim)
            cam_video_paths[cam].append(out_vid)
            # Per-episode pixel stats
            ep_stats[f"observation.images.{cam}"] = _video_pixel_stats(out_vid)

        episode_stats_rows.append({"episode_index": ep.index, "stats": ep_stats})

        state_acc.update(ep.state)
        action_acc.update(ep.action)

        episode_rows.append({
            "episode_index": ep.index,
            "tasks": list(dict.fromkeys(ftasks)),
            "length": T,
        })
        global_index += T
        total_frames += T
        total_episodes += 1

    # tasks.jsonl
    with (dst / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")

    # episodes.jsonl
    with (dst / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # episodes_stats.jsonl (per-episode stats required by v2.1 loader)
    with (dst / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for row in episode_stats_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # stats.json (aggregate; v2.1 loader can also derive from episodes_stats)
    stats = {
        "observation.state": state_acc.finalize(),
        "action": action_acc.finalize(),
    }
    with (dst / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # info.json
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        # abs: raw position targets. body_delta: action[0:3] are per-step
        # BODY-frame base deltas [forward, lateral, dtheta]; the executor must
        # integrate them onto the current base pose. Arm/gripper unchanged.
        "base_action": base_action,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_episodes * len(cameras),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(state_dim, action_dim, image_shape, cameras, fps),
    }
    with (dst / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=4)

    modality = {
        "state": {"single_arm": {"start": 0, "end": state_dim}},
        "action": {"single_arm": {"start": 0, "end": action_dim}},
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with (dst / "meta" / "modality.json").open("w") as f:
        json.dump(modality, f, indent=2)

    return {"episodes": total_episodes, "frames": total_frames,
            "tasks": len(tasks), "cameras": len(cameras)}
