"""Write LabUtopia episodes to LeRobot v3.0 on-disk layout.

Layout produced (under <dst>):
  data/chunk-000/file-000.parquet                       (all episodes bundled)
  videos/observation.images.<cam>/chunk-000/file-000.mp4  (videos concatenated)
  meta/info.json          (codebase_version="v3.0")
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/stats.json         (aggregate)

For small datasets (≤200 episodes) one chunk/file is sufficient.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .reader import discover_run, iter_episodes
from .stats import Accum

CODEBASE_VERSION = "v3.0"
CHUNK_IDX = 0
FILE_IDX = 0


def _features(state_dim: int, action_dim: int, image_shape: tuple, cameras: list[str], fps: int) -> dict:
    out = {
        "observation.state": {
            "dtype": "float32", "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32", "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
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
    return "h264" if fourcc.lower() in ("avc1", "h264") else "mpeg4"


def _transcode_to_h264(src: Path, dst: Path, fps: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-r", str(fps), "-an", str(dst)],
        check=True,
    )


def _concat_videos(parts: list[Path], out_path: Path, fps: int):
    """Concat all per-episode videos into one file.

    If any source isn't already H264/yuv420p, transcode each into a temp
    directory first so the concat demuxer + stream copy operates on
    consistent H264 inputs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    needs_transcode = any(_probe_codec(p) != "h264" for p in parts)
    with tempfile.TemporaryDirectory() as tmpdir:
        if needs_transcode:
            transcoded = []
            for i, p in enumerate(parts):
                tp = Path(tmpdir) / f"part_{i:06d}.mp4"
                _transcode_to_h264(p, tp, fps=fps)
                transcoded.append(tp)
            parts = transcoded
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for p in parts:
                f.write(f"file '{p.resolve()}'\n")
            list_file = f.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-i", list_file,
                 "-c", "copy", str(out_path)],
                check=True,
            )
        finally:
            Path(list_file).unlink(missing_ok=True)


def _video_duration_s(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    return frames / fps if fps else 0.0


def _scan_pixel_stats_from_videos(video_paths: list[Path], frame_stride: int = 10):
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
    def _r(v):
        arr = np.asarray(v, dtype=np.float32)
        return arr.reshape(3, 1, 1).tolist()
    return {"mean": _r(out["mean"]), "std": _r(out["std"]),
            "min": _r(out["min"]), "max": _r(out["max"]),
            "count": out["count"]}


def write_v30(src: Path, dst: Path, fps: int | None = None, robot_type: str = "franka") -> dict:
    info_disc = discover_run(src)
    state_dim = info_disc["state_dim"]
    action_dim = info_disc["action_dim"]
    cameras = info_disc["cameras"]
    image_shape = info_disc["image_shape"]
    if fps is None:
        fps = info_disc.get("fps") or 30

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data" / f"chunk-{CHUNK_IDX:03d}").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "episodes" / f"chunk-{CHUNK_IDX:03d}").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    tasks: list[str] = []
    episodes_meta = []         # for meta/episodes/...parquet
    all_rows_dfs = []          # for data/...parquet (one bundled file)
    cam_episode_videos: dict[str, list[Path]] = {cam: [] for cam in cameras}
    cam_durations: dict[str, list[float]] = {cam: [] for cam in cameras}

    state_acc = Accum(shape=(state_dim,))
    action_acc = Accum(shape=(action_dim,))

    global_index = 0
    total_frames = 0
    total_episodes = 0

    for ep in iter_episodes(src):
        if ep.task not in tasks:
            tasks.append(ep.task)
        task_index = tasks.index(ep.task)
        T = ep.length
        df = pd.DataFrame({
            "observation.state": [row.tolist() for row in ep.state],
            "action":            [row.tolist() for row in ep.action],
            "timestamp":         (np.arange(T, dtype=np.float32) / float(fps)),
            "frame_index":       np.arange(T, dtype=np.int64),
            "episode_index":     np.full(T, ep.index, dtype=np.int64),
            "index":             np.arange(global_index, global_index + T, dtype=np.int64),
            "task_index":        np.full(T, task_index, dtype=np.int64),
        })
        all_rows_dfs.append(df)

        # Source mp4s from LabUtopia are already H264/yuv420p; reference them
        # directly and concat without re-encoding.
        for cam, src_vid in ep.video_paths.items():
            if not src_vid.exists():
                continue
            cam_episode_videos[cam].append(src_vid)
            cam_durations[cam].append(_video_duration_s(src_vid))

        state_acc.update(ep.state)
        action_acc.update(ep.action)

        episodes_meta.append({
            "episode_index": ep.index,
            "tasks": [ep.task],
            "length": T,
            "data/chunk_index": CHUNK_IDX,
            "data/file_index": FILE_IDX,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + T,
        })
        global_index += T
        total_frames += T
        total_episodes += 1

    # Concat per-camera videos → videos/<key>/chunk-000/file-000.mp4
    final_video_paths: dict[str, Path] = {}
    cumulative_offsets: dict[str, list[float]] = {cam: [0.0] for cam in cameras}
    for cam in cameras:
        if not cam_episode_videos[cam]:
            continue
        out_vid = dst / "videos" / f"observation.images.{cam}" / f"chunk-{CHUNK_IDX:03d}" / f"file-{FILE_IDX:03d}.mp4"
        _concat_videos(cam_episode_videos[cam], out_vid, fps=fps)
        final_video_paths[cam] = out_vid
        accum = 0.0
        for d in cam_durations[cam]:
            accum += d
            cumulative_offsets[cam].append(accum)

    # Populate per-episode video time offsets in episodes_meta
    for i, ep_meta in enumerate(episodes_meta):
        for cam in cameras:
            if cam not in final_video_paths:
                continue
            ep_meta[f"videos/observation.images.{cam}/chunk_index"] = CHUNK_IDX
            ep_meta[f"videos/observation.images.{cam}/file_index"] = FILE_IDX
            ep_meta[f"videos/observation.images.{cam}/from_timestamp"] = float(cumulative_offsets[cam][i])
            ep_meta[f"videos/observation.images.{cam}/to_timestamp"] = float(cumulative_offsets[cam][i + 1])

    # Write bundled data parquet
    big_df = pd.concat(all_rows_dfs, ignore_index=True)
    big_df.to_parquet(dst / "data" / f"chunk-{CHUNK_IDX:03d}" / f"file-{FILE_IDX:03d}.parquet",
                      index=False)

    # tasks.parquet — index is task string, single column task_index
    tasks_df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks)
    tasks_df.to_parquet(dst / "meta" / "tasks.parquet")

    # episodes parquet
    episodes_df = pd.DataFrame(episodes_meta)
    episodes_df.to_parquet(dst / "meta" / "episodes" / f"chunk-{CHUNK_IDX:03d}" / f"file-{FILE_IDX:03d}.parquet",
                           index=False)

    # stats.json (aggregate)
    stats = {
        "observation.state": state_acc.finalize(),
        "action": action_acc.finalize(),
    }
    for cam in cameras:
        if cam in final_video_paths:
            ps = _scan_pixel_stats_from_videos([final_video_paths[cam]])
            if ps is not None:
                stats[f"observation.images.{cam}"] = ps
    with (dst / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # info.json
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
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
