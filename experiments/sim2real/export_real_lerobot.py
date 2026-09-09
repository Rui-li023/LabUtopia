"""Convert nuc2 real-robot recordings (DataCollectionSystemV2 layout) into a
LeRobot v2.1 dataset whose schema is byte-for-byte compatible with the simulated
``sim2real/pick_v2`` dataset, so the two can be co-trained.

Three alignments are NOT cosmetic and are the whole point of this script:

1. **Gripper action polarity AND command-vs-measured semantics.** The sim
   dataset stores ``action[7]`` as a *closedness* command in [0, 1] (0 open /
   1 closed) while the real rig records a *width in metres* (0.08 open / ~0.03
   closed) in both state and action.  ``observation.state[7]`` is a width on
   BOTH sides and is left alone; only the action channel is converted.  Beyond
   the polarity, the real "action" is not a command at all — it is the measured
   state shifted one frame — so ``--gripper-action binary`` (the default)
   rebuilds it as sim's 0/1 close command.  See ``_gripper_action``.

2. **Camera keys.** real ``observation.image.<view>`` (singular "image") maps to
   sim ``observation.images.<key>`` (plural) by physical viewpoint:
   third_view→image, second_third_view→image_2, left_wrist_view→wrist_image.

3. **Frame/row alignment.** Videos carry one more frame than the CSV rows and
   run on a slightly different clock, drifting up to 4 frames (133 ms) mid
   episode.  Every output frame is picked by nearest capture timestamp rather
   than by index.

Usage:
    python -m experiments.sim2real.export_real_lerobot \
        --src /data1/DataCollectionSystemV2/data/pick_up_beaker \
        --dst /data1/lerobot_out/sim2real_pick_real11 \
        --task "Pick up the conical bottle."
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

FPS = 30
CHUNK_SIZE = 1000
WIDTH, HEIGHT = 640, 480
PIXEL_STAT_STRIDE = 10  # matches scripts/lerobot_export/writer_v21.py

# real view name -> sim video key (physical viewpoint correspondence)
CAMERA_MAP = {
    "third_view": "image",
    "second_third_view": "image_2",
    "left_wrist_view": "wrist_image",
}


def _read_csv_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_ms, values) from a DataCollectionSystemV2 data.csv."""
    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    ts = np.asarray([int(r[0]) for r in body], dtype=np.int64)
    vals = np.asarray([[float(x) for x in r[1:]] for r in body], dtype=np.float64)
    return ts, vals


def _read_frame_timestamps(path: Path) -> np.ndarray:
    with path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    return np.asarray([int(r[0]) for r in rows[1:]], dtype=np.int64)


def _nearest_indices(frame_ts: np.ndarray, sample_ts: np.ndarray) -> np.ndarray:
    """For each sample timestamp, the index of the closest video frame.

    frame_ts is sorted, so a binary search plus a neighbour comparison is exact.
    """
    pos = np.searchsorted(frame_ts, sample_ts)
    pos = np.clip(pos, 1, len(frame_ts) - 1)
    left, right = frame_ts[pos - 1], frame_ts[pos]
    take_left = (sample_ts - left) <= (right - sample_ts)
    return np.where(take_left, pos - 1, pos)


class _Accum:
    """Streaming mean/std/min/max over rows of a (N, D) stream."""

    def __init__(self, dim: int):
        self.n = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_sq = np.zeros(dim, dtype=np.float64)
        self.lo = np.full(dim, np.inf)
        self.hi = np.full(dim, -np.inf)

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.n += x.shape[0]
        self.total += x.sum(0)
        self.total_sq += (x**2).sum(0)
        self.lo = np.minimum(self.lo, x.min(0))
        self.hi = np.maximum(self.hi, x.max(0))

    def finalize(self) -> dict:
        mean = self.total / self.n
        var = np.maximum(self.total_sq / self.n - mean**2, 0.0)
        return {
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "min": self.lo.tolist(),
            "max": self.hi.tolist(),
            "count": [self.n],
        }


def _scalar_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "mean": [float(a.mean())],
        "std": [float(a.std())],
        "min": [float(a.min())],
        "max": [float(a.max())],
        "count": [int(a.size)],
    }


def _remap_video(src: Path, dst: Path, keep: np.ndarray, n_src_frames: int) -> dict:
    """Re-encode `src` keeping exactly the frames listed in `keep` (monotone,
    may repeat), and return per-channel pixel stats over every Nth kept frame.

    Decoding and encoding are piped so memory stays at one frame regardless of
    episode length.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame_bytes = WIDTH * HEIGHT * 3
    # how many times each source frame must be emitted
    repeats = np.bincount(keep, minlength=n_src_frames)

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        bufsize=frame_bytes * 4,
    )
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst),
        ],
        stdin=subprocess.PIPE,
        bufsize=frame_bytes * 4,
    )

    acc = _Accum(3)
    written = 0
    try:
        for n in range(n_src_frames):
            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                raise RuntimeError(f"{src}: decoder ran out at frame {n}/{n_src_frames}")
            times = int(repeats[n])
            if times == 0:
                continue
            for _ in range(times):
                enc.stdin.write(raw)
                if written % PIXEL_STAT_STRIDE == 0:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.float32) / 255.0
                    acc.update(frame)
                written += 1
    finally:
        if enc.stdin:
            enc.stdin.close()
        enc.wait()
        if dec.stdout:
            dec.stdout.close()
        dec.wait()

    if written != len(keep):
        raise RuntimeError(f"{src}: wrote {written} frames, expected {len(keep)}")
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed for {dst}")

    out = acc.finalize()

    def _chw(v):
        return np.asarray(v, dtype=np.float32).reshape(3, 1, 1).tolist()

    return {
        "mean": _chw(out["mean"]),
        "std": _chw(out["std"]),
        "min": _chw(out["min"]),
        "max": _chw(out["max"]),
        "count": out["count"],
    }


def _features() -> dict:
    """Feature block identical to sim2real/pick_v2's meta/info.json."""
    video_info = {
        "video.fps": float(FPS),
        "video.height": HEIGHT,
        "video.width": WIDTH,
        "video.channels": 3,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": [f"state_{i}" for i in range(8)],
        },
        "action": {
            "dtype": "float32",
            "shape": [8],
            "names": [f"action_{i}" for i in range(8)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for key in CAMERA_MAP.values():
        feats[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": [HEIGHT, WIDTH, 3],
            "names": ["height", "width", "channel"],
            "info": dict(video_info),
        }
    return feats


CLOSE_MARGIN_M = 0.002  # width drop that counts as "the operator started closing"


def _gripper_action(width: np.ndarray, stroke: float, mode: str, ep_name: str) -> np.ndarray:
    """Map the real rig's recorded gripper channel to the sim action convention.

    The real recording has no command stream: ``actions.joint_position`` is the
    follower's *measured* state shifted one frame (verified: action[t] equals
    state[t+1] bit-for-bit on all 7 joints, 100% of frames).  So its gripper
    column is the width the fingers actually reached — which the beaker stops at
    34-37 mm, i.e. a closedness of only ~0.57.

    Sim's ``action[7]`` is a *command*: a 0-frame step from 0 to 1 (measured:
    73.2% of frames <0.05, 26.8% >0.95, nothing in between), meaning "squeeze
    shut" — the object is what stops the fingers, at a measured 55 mm.

    ``binary`` reproduces that command semantics so a real-only policy learns to
    squeeze rather than to stop at the beaker's own width.  On the rig the action
    is dispatched as ``width_m = (1 - closedness) * stroke``; a policy emitting
    0.57 would command 34 mm — exactly the beaker width, with no grip margin.
    ``measured`` keeps the recorded ramp verbatim (faithful, but that failure
    mode is expected).
    """
    if mode == "measured":
        return np.clip(1.0 - width / stroke, 0.0, 1.0)

    open_w = float(np.median(width[:5]))
    closing = width < open_w - CLOSE_MARGIN_M
    if not closing.any():
        raise RuntimeError(f"{ep_name}: gripper never closes by {CLOSE_MARGIN_M * 1000:.0f} mm")
    onset = int(np.argmax(closing))
    # A re-open would need a multi-step action, not a single step; all 11 source
    # episodes are one monotone descent, so refuse rather than silently flatten.
    if (width[onset:] > open_w - CLOSE_MARGIN_M).any():
        raise RuntimeError(f"{ep_name}: gripper re-opens after frame {onset}; binary mode cannot represent this")
    out = np.zeros_like(width)
    out[onset:] = 1.0
    return out


PARQUET_SCHEMA = pa.schema(
    [
        ("observation.state", pa.list_(pa.float64())),
        ("action", pa.list_(pa.float64())),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
        ("next.done", pa.bool_()),
    ]
)


def convert(src: Path, dst: Path, task: str, stroke: float, exclude: set[str], gripper_action: str) -> None:
    episodes = sorted(
        d for d in src.iterdir() if d.is_dir() and d.name.startswith("episode_") and d.name not in exclude
    )
    if not episodes:
        raise SystemExit(f"no episode_* directories under {src}")
    logger.info(f"{len(episodes)} source episodes under {src}")

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    state_acc, action_acc = _Accum(8), _Accum(8)
    episodes_meta: list[dict] = []
    episodes_stats: list[dict] = []
    global_index = 0
    total_frames = 0

    for ep_idx, ep_dir in enumerate(episodes):
        ts_state, state = _read_csv_matrix(ep_dir / "observation.state.joint_position" / "data.csv")
        ts_action, action = _read_csv_matrix(ep_dir / "actions.joint_position" / "data.csv")
        if len(state) != len(action):
            raise RuntimeError(f"{ep_dir.name}: state rows {len(state)} != action rows {len(action)}")
        n = len(state)

        # ---- gripper conventions -------------------------------------------
        # state[7] stays a width in metres (same as sim).
        # action[7] becomes a closedness in [0, 1] (sim convention).
        width = action[:, 7]
        if width.max() > stroke * 1.05:
            raise RuntimeError(f"{ep_dir.name}: action gripper {width.max():.4f} m exceeds stroke {stroke} m")
        action = action.copy()
        action[:, 7] = _gripper_action(width, stroke, gripper_action, ep_dir.name)

        chunk = ep_idx // CHUNK_SIZE
        ep_stats: dict = {}

        for view, key in CAMERA_MAP.items():
            cam_dir = ep_dir / f"observation.image.{view}"
            frame_ts = _read_frame_timestamps(cam_dir / "timestamps.csv")
            keep = _nearest_indices(frame_ts, ts_state)
            out_mp4 = dst / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{key}" / f"episode_{ep_idx:06d}.mp4"
            ep_stats[f"observation.images.{key}"] = _remap_video(cam_dir / "video.mp4", out_mp4, keep, len(frame_ts))

        frame_index = np.arange(n, dtype=np.int64)
        index = np.arange(global_index, global_index + n, dtype=np.int64)
        done = np.zeros(n, dtype=bool)
        done[-1] = True
        # LeRobot timestamps are the nominal 1/fps grid, not the wall clock.
        timestamp = (frame_index / FPS).astype(np.float32)

        table = pa.Table.from_pydict(
            {
                "observation.state": [row.tolist() for row in state],
                "action": [row.tolist() for row in action],
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": np.full(n, ep_idx, dtype=np.int64),
                "index": index,
                "task_index": np.zeros(n, dtype=np.int64),
                "next.done": done,
            },
            schema=PARQUET_SCHEMA,
        )
        out_parquet = dst / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_parquet)

        state_acc.update(state)
        action_acc.update(action)
        per_state, per_action = _Accum(8), _Accum(8)
        per_state.update(state)
        per_action.update(action)
        ep_stats["observation.state"] = per_state.finalize()
        ep_stats["action"] = per_action.finalize()
        ep_stats["timestamp"] = _scalar_stats(timestamp)
        ep_stats["frame_index"] = _scalar_stats(frame_index)
        ep_stats["episode_index"] = _scalar_stats(np.full(n, ep_idx))
        ep_stats["index"] = _scalar_stats(index)
        ep_stats["task_index"] = _scalar_stats(np.zeros(n))
        ep_stats["next.done"] = _scalar_stats(done.astype(np.float64))

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": n})
        episodes_stats.append({"episode_index": ep_idx, "stats": ep_stats})
        global_index += n
        total_frames += n
        logger.info(f"[{ep_idx + 1}/{len(episodes)}] {ep_dir.name} -> {n} frames")

    meta = dst / "meta"
    with (meta / "episodes.jsonl").open("w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as f:
        for row in episodes_stats:
            f.write(json.dumps(row) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    n_ep = len(episodes_meta)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "base_action": "abs",
        "total_episodes": n_ep,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_ep * len(CAMERA_MAP),
        "total_chunks": (n_ep - 1) // CHUNK_SIZE + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(),
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    # stats.json in pick_v2 holds only the two vector features.
    (meta / "stats.json").write_text(
        json.dumps({"observation.state": state_acc.finalize(), "action": action_acc.finalize()}, indent=2)
    )

    # modality.json is consumed by GR00T; harmless for openpi, keeps parity.
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": {"single_arm": {"start": 0, "end": 8}},
                "action": {"single_arm": {"start": 0, "end": 8}},
                "video": {k: {"original_key": f"observation.images.{k}"} for k in CAMERA_MAP.values()},
                "annotation": {"human.task_description": {"original_key": "task_index"}},
            },
            indent=2,
        )
    )

    logger.success(f"wrote {n_ep} episodes / {total_frames} frames to {dst}")
    summary = pd.DataFrame(
        {
            "state_min": state_acc.finalize()["min"],
            "state_max": state_acc.finalize()["max"],
            "action_min": action_acc.finalize()["min"],
            "action_max": action_acc.finalize()["max"],
        }
    )
    logger.info("\n" + summary.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="dir containing episode_*/ recordings")
    ap.add_argument("--dst", type=Path, required=True, help="output LeRobot v2.1 dataset dir")
    ap.add_argument("--task", required=True, help="task string written to tasks.jsonl")
    ap.add_argument("--gripper-stroke", type=float, default=0.08, help="full-open width in metres")
    ap.add_argument("--exclude", nargs="*", default=[], help="episode dir names to skip")
    ap.add_argument(
        "--gripper-action",
        choices=("binary", "measured"),
        default="binary",
        help="binary: sim-style 0/1 close command (default); measured: recorded width verbatim",
    )
    args = ap.parse_args()
    convert(args.src, args.dst, args.task, args.gripper_stroke, set(args.exclude), args.gripper_action)


if __name__ == "__main__":
    main()
