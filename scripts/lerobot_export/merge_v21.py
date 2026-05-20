"""Merge multiple LeRobot v2.1 datasets into a single combined dataset.

Use case: 11 per-task level1 datasets → one unified dataset so a policy can
be trained jointly across tasks (task differentiation via ``task_index``).

Assumes every source dataset has identical schema (state_dim, action_dim,
cameras, image_shape, fps). The script verifies this.

Output layout matches the per-task v2.1 writer:
  data/chunk-000/episode_NNNNNN.parquet
  videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4
  meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl,
        stats.json, modality.json}

Usage:
  python -m scripts.lerobot_export.merge_v21 \
      --src-root test_outputs/lerobot/v21 \
      --dst      test_outputs/lerobot/v21_level1_all
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import Accum

CHUNK_SIZE = 1000  # LeRobot convention: 1000 episodes per chunk-{NNN}/ folder


def _chunk_of(ep_idx: int) -> int:
    return ep_idx // CHUNK_SIZE


def _load_jsonl(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _scalar_stat(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "mean": [float(arr.mean())],
        "std": [float(arr.std(ddof=0))],
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "count": [int(arr.size)],
    }


def _vec_stat(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0, ddof=0).astype(np.float32).tolist(),
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def merge(src_dirs: list[Path], dst: Path):
    if dst.exists():
        shutil.rmtree(dst)

    # ── Validate schema compatibility from first dataset ──
    infos = [json.loads((d / "meta" / "info.json").read_text()) for d in src_dirs]
    ref = infos[0]
    ref_feats = ref["features"]
    state_dim = ref_feats["observation.state"]["shape"][0]
    action_dim = ref_feats["action"]["shape"][0]
    cameras = sorted(k[len("observation.images."):]
                     for k in ref_feats if k.startswith("observation.images."))
    fps = ref["fps"]
    image_shape = ref_feats[f"observation.images.{cameras[0]}"]["shape"]

    for i, info in enumerate(infos[1:], 1):
        fts = info["features"]
        sd = fts["observation.state"]["shape"][0]
        ad = fts["action"]["shape"][0]
        cs = sorted(k[len("observation.images."):]
                    for k in fts if k.startswith("observation.images."))
        if (sd, ad, cs, info["fps"]) != (state_dim, action_dim, cameras, fps):
            raise ValueError(
                f"Schema mismatch in {src_dirs[i]}: "
                f"got (state={sd}, action={ad}, cams={cs}, fps={info['fps']}); "
                f"expected (state={state_dim}, action={action_dim}, "
                f"cams={cameras}, fps={fps})"
            )

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    # chunk-{NNN}/ subdirs are created on demand per episode below.

    tasks: list[str] = []
    episode_rows: list[dict] = []
    episode_stats_rows: list[dict] = []
    state_acc = Accum(shape=(state_dim,))
    action_acc = Accum(shape=(action_dim,))

    new_ep_idx = 0
    global_index = 0
    total_frames = 0

    for src in src_dirs:
        print(f"[merge] reading {src.name}", flush=True)
        src_eps = _load_jsonl(src / "meta" / "episodes.jsonl")
        src_ep_stats = {r["episode_index"]: r["stats"]
                        for r in _load_jsonl(src / "meta" / "episodes_stats.jsonl")}
        src_tasks = {r["task_index"]: r["task"]
                     for r in _load_jsonl(src / "meta" / "tasks.jsonl")}

        for ep in src_eps:
            old_idx = ep["episode_index"]
            T = int(ep["length"])
            task_str = ep["tasks"][0] if ep["tasks"] else src_tasks.get(0, "")
            if task_str not in tasks:
                tasks.append(task_str)
            task_index = tasks.index(task_str)

            # ── parquet: rewrite episode_index / index / task_index ──
            src_chunk = _chunk_of(old_idx)
            dst_chunk = _chunk_of(new_ep_idx)
            src_pq = src / "data" / f"chunk-{src_chunk:03d}" / f"episode_{old_idx:06d}.parquet"
            df = pd.read_parquet(src_pq)
            df["episode_index"] = np.full(T, new_ep_idx, dtype=np.int64)
            df["index"] = np.arange(global_index, global_index + T, dtype=np.int64)
            df["task_index"] = np.full(T, task_index, dtype=np.int64)
            out_dir = dst / "data" / f"chunk-{dst_chunk:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_pq = out_dir / f"episode_{new_ep_idx:06d}.parquet"
            df.to_parquet(out_pq, index=False)

            # accumulate aggregate stats over state/action
            state_arr = np.stack(df["observation.state"].to_list()).astype(np.float32)
            action_arr = np.stack(df["action"].to_list()).astype(np.float32)
            state_acc.update(state_arr)
            action_acc.update(action_arr)

            # ── videos: copy with new episode id ──
            for cam in cameras:
                src_vid = (src / "videos" / f"chunk-{src_chunk:03d}"
                           / f"observation.images.{cam}" / f"episode_{old_idx:06d}.mp4")
                if not src_vid.exists():
                    continue
                out_cam_dir = (dst / "videos" / f"chunk-{dst_chunk:03d}"
                               / f"observation.images.{cam}")
                out_cam_dir.mkdir(parents=True, exist_ok=True)
                out_vid = out_cam_dir / f"episode_{new_ep_idx:06d}.mp4"
                shutil.copy2(src_vid, out_vid)

            # ── per-episode stats: reuse from source where possible,
            #    rebuild the index/task/episode scalar stats with new values ──
            ep_st = dict(src_ep_stats.get(old_idx, {}))
            new_indices = np.arange(global_index, global_index + T, dtype=np.float32)
            ep_st["index"] = _scalar_stat(new_indices)
            ep_st["episode_index"] = _scalar_stat(np.full(T, new_ep_idx, dtype=np.float32))
            ep_st["task_index"] = _scalar_stat(np.full(T, task_index, dtype=np.float32))
            episode_stats_rows.append({"episode_index": new_ep_idx, "stats": ep_st})

            episode_rows.append({
                "episode_index": new_ep_idx,
                "tasks": [task_str],
                "length": T,
            })
            new_ep_idx += 1
            global_index += T
            total_frames += T

    # ── Write meta ──
    with (dst / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")

    with (dst / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (dst / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for row in episode_stats_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "observation.state": state_acc.finalize(),
        "action": action_acc.finalize(),
    }
    with (dst / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    info = {
        "codebase_version": "v2.1",
        "robot_type": ref.get("robot_type", "franka"),
        "total_episodes": new_ep_idx,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": new_ep_idx * len(cameras),
        "total_chunks": (new_ep_idx + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{new_ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": ref_feats,
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

    print(f"[merge] done: {new_ep_idx} episodes / {total_frames} frames / "
          f"{len(tasks)} tasks → {dst}")
    return {"episodes": new_ep_idx, "frames": total_frames, "tasks": len(tasks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, type=Path,
                    help="Directory containing per-task v2.1 datasets")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output directory for merged dataset")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Optional subset of task subdirs (default: all level1_* in src-root)")
    args = ap.parse_args()

    if args.tasks:
        src_dirs = [args.src_root / t for t in args.tasks]
    else:
        src_dirs = sorted(d for d in args.src_root.iterdir()
                          if d.is_dir() and d.name.startswith("level1_"))
    missing = [d for d in src_dirs if not (d / "meta" / "info.json").exists()]
    if missing:
        raise SystemExit(f"Missing meta/info.json in: {missing}")
    print(f"[merge] sources ({len(src_dirs)}): {[d.name for d in src_dirs]}")
    merge(src_dirs, args.dst)


if __name__ == "__main__":
    main()
