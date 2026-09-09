"""Concatenate LeRobot v2.1 datasets that already share a schema.

Written for the sim2real co-training split: 500 simulated episodes
(``sim2real/pick_v2``) followed by the real-rig episodes exported by
``experiments/sim2real/export_real_lerobot.py``.  Episodes are appended in source
order, so ``pick_v2_real11`` episodes 0-499 are the sim ones and 500+ are real.

Tasks are merged **by string**: a task that already exists in an earlier source
keeps that source's ``task_index`` instead of getting a new one.  The real
export uses "Pick up the conical bottle.", which is already task 1 of the sim
dataset, so the merged dataset still has exactly 3 tasks.

Parquet and mp4 payloads are hard-linked when possible (same filesystem) and
copied otherwise; only the index columns are rewritten.

Usage:
    python experiments/sim2real/merge_lerobot_v21.py \
        --src .../data/v21/sim2real/pick_v2 .../data/v21/sim2real/pick_real11 \
        --dst .../data/v21/sim2real/pick_v2_real11
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 1000


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _pool(stats: list[dict], dim: int) -> dict:
    """Pool per-source mean/std/min/max/count into one aggregate."""
    n = np.array([s["count"][0] for s in stats], dtype=np.float64)
    mean = np.array([s["mean"] for s in stats], dtype=np.float64).reshape(len(stats), dim)
    std = np.array([s["std"] for s in stats], dtype=np.float64).reshape(len(stats), dim)
    total = n.sum()
    pooled_mean = (n[:, None] * mean).sum(0) / total
    second = (n[:, None] * (std**2 + mean**2)).sum(0) / total
    pooled_std = np.sqrt(np.maximum(second - pooled_mean**2, 0.0))
    return {
        "mean": pooled_mean.tolist(),
        "std": pooled_std.tolist(),
        "min": np.min([s["min"] for s in stats], axis=0).tolist(),
        "max": np.max([s["max"] for s in stats], axis=0).tolist(),
        "count": [int(total)],
    }


def merge(srcs: list[Path], dst: Path) -> None:
    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in srcs]
    ref = infos[0]
    for src, info in zip(srcs[1:], infos[1:], strict=True):
        if info["features"] != ref["features"]:
            raise SystemExit(f"{src}: feature schema differs from {srcs[0]}")
        if info["fps"] != ref["fps"]:
            raise SystemExit(f"{src}: fps {info['fps']} != {ref['fps']}")

    video_keys = [k for k in ref["features"] if k.startswith("observation.images.")]

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    task_to_index: dict[str, int] = {}
    tasks_out: list[dict] = []
    episodes_out: list[dict] = []
    stats_out: list[dict] = []
    state_stats, action_stats = [], []
    ep_counter = 0
    global_index = 0

    for src in srcs:
        src_tasks = {t["task_index"]: t["task"] for t in _load_jsonl(src / "meta" / "tasks.jsonl")}
        remap: dict[int, int] = {}
        for old_idx, text in sorted(src_tasks.items()):
            if text not in task_to_index:
                task_to_index[text] = len(task_to_index)
                tasks_out.append({"task_index": task_to_index[text], "task": text})
            remap[old_idx] = task_to_index[text]

        episodes = _load_jsonl(src / "meta" / "episodes.jsonl")
        ep_stats = {r["episode_index"]: r["stats"] for r in _load_jsonl(src / "meta" / "episodes_stats.jsonl")}

        for ep in episodes:
            old_idx, length = ep["episode_index"], ep["length"]
            new_idx = ep_counter
            old_chunk, new_chunk = old_idx // CHUNK_SIZE, new_idx // CHUNK_SIZE

            table = pq.read_table(src / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_idx:06d}.parquet")
            n = table.num_rows
            if n != length:
                raise SystemExit(f"{src} ep{old_idx}: parquet rows {n} != episodes.jsonl length {length}")
            old_task = table.column("task_index").to_pylist()
            new_task = np.array([remap[t] for t in old_task], dtype=np.int64)
            cols = {name: table.column(name) for name in table.schema.names}
            cols["episode_index"] = pa.array(np.full(n, new_idx, dtype=np.int64))
            cols["index"] = pa.array(np.arange(global_index, global_index + n, dtype=np.int64))
            cols["task_index"] = pa.array(new_task)
            out = dst / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_idx:06d}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pydict(cols, schema=table.schema), out)

            for key in video_keys:
                _link_or_copy(
                    src / "videos" / f"chunk-{old_chunk:03d}" / key / f"episode_{old_idx:06d}.mp4",
                    dst / "videos" / f"chunk-{new_chunk:03d}" / key / f"episode_{new_idx:06d}.mp4",
                )

            episodes_out.append({"episode_index": new_idx, "tasks": ep["tasks"], "length": length})

            st = dict(ep_stats[old_idx])
            st["episode_index"] = {
                "mean": [float(new_idx)],
                "std": [0.0],
                "min": [float(new_idx)],
                "max": [float(new_idx)],
                "count": [n],
            }
            st["index"] = {
                "mean": [float(global_index + (n - 1) / 2)],
                "std": [float(np.arange(n).std())],
                "min": [float(global_index)],
                "max": [float(global_index + n - 1)],
                "count": [n],
            }
            st["task_index"] = {
                "mean": [float(new_task.mean())],
                "std": [float(new_task.std())],
                "min": [float(new_task.min())],
                "max": [float(new_task.max())],
                "count": [n],
            }
            stats_out.append({"episode_index": new_idx, "stats": st})

            global_index += n
            ep_counter += 1

        src_stats = json.loads((src / "meta" / "stats.json").read_text())
        state_stats.append(src_stats["observation.state"])
        action_stats.append(src_stats["action"])
        print(f"[merge] {src.name}: {len(episodes)} episodes -> running total {ep_counter}")

    _dump_jsonl(dst / "meta" / "tasks.jsonl", tasks_out)
    _dump_jsonl(dst / "meta" / "episodes.jsonl", episodes_out)
    _dump_jsonl(dst / "meta" / "episodes_stats.jsonl", stats_out)

    dim = ref["features"]["observation.state"]["shape"][0]
    (dst / "meta" / "stats.json").write_text(
        json.dumps(
            {"observation.state": _pool(state_stats, dim), "action": _pool(action_stats, dim)},
            indent=2,
        )
    )

    info_out = dict(ref)
    info_out.update(
        total_episodes=ep_counter,
        total_frames=global_index,
        total_tasks=len(tasks_out),
        total_videos=ep_counter * len(video_keys),
        total_chunks=(ep_counter - 1) // CHUNK_SIZE + 1,
        splits={"train": f"0:{ep_counter}"},
    )
    (dst / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))

    modality = srcs[0] / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, dst / "meta" / "modality.json")

    print(f"[merge] wrote {ep_counter} episodes / {global_index} frames / {len(tasks_out)} tasks to {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", type=Path, required=True, help="source datasets, in output order")
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    merge(args.src, args.dst)


if __name__ == "__main__":
    main()
