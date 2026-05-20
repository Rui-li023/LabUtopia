"""In-place repair of LeRobot v2.1 datasets that were exported under the wrong
fps/codec assumptions.

Background: LabUtopia's DataCollector wrote mp4s as `mp4v` (codec_name=mpeg4)
at 30 fps, but earlier writers hardcoded fps=60 and labelled the codec h264
in info.json. This produced:
  - info.json fps=60 (wrong, real is 30)
  - parquet `timestamp = frame_index/60` (wrong, should be /30)
  - mp4 codec=mpeg4 (wrong, downstream loaders expect h264)

This script fixes a given dataset directory by:
  1. Re-encoding every mp4 to H264/yuv420p at the real source fps
  2. Rewriting `timestamp` in every parquet to `frame_index / real_fps`
  3. Updating meta/info.json fps and video.* info blocks

Usage:
  python -m scripts.lerobot_export.repair_fps_codec --dst <dataset_root> [--fps 30]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _probe_codec(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    return "h264" if fourcc.lower() in ("avc1", "h264") else "mpeg4"


def _probe_fps(path: Path) -> int | None:
    cap = cv2.VideoCapture(str(path))
    f = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(f)) if f and f > 0 else None


def _transcode_one(src: Path, fps: int) -> tuple[Path, bool, str]:
    if _probe_codec(src) == "h264":
        return src, True, "already h264"
    tmp = src.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-r", str(fps), "-an", str(tmp)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        return src, False, e.stderr.decode("utf-8", "replace")[-300:]
    shutil.move(str(tmp), str(src))
    return src, True, "transcoded"


def repair(dst: Path, fps_override: int | None = None, jobs: int = 8):
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    # ── 1. detect real fps from a sample video ──
    sample = next(dst.glob("videos/**/*.mp4"), None)
    if sample is None:
        raise SystemExit(f"No videos found under {dst}")
    real_fps = fps_override or _probe_fps(sample)
    if not real_fps:
        raise SystemExit(f"Could not probe fps from {sample}")
    print(f"[repair] real fps = {real_fps} (info.json had {info['fps']})")

    # ── 2. transcode all mp4s to h264 in parallel ──
    mp4s = sorted(dst.glob("videos/**/*.mp4"))
    print(f"[repair] transcoding {len(mp4s)} mp4s to h264 (jobs={jobs})...")
    n_done = n_skip = n_fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = [ex.submit(_transcode_one, p, real_fps) for p in mp4s]
        for i, fut in enumerate(as_completed(futures)):
            path, ok, msg = fut.result()
            if not ok:
                n_fail += 1
                print(f"  ✗ {path.name}: {msg}")
            elif msg == "already h264":
                n_skip += 1
            else:
                n_done += 1
            if (i + 1) % 200 == 0:
                print(f"  …{i+1}/{len(mp4s)}")
    print(f"[repair] mp4 transcode: {n_done} done, {n_skip} already-ok, {n_fail} failed")

    # ── 3. rewrite parquet timestamps ──
    parquets = sorted(dst.glob("data/**/episode_*.parquet"))
    print(f"[repair] rewriting timestamps in {len(parquets)} parquets…")
    old_fps = float(info.get("fps", 60))
    for i, pq in enumerate(parquets):
        df = pd.read_parquet(pq)
        # Always recompute from frame_index to be safe (idempotent).
        df["timestamp"] = (df["frame_index"].astype(np.float32) / float(real_fps)).astype(np.float32)
        df.to_parquet(pq, index=False)
        if (i + 1) % 200 == 0:
            print(f"  …{i+1}/{len(parquets)}")

    # ── 4. update meta/info.json ──
    info["fps"] = int(real_fps)
    feats = info.get("features", {})
    for k, v in feats.items():
        if k.startswith("observation.images.") and "info" in v:
            v["info"]["video.fps"] = float(real_fps)
            v["info"]["video.codec"] = "h264"
            v["info"]["video.pix_fmt"] = "yuv420p"
    info_path.write_text(json.dumps(info, indent=4))
    print(f"[repair] wrote {info_path} (fps={real_fps}, codec=h264)")

    # ── 5. per-episode stats timestamp range (optional) ──
    stats_path = dst / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        print(f"[repair] updating per-episode timestamp stats in {stats_path.name}…")
        out_lines = []
        for line in stats_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            T = None
            # episode length from frame_index max + 1
            fi = row.get("stats", {}).get("frame_index")
            if fi and isinstance(fi.get("max"), list):
                T = int(fi["max"][0]) + 1
            if T:
                ts = np.arange(T, dtype=np.float32) / float(real_fps)
                row["stats"]["timestamp"] = {
                    "mean": [float(ts.mean())],
                    "std": [float(ts.std(ddof=0))],
                    "min": [float(ts.min())],
                    "max": [float(ts.max())],
                    "count": [int(T)],
                }
            out_lines.append(json.dumps(row, ensure_ascii=False))
        stats_path.write_text("\n".join(out_lines) + "\n")

    print("[repair] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=None,
                    help="Override real fps (otherwise probe from first mp4)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    repair(args.dst, fps_override=args.fps, jobs=args.jobs)


if __name__ == "__main__":
    main()
