"""Verify a Level-5 mobile episode's on-disk format.

Usage:
    python scripts/level5_tasks/verify_episode.py <episode_dir> --dims 11 --phases 0,1
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def verify(episode_dir: str, dims: int, phases: list) -> list:
    errors = []
    name = os.path.basename(os.path.normpath(episode_dir))
    h5_path = os.path.join(episode_dir, f"{name}.h5")
    if not os.path.exists(h5_path):
        return [f"missing {h5_path}"]
    with h5py.File(h5_path, "r") as f:
        for key in ("actions", "agent_pose"):
            if key not in f:
                errors.append(f"missing dataset {key}")
                continue
            shape = f[key].shape
            if len(shape) != 2 or shape[1] != dims:
                errors.append(f"{key} shape {shape}, expected (T, {dims})")
        if "phase" not in f:
            errors.append("missing dataset phase")
        else:
            ph = np.asarray(f["phase"][:])
            if sorted(set(ph.tolist())) != sorted(phases):
                errors.append(f"phase values {sorted(set(ph.tolist()))}, expected {sorted(phases)}")
            if np.any(np.diff(ph) < 0):
                errors.append("phase sequence is not non-decreasing")
            if "actions" in f and len(ph) != f["actions"].shape[0]:
                errors.append(f"phase length {len(ph)} != actions length {f['actions'].shape[0]}")
    if not glob.glob(os.path.join(episode_dir, "*.mp4")):
        errors.append("no camera mp4 found")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir")
    parser.add_argument("--dims", type=int, default=11)
    parser.add_argument("--phases", type=str, default="0,1")
    args = parser.parse_args()
    phases = [int(v) for v in args.phases.split(",")]
    errors = verify(args.episode_dir, args.dims, phases)
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
