"""Navigation motion-quality metrics for Level-5 episodes.

Reads an episode HDF5 and reports, over the navigation phases (0 and 2):
- heading error while moving (mean / p90): does the base face where it drives?
- stall fraction: moving-phase frames stuck below 30% of the median speed
  (excluding the natural start/end ramps) — the "move a bit, stop, rotate,
  move again" signature.
- rotate/translate blending: fraction of turning frames that also translate
  (pure-pursuit should turn while driving, not alternate), plus the number of
  alternations between rotate-only and translate-only states.

Usage: python nav_quality.py <episode_dir_or_h5> [--phases 0,2]
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def _resolve(path: str) -> str:
    if os.path.isdir(path):
        matches = sorted(glob.glob(os.path.join(path, "*.h5")))
        if not matches:
            sys.exit(f"no .h5 file in {path}")
        return matches[0]
    return path


def analyze(h5_path: str, phases: list) -> dict:
    with h5py.File(h5_path, "r") as f:
        ph = f["phase"][:]
        ap = f["agent_pose"][:]

    out = {}
    for phase in phases:
        nav = np.where(ph == phase)[0]
        if len(nav) < 10:
            continue
        xy = ap[nav][:, :2]
        th = ap[nav][:, 2]
        v = np.diff(xy, axis=0)
        speed = np.linalg.norm(v, axis=1)
        dth = np.abs(np.diff(th))
        moving = speed > 1e-4
        turning = dth > 2e-3

        travel = np.arctan2(v[moving, 1], v[moving, 0])
        err = np.degrees(np.abs(
            (travel - th[:-1][moving] + np.pi) % (2 * np.pi) - np.pi))

        # Stall metric over the cruise window (skip start/end ramps).
        n = len(speed)
        core = slice(int(0.1 * n), int(0.9 * n))
        s_core = speed[core]
        med = np.median(s_core[s_core > 1e-4]) if (s_core > 1e-4).any() else 0.0
        stall = float((s_core < 0.3 * med).mean()) if med > 0 else 1.0

        # Blending: turning frames that also translate; alternation count
        # between rotate-only and translate-only states.
        blend = float((turning & moving).sum() / max(1, turning.sum()))
        state = np.zeros(len(speed), dtype=int)   # 0 idle, 1 rot-only, 2 trans-only, 3 both
        state[turning & ~moving] = 1
        state[~turning & moving] = 2
        state[turning & moving] = 3
        s = state[state != 0]
        alternations = int(((s[1:] == 1) & (s[:-1] == 2)).sum()
                           + ((s[1:] == 2) & (s[:-1] == 1)).sum())

        out[phase] = dict(
            frames=len(nav), moving=int(moving.sum()),
            heading_mean=float(err.mean()), heading_p90=float(np.percentile(err, 90)),
            stall_frac=stall, blend=blend, alternations=alternations,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episode")
    p.add_argument("--phases", default="0,2")
    args = p.parse_args()
    phases = [int(x) for x in args.phases.split(",")]

    results = analyze(_resolve(args.episode), phases)
    for phase, m in results.items():
        print(f"phase {phase}: frames={m['frames']} moving={m['moving']} "
              f"heading_err mean={m['heading_mean']:.1f}deg p90={m['heading_p90']:.1f}deg "
              f"stall={m['stall_frac']:.0%} blend={m['blend']:.0%} "
              f"rot<->trans alternations={m['alternations']}")


if __name__ == "__main__":
    main()
