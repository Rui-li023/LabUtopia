"""CLI: convert a LabUtopia run dir to LeRobot v2.1 or v3.0 on-disk layout.

Usage:
  python -m scripts.lerobot_export.cli \
      --src test_outputs/level1_collect100/level1_pick/run \
      --dst /tmp/lerobot_v21_pick \
      --version v2.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .writer_v21 import write_v21
from .writer_v30 import write_v30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="LabUtopia run dir (contains dataset/)")
    ap.add_argument("--dst", required=True, type=Path, help="Output LeRobot dataset root")
    ap.add_argument("--version", required=True, choices=["v2.1", "v3.0"])
    ap.add_argument("--fps", type=int, default=None,
                    help="Override fps; default: probe from source mp4 (LabUtopia=30)")
    ap.add_argument("--robot", default="franka")
    ap.add_argument("--base-action", default="abs", choices=["abs", "body_delta"],
                    help="Mobile-base action dims (0:3): 'abs' = raw spawn-frame position "
                         "targets; 'body_delta' = per-step body-frame [forward, lateral, dtheta]")
    ap.add_argument("--nav-only", action="store_true",
                    help="Trim each episode to its leading navigation segment (phase == 0 "
                         "prefix): state/action sliced, videos cut to the same frame count. "
                         "For training dedicated navigation policies on mobile tasks.")
    args = ap.parse_args()

    if args.version == "v2.1":
        result = write_v21(args.src, args.dst, fps=args.fps, robot_type=args.robot,
                           base_action=args.base_action, nav_only=args.nav_only)
    else:
        if args.base_action != "abs":
            raise SystemExit("--base-action body_delta is only implemented for v2.1")
        if args.nav_only:
            raise SystemExit("--nav-only is only implemented for v2.1")
        result = write_v30(args.src, args.dst, fps=args.fps, robot_type=args.robot)
    print(f"OK [{args.version}]: {result}  →  {args.dst}")


if __name__ == "__main__":
    main()
