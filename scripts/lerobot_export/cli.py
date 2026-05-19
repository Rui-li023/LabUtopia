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
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--robot", default="franka")
    args = ap.parse_args()

    if args.version == "v2.1":
        result = write_v21(args.src, args.dst, fps=args.fps, robot_type=args.robot)
    else:
        result = write_v30(args.src, args.dst, fps=args.fps, robot_type=args.robot)
    print(f"OK [{args.version}]: {result}  →  {args.dst}")


if __name__ == "__main__":
    main()
