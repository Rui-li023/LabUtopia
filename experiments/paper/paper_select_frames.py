"""Turn a captured sub-stage strip into the delivered ``<task>/01.png`` sequence.

``experiments/main.py --keyframe-dir`` writes every controller sub-stage boundary, which is
usually one or two frames more than a figure wants (an atomic action's "retreat"
and "done" states often read the same).  This script picks ``--count`` frames —
always keeping the first and the last, and spreading the rest evenly over the
remaining sub-stages — and copies them out under the delivery naming scheme.

With no ``--count`` every captured frame is delivered, just renamed.
"""

import argparse
import os
import shutil


def pick_indices(total: int, count: int | None) -> list[int]:
    """Evenly spaced indices over ``total`` frames, always including both ends."""
    if count is None or count >= total:
        return list(range(total))
    if count <= 1:
        return [0]
    if count == 2:
        return [0, total - 1]
    step = (total - 1) / (count - 1)
    return sorted({round(i * step) for i in range(count)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="keyframe dir written by --keyframe-dir (contains frames/)")
    parser.add_argument("dst", help="delivery dir; receives 01.png, 02.png, ...")
    parser.add_argument("--count", type=int, default=None, help="how many frames to deliver")
    parser.add_argument(
        "--drop", type=int, nargs="*", default=[], help="0-based indices of captured frames to exclude before selection"
    )
    args = parser.parse_args()

    frames_dir = os.path.join(args.src, "frames")
    if not os.path.isdir(frames_dir):
        frames_dir = args.src
    names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".png"))
    if not names:
        raise SystemExit(f"no frames in {frames_dir}")
    names = [n for i, n in enumerate(names) if i not in set(args.drop)]

    chosen = pick_indices(len(names), args.count)
    os.makedirs(args.dst, exist_ok=True)
    for old in os.listdir(args.dst):
        if old.lower().endswith(".png"):
            os.remove(os.path.join(args.dst, old))

    lines = []
    for position, index in enumerate(chosen, start=1):
        target = os.path.join(args.dst, f"{position:02d}.png")
        shutil.copy2(os.path.join(frames_dir, names[index]), target)
        lines.append(f"{position:02d}.png\t{names[index]}")
    with open(os.path.join(args.dst, "frames.tsv"), "w", encoding="utf-8") as handle:
        handle.write("delivered\tcaptured\n")
        handle.write("\n".join(lines) + "\n")
    print(f"{args.dst}: {len(chosen)} frame(s) of {len(names)} captured")
    for line in lines:
        print("  " + line.replace("\t", "  <-  "))


if __name__ == "__main__":
    main()
