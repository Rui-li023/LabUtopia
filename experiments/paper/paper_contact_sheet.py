"""Tile a directory of render candidates into one labelled contact sheet.

Judging framing one 1920x1080 PNG at a time is slow; a grid makes the choice a
single glance.
"""

import argparse
import os

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="directory of PNGs to tile")
    parser.add_argument("dst", help="output contact-sheet PNG")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cell-width", type=int, default=640)
    args = parser.parse_args()

    names = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".png"))
    if not names:
        raise SystemExit(f"no PNGs in {args.src}")

    cells = []
    for name in names:
        image = cv2.imread(os.path.join(args.src, name))
        if image is None:
            continue
        scale = args.cell_width / image.shape[1]
        cell = cv2.resize(image, (args.cell_width, max(1, int(image.shape[0] * scale))))
        cv2.rectangle(cell, (0, 0), (cell.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(cell, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cells.append(cell)

    height = max(cell.shape[0] for cell in cells)
    cells = [
        cv2.copyMakeBorder(cell, 0, height - cell.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for cell in cells
    ]
    rows = []
    for start in range(0, len(cells), args.cols):
        row = cells[start : start + args.cols]
        while len(row) < args.cols:
            row.append(np.full_like(cells[0], 30))
        rows.append(np.hstack(row))
    cv2.imwrite(args.dst, np.vstack(rows))
    print(f"{args.dst} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
