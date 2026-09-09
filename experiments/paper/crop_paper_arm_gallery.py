"""Crop the 24-arm publication frames and tile them into one figure.

The gallery is shot wide so the bench's front edge and a slice of the lab room
stay in frame, which leaves each arm occupying only ~10% of the frame. Rather
than re-render, this crops after the fact.

Finding the crop needs no hand-tuning. Every frame is shot with the same camera,
backdrop and light rig, so the ONLY thing that differs between them is the arm
and the vessel it holds. The per-pixel median across the whole set therefore
reconstructs the arm-free background, and any pixel that deviates from it belongs
to some arm.

Two modes:

``--mode square`` (default)
    One tight 1:1 crop per arm, centred on that arm alone, so the arm fills its
    panel. Note this normalises away the real size differences between arms
    (silhouettes range from 203x819 px for unitree_z1 to 799x1272 px for
    doosan_m1013) -- every arm ends up looking about the same size.

``--mode shared``
    One box, measured from the union of all silhouettes and applied identically
    to every frame. Preserves relative size, at the cost of making the small arms
    genuinely small.

Usage:
    python -m experiments.paper.crop_paper_arm_gallery                      # square + merged grid
    python -m experiments.paper.crop_paper_arm_gallery --mode shared
    python -m experiments.paper.crop_paper_arm_gallery --print-box          # report, write nothing
"""

import argparse
import os

import cv2
import numpy as np
from loguru import logger

# Arms whose silhouette must not drag a SHARED box open. Split ALOHA stands on the
# floor beside the bench and reaches over its edge, so its platform sits a metre
# further back and much lower than every table-mounted arm. Irrelevant in square
# mode, where each arm is measured on its own.
SOLO_BOX_ROBOTS = {"split_aloha_fl"}

# A pixel this far from the reconstructed background (0-255, per channel max) is
# treated as arm. Low enough to catch the light grey arms against the light grey
# cabinets, high enough to ignore renderer noise.
DIFF_THRESHOLD = 14
# Fraction of the measured silhouette added as breathing room on every side.
PADDING_RATIO = 0.08

# Arms the silhouette method cannot frame at all, mapped to the world-space box
# to project instead.
#
# The median-background trick assumes every frame shares a background. Split ALOHA
# breaks that assumption by construction: it is the one arm rendered with the
# backdrop shifted (--paper-background-translation -0.19), so its ENTIRE
# background differs from the median and the mask's largest blobs are strips of
# wall and bench, not the robot. Framing it therefore has to come from geometry.
#
# The box is the platform's own extent: base x, dual-arm lateral spread y, floor
# to arm top z.
WORLD_FRAMED_ROBOTS = {
    "split_aloha_fl": {"x": (-2.03, -0.98), "y": (-0.95, 0.35), "z": (0.10, 1.55)},
}

# Render camera, kept in sync with experiments/paper/render_paper_arm_gallery.sh. The USD
# focalLength is `--camera-focal-length` x 10 (Isaac takes stage units), against
# USD's default horizontal aperture; Isaac keeps pixels square, so the vertical
# aperture follows the output aspect ratio. Validated against a rendered frame:
# the bench's far edge (x=-0.735, z=0.720) projects to y=1420.7 px, which is
# where it lands in outputs/paper_arm_gallery/franka.
CAMERA_EYE = (3.40, 0.00, 1.80)
CAMERA_TARGET = (-0.30, 0.00, 1.10)
FOCAL_LENGTH = 27.1
HORIZONTAL_APERTURE = 20.955


def _find_frames(gallery_dir: str) -> dict[str, str]:
    """Map robot name -> its success frame."""
    skip = {"logs", "cropped", "cropped_square"}
    frames: dict[str, str] = {}
    for name in sorted(os.listdir(gallery_dir)):
        robot_dir = os.path.join(gallery_dir, name)
        if not os.path.isdir(robot_dir) or name in skip:
            continue
        candidates = sorted(f for f in os.listdir(robot_dir) if f.endswith("_success.png"))
        if candidates:
            frames[name] = os.path.join(robot_dir, candidates[0])
        else:
            logger.warning(f"{name}: no success frame; skipping")
    return frames


def _silhouette_mask(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels where *image* departs from the shared background."""
    deviation = np.abs(image.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    mask = (deviation > DIFF_THRESHOLD).astype(np.uint8)
    # Renderer noise and anti-aliasing leave isolated speckle that would otherwise
    # pin the bounding box to a stray pixel in the far corner of the room.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool)


def _box_from_mask(mask: np.ndarray, width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
    """Pad a silhouette mask out to a clamped box of the requested aspect ratio."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("Empty silhouette mask")
    y0, y1 = float(rows.min()), float(rows.max())
    x0, x1 = float(cols.min()), float(cols.max())

    pad_x = (x1 - x0) * PADDING_RATIO
    pad_y = (y1 - y0) * PADDING_RATIO
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y

    # Grow the shorter side: squeezing would misreport the arm's proportions.
    box_w, box_h = x1 - x0, y1 - y0
    if box_w / box_h < aspect:
        box_w = box_h * aspect
    else:
        box_h = box_w / aspect
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # A box larger than the frame has to shrink, but both sides must shrink by the
    # SAME factor -- clamping each axis independently would silently hand back a
    # non-square box in square mode (split_aloha_fl's silhouette fills the frame).
    fit = min(1.0, width / box_w, height / box_h)
    box_w, box_h = box_w * fit, box_h * fit
    cx = float(np.clip(cx, box_w / 2.0, width - box_w / 2.0))
    cy = float(np.clip(cy, box_h / 2.0, height - box_h / 2.0))
    return (
        round(cx - box_w / 2.0),
        round(cy - box_h / 2.0),
        round(cx + box_w / 2.0),
        round(cy + box_h / 2.0),
    )


def _world_box_mask(roi: dict, width: int, height: int) -> np.ndarray:
    """Mask covering the projection of a world-space box, for _box_from_mask."""
    eye = np.asarray(CAMERA_EYE, dtype=float)
    forward = np.asarray(CAMERA_TARGET, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    vertical_aperture = HORIZONTAL_APERTURE * height / width

    pixels = []
    for x in roi["x"]:
        for y in roi["y"]:
            for z in roi["z"]:
                delta = np.asarray((x, y, z), dtype=float) - eye
                depth = float(delta @ forward)
                ndc_x = (float(delta @ right) / depth) * (FOCAL_LENGTH / (HORIZONTAL_APERTURE / 2.0))
                ndc_y = (float(delta @ up) / depth) * (FOCAL_LENGTH / (vertical_aperture / 2.0))
                pixels.append(((ndc_x + 1.0) / 2.0 * width, (1.0 - ndc_y) / 2.0 * height))

    corners = np.array(pixels)
    x0, y0 = np.clip(corners.min(axis=0), [0, 0], [width - 1, height - 1]).astype(int)
    x1, y1 = np.clip(corners.max(axis=0), [0, 0], [width - 1, height - 1]).astype(int)
    mask = np.zeros((height, width), bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _merge(panels: dict[str, np.ndarray], cols: int, cell: int, out_path: str) -> None:
    """Tile panels into one image: no labels, no gutters."""
    names = sorted(panels)
    rows = (len(names) + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for index, name in enumerate(names):
        tile = cv2.resize(panels[name], (cell, cell), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, cols)
        sheet[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = tile
    cv2.imwrite(out_path, sheet)
    logger.success(f"Merged {len(names)} panels into {out_path} ({sheet.shape[1]}x{sheet.shape[0]}, {cols}x{rows})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-dir", default="outputs/paper_arm_gallery")
    parser.add_argument("--out-dir", default=None, help="default: <gallery-dir>/cropped_square or /cropped")
    parser.add_argument("--mode", choices=["square", "shared"], default="square")
    parser.add_argument("--cols", type=int, default=6, help="columns in the merged grid")
    parser.add_argument("--cell", type=int, default=900, help="merged grid cell size in pixels")
    parser.add_argument("--merge", default=None, help="default: <gallery-dir>/gallery_grid.png")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--print-box", action="store_true", help="report boxes, write nothing")
    args = parser.parse_args()

    gallery_dir = os.path.abspath(args.gallery_dir)
    default_out = "cropped_square" if args.mode == "square" else "cropped"
    out_dir = os.path.abspath(args.out_dir or os.path.join(gallery_dir, default_out))
    merge_path = os.path.abspath(args.merge or os.path.join(gallery_dir, "gallery_grid.png"))

    frames = _find_frames(gallery_dir)
    if len(frames) < 3:
        logger.error(f"Need at least 3 frames to reconstruct the background; found {len(frames)}")
        return

    images = {robot: cv2.imread(path) for robot, path in frames.items()}
    for robot in [r for r, image in images.items() if image is None]:
        logger.error(f"{robot}: could not read {frames[robot]}")
        del images[robot]

    shapes = {image.shape for image in images.values()}
    if len(shapes) != 1:
        logger.error(f"Frames disagree on resolution: {shapes}")
        return
    height, width = next(iter(shapes))[:2]

    background = np.median(np.stack(list(images.values())), axis=0).astype(np.uint8)
    masks = {robot: _silhouette_mask(image, background) for robot, image in images.items()}
    logger.info(f"Background reconstructed from {len(images)} frames at {width}x{height}")

    boxes: dict[str, tuple[int, int, int, int]] = {}
    if args.mode == "square":
        for robot, mask in masks.items():
            roi = WORLD_FRAMED_ROBOTS.get(robot)
            fitted = _world_box_mask(roi, width, height) if roi else mask
            boxes[robot] = _box_from_mask(fitted, width, height, aspect=1.0)
    else:
        shared_sources = [r for r in masks if r not in SOLO_BOX_ROBOTS]
        shared = _box_from_mask(
            np.logical_or.reduce([masks[r] for r in shared_sources]), width, height, aspect=width / height
        )
        logger.info(f"Shared box {shared} measured from {len(shared_sources)} arms")
        for robot, mask in masks.items():
            boxes[robot] = _box_from_mask(mask, width, height, width / height) if robot in SOLO_BOX_ROBOTS else shared

    for robot in sorted(boxes):
        x0, y0, x1, y1 = boxes[robot]
        logger.info(f"{robot:26s} box=({x0},{y0})-({x1},{y1}) size={x1 - x0}x{y1 - y0}")
    if args.print_box:
        return

    os.makedirs(out_dir, exist_ok=True)
    panels: dict[str, np.ndarray] = {}
    for robot, image in images.items():
        x0, y0, x1, y1 = boxes[robot]
        panel = image[y0:y1, x0:x1]
        panels[robot] = panel
        if not cv2.imwrite(os.path.join(out_dir, f"{robot}.png"), panel):
            logger.error(f"{robot}: failed to write panel")
    logger.success(f"Wrote {len(panels)} panel(s) to {out_dir}")

    if not args.no_merge:
        _merge(panels, args.cols, args.cell, merge_path)


if __name__ == "__main__":
    main()
