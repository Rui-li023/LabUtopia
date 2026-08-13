"""Contact sheet: the real rig's frames next to the sim renders, per camera.

Every tile is 640x480 and every column uses the SAME calibrated intrinsics, so
they are comparable pixel for pixel. Default layout is three columns:

    REAL live      a fresh frame pulled off the rig (GET /cameras/<name>/snapshot)
    REAL overlay   the rig's own verify_calibration_overlay.py output
    SIM            build_table_scene.py --render, cameras placed from the same
                   calibration

WHAT TO ACTUALLY COMPARE. The scene contents will NOT match --- the rig usually
has a calibration board, cables and whatever is clamped in the gripper, while the
sim bench is clean. Compare GEOMETRY, and specifically the anchor glyphs:

    yellow ring    where the RIG's overlay tool draws the frame origin (measured)
    cyan cross     where OUR extrinsics put that same origin (computed)

Both anchors are projections of a frame ORIGIN and are pose-independent --- the
wrist camera is rigid to the gripper, and the robot base does not move --- so they
are valid on any frame from that camera, whatever the arm is doing. If the ring and
the cross sit on top of each other, the extrinsics agree. The numeric delta is
printed and drawn in the caption.

    python scripts/sim2real/compare_real_vs_sim.py --open
    python scripts/sim2real/compare_real_vs_sim.py --snapshots      # old 08-06 stills
    python scripts/sim2real/compare_real_vs_sim.py --pull           # re-pull from nuc2

Run build_table_scene.py --render first; this only composites what is on disk.
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from usd_extrinsics import EXTRINSICS, OVERLAY_ANCHORS  # noqa: E402
from usd_intrinsics import INTRINSICS, hfov_deg, vfov_deg  # noqa: E402

# The Franka station. docs/real2sim.md in DataCollectionSystemV2 is its handover.
NUC2 = "haomingsong@192.168.31.157"
NUC2_CAMERA = {  # scene camera name -> collector channel name
    "third_view_orbbec335L": "observation.image.third_view",
    "second_third_realsense435i": "observation.image.second_third_view",
    "left_wrist_orbbec335": "observation.image.left_wrist_view",
}
SHORT = {k: v.rsplit(".", 1)[-1] for k, v in NUC2_CAMERA.items()}

LIVE_DIR = "/home/ubuntu/Downloads/nuc2_live"
OVERLAY_DIR = "/home/ubuntu/Downloads/nuc2_overlay"
SNAPSHOT_DIR = "/data1/DataCollectionSystemV2/data/snapshots/20260806-164844"
SNAPSHOT_FRAME = {
    "third_view_orbbec335L": "third_view_orbbec335L_CP2R5530009H.jpg",
    "second_third_realsense435i": "second_third_realsense435i_040322071795.jpg",
    "left_wrist_orbbec335": "left_wrist_orbbec335_CP02653000Z2.jpg",
}
CAL_DATE = {
    "third_view_orbbec335L": "2026-08-08 16:59",
    "second_third_realsense435i": "2026-08-08 17:27",
    "left_wrist_orbbec335": "2026-08-07 16:03",
}

TILE_W, TILE_H = 640, 480
CAPTION_H, TITLE_H, GAP = 26, 46, 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
RING = (60, 220, 255)  # BGR, the rig's measured marker
CROSS = (255, 210, 60)  # BGR, our computed projection


def label(width, height, text, scale=0.45, bg=(38, 38, 38), fg=(235, 235, 235)):
    bar = np.full((height, width, 3), bg, np.uint8)
    cv2.putText(bar, text, (8, height - 8), FONT, scale, fg, 1, cv2.LINE_AA)
    return bar


def anchor_pixels(name):
    """(rig measured (u,v), ours (u,v)) for this camera, or None if no anchor."""
    if name not in OVERLAY_ANCHORS:
        return None
    point, measured = OVERLAY_ANCHORS[name]
    intr = INTRINSICS[name]
    p_cam = np.linalg.inv(EXTRINSICS[name].cam_in_ref) @ np.array([*point, 1.0])
    ours = (intr.fx * p_cam[0] / p_cam[2] + intr.cx, intr.fy * p_cam[1] / p_cam[2] + intr.cy)
    return measured, ours


def draw_anchor(img, anchor):
    """Ring = what the rig drew, cross = what we compute. Overlapping is the pass."""
    if anchor is None:
        return img
    (mu, mv), (ou, ov) = anchor
    out = img.copy()
    cv2.circle(out, (round(mu), round(mv)), 11, RING, 2, cv2.LINE_AA)
    x, y = round(ou), round(ov)
    cv2.line(out, (x - 16, y), (x + 16, y), CROSS, 1, cv2.LINE_AA)
    cv2.line(out, (x, y - 16), (x, y + 16), CROSS, 1, cv2.LINE_AA)
    return out


def load(path, missing_text):
    img = cv2.imread(path) if path else None
    if img is None:
        img = np.full((TILE_H, TILE_W, 3), 24, np.uint8)
        for i, line in enumerate(missing_text.split("\n")):
            cv2.putText(img, line, (16, TILE_H // 2 + 20 * i), FONT, 0.45, (90, 90, 200), 1, cv2.LINE_AA)
        return img, False
    if img.shape[:2] != (TILE_H, TILE_W):
        img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    return img, True


def pull_from_nuc2(live_dir, overlay_dir):
    """Fresh snapshot off every camera, plus whatever overlays are on the box."""
    os.makedirs(live_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    fetch = " ".join(
        f'curl -s --max-time 15 -o /tmp/live_snap/{SHORT[k]}.jpg "http://127.0.0.1:9527/cameras/{v}/snapshot";'
        for k, v in NUC2_CAMERA.items()
    )
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", NUC2, f"mkdir -p /tmp/live_snap && {fetch} ls /tmp/live_snap/"],
        check=True,
        timeout=180,
    )
    for remote, local in ((" /tmp/live_snap/*.jpg", live_dir), (" /tmp/overlay/*.jpg", overlay_dir)):
        subprocess.run(f"scp -o BatchMode=yes {NUC2}:'{remote.strip()}' {local}/", shell=True, timeout=180)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", default="/home/ubuntu/Downloads/sim2real_renders")
    ap.add_argument("--live-dir", default=LIVE_DIR)
    ap.add_argument("--overlay-dir", default=OVERLAY_DIR)
    ap.add_argument("--snapshots", action="store_true", help="use the stale 2026-08-06 stills instead of live")
    ap.add_argument("--pull", action="store_true", help="re-pull live frames + overlays off nuc2 first")
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if args.pull:
        pull_from_nuc2(args.live_dir, args.overlay_dir)

    stem = "compare_snapshots_vs_sim" if args.snapshots else "compare_live_vs_sim"
    out_path = args.out or os.path.join(args.render_dir, stem + ".png")

    def columns(name):
        sim = (os.path.join(args.render_dir, name + ".png"), "SIM  calibrated extrinsics, clean bench")
        if args.snapshots:
            return [
                (os.path.join(SNAPSHOT_DIR, SNAPSHOT_FRAME[name]), "REAL stills 2026-08-06 -- PREDATE the calibration"),
                sim,
            ]
        return [
            (os.path.join(args.live_dir, SHORT[name] + ".jpg"), "REAL live  /cameras/<name>/snapshot"),
            (os.path.join(args.overlay_dir, SHORT[name] + ".jpg"), "REAL overlay  verify_calibration_overlay.py"),
            sim,
        ]

    names = [n for n in NUC2_CAMERA if n in EXTRINSICS]
    ncol = len(columns(names[0]))
    width = ncol * TILE_W + (ncol + 1) * GAP
    rows = [
        label(
            width,
            TITLE_H,
            "sim2real camera check   |   yellow ring = frame origin as the RIG projects it"
            "   |   cyan cross = same origin as OUR extrinsics project it   |   they should coincide",
            scale=0.52,
            bg=(20, 20, 20),
        )
    ]

    print(f"  {'camera':28s} {'fov':>13s}  anchor delta (ours vs rig)")
    for name in names:
        intr, extr = INTRINSICS[name], EXTRINSICS[name]
        anchor = anchor_pixels(name)
        if anchor:
            (mu, mv), (ou, ov) = anchor
            delta = f"d=({abs(ou - mu):.1f},{abs(ov - mv):.1f})px"
        else:
            delta = "no anchor: origin projects off-frame, UNVERIFIED"

        caps = np.full((CAPTION_H, width, 3), 38, np.uint8)
        band = np.full((TILE_H, width, 3), 38, np.uint8)
        for i, (path, cap) in enumerate(columns(name)):
            tile, ok = load(path, f"missing:\n{os.path.basename(path or '?')}")
            if ok and "overlay" not in cap:  # the rig already drew its own on the overlay
                tile = draw_anchor(tile, anchor)
            x0 = GAP + i * (TILE_W + GAP)
            band[:, x0 : x0 + TILE_W] = tile
            caps[:, x0 - GAP : x0 + TILE_W] = label(TILE_W + GAP, CAPTION_H, f"{cap}   [{name}]")
        rows += [caps, band, np.full((GAP, width, 3), 20, np.uint8)]

        borrowed = f" borrowed<-{extr.borrowed_from}" if extr.borrowed_from else ""
        print(
            f"  {name:28s} {hfov_deg(intr.fx):5.1f}x{vfov_deg(intr.fy):4.1f}deg  "
            f"{delta}  cal {CAL_DATE.get(name, '?')} {extr.trans_err_mm:.1f}mm/{extr.rot_err_deg:.2f}deg{borrowed}"
        )

    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, sheet)
    print(f"WROTE {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")
    if args.snapshots:
        print("NOTE: the 08-06 stills predate every calibration -- framing differences there are NOT bugs.")
    if args.open:
        subprocess.Popen(["xdg-open", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
