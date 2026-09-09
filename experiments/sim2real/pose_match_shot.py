"""Same arm pose, both third-person cameras: the real rig next to the sim render.

Answers one question --- with the Franka at a GIVEN joint pose, does the sim put
the arm where the real cameras see it? Both columns use the same calibrated
intrinsics/extrinsics, so a mismatch is geometry, not framing.

    # 1. grab the rig's frames AND the joint angles at the same instant
    python experiments/sim2real/pose_match_shot.py --pull

    # 2. render the sim at those exact angles (prints the command for you)
    #    NEVER while another Isaac Sim is up --- one instance saturates the GPU
    python scripts/sim2real/build_table_scene.py --render --only third_view \
        second_third --joints <7 angles> --finger <m>

    # 3. composite
    python experiments/sim2real/pose_match_shot.py --open

The pose is read back out of the articulation by build_table_scene.py and printed
as ``reached=``; if that disagrees with ``cmd=`` the sim clamped a joint and the
two columns are NOT the same pose, whatever this sheet looks like.
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "sim2real"))

# scene camera name (sim png stem) -> collector channel suffix (real jpg stem)
PAIRS = [
    ("third_view_orbbec335L", "third_view", "Orbbec Gemini 335L"),
    ("second_third_realsense435i", "second_third_view", "RealSense D435i"),
]

TILE_W, TILE_H = 640, 480
CAPTION_H, TITLE_H, GAP = 26, 52, 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(width, height, text, scale=0.45, bg=(38, 38, 38), fg=(235, 235, 235)):
    bar = np.full((height, width, 3), bg, np.uint8)
    cv2.putText(bar, text, (8, height - 9), FONT, scale, fg, 1, cv2.LINE_AA)
    return bar


def project_eef(sim_name, eef_xyz):
    """Where the calibration says the gripper origin lands, in pixels.

    Drawn on BOTH columns: the real frame and the sim render are different
    images of the same claim, so a cross that sits on the gripper in one and
    misses in the other localises the disagreement to that camera.
    """
    from usd_extrinsics import EXTRINSICS
    from usd_intrinsics import INTRINSICS

    if sim_name not in EXTRINSICS or sim_name not in INTRINSICS:
        return None
    extr, intr = EXTRINSICS[sim_name], INTRINSICS[sim_name]
    p_cam = np.linalg.inv(extr.cam_in_ref) @ np.array([*eef_xyz, 1.0])
    if p_cam[2] <= 0:  # behind the camera
        return None
    return (intr.fx * p_cam[0] / p_cam[2] + intr.cx, intr.fy * p_cam[1] / p_cam[2] + intr.cy)


def draw_cross(img, uv, colour=(255, 210, 60)):
    if uv is None:
        return img
    out, (u, v) = img.copy(), (round(uv[0]), round(uv[1]))
    cv2.line(out, (u - 18, v), (u + 18, v), colour, 1, cv2.LINE_AA)
    cv2.line(out, (u, v - 18), (u, v + 18), colour, 1, cv2.LINE_AA)
    cv2.circle(out, (u, v), 4, colour, 1, cv2.LINE_AA)
    return out


def load(path, row_h):
    """Read a tile scaled to *row_h*, preserving its own aspect ratio.

    The rig runs the two third-person cameras in DIFFERENT modes (1280x800 vs
    1920x1080), so a fixed tile size would letterbox one and squash the other.
    Returns the scale factor too, so pixel overlays computed in the camera's
    native frame can follow.
    """
    img = cv2.imread(path) if path else None
    if img is None:
        img = np.full((row_h, int(row_h * 4 / 3), 3), 24, np.uint8)
        cv2.putText(img, "missing:", (16, row_h // 2), FONT, 0.7, (90, 90, 200), 2, cv2.LINE_AA)
        cv2.putText(img, os.path.basename(path or "?"), (16, row_h // 2 + 30), FONT, 0.6, (90, 90, 200), 1)
        return img, False, 1.0
    h, w = img.shape[:2]
    s = row_h / float(h)
    if s != 1.0:
        img = cv2.resize(img, (round(w * s), row_h), interpolation=cv2.INTER_AREA)
    return img, True, s


def pull(live_dir, remote_host):
    """Snapshot both third-person cameras AND the joint state, in one ssh round trip.

    The state is read before and after the frames so a moving arm is detectable;
    a pose sheet built from frames taken while the arm was in motion is a lie.
    """
    os.makedirs(live_dir, exist_ok=True)
    fetch = " ".join(
        f"curl -s --max-time 15 -o /tmp/pose_snap/{short}.jpg "
        f'"http://127.0.0.1:9527/cameras/observation.image.{short}/snapshot";'
        for _, short, _ in PAIRS
    )
    remote = (
        "mkdir -p /tmp/pose_snap && "
        "curl -s --max-time 10 http://127.0.0.1:9527/robot/live-state > /tmp/pose_snap/state_before.json; "
        f"{fetch} "
        "curl -s --max-time 10 http://127.0.0.1:9527/robot/live-state > /tmp/pose_snap/state_after.json"
    )
    subprocess.run(["ssh", "-o", "BatchMode=yes", remote_host, remote], check=True, timeout=180)
    subprocess.run(
        ["scp", "-o", "BatchMode=yes", f"{remote_host}:/tmp/pose_snap/*", f"{live_dir}/"],
        check=True,
        timeout=180,
    )

    with open(os.path.join(live_dir, "state_before.json")) as fh:
        before = json.load(fh)
    with open(os.path.join(live_dir, "state_after.json")) as fh:
        after = json.load(fh)
    # The cameras do not need the robot: an E-stop drops /robot/live-state to a
    # 409 body while the arm stays exactly where its brakes caught it. Keep the
    # frames, drop the pose caption, and say which one this is.
    if "joint_position" not in before:
        print(f"  ! robot not connected ({before.get('detail', before)}) -- frames kept, pose unknown")
        return None
    drift = np.abs(np.array(before["joint_position"][:7]) - np.array(after["joint_position"][:7])).max()
    print(
        f"  pose drift during capture: {np.degrees(drift):.4f} deg" + ("  <-- ARM WAS MOVING" if drift > 1e-3 else "")
    )
    deg = [round(v, 2) for v in np.degrees(before["joint_position"][:7])]
    finger = before["gripper_m"] / 2.0
    print("\n  render the sim at this pose:\n")
    print(
        f"    python scripts/sim2real/build_table_scene.py --render --only third_view second_third \\\n"
        f"        --joints {' '.join(str(v) for v in deg)} --finger {finger:.4f}\n"
    )
    return before


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default=os.path.expanduser("~/Downloads/nuc2_pose_match"))
    ap.add_argument("--render-dir", default=os.path.expanduser("~/Downloads/sim2real_renders"))
    ap.add_argument(
        "--remote-host",
        default=os.environ.get("LABUTOPIA_CAPTURE_HOST"),
        help="SSH host used by --pull (or set LABUTOPIA_CAPTURE_HOST)",
    )
    ap.add_argument("--pull", action="store_true", help="re-grab frames + joint state off nuc2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--row-height", type=int, default=800, help="tile height in the sheet; 0 keeps native")
    # the cross is projected with the intrinsics of the mode the rig is IN
    ap.add_argument("--hires", action="store_true", help="project using the high-resolution intrinsics")
    args = ap.parse_args()
    if args.pull and not args.remote_host:
        ap.error("--pull requires --remote-host or LABUTOPIA_CAPTURE_HOST")
    if args.hires:
        from usd_intrinsics import use_hires

        use_hires()

    state = pull(args.live_dir, args.remote_host) if args.pull else None
    if state is None:
        path = os.path.join(args.live_dir, "state_before.json")
        if os.path.exists(path):
            with open(path) as fh:
                state = json.load(fh)
    if state is not None and "joint_position" not in state:
        state = None  # a 409 error body, not a pose

    if state is not None:
        deg = np.degrees(state["joint_position"][:7])
        pose_txt = (
            "q=[" + ", ".join(f"{v:.2f}" for v in deg) + "] deg   gripper=" + f"{state['gripper_m'] * 1000:.1f} mm"
        )
        eef = state.get("eef_pose")
        if eef:
            pose_txt += f"   EEF xyz=({eef[0]:.3f}, {eef[1]:.3f}, {eef[2]:.3f})"
    elif os.path.exists(os.path.join(args.live_dir, "state_before.json")):
        # frames pulled fine, the ROBOT was unreadable (E-stop / not connected)
        pose_txt = "pose NOT read back -- robot offline when these frames were taken"
    else:
        pose_txt = "pose unknown (no state json; run with --pull)"

    bands, missing, stale = [], [], []
    for sim_name, short, model in PAIRS:
        real_path = os.path.join(args.live_dir, short + ".jpg")
        sim_path = os.path.join(args.render_dir, sim_name + ".png")
        # A sim render older than the real frame was made at a DIFFERENT pose --- the
        # sheet would look fine and mean nothing. Loud, because it is not visible.
        if os.path.exists(sim_path) and os.path.exists(real_path):
            age_h = (os.path.getmtime(real_path) - os.path.getmtime(sim_path)) / 3600.0
            if age_h > 0:
                stale.append(f"{sim_name}.png predates the real frame by {age_h:.1f} h")
        cols = [
            (real_path, f"REAL  {model}  [{short}]"),
            (sim_path, f"SIM   calibrated extrinsics  [{sim_name}]"),
        ]
        uv = project_eef(sim_name, state["eef_pose"][:3]) if state and state.get("eef_pose") else None
        if uv:
            print(f"  {sim_name:28s} gripper origin projects to ({uv[0]:.0f}, {uv[1]:.0f}) px")
        tiles, caps_txt = [], []
        for path, cap in cols:
            tile, ok, s = load(path, args.row_height)
            if not ok:
                missing.append(path)
            elif uv:
                tile = draw_cross(tile, (uv[0] * s, uv[1] * s))
            tiles.append(tile)
            caps_txt.append(f"{cap}  [{tile.shape[1]}x{tile.shape[0]} shown]")
        bw = sum(t.shape[1] for t in tiles) + (len(tiles) + 1) * GAP
        caps = np.full((CAPTION_H, bw, 3), 38, np.uint8)
        band = np.full((args.row_height, bw, 3), 38, np.uint8)
        x0 = GAP
        for tile, cap in zip(tiles, caps_txt):
            band[:, x0 : x0 + tile.shape[1]] = tile
            caps[:, x0 - GAP : x0 + tile.shape[1]] = label(tile.shape[1] + GAP, CAPTION_H, cap)
            x0 += tile.shape[1] + GAP
        bands += [caps, band, np.full((GAP, bw, 3), 20, np.uint8)]

    width = max(b.shape[1] for b in bands)
    rows = [
        label(
            width,
            TITLE_H,
            "REAL rig  vs  SIM at the SAME joint pose   |   cross = gripper origin per the calibration",
            scale=0.6,
            bg=(20, 20, 20),
        ),
        label(width, CAPTION_H, pose_txt, scale=0.45, bg=(20, 20, 20), fg=(120, 220, 255)),
    ]
    # rows come from cameras with different aspect ratios, so pad to a common width
    for b in bands:
        if b.shape[1] < width:
            b = np.hstack([b, np.full((b.shape[0], width - b.shape[1], 3), 20, np.uint8)])
        rows.append(b)

    out_path = args.out or os.path.join(args.render_dir, "pose_match_real_vs_sim.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.vstack(rows))
    print(f"WROTE {out_path}")
    for m in missing:
        print(f"  ! missing tile: {m}")
    for s in stale:
        print(f"  ! STALE, re-render before trusting this sheet: {s}")
    if args.open:
        subprocess.run(["xdg-open", out_path], check=False)
    return 1 if (missing or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
