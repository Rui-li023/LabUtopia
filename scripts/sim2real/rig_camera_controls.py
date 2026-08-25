"""Read, back up, restore and unify the REAL rig's camera exposure / white balance.

This is the only script in the tree that WRITES to the real data-collection rig
(nuc2, the Franka station). Everything it changes is captured by ``backup`` first
and undone by ``restore``, and the collector persists controls to its own
``data/camera_controls.json``, so a write here survives a restart --- which is
exactly why the backup is not optional.

    python scripts/sim2real/rig_camera_controls.py show
    python scripts/sim2real/rig_camera_controls.py backup
    python scripts/sim2real/rig_camera_controls.py measure         # tabletop patch per camera
    python scripts/sim2real/rig_camera_controls.py unify --wb 4600 --apply
    python scripts/sim2real/rig_camera_controls.py set third_view --exposure 150 --apply
    python scripts/sim2real/rig_camera_controls.py restore --apply  # put it all back

Nothing writes without ``--apply``; the default is a dry run that prints the
request it would send.

WHY UNIFY. The three cameras shipped with three different white balances
(4000 / 3060 / 4600 K) and three exposures, so the same physical tabletop read
BGR (189,223,250) / (198,195,208) / (173,174,172) --- a 31 % luma spread and a
colour cast ranging from strongly warm to neutral. That spread is larger than the
gap between any one of them and the simulator, so there is no single sim tabletop
colour that can match all three. Unify the cameras first, then match the sim to
the one common target.

CAVEAT on the wrist camera: its tabletop patch is partly shadowed by the gripper
itself and it sees a different piece of bench than the third-person pair, so
equalising its patch reading is weaker evidence than for the other two. To do it
properly, park the arm so the wrist camera and third_view share a clean patch of
bench and equalise there.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUC2 = "haomingsong@192.168.31.157"
API = "http://127.0.0.1:9527"
CAMERAS = ("third_view", "second_third_view", "left_wrist_view")
CHANNEL = "observation.image.{}"
BACKUP = os.path.join(REPO, "assets/sim2real/calibration/camera_controls_backup.json")
LIVE_DIR = "/home/ubuntu/Downloads/nuc2_live"

# Clean tabletop patches (x0, y0, x1, y1) used to compare cameras photometrically.
TABLE_PATCH = {
    "third_view": [(30, 380, 160, 465), (250, 395, 420, 465), (520, 150, 620, 300)],
    "second_third_view": [(150, 410, 280, 470), (420, 400, 600, 470), (40, 400, 140, 470)],
    "left_wrist_view": [(350, 40, 600, 180), (380, 200, 600, 320), (420, 330, 600, 420)],
}
CONTROL_KEYS = (
    "exposure",
    "gain",
    "brightness",
    "white_balance",
    "saturation",
    "auto_exposure",
    "auto_white_balance",
    "hue",
    "sharpness",
)


def ssh(script, timeout=180):
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", NUC2, script], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ssh failed ({out.returncode}): {out.stderr.strip()[:400]}")
    return out.stdout


def get_controls():
    script = (
        'python3 -c "import json,urllib.request;'
        f"print(json.dumps({{n: json.load(urllib.request.urlopen('{API}/cameras/{CHANNEL.format('%s')}'.replace('%s',n)+'/controls',timeout=10)) for n in {list(CAMERAS)}}}))\""
    )
    return json.loads(ssh(script))


def post_controls(camera, payload):
    body = shlex.quote(json.dumps(payload))
    url = f"{API}/cameras/{CHANNEL.format(camera)}/controls"
    return ssh(f"curl -s --max-time 20 -X POST -H 'Content-Type: application/json' -d {body} {shlex.quote(url)}")


def snapshot(dest=LIVE_DIR):
    os.makedirs(dest, exist_ok=True)
    fetch = " ".join(
        f"curl -s --max-time 15 -o /tmp/live_snap/{c}.jpg {shlex.quote(f'{API}/cameras/{CHANNEL.format(c)}/snapshot')};"
        for c in CAMERAS
    )
    ssh(f"mkdir -p /tmp/live_snap && rm -f /tmp/live_snap/*.jpg; {fetch} true")
    subprocess.run(f"scp -o BatchMode=yes {NUC2}:'/tmp/live_snap/*.jpg' {dest}/", shell=True, timeout=180, check=True)
    return dest


def measure(src=LIVE_DIR):
    """Median BGR of the tabletop patches, per camera."""
    import cv2

    rows = {}
    for cam, boxes in TABLE_PATCH.items():
        img = cv2.imread(os.path.join(src, cam + ".jpg"))
        if img is None:
            rows[cam] = None
            continue
        px = np.vstack([img[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in boxes]).astype(float)
        med = np.median(px, 0)
        rows[cam] = {
            "bgr": med.tolist(),
            "luma": float(0.114 * med[0] + 0.587 * med[1] + 0.299 * med[2]),
            "r_over_g": float(med[2] / med[1]),
            "b_over_g": float(med[0] / med[1]),
            "clipped_pct": float(100.0 * (px.max(1) >= 254).mean()),
        }
    return rows


def print_measure(rows, note=""):
    print(f"  {'camera':20s} {'B':>6s}{'G':>6s}{'R':>6s} {'luma':>7s} {'R/G':>7s} {'B/G':>7s} {'clip%':>7s}  {note}")
    for cam, r in rows.items():
        if r is None:
            print(f"  {cam:20s} (no frame)")
            continue
        b, g, x = r["bgr"]
        print(
            f"  {cam:20s} {b:6.1f}{g:6.1f}{x:6.1f} {r['luma']:7.1f} "
            f"{r['r_over_g']:7.3f} {r['b_over_g']:7.3f} {r['clipped_pct']:7.2f}"
        )
    lumas = [r["luma"] for r in rows.values() if r]
    if len(lumas) > 1:
        print(f"  {'':20s} luma spread {max(lumas) - min(lumas):.1f} ({100 * (max(lumas) / min(lumas) - 1):.1f} %)")


def print_controls(state, title):
    print(f"\n{title}")
    for cam, v in state.items():
        vals = v.get("values", v)
        print(f"  {cam:20s} " + "  ".join(f"{k}={vals.get(k)}" for k in CONTROL_KEYS))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    p_backup = sub.add_parser("backup")
    p_backup.add_argument("--out", default=BACKUP)
    p_restore = sub.add_parser("restore")
    p_restore.add_argument("--source", default=BACKUP)
    p_restore.add_argument("--apply", action="store_true")
    p_meas = sub.add_parser("measure")
    p_meas.add_argument("--no-snapshot", action="store_true", help="measure frames already on disk")
    p_uni = sub.add_parser("unify")
    p_uni.add_argument("--wb", type=int, default=4600, help="common colour temperature, K")
    p_uni.add_argument("--apply", action="store_true")
    p_set = sub.add_parser("set")
    p_set.add_argument("camera", choices=CAMERAS)
    for key in ("exposure", "gain", "brightness", "white_balance", "hue", "saturation"):
        p_set.add_argument(f"--{key.replace('_', '-')}", type=int, default=None)
    p_set.add_argument("--lock-awb", action="store_true")
    p_set.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.cmd == "show":
        print_controls(get_controls(), "current rig camera controls")
        return 0

    if args.cmd == "backup":
        state = get_controls()
        saved = ssh("cat /home/haomingsong/dc/collector/data/camera_controls.json")
        blob = {"live": state, "saved_camera_controls_json": json.loads(saved)}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=2)
        print_controls(state, f"backed up to {args.out}")
        return 0

    if args.cmd == "measure":
        if not args.no_snapshot:
            snapshot()
        print_measure(measure())
        return 0

    if args.cmd == "restore":
        with open(args.source) as fh:
            blob = json.load(fh)
        for cam, v in blob["live"].items():
            vals = v["values"]
            payload = {k: vals[k] for k in CONTROL_KEYS if k in vals}
            # auto_* must be restored last or it would clobber the manual values
            print(f"  {cam:20s} <- {payload}")
            if args.apply:
                print("      " + post_controls(cam, payload).strip()[:200])
        print("\nrestored" if args.apply else "\nDRY RUN - pass --apply to write")
        return 0

    if args.cmd == "unify":
        state = get_controls()
        print_controls(state, "before")
        for cam in CAMERAS:
            payload = {"white_balance": args.wb, "auto_white_balance": False, "auto_exposure": False}
            print(f"  {cam:20s} <- {payload}")
            if args.apply:
                print("      " + post_controls(cam, payload).strip()[:200])
        if args.apply:
            print_controls(get_controls(), "after")
        else:
            print("\nDRY RUN - pass --apply to write")
        return 0

    if args.cmd == "set":
        payload = {
            k: getattr(args, k) for k in ("exposure", "gain", "brightness", "white_balance", "hue", "saturation")
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if args.lock_awb:
            payload["auto_white_balance"] = False
        if not payload:
            print("nothing to set")
            return 1
        print(f"  {args.camera:20s} <- {payload}")
        if args.apply:
            print("      " + post_controls(args.camera, payload).strip()[:300])
        else:
            print("\nDRY RUN - pass --apply to write")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
