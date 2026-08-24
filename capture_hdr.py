"""
Standalone HDR panorama capture script for Isaac Sim 4.x / 5.x.

Usage:
    # headless, default settings
    ./python.sh capture_hdr.py --usd /path/to/scene.usd

    # GUI, custom position and output
    ./python.sh capture_hdr.py --usd /path/to/scene.usd \
        --out ./hdr_output --pos 0 0 1.2 \
        --face-size 2048 --mode PathTracing

    # multiple capture positions in one run
    ./python.sh capture_hdr.py --usd /path/to/scene.usd \
        --pos 0 0 1.5   --pos 1.0 0 1.5   --pos -1.0 0 1.5
"""

import argparse
import os
import sys

# ── CLI arguments (must be parsed BEFORE SimulationApp is created) ──────────


def _parse_args():
    p = argparse.ArgumentParser(description="Capture equirectangular HDR panoramas from an Isaac Sim scene")
    p.add_argument("--usd", required=True, help="Path to the USD scene file to load")
    p.add_argument("--out", default="./hdr_output", help="Output directory for .exr files (default: ./hdr_output)")
    p.add_argument(
        "--pos",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help=("Capture position in world space. Repeat to capture from multiple positions. Default: 0 0 1.5"),
    )
    p.add_argument(
        "--face-size",
        type=int,
        default=1024,
        help="Resolution of each cube face in pixels (default: 1024). Output EXR will be (4N x 2N).",
    )
    p.add_argument(
        "--mode",
        default="RaytracedLighting",
        choices=["RaytracedLighting", "PathTracing"],
        help=(
            "RTX render mode. "
            "RaytracedLighting = RTX-Interactive (fast), "
            "PathTracing = RTX-Accurate (high quality). "
            "Default: RaytracedLighting"
        ),
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=60,
        help="Number of simulation_app.update() frames before capture for scene loading (default: 60)",
    )
    p.add_argument("--headless", action="store_true", help="Run without GUI (default: show window)")
    parsed = p.parse_args()
    if parsed.face_size <= 0:
        p.error("--face-size must be positive")
    if parsed.warmup < 0:
        p.error("--warmup must be non-negative")
    return parsed


args = _parse_args()

# ── 1. SimulationApp — must be first, supports both 4.x and 5.x ─────────────

try:
    from isaacsim import SimulationApp  # Isaac Sim 5.x
except ImportError:
    from omni.isaac.kit import SimulationApp  # Isaac Sim 4.x

simulation_app = SimulationApp({"headless": args.headless})

# ── 2. Make project utilities importable ────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 3. All other imports (after SimulationApp) ───────────────────────────────

import carb  # noqa: E402
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: E402

from utils.hdr_capture_utils import capture_hdr_panorama  # noqa: E402

# ── 4. RTX render mode ───────────────────────────────────────────────────────


def _set_rtx_mode(mode: str) -> None:
    """Switch the renderer to the requested RTX mode."""
    carb.settings.get_settings().set("/rtx/rendermode", mode)
    print(f"[capture_hdr] Render mode set to: {mode}")


# ── 5. Main capture routine ──────────────────────────────────────────────────


def main() -> int:
    # Resolve positions (default to a single centre-room position)
    positions = args.pos if args.pos else [[0.0, 0.0, 1.5]]
    face_size = args.face_size
    output_dir = os.path.abspath(args.out)

    print(f"[capture_hdr] USD        : {args.usd}")
    print(f"[capture_hdr] Output dir : {output_dir}")
    print(f"[capture_hdr] Face size  : {face_size}  → EXR {face_size * 4}x{face_size * 2}")
    print(f"[capture_hdr] Positions  : {positions}")
    print(f"[capture_hdr] Render mode: {args.mode}")
    print(f"[capture_hdr] Warmup     : {args.warmup} frames")

    # Switch to RTX before loading the scene so the renderer initialises correctly
    _set_rtx_mode(args.mode)

    # Load the USD scene
    usd_path = os.path.abspath(args.usd)
    add_reference_to_stage(usd_path=usd_path, prim_path="/World")
    print(f"[capture_hdr] Scene loaded: {usd_path}")

    # Warm-up: let the renderer finish loading textures and geometry
    print(f"[capture_hdr] Warming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        simulation_app.update()

    # Capture one panorama per requested position
    saved = []
    for i, pos in enumerate(positions):
        camera_prim = f"/World/HDRCaptureCam_{i}"
        print(f"[capture_hdr] Capturing position {i + 1}/{len(positions)}: {pos}")

        exr_path = capture_hdr_panorama(
            output_dir=output_dir,
            position=tuple(pos),
            face_size=face_size,
            camera_prim_base=camera_prim,
            cleanup_cameras=True,
            render_warmup_frames=args.warmup,
        )

        if exr_path:
            saved.append(exr_path)
            print(f"[capture_hdr]   Saved → {exr_path}")
        else:
            print(f"[capture_hdr]   WARNING: capture at position {pos} produced no file")

    # Summary
    print(f"\n[capture_hdr] Done. {len(saved)}/{len(positions)} file(s) saved:")
    for p in saved:
        print(f"  {p}")

    return 0 if len(saved) == len(positions) else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
