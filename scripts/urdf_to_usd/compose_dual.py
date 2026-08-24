"""Compose a dual-arm USD by referencing one arm asset twice at given poses.

NVIDIA publishes single-arm FR3 only, so a "FR3 Duo" has to be assembled. Rather than
duplicating geometry, both arms are USD *references* to the same source asset: the file
stays a few KB and any fix to the source propagates to both arms.

ASSUMED LAYOUT (override with --separation / --yaw-deg / --mount-*): the two arms sit
side by side on a bench, separated along Y, both facing +X. Real bimanual rigs vary --
some angle the arms inward, some mount them facing each other. Adjust to match the
actual hardware before using this for anything but visualisation.

Run with the Isaac Sim interpreter::

    python scripts/urdf_to_usd/compose_dual.py \
        --source assets/robots/official/franka_fr3/FrankaFR3.usd \
        --out assets/robots/fr3_duo.usd --separation 0.9
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--source", required=True, help="Single-arm USD to reference twice")
_parser.add_argument("--out", required=True, help="Output composed USD")
_parser.add_argument("--name", default=None, help="Root prim name (defaults to output stem)")
_parser.add_argument("--separation", type=float, default=0.9, help="Distance between arm bases (m)")
_parser.add_argument("--yaw-deg", type=float, default=0.0, help="Inward yaw applied to each arm (deg)")
_parser.add_argument("--mount-height", type=float, default=0.0, help="Base height for both arms (m)")
_args = _parser.parse_args()

_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom  # noqa: E402


def say(message: str) -> None:
    print(f"[compose] {message}", file=sys.stderr, flush=True)


def main() -> int:
    source = Path(_args.source).resolve()
    out_path = Path(_args.out).resolve()
    if not source.is_file():
        say(f"ERROR: source not found: {source}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root_name = _args.name or out_path.stem
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())

    half = _args.separation / 2.0
    for side, y, yaw in (("left", +half, -_args.yaw_deg), ("right", -half, +_args.yaw_deg)):
        # Two prims per arm on purpose: the placement Xform must stay separate from the
        # referenced prim. A reference carries the source's own xformOpOrder, so adding
        # a translate to the same prim raises "xformOp:translate already exists".
        mount = UsdGeom.Xform.Define(stage, f"/{root_name}/{side}_arm")
        mount.AddTranslateOp().Set(Gf.Vec3d(0.0, y, _args.mount_height))
        mount.AddRotateZOp().Set(yaw)

        # Reference rather than copy: one source of truth, and the composed file stays tiny.
        robot = stage.DefinePrim(f"/{root_name}/{side}_arm/robot")
        robot.GetReferences().AddReference(str(source))
        say(f"{side}_arm at y={y:+.3f} yaw={yaw:+.1f} deg")

    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(str(out_path))
    prim_count = sum(1 for _ in Usd.PrimRange.Stage(check, Usd.TraverseInstanceProxies()))
    say(f"OK  {out_path}  default_prim={check.GetDefaultPrim().GetPath()}  prims={prim_count}")
    say(f"    references -> {source}")
    say("    NOTE: layout is assumed (side by side, facing +X). Adjust to the real rig.")
    return 0


if __name__ == "__main__":
    code = main()
    _app.close()
    sys.exit(code)
