"""Gravity-only original-mesh seating comparison, NOT a controller success test.

All case poses are authored before starting physics. No attachments, repeated
teleports, rotation locks, added surfaces, or mass/friction changes are allowed.
"""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--asset", type=Path, default=Path("assets/chemistry_lab/lab_002/lab_002.usd"))
parser.add_argument(
    "--profiles",
    nargs="+",
    choices=("baseline", "support_mesh", "refined_hulls"),
    default=["baseline", "support_mesh", "refined_hulls"],
)
parser.add_argument("--offsets-mm", nargs="+", type=float, default=[-3, 0, 3])
parser.add_argument("--physics-hz", type=int, choices=(60, 120, 240), default=60)
parser.add_argument("--seconds", type=int, default=8)
args = parser.parse_args()
if args.out.exists() or args.seconds < 3 or any(abs(x) > 10 for x in args.offsets_mm):
    parser.error("Use a new output, >=3 seconds and bounded +/-10 mm offsets")
app = SimulationApp({"headless": True})

# Isaac SDK must load after SimulationApp.
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from experiments.support.flask_support_collision import configure_original_flask_collisions  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
asset = (ROOT / args.asset).resolve()
world = World(stage_units_in_meters=1.0, physics_dt=1 / args.physics_hz, rendering_dt=1 / args.physics_hz)
world.scene.add_default_ground_plane(z_position=0)
stage = world.stage


def bounds(path):
    box = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array([box.GetMin(), box.GetMax()])


def place_geometry_center(path, center):
    prim = stage.GetPrimAtPath(path)
    before = bounds(path).mean(0)
    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
    if op.GetOpType() != UsdGeom.XformOp.TypeTranslate:
        raise ValueError("Original root must start with a translation op")
    op.Set(Gf.Vec3d(*(np.asarray(op.Get()) + np.asarray(center) - before)))


cases = []
for index, (support, profile, x, y) in enumerate(
    product(("pipeclay_triangle", "cork_ring"), args.profiles, args.offsets_mm, args.offsets_mm)
):
    shift = np.array([(index % 9) * 0.4, (index // 9) * 0.4, 0])
    support_path, flask_path = f"/World/Case{index}/Support", f"/World/Case{index}/Flask"
    for path, source in ((support_path, support), (flask_path, "round_bottom_flask")):
        stage.DefinePrim(path, "Xform").GetReferences().AddReference(str(asset), "/World/" + source)
    height = np.ptp(bounds(support_path), axis=0)[2]
    place_geometry_center(support_path, shift + np.array([0, 0, height / 2]))
    mesh = stage.GetPrimAtPath(support_path + "/mesh_000")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh)
    vertices = np.array([transform.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
    top = vertices[vertices[:, 2] > vertices[:, 2].max() - 0.001]
    # The stand's foot bounding box is asymmetric. Use the actual upper ring's
    # XY bounds for the release axis, not the whole object's box center.
    center = (top.min(0) + top.max(0)) / 2
    flask_height = np.ptp(bounds(flask_path), axis=0)[2]
    release = np.array([center[0] + x / 1000, center[1] + y / 1000, height + flask_height / 2 + 0.010])
    place_geometry_center(flask_path, release)
    configure_original_flask_collisions(stage, flask_path=flask_path, support_path=support_path, profile=profile)
    cases.append(
        {
            "index": index,
            "support": support,
            "profile": profile,
            "offset_mm": [x, y],
            "shift": shift,
            "flask": flask_path,
            "release": release,
            "axis_xy": center[:2],
        }
    )

world.reset()
view = RigidPrim("/World/Case.*/Flask")
view.initialize()
mass = view.get_masses().reshape(-1).tolist()
history = {c["index"]: [] for c in cases}
for step in range(args.seconds * args.physics_hz):
    world.step(render=False)
    if step >= (args.seconds - 1) * args.physics_hz:
        cache = UsdGeom.XformCache()
        for case in cases:
            matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(case["flask"]))
            up = np.asarray(matrix.TransformDir(Gf.Vec3d(0, 1, 0)))
            tilt = float(np.degrees(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1, 1))))
            history[case["index"]].append([*bounds(case["flask"]).mean(0), tilt])
results = []
for case in cases:
    data = np.asarray(history[case["index"]])
    final = data[-1, :3]
    xy_error = float(np.linalg.norm(final[:2] - case["axis_xy"]))
    motion = float(np.linalg.norm(data[:, :3] - final, axis=1).max())
    # No stand's ground-resting flask can pass this support-height check.
    supported = final[2] > (0.10 if case["support"] == "pipeclay_triangle" else 0.078)
    results.append(
        {
            "support": case["support"],
            "profile": case["profile"],
            "offset_mm": case["offset_mm"],
            "final_center": (final - case["shift"]).tolist(),
            "release_axis_xy": (case["axis_xy"] - case["shift"][:2]).tolist(),
            "tilt_deg": float(data[-1, 3]),
            "motion_last_second_m": motion,
            "xy_error_m": xy_error,
            "stable_supported": bool(supported and xy_error < 0.015 and data[-1, 3] < 25 and motion < 0.001),
        }
    )
report = {
    "diagnostic_only": True,
    "controller_tested": False,
    "asset": str(asset),
    "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
    "physics_hz": args.physics_hz,
    "seconds": args.seconds,
    "actual_flask_masses_kg": mass,
    "results": results,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("x") as output:
    json.dump(report, output, indent=2, allow_nan=False)
for support, profile in product(("pipeclay_triangle", "cork_ring"), args.profiles):
    selected = [r for r in results if r["support"] == support and r["profile"] == profile]
    print(
        "FIT_SUMMARY "
        + json.dumps(
            {
                "support": support,
                "profile": profile,
                "stable": sum(r["stable_supported"] for r in selected),
                "count": len(selected),
                "max_tilt": max(r["tilt_deg"] for r in selected),
                "max_motion": max(r["motion_last_second_m"] for r in selected),
            }
        ),
        flush=True,
    )
app.close()
