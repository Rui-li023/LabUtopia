"""Gravity-only fit probe; no robot, attachment, or task-success override.

Run with the Isaac Sim Python interpreter from the repository root.
"""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--asset", default="assets/chemistry_lab/lab_002/realistic_pipette/lab_002_pipette.usda")
parser.add_argument("--depths", nargs="+", type=float, default=[-0.033, -0.034, -0.035, -0.036])
parser.add_argument("--yaws-deg", nargs="+", type=float, default=[0.0])
parser.add_argument("--physics-hz", type=int, choices=(120, 240, 480), default=120)
parser.add_argument("--slot-index", type=int, choices=range(4))
parser.add_argument("--release-clearance", type=float, default=0.030)
parser.add_argument("--rack-friction", type=float, nargs=2, metavar=("STATIC", "DYNAMIC"))
parser.add_argument("--report", type=Path)
options = parser.parse_args()
if options.report is not None and options.report.exists():
    parser.error("Report already exists; use a new diagnostic output path")
if any(not -20 <= yaw <= 20 for yaw in options.yaws_deg) or not 0 <= options.release_clearance <= 0.030:
    parser.error("Heading fit probe requires yaw in [-20,20] deg and release clearance in [0,30] mm")
app = SimulationApp({"headless": True})

# Isaac extensions must be imported after SimulationApp starts.
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from loguru import logger  # noqa: E402
from omni.physx import get_physx_cooking_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdShade, UsdUtils  # noqa: E402

from utils.pipette_rack_material import configure_acrylic_rack_contact  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1 / options.physics_hz)
world.scene.add_default_ground_plane(z_position=-0.01)
stage = world.stage
asset = str(Path(options.asset).resolve())
source_stage = Usd.Stage.Open(asset)
seat_name = "PipetteSeat" if options.slot_index is None else f"Seat{options.slot_index}"
seat_local = (
    UsdGeom.Xformable(source_stage.GetPrimAtPath("/World/pipette_rack/" + seat_name))
    .GetLocalTransformation()
    .ExtractTranslation()
)
reference_prim = source_stage.GetPrimAtPath("/World/pipette_01/Reference")
reference_local = np.asarray(
    UsdGeom.Xformable(reference_prim).GetLocalTransformation().ExtractTranslation()
    if reference_prim.IsValid()
    else [0, 0, 0.110]
)
# Preserve original material references on all rack pieces, including feet.
stage.DefinePrim("/World/RealisticPipetteMaterials", "Scope").GetReferences().AddReference(
    asset, "/World/RealisticPipetteMaterials"
)
contact_material = UsdShade.Material.Define(stage, "/World/PipetteContact")
contact_material.GetPrim().GetReferences().AddReference(asset, "/World/RealisticPipetteMaterials/PipetteContact")
cases = []
for i, (x, yaw) in enumerate(product(options.depths, options.yaws_deg)):
    shift = np.array([(i % 4) * 0.5, (i // 4) * 0.5, 0])
    angle = np.radians(yaw)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    # Rotate the real reference offset; keep the SAME intended reference point.
    origin = (
        shift + np.array([x, seat_local[1], seat_local[2] + options.release_clearance]) - rotation @ reference_local
    )
    for name, source, position, angle in (
        (f"Rack{i}", "pipette_rack", shift, 0),
        (f"Pipette{i}", "pipette_01", origin, yaw),
    ):
        prim = stage.DefinePrim("/World/" + name, "Xform")
        prim.GetReferences().AddReference(asset, "/World/" + source)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, angle))
    cases.append((i, x, yaw, shift))
    if options.rack_friction is not None:
        configure_acrylic_rack_contact(
            stage,
            f"/World/Rack{i}",
            static_friction=options.rack_friction[0],
            dynamic_friction=options.rack_friction[1],
        )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/World/Pipette{i}/Body")).Bind(
        contact_material, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
world.reset()
body_view = RigidPrim("/World/Pipette0")
body_view.initialize()
logger.info("MASS body={} com={} inertia={}", body_view.get_masses(), body_view.get_coms(), body_view.get_inertias())


def inspect_hulls(result, convexes):
    logger.info("Cooked pipette: result={}, hulls={}", result, len(convexes))
    for height in (0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.215, 0.2245):
        section = []
        for convex in convexes:
            points = np.array([(v.x, v.y, v.z) for v in convex.vertices])
            for polygon in convex.polygons:
                indices = convex.indices[polygon.index_base : polygon.index_base + polygon.num_vertices]
                for a, b in zip(indices, [*indices[1:], indices[0]]):
                    p, q = points[a], points[b]
                    if (p[2] - height) * (q[2] - height) < 0:
                        section.append(p + (height - p[2]) / (q[2] - p[2]) * (q - p))
        if section:
            vertices = np.asarray(section)
            logger.info("Cooked section z={}: min={}, max={}", height, vertices.min(0), vertices.max(0))


get_physx_cooking_interface().request_convex_collision_representation(
    stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
    collision_prim_id=PhysicsSchemaTools.sdfPathToInt("/World/Pipette0/Body"),
    run_asynchronously=False,
    on_result=inspect_hulls,
)
before = {}
results = []
for step in range(6 * options.physics_hz):
    world.step(render=False)
    if step in (5 * options.physics_hz - 1, 6 * options.physics_hz - 1):
        for i, x, yaw, shift in cases:
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/Pipette{i}"))
            pos = np.array(matrix.ExtractTranslation()) - shift
            axis = np.array(matrix.TransformDir(Gf.Vec3d(0, 0, 1)))
            if step == 5 * options.physics_hz - 1:
                before[i] = pos
            else:
                marker = stage.GetPrimAtPath(f"/World/Pipette{i}/Reference")
                reference = (
                    np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(marker).ExtractTranslation()) - shift
                    if marker.IsValid()
                    else np.asarray(matrix.Transform(Gf.Vec3d(0, 0, 0.110))) - shift
                )
                target = (
                    np.asarray(
                        UsdGeom.XformCache()
                        .GetLocalToWorldTransform(stage.GetPrimAtPath(f"/World/Rack{i}/" + seat_name))
                        .ExtractTranslation()
                    )
                    - shift
                )
                logger.info(
                    "FIT x={} yaw={} origin={} reference={} axis={} motion_1s={} seat_error={}",
                    x,
                    yaw,
                    pos,
                    reference,
                    axis,
                    np.linalg.norm(pos - before[i]),
                    reference - target,
                )
                results.append(
                    {
                        "x": x,
                        "initial_yaw_deg": yaw,
                        "origin": pos.tolist(),
                        "reference": reference.tolist(),
                        "axis": axis.tolist(),
                        "motion_1s_m": float(np.linalg.norm(pos - before[i])),
                        "seat_error_m": (reference - target).tolist(),
                    }
                )
if options.report is not None:
    options.report.parent.mkdir(parents=True, exist_ok=True)
    with options.report.open("x") as report:
        json.dump(
            {
                "diagnostic_only": True,
                "robot_grasp_or_insertion_tested": False,
                "poses_authored_only_before_simulation": True,
                "seconds": 6,
                "physics_hz": options.physics_hz,
                "rack_friction": options.rack_friction,
                "release_clearance": options.release_clearance,
                "slot_index": options.slot_index,
                "asset": asset,
                "asset_sha256": hashlib.sha256(Path(asset).read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "results": results,
            },
            report,
            indent=2,
            allow_nan=False,
        )
app.close()
