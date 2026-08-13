"""Build the sim2real bench cell: plywood bench + dark backdrop + two Frankas.

Geometry mirrors the real data-collection rig captured in
``/data1/DataCollectionSystemV2/data/snapshots/``. Measured values are marked
MEASURED; everything else is a placeholder until the numbers land — change the
constant and re-run, the whole stage is regenerated.

    python scripts/sim2real/build_table_scene.py             # write the USD
    python scripts/sim2real/build_table_scene.py --render    # + preview PNGs

The tabletop diffuse map is cut from the real photos by
``scripts/sim2real/extract_table_texture.py`` so the benchtop matches the real
cameras rather than a stock wood material.

Frame: Z up, metres, floor at z=0. +y is BEHIND the bench (toward the backdrop),
-y is the front where the third-view camera sits. The origin sits on the left
arm's base.
"""

import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── mounting plates (MEASURED) ───────────────────────────────────────────────
# Steel plate let into the benchtop, flush with the wood, with a groove around it.
PLATE_W, PLATE_D, PLATE_T = 0.40, 0.30, 0.012  # MEASURED 40 x 30 cm
PLATE_GAP = 0.40  # MEASURED edge-to-edge gap between the two plates
PLATE_GROOVE = 0.008  # visible gap between plate and wood
PLATE_Y = 0.20  # plate centre in y (origin choice, see below)
# The robot base centre sits on the plate's front edge.
BASE_BEHIND_PLATE_FRONT = 0.00
BASE_Y = PLATE_Y - PLATE_D / 2.0 + BASE_BEHIND_PLATE_FRONT

# Left arm on the origin, right arm PLATE_W + PLATE_GAP to its right. The bench
# faces -y, so the robots' right-hand side is -x.
ARM_PITCH = PLATE_W + PLATE_GAP  # 0.80 m centre-to-centre
ARMS = [
    # name, plate centre x, wrist camera name
    ("Franka_left", 0.0, "left_wrist_orbbec335"),
    ("Franka_right", -ARM_PITCH, "right_wrist_orbbec335"),
]

# ── bench ────────────────────────────────────────────────────────────────────
# MEASURED layout, both axes derived from the plates so nothing is set twice:
#   long axis (x):  70 + 40 + 40(gap) + 40 + 70 = 260 cm
#     the plates' SHORT edges sit 70 cm from the bench's short (end) edges
#   short axis (y): 30(plate) + 100 = 130 cm
#     the plates' LONG edge is flush with the bench's back long edge
PLATE_SHORT_EDGE_TO_BENCH_END = 0.70  # MEASURED
TABLE_D = 1.30  # MEASURED bench 宽 (front-to-back)
TABLE_W = 2 * PLATE_SHORT_EDGE_TO_BENCH_END + 2 * PLATE_W + PLATE_GAP  # -> 2.60 m
TABLE_H = 0.75  # top surface height — NOT measured yet
TOP_T = 0.03
LEG_T = 0.06

TABLE_BACK_Y = PLATE_Y + PLATE_D / 2.0  # plate long edge flush with the back edge
TABLE_FRONT_Y = TABLE_BACK_Y - TABLE_D
TABLE_Y = (TABLE_FRONT_Y + TABLE_BACK_Y) / 2.0
TABLE_CX = -ARM_PITCH / 2.0  # bench centred on the pair of arms

# metres covered by one texture tile, (along-grain, across-grain). Real plywood
# grain runs in long streaks down the sheet, so the tile is stretched on one axis.
TEX_TILE_M = (1.10, 0.42)
# Gain on the benchtop albedo, and the light levels below it, are jointly MEASURED
# against the real rig, 2026-08-08. Both are only meaningful together with the
# rig's camera settings, which had to be unified FIRST: the three real cameras
# shipped with three different white balances and exposures and read the same
# bench at luma 227 / 199 / 173 --- a spread larger than the gap to the simulator,
# so no single sim tabletop could have matched all three.
# scripts/sim2real/rig_camera_controls.py pinned them to a common photometry
# (archived in camera_controls_unified.json); they now agree to 2.5 % and the
# bench reads BGR (158.7, 187.0, 202.7), luma 188.5, R/G 1.084, B/G 0.848.
#
# Matching that takes two knobs:
#   STRIP_INTENSITY  the level        TOP_TINT  the colour ratios
# ORDER MATTERS, they are not independent. The first attempt sat at luma 210,
# where the tonemapper compresses all three channels toward white: a -17 % tint
# change on green moved R/G by only +3.6 %. Set the LEVEL first, then the colour.
# Converged: B +2.3 %, G 0.0 %, R -1.5 %, luma -0.3 % against the real bench.
#
# The residual is a floor, not sloppiness: the three real cameras still differ
# from EACH OTHER in B/G by 10 % (0.807 / 0.853 / 0.886) because each has only a
# colour-temperature knob and no green-magenta axis. One sim bench cannot sit on
# all three, so it is tuned to the middle. Re-run `rig_camera_controls.py measure`
# and redo the two rounds if the rig's exposure or lighting changes.
TOP_TINT = (1.0, 0.762, 0.473)

WALL_Y = TABLE_Y + TABLE_D / 2.0 + 0.15  # dark backdrop behind the bench
WALL_W, WALL_H, WALL_T = 4.00, 2.00, 0.03
FLOOR_SIZE = 6.0
CEILING_Z = 2.45  # white reflector above the bench (below it: the luminaires)

# Franka home pose (matches robots/franka/franka.py DEFAULT_JOINT_POSITIONS)
FRANKA_HOME_DEG = [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0]
FRANKA_FINGER = 0.04
# Base yaw about Z. At 0 the arm reaches along +x; the real cell has both arms
# reaching toward the front of the bench (-y).
FRANKA_YAW_DEG = -90.0

# ── cameras ──────────────────────────────────────────────────────────────────
# Intrinsics are MEASURED off the real rig and live in usd_intrinsics.py, which
# is the single source of truth shared with verify_camera_render.py.
from usd_intrinsics import (  # noqa: E402  (script dir is on sys.path)
    APERTURE,
    INTRINSICS,
    OPENCV_PINHOLE_COEFFS,
    RES_H,
    RES_W,
    Intrinsics,
    describe,
    usd_projection,
)

# Author the principal point BOTH as USD aperture offsets (what a plain USD
# consumer understands) and in the lens-distortion schema (what RTX reads).
# MEASURED 2026-08-07 by scripts/sim2real/verify_camera_render.py: RTX applies
# the principal point exactly ONCE (an on-axis marker lands on (cx, cy) to within
# 0.14-0.62 px on all four cameras), so the two representations do NOT compose
# into a double shift. That same run proved RTX really does render the lens
# distortion in a plain viewport capture: on the third-view camera (the largest
# coefficients) five probe markers matched the OpenCV-distorted prediction to
# 0.10-0.22 px while sitting 1.85-2.14 px away from the pinhole prediction.
PRINCIPAL_POINT_MODE = "both"

# Extrinsics are MEASURED too, by the rig's own hand-eye calibration (archived at
# assets/sim2real/calibration/camera_params.json) and turned into USD transforms
# by usd_extrinsics.py. Nothing about camera placement is eyeballed any more:
# there are no eye/look-at tuples left, only the calibrated 4x4s.
from usd_extrinsics import (  # noqa: E402  (script dir is on sys.path)
    EXTRINSICS,
    GRIPPER_Z_IN_HAND,
    to_gf_rows,
    usd_cam_in_hand,
    usd_cam_in_world,
)
from usd_extrinsics import describe as describe_pose  # noqa: E402

# The two third-person cameras were calibrated in the robot BASE frame, so they
# follow whichever arm CAMERA_TARGET_ARM names; the wrist cameras were calibrated
# in the gripper frame and ride panda_hand.
STATIC_CAMERAS = ["third_view_orbbec335L", "second_third_realsense435i"]


# Official Orbbec models from the Isaac asset library. Both are Y-up with the
# optical axis along local +Z (front_cover / COVER_GLASS sit on the +Z face),
# which is a 180 deg turn about Y from USD's camera convention (-Z forward).
SENSOR_335 = "assets/sim2real/sensors/orbbec_gemini_335.usd"
SENSOR_335L = "assets/sim2real/sensors/orbbec_gemini_335L.usd"
SENSOR_FRONT_OFFSET = 0.012  # pull the body back so its front glass is at the camera origin
SENSOR_FOR_CAMERA = {
    # third-view / second-third bodies intentionally omitted
    "left_wrist_orbbec335": SENSOR_335,
    "right_wrist_orbbec335": SENSOR_335,
}

# ── task object ──────────────────────────────────────────────────────────────
# The Level-1 pick task does not spawn its target; it repositions a prim that
# must already exist in the scene. Reference just that subtree out of the lab
# asset. The prim NAME is load-bearing: BaseTask keys its grasp-height table and
# the language instruction off the last path component.
PICK_OBJECT_PATH = "/World/conical_bottle02"
PICK_OBJECT_SOURCE = "assets/chemistry_lab/lab_001/lab_001.usd"
PICK_OBJECT_START = (0.0, -0.48, TABLE_H + 0.02)  # task re-randomises this per episode
# Squash the flask in z to match the cup on the real rig, which is ~2 cm shorter.
# Measured bbox of the source asset: 92.6 x 92.6 x 164.5 mm, so 0.878 takes 2 cm
# off the height.
#
# Baked here, at build time, so the collider is cooked at the final size.
#
# ★ It is a MULTIPLIER on the existing scale op, never a replacement. That op is
# already 1e-4 — a unit conversion, because the source asset is not authored in
# metres. Writing 0.878 over it blows the flask up 10000x; done at runtime that
# also detonates physics (measured: arm-tracking error 0.067 -> 2.4e6, every
# episode failed). Read the op, multiply, write back.
PICK_OBJECT_Z_SCALE = 0.878
# Its mesh binds /World/Looks/OmniGlass by path; that material lives in the lab
# asset and does NOT travel with a sub-prim reference, so author one here or the
# bottle renders as untextured grey.
GLASS_MATERIAL_PATH = "/World/Looks/OmniGlass"
# Glass look. OmniGlass defaults to thin_walled=FALSE, which refracts through the
# whole mesh volume as if the flask were a solid glass paperweight — that is what
# made the see-through look wrong. A flask is a thin shell.
#   ior 1.47      = borosilicate, real lab glassware (OmniGlass default 1.491)
#   frosting 0.02 = just enough to keep the silhouette readable at the task
#                   camera distance; OmniSurface_Glass looks better in close-up
#                   but goes nearly invisible at 0.9 m, which is bad for a
#                   vision policy that has to see the object.
# Alternatives, one line each (rendered comparison in sim2real_renders/glass/):
#   plain thin glass : GLASS_INPUTS = {"thin_walled": True}
#   OmniSurface glass: GLASS_MDL = ("OmniSurfacePresets.mdl", "OmniSurface_Glass")
GLASS_MDL = ("OmniGlass.mdl", "OmniGlass")
GLASS_INPUTS = {"thin_walled": True, "glass_ior": 1.47, "frosting_roughness": 0.02}

# Which arm the two static cameras look at. The extrinsics are placeholders
# until the real calibration lands, and a camera framed on the left arm cannot
# see a right-arm task at all, so this shifts both by the arm pitch rather than
# inventing extra hardware the real rig does not have.
CAMERA_TARGET_ARM = "left"

TEXTURE = "assets/sim2real/textures/plywood_top_diffuse.png"
FRANKA_USD = "assets/robots/Franka.usd"
OUT_USD = "assets/sim2real/table_cell.usd"

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=os.path.join(REPO, OUT_USD))
parser.add_argument("--render", action="store_true", help="also write preview PNGs")
parser.add_argument("--render-dir", default="/home/ubuntu/Downloads/sim2real_renders")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": args.width, "height": args.height})


import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────
def make_box(stage, path, center, size, uv_tile=None):
    """Axis-aligned box mesh with per-face UVs (so textures tile in metres)."""
    cx, cy, cz = center
    sx, sy, sz = (s / 2.0 for s in size)
    pts = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    faces = [(4, 5, 6, 7), (1, 0, 3, 2), (0, 1, 5, 4), (2, 3, 7, 6), (3, 0, 4, 7), (1, 2, 6, 5)]
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr([Gf.Vec3f(cx - sx, cy - sy, cz - sz), Gf.Vec3f(cx + sx, cy + sy, cz + sz)])

    if uv_tile:
        tu, tv = uv_tile if isinstance(uv_tile, tuple) else (uv_tile, uv_tile)
        st = []
        for f in faces:
            # project each corner on the face's two dominant world axes
            p0, p1, p2 = (pts[f[0]], pts[f[1]], pts[f[3]])
            u_ax = max(range(3), key=lambda i: abs(p1[i] - p0[i]))
            v_ax = max(range(3), key=lambda i: abs(p2[i] - p0[i]))
            for idx in f:
                p = pts[idx]
                st.append(Gf.Vec2f(p[u_ax] / tu, p[v_ax] / tv))
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        pv.Set(st)
    return mesh


def make_material(stage, path, color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, texture=None, tint=None, mdl=None):
    """OmniPBR by default; pass mdl=("OmniGlass.mdl", "OmniGlass") for real glass.

    OmniPBR cannot fake transparency here: its opacity_constant is ignored unless
    enable_opacity is also set, and even then it is a cutout, not refraction.
    """
    mat = UsdShade.Material.Define(stage, Sdf.Path(path))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(path + "/Shader"))
    asset, sub = mdl if mdl else ("OmniPBR.mdl", "OmniPBR")
    sh.SetSourceAsset(Sdf.AssetPath(asset), "mdl")
    sh.SetSourceAssetSubIdentifier(sub, "mdl")
    if mdl:
        # match the lab asset exactly: OmniGlass with every input left at default
        mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return mat
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    if tint:
        # OmniPBR ignores diffuse_color_constant once a diffuse_texture is bound;
        # diffuse_tint is the multiplier that actually reaches the texture.
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def add_collider(prim):
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("boundingCube")


def look_at_xform(eye, target, up=Gf.Vec3d(0, 0, 1)):
    fwd = Gf.Vec3d(*[target[i] - eye[i] for i in range(3)]).GetNormalized()
    if abs(fwd[2]) > 0.985:  # straight-down view is degenerate against +Z up
        up = Gf.Vec3d(0, 1, 0)
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()


def make_camera(stage, path, intr, xform=None, distortion=True):
    """Author a UsdGeom.Camera from calibrated OpenCV intrinsics.

    Two representations are written on purpose:
      * the five standard USD projection attrs — what any USD consumer reads,
        and the only place the exact fy (non-square pixels) survives as geometry;
      * OmniLensDistortionOpenCvPinholeAPI — what the RTX delegate reads. It is
        the only path to distortion; there is no separate "rational" model in
        5.1, opencvPinhole with k1..k6 IS the rational polynomial.
    """
    # RealSense reports inverse_brown_conrady: coefficients that map DISTORTED ->
    # undistorted, the opposite direction from OpenCV's forward model the schema
    # expects. Identical only while they are all zero (a rectified colour stream),
    # which is the case today. Refuse to silently mis-feed a non-zero set.
    if intr.model == "inverse_brown_conrady" and any(c != 0.0 for c in intr.coeffs):
        raise ValueError(
            f"{path}: inverse_brown_conrady coefficients must be inverted to OpenCV forward "
            "form before they can go into OmniLensDistortionOpenCvPinholeAPI"
        )
    a = usd_projection(intr.fx, intr.fy, intr.cx, intr.cy, RES_W, RES_H, APERTURE)
    if distortion and PRINCIPAL_POINT_MODE == "schema":
        a["horizontalApertureOffset"] = 0.0
        a["verticalApertureOffset"] = 0.0

    cam = UsdGeom.Camera.Define(stage, Sdf.Path(path))
    cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    cam.CreateFocalLengthAttr(a["focalLength"])
    cam.CreateHorizontalApertureAttr(a["horizontalAperture"])
    cam.CreateVerticalApertureAttr(a["verticalAperture"])
    cam.CreateHorizontalApertureOffsetAttr(a["horizontalApertureOffset"])
    cam.CreateVerticalApertureOffsetAttr(a["verticalApertureOffset"])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))

    if distortion:
        prim = cam.GetPrim()
        # Codeless schema: applied by STRING name, there is no pxr binding module.
        if not prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI"):
            raise RuntimeError(f"could not apply lens distortion schema on {path}")
        prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
        ns = "omni:lensdistortion:opencvPinhole:"
        # imageSize MUST be authored: the schema default is (2048, 1024), which
        # would reinterpret fx/fy/cx/cy against the wrong frame.
        prim.CreateAttribute(ns + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(RES_W, RES_H))
        for attr_name, value in (("fx", intr.fx), ("fy", intr.fy), ("cx", intr.cx), ("cy", intr.cy)):
            prim.CreateAttribute(ns + attr_name, Sdf.ValueTypeNames.Float).Set(float(value))
        padded = tuple(intr.coeffs) + (0.0,) * (len(OPENCV_PINHOLE_COEFFS) - len(intr.coeffs))
        for attr_name, value in zip(OPENCV_PINHOLE_COEFFS, padded):
            prim.CreateAttribute(ns + attr_name, Sdf.ValueTypeNames.Float).Set(float(value))
        # provenance, so a consumer can tell measured from borrowed
        prim.CreateAttribute("labutopia:calibration:serial", Sdf.ValueTypeNames.String).Set(intr.serial)
        prim.CreateAttribute("labutopia:calibration:vendorModel", Sdf.ValueTypeNames.String).Set(intr.model)
        if intr.borrowed_from:
            prim.CreateAttribute("labutopia:calibration:borrowedFrom", Sdf.ValueTypeNames.String).Set(
                intr.borrowed_from
            )

    if xform is not None:
        cam.MakeMatrixXform().Set(xform)
    return cam


def place_sensor_body(stage, path, usd_rel, cam_xform):
    """Drop a camera body so its front glass sits at the camera origin.

    The reference goes on a child prim: composing it onto the prim that carries
    our transform would inherit the asset's own xformOps and conflict with them.
    """
    holder = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    inner = UsdGeom.Xform.Define(stage, Sdf.Path(path + "/model"))
    inner.GetPrim().GetReferences().AddReference(os.path.join(REPO, usd_rel))
    flip = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 180.0))
    back = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -SENSOR_FRONT_OFFSET))
    holder.MakeMatrixXform().Set(back * flip * cam_xform)
    return holder


def place_franka(stage, path, base_xy):
    """Reference the Franka and drive its existing xform ops (it ships its own)."""
    xf = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    xf.GetPrim().GetReferences().AddReference(os.path.join(REPO, FRANKA_USD))
    x = UsdGeom.Xformable(xf.GetPrim())
    pos = Gf.Vec3d(base_xy[0], base_xy[1], TABLE_H)
    half = math.radians(FRANKA_YAW_DEG) / 2.0
    yw, yz = math.cos(half), math.sin(half)
    got_t = got_r = False
    for op in x.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(pos)
            got_t = True
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            is_f = op.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Quatf
            op.Set(Gf.Quatf(yw, Gf.Vec3f(0, 0, yz)) if is_f else Gf.Quatd(yw, Gf.Vec3d(0, 0, yz)))
            got_r = True
    if not got_t:
        x.AddTranslateOp().Set(pos)
    if not got_r:
        x.AddOrientOp().Set(Gf.Quatf(yw, Gf.Vec3f(0, 0, yz)))
    return xf


def author_home_pose(stage, root):
    """Write the home joint state so the stage opens posed once physics runs."""
    for i, deg in enumerate(FRANKA_HOME_DEG, start=1):
        matches = [
            p for p in stage.Traverse() if p.GetName() == f"panda_joint{i}" and str(p.GetPath()).startswith(root)
        ]
        if not matches:
            print(f"  ! {root}/panda_joint{i} not found")
            continue
        PhysxSchema.JointStateAPI.Apply(matches[0], "angular").CreatePositionAttr(deg)
        UsdPhysics.DriveAPI.Apply(matches[0], "angular").CreateTargetPositionAttr(deg)
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        matches = [p for p in stage.Traverse() if p.GetName() == name and str(p.GetPath()).startswith(root)]
        if matches:
            PhysxSchema.JointStateAPI.Apply(matches[0], "linear").CreatePositionAttr(FRANKA_FINGER)
            UsdPhysics.DriveAPI.Apply(matches[0], "linear").CreateTargetPositionAttr(FRANKA_FINGER)


# ── build ────────────────────────────────────────────────────────────────────
ctx = omni.usd.get_context()
ctx.new_stage()
for _ in range(10):
    sim_app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
stage.SetDefaultPrim(world.GetPrim())
UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))

# The MDL texture loader resolves diffuse_texture against the process CWD, not
# the USD layer, so a relative path breaks when the stage is opened elsewhere.
# Write an absolute path; re-run this builder to re-anchor it after a move.
tex_abs = os.path.join(REPO, TEXTURE)
m_wood = make_material(stage, "/World/Looks/PlywoodTop", (1.0, 1.0, 1.0), 0.45, 0.0, tex_abs, tint=TOP_TINT)
m_frame = make_material(stage, "/World/Looks/TableFrame", (0.55, 0.55, 0.57), 0.55, 0.0)
# Brushed steel, CALIBRATED against the real third-view photo (the real plate
# measures BGR 230,207,194 there -- brighter than the plywood, and blue-dominant
# because it reflects the cool overhead light, i.e. a satin surface, not a mirror).
#
# The knob that matters is ROUGHNESS, not metallic. At 0.45 the plate acted as a
# mirror: from the low front camera the reflection direction points at the dark
# backdrop, so it rendered BGR (84,93,71) -- a navy hole in the bench. Dropping
# metallic would "fix" the brightness by turning the steel into plastic; raising
# roughness fixes it while keeping it metal.
#   sweep at 640x480 through the real camera, masked to the plate's own pixels:
#     rough 0.45 -> ( 84, 93, 71)  dE 146      rough 0.80 -> (209,208,206) dE 21
#     rough 0.75 -> (199,198,195)  dE  31      + cool tint -> (217,212,209) dE 15
# Albedo is RGB here (Gf.Vec3f), so the cool tint is B > R.
# MEASURED 2026-08-08, off the real plate as seen in second_third next to the
# robot base --- the only view where it is not buried under the arm, cables and
# a clamp. Five patches agreed closely (R/G 1.14-1.16, B/G 0.86-0.93):
#     real plate BGR (60, 66, 76), luma 68.6, R/G 1.15, B/G 0.90
# It is a DARK WARM grey, not the near-white this started as (luma 155) nor the
# neutral mid-grey it was guessed at (luma ~100) before the measurement existed.
# Do not try to measure it in third_view: the plate quad projects to
# (380,91) (221,89) (233,14) (372,16) there, but that region is occluded, and
# sampling it once grabbed the cardboard standing in the workspace and turned
# the plate orange.
m_plate = make_material(stage, "/World/Looks/MountPlate", (0.401, 0.373, 0.337), 0.80, 0.90)
m_groove = make_material(stage, "/World/Looks/Groove", (0.05, 0.05, 0.05), 0.95, 0.0)
# backdrop albedo solved from the real third-view frame (dark blue-grey fabric)
m_wall = make_material(stage, "/World/Looks/Backdrop", (0.028, 0.035, 0.080), 0.92, 0.0)
m_floor = make_material(stage, "/World/Looks/Floor", (0.34, 0.34, 0.35), 0.80, 0.0)

floor = make_box(stage, "/World/Floor", (TABLE_CX, TABLE_Y, -0.02), (FLOOR_SIZE, FLOOR_SIZE, 0.04), uv_tile=1.0)

# Black cloth hung in front of the bench. This is NOT a lighting device --- it is a
# clean BACKGROUND, so that when the wrist camera looks forward off the bench it
# sees uniform black instead of whatever lab clutter happens to be there. That is
# the one kind of background a simulator can reproduce exactly, which is the whole
# point: room clutter can never be matched, flat black is free.
#
# Placement is constrained on both sides and is not free to move:
#   y = TABLE_FRONT_Y - 0.10 puts it IN FRONT of the working strip light at
#   y = -0.92, so it does not shadow the bench, and BEHIND both third-person
#   cameras (they sit at y > -0.71 looking toward +y), so it never enters their
#   frames. Moving it back past -0.92 would black out the bench lighting.
CLOTH_Y = TABLE_FRONT_Y - 0.10
CLOTH_H = 1.60  # tall enough to fill a forward-looking wrist view
CLOTH_W = TABLE_W + 0.60  # wider than the bench so the wrist cannot see past its edges
m_cloth = make_material(stage, "/World/Looks/BlackCloth", (0.025, 0.025, 0.027), 0.95, 0.0)
cloth = make_box(stage, "/World/BlackCloth", (TABLE_CX, CLOTH_Y, CLOTH_H / 2.0), (CLOTH_W, 0.02, CLOTH_H))
bind(cloth, m_cloth)
add_collider(cloth)
bind(floor, m_floor)
add_collider(floor)

top = make_box(
    stage,
    "/World/Table/Top",
    (TABLE_CX, TABLE_Y, TABLE_H - TOP_T / 2.0),
    (TABLE_W, TABLE_D, TOP_T),
    uv_tile=TEX_TILE_M,
)
bind(top, m_wood)
add_collider(top)

inset = 0.08
for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
    leg = make_box(
        stage,
        f"/World/Table/Leg_{i}",
        (
            TABLE_CX + sx * (TABLE_W / 2 - inset - LEG_T / 2),
            TABLE_Y + sy * (TABLE_D / 2 - inset - LEG_T / 2),
            (TABLE_H - TOP_T) / 2.0,
        ),
        (LEG_T, LEG_T, TABLE_H - TOP_T),
    )
    bind(leg, m_frame)
    add_collider(leg)

# plates are let into the benchtop: groove ring sits just below the wood, the
# plate itself is flush with it
for name, px, _ in ARMS:
    tag = name.split("_")[-1]
    groove = make_box(
        stage,
        f"/World/Table/Groove_{tag}",
        (px, PLATE_Y, TABLE_H - 0.012),
        (PLATE_W + 2 * PLATE_GROOVE, PLATE_D + 2 * PLATE_GROOVE, 0.022),
    )
    bind(groove, m_groove)
    plate = make_box(
        stage,
        f"/World/Table/Plate_{tag}",
        (px, PLATE_Y, TABLE_H - PLATE_T / 2.0 + 0.0005),
        (PLATE_W, PLATE_D, PLATE_T),
    )
    bind(plate, m_plate)
    add_collider(plate)

# A flat white dome gives metal a flat grey reflection, which reads as plastic.
# A real lab has a bright ceiling directly above the bench; this panel is that
# reflector, and it is what puts a broad highlight on the steel plates.
# Albedo 0.15, NOT the 0.85 it started at. A 3.6 x 2.6 m white plane directly
# over the bench is a giant softbox: it was the single thing flattening the
# front-to-back falloff. MEASURED -- dropping it 0.85 -> 0.15 took the sim
# gradient from 7.6 % to 20.1 % in one step, while cutting the dome 8x had
# done nothing at all. Raise it and the bench goes evenly lit again.
m_ceiling = make_material(stage, "/World/Looks/Ceiling", (0.15, 0.15, 0.15), 0.85, 0.0)
ceiling = make_box(stage, "/World/Ceiling", (TABLE_CX, TABLE_Y, CEILING_Z), (3.60, 2.60, 0.06))
bind(ceiling, m_ceiling)

wall = make_box(stage, "/World/Backdrop", (TABLE_CX, WALL_Y, WALL_H / 2.0), (WALL_W, WALL_T, WALL_H))
bind(wall, m_wall)
add_collider(wall)

# ── robots ───────────────────────────────────────────────────────────────────
arm_roots = []
for name, px, wrist_name in ARMS:
    root = f"/World/{name}"
    place_franka(stage, root, (px, BASE_Y))
    arm_roots.append((root, wrist_name))
for _ in range(30):
    sim_app.update()
for root, _ in arm_roots:
    author_home_pose(stage, root)

# ── task object ──────────────────────────────────────────────────────────────
glass = make_material(stage, GLASS_MATERIAL_PATH, mdl=GLASS_MDL)
_gsh = UsdShade.Shader(stage.GetPrimAtPath(GLASS_MATERIAL_PATH + "/Shader"))
for _k, _v in GLASS_INPUTS.items():
    _t = Sdf.ValueTypeNames.Bool if isinstance(_v, bool) else Sdf.ValueTypeNames.Float
    _gsh.CreateInput(_k, _t).Set(_v)
print(f"  glass = {GLASS_MDL[1]} {GLASS_INPUTS}", flush=True)

bottle = UsdGeom.Xform.Define(stage, Sdf.Path(PICK_OBJECT_PATH))
bottle.GetPrim().GetReferences().AddReference(os.path.join(REPO, PICK_OBJECT_SOURCE), PICK_OBJECT_PATH)
for _ in range(20):
    sim_app.update()
_bx = UsdGeom.Xformable(bottle.GetPrim())
for _op in _bx.GetOrderedXformOps():
    if _op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        _op.Set(Gf.Vec3d(*PICK_OBJECT_START))
        break
else:
    _bx.AddTranslateOp().Set(Gf.Vec3d(*PICK_OBJECT_START))
for _op in _bx.GetOrderedXformOps():
    if _op.GetOpType() == UsdGeom.XformOp.TypeScale:
        _cur = _op.Get()
        # ★ 压的是**局部 y**，不是局部 z：这个 prim 带 xformOp:orient 绕 X 转 90°，
        # 局部 z 映射到世界 -y，世界高度来自局部 y。压局部 z 只会把瓶子压扁成椭圆
        # 截面而高度纹丝不动（实测：世界 z 仍 0.1645，y 从 0.0926 掉到 0.0813）。
        _op.Set(type(_cur)(_cur[0], _cur[1] * PICK_OBJECT_Z_SCALE, _cur[2]))
        print(f"[bottle] world-height scale via local-y {_cur[1]:.3e} -> {_cur[1] * PICK_OBJECT_Z_SCALE:.3e}")
        break
else:
    _bx.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, PICK_OBJECT_Z_SCALE))
# The bottle's mesh binds /World/Looks/OmniGlass by path, but that target is
# OUTSIDE the scope of a sub-prim reference, so USD drops the relationship
# ("refers to a path outside the scope of the reference"). Re-bind it here, in
# this stage, or the bottle renders with the default grey material.
_mesh = stage.GetPrimAtPath(PICK_OBJECT_PATH + "/mesh")
if _mesh.IsValid():
    UsdShade.MaterialBindingAPI.Apply(_mesh).Bind(glass)
    print(f"  re-bound {GLASS_MATERIAL_PATH} onto {_mesh.GetPath()}", flush=True)
else:
    print(f"  ! {PICK_OBJECT_PATH}/mesh not found; glass not bound", flush=True)
print(f"  pick object {PICK_OBJECT_PATH} at {PICK_OBJECT_START}", flush=True)

# ── lights ───────────────────────────────────────────────────────────────────
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome.CreateIntensityAttr(60.0)
dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
# NOTE: an HDRI on this dome (texture:file) killed RTX lighting outright — the
# whole frame rendered black at any intensity. Metal instead gets something to
# reflect from the ceiling panel below.

# Long strip luminaires running along the bench, like a real lab ceiling. Narrow
# sources also read better on the steel plates and the glass: they leave a long
# specular streak instead of a square blob.
STRIP_LEN, STRIP_W = 2.40, 0.12  # m
STRIP_Z = (1.10, 1.40)  # height drives the front-to-back falloff: lower = steeper
# The real bench is NOT lit symmetrically: MEASURED on third_view, its top reads
# +15 % (left band) to +27 % (right band) brighter near the front edge than 0.55 m
# back, on the clean 08-06 bench. The live frame shows +27/+32 % because a
# calibration board standing in the workspace shadows the rear --- that extra
# 5-11 points is scene, not lighting, so the target here is the 15-27 % band.
# Reproducing it needs THREE things, and only the third one mattered much:
#   1. bias the strips forward and dim the rear one (below)
#   2. lower them  (STRIP_Z)
#   3. kill the ceiling bounce (see the Ceiling material -- this was the big one)
# Beyond that the tonemapper shoulder fights back: the SAME geometry measured
# 20.1 % at bench luma 170 and 12.2 % at luma 198, because the shoulder compresses
# the ratio as the level rises. Front strip y first, then rear.
STRIP_Y = (TABLE_FRONT_Y - 0.10, TABLE_Y - 0.62)
STRIP_GAIN = (0.0, 1.0)
# MEASURED against the unified real cameras -- see the TOP_TINT block. Response is
# roughly +52 luma per decade of intensity, so this is a coarse knob: it moved the
# bench 229 -> 210 -> 169 -> 172 over three rounds. dome and fill below are scaled
# with it; changing one alone re-breaks the colour match.
STRIP_INTENSITY = 785000.0
for _i, (_y, _z, _gain) in enumerate(zip(STRIP_Y, STRIP_Z, STRIP_GAIN)):
    strip = UsdLux.RectLight.Define(stage, Sdf.Path(f"/World/Lights/Strip_{_i}"))
    strip.CreateIntensityAttr(STRIP_INTENSITY * _gain)
    strip.CreateWidthAttr(STRIP_LEN)
    strip.CreateHeightAttr(STRIP_W)
    _sx = UsdGeom.Xformable(strip)
    _sx.AddTranslateOp().Set(Gf.Vec3d(TABLE_CX, _y, _z))
    _sx.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))

fill = UsdLux.RectLight.Define(stage, Sdf.Path("/World/Lights/Fill"))
fill.CreateIntensityAttr(600.0)
fill.CreateWidthAttr(1.4)
fill.CreateHeightAttr(1.0)
fx = UsdGeom.Xformable(fill)
fx.AddTranslateOp().Set(Gf.Vec3d(TABLE_CX - 0.9, -1.4, 1.9))
fx.AddRotateXYZOp().Set(Gf.Vec3f(-140.0, 0.0, -35.0))

# ── cameras ──────────────────────────────────────────────────────────────────
UsdGeom.Xform.Define(stage, Sdf.Path("/World/Cameras"))
cam_base_x = -ARM_PITCH if CAMERA_TARGET_ARM == "right" else 0.0
cam_base_xyz = (cam_base_x, BASE_Y, TABLE_H)
print(f"  static cameras from calibration, on the {CAMERA_TARGET_ARM} arm's base at {cam_base_xyz}", flush=True)
for name in STATIC_CAMERAS:
    # calibrated in the robot base frame -> push through the arm's own placement
    world = usd_cam_in_world(name, cam_base_xyz, FRANKA_YAW_DEG)
    make_camera(stage, f"/World/Cameras/{name}", INTRINSICS[name], xform=Gf.Matrix4d(*to_gf_rows(world)))
    eye = world[:3, 3]
    print(f"  {name}: world eye ({eye[0]:+.3f},{eye[1]:+.3f},{eye[2]:+.3f})", flush=True)

wrist_paths = []
for root, wrist_name in arm_roots:
    hand = stage.GetPrimAtPath(f"{root}/panda_hand")
    if not hand.IsValid():
        raise RuntimeError(f"{root}/panda_hand missing: the gripper-frame calibration has nothing to hang off")
    parent = f"{root}/panda_hand"
    # The asset ships the libfranka EE frame as a prim. Tie GRIPPER_Z_IN_HAND to
    # it so the calibration and the robot cannot silently drift apart.
    tool = stage.GetPrimAtPath(f"{parent}/tool_center")
    if not tool.IsValid():
        raise RuntimeError(f"{parent}/tool_center missing: cannot verify GRIPPER_Z_IN_HAND against the asset")
    tool_t = UsdGeom.Xformable(tool).GetLocalTransformation().ExtractTranslation()
    if abs(tool_t[2] - GRIPPER_Z_IN_HAND) > 1e-4 or max(abs(tool_t[0]), abs(tool_t[1])) > 1e-4:
        raise RuntimeError(
            f"{parent}/tool_center is at {tuple(round(v, 6) for v in tool_t)} but GRIPPER_Z_IN_HAND "
            f"is {GRIPPER_Z_IN_HAND}: the wrist calibration would be applied against the wrong frame"
        )
    # calibrated in the gripper frame, which is panda_hand shifted along its
    # approach axis (see GRIPPER_Z_IN_HAND in usd_extrinsics.py)
    local = Gf.Matrix4d(*to_gf_rows(usd_cam_in_hand(wrist_name)))
    make_camera(stage, f"{parent}/{wrist_name}", INTRINSICS[wrist_name], xform=local)
    wrist_paths.append(f"{parent}/{wrist_name}")
    if wrist_name in SENSOR_FOR_CAMERA:
        place_sensor_body(stage, f"{parent}/{wrist_name}_body", SENSOR_FOR_CAMERA[wrist_name], local)

print(
    f"  bench x[{TABLE_CX - TABLE_W / 2:.2f},{TABLE_CX + TABLE_W / 2:.2f}] y[{TABLE_FRONT_Y:.2f},{TABLE_Y + TABLE_D / 2:.2f}]"
)
print(
    f"  plates {PLATE_W}x{PLATE_D} at x={[a[1] for a in ARMS]}, gap={PLATE_GAP}, front edge y={PLATE_Y - PLATE_D / 2:.3f}"
)
print(f"  bases y={BASE_Y:.3f} z={TABLE_H:.3f}, wrist cams {wrist_paths}")
for _name in INTRINSICS:
    print("  " + describe(_name), flush=True)
for _name in EXTRINSICS:
    print("  " + describe_pose(_name), flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
stage.GetRootLayer().Export(args.out)
print(f"WROTE {args.out}")

# ── optional preview render ──────────────────────────────────────────────────
if args.render:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    os.makedirs(args.render_dir, exist_ok=True)

    # render from the exported layer, not the in-memory anonymous stage, so the
    # saved asset itself gets verified
    ctx.open_stage(args.out)
    for _ in range(40):
        sim_app.update()
    stage = ctx.get_stage()

    # joint state authored on the stage only takes effect once physics steps,
    # so drive both articulations to the home pose before capturing
    sim_world = World(stage_units_in_meters=1.0)
    sim_world.reset()
    q = np.deg2rad(np.array([*FRANKA_HOME_DEG, 0.0, 0.0]))
    q[7:] = FRANKA_FINGER
    for root, _ in arm_roots:
        art = SingleArticulation(root, name=root.split("/")[-1])
        art.initialize()
        art.set_joint_positions(q)
        art.set_joint_velocities(np.zeros_like(q))
    for _ in range(30):
        sim_world.step(render=True)
    print("  posed both arms", flush=True)

    previews = {
        "preview_front": look_at_xform((TABLE_CX, -3.0, 1.9), (TABLE_CX, -0.2, 0.85)),
        "preview_iso": look_at_xform((TABLE_CX + 2.4, -2.8, 2.2), (TABLE_CX, -0.15, 0.85)),
        "preview_side": look_at_xform((TABLE_CX - 3.0, -0.9, 1.7), (TABLE_CX, -0.15, 0.85)),
        "preview_wrist_left": look_at_xform((0.55, -0.75, 1.45), (0.16, -0.30, 1.28)),
        "preview_wrist_right": look_at_xform((-0.25, -0.75, 1.45), (-0.64, -0.30, 1.28)),
        "preview_top": look_at_xform((TABLE_CX, -0.30, CEILING_Z - 0.15), (TABLE_CX, -0.30, 0.75)),
        # what a forward-looking wrist camera sees off the front of the bench
        "preview_forward": look_at_xform((0.0, -0.45, 0.95), (0.0, -2.20, 0.80)),
    }
    for pname, xf in previews.items():
        # preview cams are free-look, use a wide-ish 55 deg equivalent
        # free-look previews: centred principal point, no distortion, so they
        # stay comparable across runs regardless of the rig calibration
        fov = 100.0 if pname == "preview_top" else 55.0
        _f = RES_W / (2 * math.tan(math.radians(fov) / 2))
        make_camera(
            stage,
            f"/World/Cameras/{pname}",
            Intrinsics(_f, _f, RES_W / 2.0, RES_H / 2.0),
            xform=xf,
            distortion=False,
        )

    shots = [f"/World/Cameras/{n}" for n in previews]
    shots += [f"/World/Cameras/{n}" for n in STATIC_CAMERAS]
    shots += wrist_paths

    vp = get_active_viewport()
    calibrated = set(INTRINSICS)
    for cam_path in shots:
        cam_name = cam_path.split("/")[-1]
        out_png = os.path.join(args.render_dir, cam_name + ".png")
        # a calibrated camera renders at the resolution its intrinsics were
        # measured at; anything else would reinterpret cx/cy and the lens schema
        vp.resolution = (RES_W, RES_H) if cam_name in calibrated else (args.width, args.height)
        vp.camera_path = cam_path
        for _ in range(70):
            sim_world.step(render=True)
        capture_viewport_to_file(vp, out_png)
        ok = False
        for _ in range(60):
            sim_world.step(render=True)
            if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
                ok = True
                break
        print(f"  [{'ok' if ok else 'FAIL'}] {out_png}", flush=True)

print("DONE", flush=True)
sys.stdout.flush()
sim_app.close()
os._exit(0)
