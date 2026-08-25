"""Measure what the RTX renderer actually does with the calibrated cameras.

Two things about Isaac Sim 5.1 cannot be settled by reading the code, and both
change how the sim images must be interpreted:

  A. Is the principal point applied once, twice, or not at all?
     The camera carries cx/cy in BOTH the USD aperture offsets and the
     OmniLensDistortionOpenCvPinholeAPI schema. If RTX composes them the image
     is shifted 2x. Probe: put a marker exactly on the optical axis. Distortion
     is identically zero on the axis, so where that marker lands IS the
     principal point, isolated from every other effect.

  B. Does RTX actually apply the lens distortion?
     Probe: put markers near the frame corners, where the measured k1..k3 move a
     point by a few pixels, and compare the rendered centroid against the
     pinhole prediction and the OpenCV-distorted prediction.

Run AFTER build_table_scene.py (it reads the exported stage):

    python scripts/sim2real/verify_camera_render.py

Never run this while another Isaac Sim process is up.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usd_intrinsics import INTRINSICS, RES_H, RES_W

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=os.path.join(REPO, "assets/sim2real/table_cell.usd"))
parser.add_argument("--out-dir", default="/home/ubuntu/Downloads/sim2real_renders/verify")
parser.add_argument("--marker-radius", type=float, default=0.005)
parser.add_argument("--marker-dist", type=float, default=0.30)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": RES_W, "height": RES_H})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)
MARKER = "/World/_probe_marker"


def distort_opencv(x, y, coeffs):
    """OpenCV forward distortion, rational polynomial (k1 k2 p1 p2 k3 k4 k5 k6)."""
    k1, k2, p1, p2, k3, k4, k5, k6 = (list(coeffs) + [0.0] * 8)[:8]
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def make_marker(stage):
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(MARKER))
    sphere.CreateRadiusAttr(args.marker_radius)
    sphere.CreateExtentAttr([Gf.Vec3f(-args.marker_radius) * 1.0, Gf.Vec3f(args.marker_radius) * 1.0])
    mat = UsdShade.Material.Define(stage, Sdf.Path(MARKER + "/Mat"))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(MARKER + "/Mat/Shader"))
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 1, 0))
    sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 1, 0))
    sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(6000.0)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(mat)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(0, 0, -100))
    return sphere


def set_marker(sphere, world_pos):
    for op in UsdGeom.Xformable(sphere).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*world_pos))
            return
    raise RuntimeError("marker translate op missing")


def cam_local_to_world(cam_xf, p_local):
    v = Gf.Vec4d(p_local[0], p_local[1], p_local[2], 1.0) * cam_xf
    return (v[0], v[1], v[2])


def shoot(vp, world, cam_path, png):
    vp.camera_path = cam_path
    for _ in range(45):
        world.step(render=True)
    capture_viewport_to_file(vp, png)
    for _ in range(60):
        world.step(render=True)
        if os.path.exists(png) and os.path.getsize(png) > 0:
            return True
    return False


def find_marker(png):
    """Sub-pixel centroid of the green marker, or None."""
    img = cv2.imread(png).astype(np.float32)
    if img is None:
        return None
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    score = g - np.maximum(b, r)
    mask = score > 40
    if mask.sum() < 4:
        return None
    ys, xs = np.nonzero(mask)
    w = score[ys, xs]
    return float((xs * w).sum() / w.sum()) + 0.5, float((ys * w).sum() / w.sum()) + 0.5


# ── scene ────────────────────────────────────────────────────────────────────
ctx = omni.usd.get_context()
ctx.open_stage(args.usd)
for _ in range(40):
    sim_app.update()
stage = ctx.get_stage()

world = World(stage_units_in_meters=1.0)
world.reset()
for root in ("/World/Franka_left", "/World/Franka_right"):
    if stage.GetPrimAtPath(root).IsValid():
        art = SingleArticulation(root, name=root.split("/")[-1])
        art.initialize()
        q = np.deg2rad(np.array([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0, 0.0, 0.0]))
        q[7:] = 0.04
        art.set_joint_positions(q)
        art.set_joint_velocities(np.zeros_like(q))
for _ in range(30):
    world.step(render=True)

marker = make_marker(stage)
vp = get_active_viewport()
vp.resolution = (RES_W, RES_H)
xc = UsdGeom.XformCache(Usd.TimeCode.Default())

cam_paths = {}
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera) and prim.GetName() in INTRINSICS:
        cam_paths[prim.GetName()] = str(prim.GetPath())

print(f"\n{'=' * 78}\nA. PRINCIPAL POINT — marker on the optical axis (distortion is 0 there)\n{'=' * 78}")
print(f"{'camera':28s} {'expected (cx,cy)':>20s} {'measured':>18s} {'verdict':>22s}")
verdict_pp = {}
for name, path in sorted(cam_paths.items()):
    intr = INTRINSICS[name]
    xc.Clear()
    cam_xf = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    set_marker(marker, cam_local_to_world(cam_xf, (0.0, 0.0, -args.marker_dist)))
    png = os.path.join(args.out_dir, f"pp_{name}.png")
    got = find_marker(png) if shoot(vp, world, path, png) else None
    if got is None:
        print(f"{name:28s} {'-':>20s} {'MARKER NOT FOUND':>18s}")
        continue
    u, v = got
    cand = {
        "correct (1x)": (intr.cx, intr.cy),
        "double shift (2x)": (RES_W / 2 + 2 * (intr.cx - RES_W / 2), RES_H / 2 + 2 * (intr.cy - RES_H / 2)),
        "ignored (centre)": (RES_W / 2.0, RES_H / 2.0),
    }
    best = min(cand, key=lambda k: max(abs(u - cand[k][0]), abs(v - cand[k][1])))
    err = max(abs(u - cand[best][0]), abs(v - cand[best][1]))
    verdict_pp[name] = (best, err)
    print(f"{name:28s} ({intr.cx:7.3f},{intr.cy:7.3f}) ({u:7.2f},{v:7.2f}) {best:>22s} err {err:.2f} px")

print(f"\n{'=' * 78}\nB. DISTORTION — off-axis markers, pinhole vs OpenCV-distorted prediction\n{'=' * 78}")
TARGETS = [(0.16, 0.16), (0.84, 0.16), (0.16, 0.84), (0.84, 0.84), (0.5, 0.5)]
print(f"{'camera':24s} {'target px':>14s} {'pinhole':>16s} {'distorted':>16s} {'measured':>16s}  {'closer to':>10s}")
verdict_d = {}
for name in sorted(cam_paths):
    intr = INTRINSICS[name]
    if not any(c != 0.0 for c in intr.coeffs):
        print(f"{name:24s}  (all distortion coefficients are zero — nothing to detect)")
        continue
    path = cam_paths[name]
    xc.Clear()
    cam_xf = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    votes = {"pinhole": 0, "distorted": 0, "neither": 0}
    for fu, fv in TARGETS:
        u_t, v_t = fu * RES_W, fv * RES_H
        x = (u_t - intr.cx) / intr.fx
        y = (v_t - intr.cy) / intr.fy
        z = args.marker_dist
        # OpenCV camera frame (+Y down, +Z fwd) -> USD camera local (+Y up, -Z fwd)
        set_marker(marker, cam_local_to_world(cam_xf, (x * z, -y * z, -z)))
        png = os.path.join(args.out_dir, f"d_{name}_{int(fu * 100)}_{int(fv * 100)}.png")
        got = find_marker(png) if shoot(vp, world, path, png) else None
        if got is None:
            print(f"{name:24s} ({u_t:6.1f},{v_t:6.1f})  marker not found (off frame or occluded)")
            continue
        xd, yd = distort_opencv(x, y, intr.coeffs)
        u_d, v_d = intr.fx * xd + intr.cx, intr.fy * yd + intr.cy
        e_p = max(abs(got[0] - u_t), abs(got[1] - v_t))
        e_d = max(abs(got[0] - u_d), abs(got[1] - v_d))
        closer = "pinhole" if e_p < e_d else "distorted"
        if min(e_p, e_d) > 3.0:
            closer = "neither"
        votes[closer] += 1
        print(
            f"{name:24s} ({u_t:6.1f},{v_t:6.1f}) ({u_t:7.2f},{v_t:6.2f}) ({u_d:7.2f},{v_d:6.2f}) "
            f"({got[0]:7.2f},{got[1]:6.2f})  {closer:>10s}  dp {e_p:5.2f} dd {e_d:5.2f}"
        )
    verdict_d[name] = votes
    print(f"{name:24s} VOTES {votes}")

print(f"\n{'=' * 78}\nCONCLUSIONS\n{'=' * 78}")
for name, (best, err) in verdict_pp.items():
    print(f"  principal point  {name:28s} -> {best} ({err:.2f} px)")
for name, votes in verdict_d.items():
    winner = max(votes, key=votes.get)
    print(f"  distortion       {name:28s} -> {winner} {votes}")
if not verdict_d:
    print("  distortion       no camera with non-zero coefficients was measurable")
print(f"\nprobe renders: {args.out_dir}")

sys.stdout.flush()
sim_app.close()
os._exit(0)
