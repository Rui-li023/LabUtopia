"""Compare composed, world-scaled original and restored pipette dimensions.

Run with USD Python bindings available. Rotation is removed before comparing
the body; the original disposable tip is reported separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def mesh_points(stage, path):
    prim = stage.GetPrimAtPath(path)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    return np.asarray([matrix.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]) * scale


def audit(asset="assets/chemistry_lab/lab_002/realistic_pipette/lab_002_pipette.usda"):
    original = Usd.Stage.Open(str(ROOT / "assets/chemistry_lab/lab_002/lab_002.usd"))
    current = Usd.Stage.Open(str(ROOT / asset))
    body = mesh_points(original, "/World/pipette_01/mesh_003")
    tip = mesh_points(original, "/World/pipette_01/mesh_004")
    center = body.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((body - center).T))
    direction = axes[:, -1]
    if np.dot(direction, center - (tip.min(0) + tip.max(0)) / 2) < 0:
        direction = -direction
    rotation, _ = Rotation.align_vectors([[0, 0, 1]], [direction])
    original_size = np.ptp(rotation.apply(body - center), axis=0)
    with_tip_size = np.ptp(rotation.apply(np.concatenate([body, tip]) - center), axis=0)
    # Remove the authored root rotation, retaining composed world scale.
    current_points = mesh_points(current, "/World/pipette_01/Body")
    root_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(current.GetPrimAtPath("/World/pipette_01"))
    inverse_rotation = root_matrix.ExtractRotation().GetInverse()
    current_size = np.ptp(
        np.asarray([inverse_rotation.TransformDir(Gf.Vec3d(*p)) for p in current_points - current_points.mean(0)]),
        axis=0,
    )
    return {
        "axis_order": ["width_x", "thickness_y", "body_length_z"],
        "original_body_mm": (original_size * 1000).tolist(),
        "original_with_tip_mm": (with_tip_size * 1000).tolist(),
        "current_body_mm": (current_size * 1000).tolist(),
        "body_difference_percent": ((current_size / original_size - 1) * 100).tolist(),
        "builder_uniform_scale_before_smoothing": float(
            current.GetPrimAtPath("/World/pipette_01").GetAttribute("reference:estimatedLength").Get()
            / original_size[2]
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="assets/chemistry_lab/lab_002/realistic_pipette/lab_002_pipette.usda")
    print(json.dumps(audit(parser.parse_args().asset), indent=2))
