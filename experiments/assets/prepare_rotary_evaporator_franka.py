"""Author a reversible USD variant with a grippable dial and missing fascia contacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ASSET = Path(__file__).resolve().parents[2] / "assets/chemistry_lab/rotary_evaporator"
BODY_SCALE = 0.7
ANCHORS = {
    "upper": (np.array([-0.049, 0, 0.691]), np.array([0, -0.006, 0.030])),
    "bath": (np.array([0.204, -0.172, 0.090]), np.zeros(3)),
    "switch": (np.array([0.360, -0.055, 0.0756]), np.zeros(3)),
    "ruler": (np.array([0.078, 0.039, 0.472]), np.zeros(3)),
}


def remap(points, group: str = "body") -> np.ndarray:
    """Shrink the instrument while preserving the controls' usable dimensions."""
    points = np.asarray(points, dtype=float)
    if group == "body":
        return points * BODY_SCALE
    anchor, offset = ANCHORS[group]
    return points + (BODY_SCALE - 1) * anchor + offset


def control_group(body: str) -> str:
    if body in {"motor_power", "motor_run", "speed_dial"}:
        return "upper"
    if body in {"bath_mode", "bath_down", "bath_up", "bath_set"}:
        return "bath"
    return "switch" if body == "mains_switch" else "body"


def mesh_group(body: str, name: str, world_points: np.ndarray) -> str:
    group = control_group(body)
    if group != "body":
        return group
    if name.startswith("Controller___"):
        return "upper"
    if "ruler" in name:
        return "ruler"
    if name in {"Bath___power_switch_surround", "Bath___mains_label"}:
        return "switch"
    if name.startswith("Bath___") and world_points[:, 1].mean() < -0.14 and world_points[:, 2].max() < 0.15:
        return "bath"
    return "body"


def bake_proportions(stage: Usd.Stage) -> dict:
    """Bake into geometry and anchors, never scale an active articulation root.

    The source exports translation-only bodies and identity visual mesh frames.
    Button travel stays 2.2 mm; the lift and its ruler stay calibrated to 120 mm.
    """
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    old = {}
    for body in bodies:
        ops = UsdGeom.Xformable(body).GetOrderedXformOps()
        assert len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate
        old[str(body.GetPath())] = np.array(ops[0].Get())
    new = {path: remap(origin, control_group(path.split("/")[-1])) for path, origin in old.items()}
    for body in bodies:
        path, name = str(body.GetPath()), body.GetName()
        body_group = control_group(name)
        UsdGeom.Xformable(body).GetOrderedXformOps()[0].Set(Gf.Vec3d(*new[path]))
        for prim in Usd.PrimRange(body):
            if prim.IsA(UsdGeom.Mesh) and "/Visuals/" in str(prim.GetPath()):
                assert not UsdGeom.Xformable(prim).GetOrderedXformOps()
                mesh = UsdGeom.Mesh(prim)
                points = np.array(mesh.GetPointsAttr().Get(), dtype=float) + old[path]
                group = mesh_group(name, prim.GetName(), points)
                points = remap(points, group) - new[path]
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                group = "upper" if name == "lift" and prim.GetName() == "controller" else body_group
                ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                translate = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
                center = remap(np.array(translate.Get()) + old[path], group) - new[path]
                translate.Set(Gf.Vec3d(*center))
                if group == "body":
                    scale = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
                    if scale:
                        scale.Set(Gf.Vec3f(*(np.array(scale.Get(), dtype=float) * BODY_SCALE)))
                    else:
                        for attr_name in ("radius", "height", "size"):
                            attr = prim.GetAttribute(attr_name)
                            if attr:
                                attr.Set(attr.Get() * BODY_SCALE)
                    # Primitive extents may be cached by the exporter. Recompute
                    # from the resized schema so planner bounds match PhysX.
                    prim.GetAttribute("extent").Set(
                        UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
                    )
            elif prim.GetName() in {"Display_rpm", "Display_bath"}:
                group = "upper" if prim.GetName() == "Display_rpm" else "bath"
                translate = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
                center = remap(np.array(translate.Get()) + old[path], group) - new[path]
                translate.Set(Gf.Vec3d(*center))
        if body_group == "body":
            mass = UsdPhysics.MassAPI(body)
            for attr, factor in ((mass.GetMassAttr(), BODY_SCALE**3), (mass.GetDiagonalInertiaAttr(), BODY_SCALE**5)):
                if attr.HasAuthoredValue():
                    attr.Set(attr.Get() * factor)
            if mass.GetCenterOfMassAttr().HasAuthoredValue():
                mass.GetCenterOfMassAttr().Set(mass.GetCenterOfMassAttr().Get() * BODY_SCALE)
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        child_path = str(joint.GetBody1Rel().GetTargets()[0])
        group = control_group(child_path.split("/")[-1])
        pivot = remap(old[child_path] + np.array(joint.GetLocalPos1Attr().Get()), group)
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(*(pivot - new[child_path])))
        parents = joint.GetBody0Rel().GetTargets()
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*(pivot - new[str(parents[0])] if parents else pivot)))
    return {"body_linear_scale": BODY_SCALE, "button_travel_m": 0.0022, "lift_travel_m": 0.120}


def build_variant() -> Path:
    destination = ASSET / "rotary_evaporator_franka.usda"
    stage = Usd.Stage.CreateNew(str(destination))
    root = UsdGeom.Xform.Define(stage, "/RotaryEvaporator")
    root.GetPrim().GetReferences().AddReference("./rotary_evaporator.usdc")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    prefix = str(root.GetPath())
    dimensions = bake_proportions(stage)
    # Keep the engraved X/Z calibration and joint origin; extend the blue grip
    # forward 26 mm so Panda fingertips can close without entering the scale.
    for name in ("Controller___speed_dial", "Controller___speed_dial_inset", "Controller___dial_index"):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prefix + "/speed_dial/Visuals/" + name))
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        if name == "Controller___speed_dial":
            rear = points[:, 1].max()
            factor = (np.ptp(points[:, 1]) + 0.026) / np.ptp(points[:, 1])
            points[:, 1] = rear + (points[:, 1] - rear) * factor
            normals = np.array(mesh.GetNormalsAttr().Get(), dtype=float)
            normals[:, 1] /= factor
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            mesh.GetNormalsAttr().Set([Gf.Vec3f(*row) for row in normals])
        else:
            points[:, 1] -= 0.026
        mesh.GetPointsAttr().Set([Gf.Vec3f(*row) for row in points])
        mesh.GetExtentAttr().Set([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
    knob = UsdGeom.Cylinder(stage.GetPrimAtPath(prefix + "/speed_dial/Collisions/knob"))
    knob.GetHeightAttr().Set(0.043)
    UsdGeom.Xformable(knob).GetOrderedXformOps()[0].Set(Gf.Vec3d(0, -0.013, 0))
    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(prefix + "/speed_dial"))
    mass.GetMassAttr().Set(0.060)
    mass.GetCenterOfMassAttr().Set(Gf.Vec3f(0, -0.013, 0))
    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.000014, 0.000012, 0.000014))
    contact = UsdShade.Material.Define(stage, prefix + "/Materials/FrankaGrip")
    material = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    material.CreateStaticFrictionAttr(0.9)
    material.CreateDynamicFrictionAttr(0.7)
    material.CreateRestitutionAttr(0)
    UsdShade.MaterialBindingAPI.Apply(knob.GetPrim()).Bind(contact, materialPurpose="physics")

    def collider(body: str, name: str, center, size, angle=0):
        origin = (
            UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(prefix + "/" + body)).ExtractTranslation()
        )
        shape = UsdGeom.Cube.Define(stage, f"{prefix}/{body}/Collisions/{name}")
        shape.CreateSizeAttr(1)
        shape.AddTranslateOp().Set(Gf.Vec3d(*(np.array(center) - origin)))
        shape.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), angle).GetQuat()))
        shape.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        shape.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        shape.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0003)
        shape.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0)

    for i in range(24):
        angle = i * math.tau / 24
        collider(
            "lift",
            f"dial_surround_{i:02d}",
            remap((-0.049 + 0.029 * math.sin(angle), -0.0415, 0.643 + 0.029 * math.cos(angle)), "upper"),
            (0.008, 0.0012, 0.016),
            math.degrees(angle),
        )
    collider("base", "front_fascia", remap((0.204, -0.1515, 0.090), "bath"), (0.220, 0.032, 0.090))
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
    )
    dimensions["visual_bounds_size_m"] = list(bounds.GetSize())
    root.GetPrim().SetCustomDataByKey("variant", "Franka physical contact development")
    stage.GetRootLayer().Save()
    (ASSET / "franka_affordances.json").write_text(
        json.dumps(
            {
                "asset": destination.name,
                "dimensions": dimensions,
                "dial": {
                    "center_local_m": remap([-0.049, -0.072, 0.643], "upper").tolist(),
                    "grip_diameter_m": 0.040,
                    "added_projection_m": 0.026,
                    "fixed_calibration_deg_per_rpm": 2.5,
                },
                "changes": [
                    "Body, glassware and supports reduced to 70% with baked collision geometry and joint anchors",
                    "Control panels, labels, button travel, dial calibration and 120 mm lift ruler retain usable dimensions",
                    "Extended blue dial grip 26 mm forward; zero and scale unchanged",
                    "Added contacts for the stationary dial surround and bath front fascia",
                    "Dial contact friction 0.9 static / 0.7 dynamic",
                ],
                "scope": "Geometry variant for simulated Franka contact; original USD and Blender source retained",
            },
            indent=2,
        )
        + "\n"
    )
    return destination


if __name__ == "__main__":
    print(build_variant())
