"""Collision-only candidates for the ORIGINAL flask, metal stand and cork mesh.

No visual geometry, material, pose, mass, friction or rigid-body lock is changed.
Use before physics initialization. Static supports can retain their real concavity
with triangle-mesh collision; the dynamic flask must use convex decomposition.
"""

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def configure_original_flask_collisions(
    stage: Usd.Stage,
    *,
    flask_path: str = "/World/round_bottom_flask",
    support_path: str = "/World/pipeclay_triangle",
    profile: str = "support_mesh",
) -> dict:
    if profile not in ("baseline", "support_mesh", "refined_hulls"):
        raise ValueError("Unknown original-flask collision candidate")
    support = stage.GetPrimAtPath(support_path)
    support_mesh = stage.GetPrimAtPath(support_path + "/mesh_000")
    body_mesh = stage.GetPrimAtPath(flask_path + "/mesh_001")
    if not support_mesh.IsA(UsdGeom.Mesh) or not body_mesh.IsA(UsdGeom.Mesh):
        raise ValueError("Collision repair requires the original visible USD meshes")
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(support)):
        raise ValueError("Exact triangle-mesh support must be static")
    if not stage.GetPrimAtPath(flask_path).HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("The original flask must remain a free dynamic rigid body")
    if profile != "baseline":
        UsdPhysics.CollisionAPI.Apply(support_mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(support_mesh).CreateApproximationAttr("none")
        support_mesh.AddAppliedSchema("PhysxCollisionAPI")
        support_mesh.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0005)
        support_mesh.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    if profile == "refined_hulls":
        for prim in Usd.PrimRange(stage.GetPrimAtPath(flask_path)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0005)
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        UsdPhysics.MeshCollisionAPI.Apply(body_mesh).CreateApproximationAttr("convexDecomposition")
        body_mesh.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
        for name, kind, value in (
            ("errorPercentage", Sdf.ValueTypeNames.Float, 0.5),
            ("maxConvexHulls", Sdf.ValueTypeNames.Int, 128),
            ("minThickness", Sdf.ValueTypeNames.Float, 0.0001),
            ("shrinkWrap", Sdf.ValueTypeNames.Bool, True),
        ):
            body_mesh.CreateAttribute("physxConvexDecompositionCollision:" + name, kind).Set(value)
    return {
        "profile": profile,
        "flask_path": flask_path,
        "support_path": support_path,
        "visual_geometry_changed": False,
    }
