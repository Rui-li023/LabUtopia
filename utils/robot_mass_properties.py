"""Opt-in, geometry-derived robot inertia without changing mass or visuals.

The Isaac 5.1 Panda asset authors kg-scale links with sub-millimetre radii of
gyration. Keep its masses/COMs, but let PhysX infer inertia from the existing
colliders. This is a uniform-density approximation, not identified hardware
inertia. Never silently enable it for another task or accept an old replay.
"""

import math
from typing import Any

from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def prepare_robot_mass_properties(stage: Any, world: Any, robot_path: str, mode: str = "asset") -> dict:
    """Author local automatic-inertia overrides before physics initialization."""
    if mode == "asset":
        return {}
    if mode != "collision_geometry":
        raise ValueError(f"Unknown robot inertia mode: {mode}")
    if not world.is_stopped():
        raise RuntimeError("Robot inertia must be prepared before simulation starts")
    path = Sdf.Path(robot_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or str(path) in ("/", "/World"):
        raise ValueError("Expected a specific absolute robot prim path")
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        raise ValueError(f"Robot prim does not exist: {robot_path}")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or not math.isclose(
        UsdPhysics.GetStageKilogramsPerUnit(stage), 1.0
    ):
        raise ValueError("Robot geometry-derived inertia requires metre/kilogram stage units")
    bodies = []
    # Validate everything before authoring, so a missing collider cannot leave
    # half of the robot changed. Geometry must already be de-instanced.
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get() if mass_api else None
        if mass is None or not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"Preserving robot mass requires a positive authored mass: {prim.GetPath()}")
        colliders = []
        descendants = iter(Usd.PrimRange(prim))
        for child in descendants:
            if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
                descendants.PruneChildren()
                continue
            if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get():
                colliders.append(child)
        if not colliders:
            raise ValueError(f"Robot body has no exposed enabled collision geometry: {prim.GetPath()}")
        center = list(mass_api.GetCenterOfMassAttr().Get())
        bodies.append(
            (
                mass_api,
                {
                    "path": str(prim.GetPath()),
                    "mass_kg": float(mass),
                    "original_diagonal_inertia": list(mass_api.GetDiagonalInertiaAttr().Get()),
                    "preserved_center_of_mass": center
                    if all(math.isfinite(value) for value in center)
                    else "automatic",
                    "colliders": [str(child.GetPath()) for child in colliders],
                },
            )
        )
    if not bodies:
        raise ValueError(f"No rigid robot bodies found: {robot_path}")
    for mass_api, _ in bodies:
        # USD's zero sentinel requests automatic inertia, not a zero-inertia
        # dynamic body. Its existing mass and COM remain explicitly authored.
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(0.0))
    logger.info("[robot-inertia] {} bodies: geometry-derived inertia; mass/COM/mesh unchanged", len(bodies))
    return {"version": 1, "mode": mode, "bodies": [record for _, record in bodies]}


def validate_robot_mass_replay(recorded: dict, current: dict) -> None:
    """Old inertia and corrected inertia must never be mixed in strict replay."""
    if recorded != current:
        raise ValueError("Replay robot inertia configuration differs from collection; collect a fresh dataset")
