"""Scoped physical contact material for the acrylic pipette stand surfaces."""

import math
import re

from loguru import logger
from pxr import Usd, UsdPhysics, UsdShade


def configure_acrylic_rack_contact(
    stage: Usd.Stage, rack_path: str, *, static_friction: float, dynamic_friction: float
) -> tuple[str, ...]:
    """Bind physical material to the four acrylic pieces, preserving the feet.

    Coefficients are explicitly supplied engineering estimates, not factory
    calibration. The function never changes visual bindings, geometry, mass,
    collision enablement, or friction on the pipette and robot. It authors
    only the runtime stage; it never saves an asset layer.
    """
    if not all(math.isfinite(value) for value in (static_friction, dynamic_friction)):
        raise ValueError("Rack contact friction must be finite")
    if not 0 <= dynamic_friction <= static_friction <= 1.2:
        raise ValueError("Rack contact friction must satisfy 0 <= dynamic <= static <= 1.2")
    rack = stage.GetPrimAtPath(rack_path)
    if not rack.IsValid():
        raise ValueError(f"Pipette rack does not exist: {rack_path}")
    parts = [
        prim
        for prim in Usd.PrimRange(rack)
        if prim.GetName() in {"Base_plate", "Inclined_back"}
        or re.fullmatch(r"(?:Upper|Lower)_\d+(?:\.\d+)?mm_slots", prim.GetName())
    ]
    names = [prim.GetName() for prim in parts]
    if (
        len(parts) != 4
        or "Base_plate" not in names
        or "Inclined_back" not in names
        or sum(name.startswith("Upper_") for name in names) != 1
        or sum(name.startswith("Lower_") for name in names) != 1
        or any(not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsInstanceProxy() for prim in parts)
    ):
        raise ValueError("Expected four editable acrylic rack colliders; contact material was not changed")
    material_path = rack.GetPath().AppendChild("AcrylicContact")
    existing = stage.GetPrimAtPath(material_path)
    if existing.IsValid() and not existing.IsA(UsdShade.Material):
        raise ValueError(f"Rack contact material path is occupied: {material_path}")
    material = UsdShade.Material.Define(stage, material_path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(0.0)
    for prim in parts:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
    logger.info(
        "Configured acrylic contact on {} rack colliders: static={}, dynamic={}; silicone feet unchanged",
        len(parts),
        static_friction,
        dynamic_friction,
    )
    return tuple(str(prim.GetPath()) for prim in parts)
