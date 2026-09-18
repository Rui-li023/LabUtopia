"""Prepare robot geometry for the particle scene's USD-backed renderer.

Do this while the world is stopped, before cameras/physics initialize. Changing
instance topology on a running scene can invalidate PhysX and render products.
The source USD, mesh points, materials, visibility and physics are not edited.
"""

from typing import Any

from loguru import logger
from pxr import Sdf, Usd


def prepare_robot_visuals(stage: Any, world: Any, robot_path: str) -> dict[str, Any]:
    """Locally de-instance this robot only, preserving all authored properties."""
    if not world.is_stopped():
        raise RuntimeError("Robot rendering must be prepared before simulation starts")
    path = Sdf.Path(robot_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or str(path) in ("/", "/World"):
        raise ValueError("Expected a specific, absolute robot prim path")
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        raise ValueError(f"Robot prim does not exist: {robot_path}")
    changed = []
    # A new traversal exposes children of each former instance, including
    # nested geometry references. Do not walk prototypes shared by other robots.
    while True:
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            changed.append(str(prim.GetPath()))
            prim.SetInstanceable(False)
    logger.info("[robot-rendering] {}: de-instanced {} local prims before initialization", robot_path, len(changed))
    return {"version": 1, "representation": "noninstanced", "robot_path": robot_path, "changed_paths": changed}
