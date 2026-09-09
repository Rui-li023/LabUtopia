"""Read-only geometry measurements for liquid-container calibration."""
import json

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

stage = Usd.Stage.Open("assets/chemistry_lab/lab_001/lab_001.usd")
cache = UsdGeom.XformCache()
for root in ("/World/beaker1", "/World/beaker2"):
    print(root)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            matrix = cache.GetLocalToWorldTransform(prim)
            world = points @ np.asarray(matrix)[:3, :3] + np.asarray(matrix)[3, :3]
            print(json.dumps({
                "path": str(prim.GetPath()), "vertices": len(points),
                "local_min": points.min(0).tolist(), "local_max": points.max(0).tolist(),
                "world_min": world.min(0).tolist(), "world_max": world.max(0).tolist(),
                "matrix": np.asarray(matrix).tolist(),
                "collision": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                "rigid": prim.HasAPI(UsdPhysics.RigidBodyAPI),
            }))
            height_axis = int(np.argmax(np.abs(np.asarray(matrix)[:3, 2])))
            radial_axes = [i for i in range(3) if i != height_axis]
            radii = np.linalg.norm(points[:, radial_axes], axis=1)
            for z in np.unique(np.round(points[:, height_axis], 3)):
                subset = radii[np.isclose(points[:, height_axis], z, atol=0.0006, rtol=0)]
                if len(subset) >= 8:
                    print("profile", float(z), float(subset.min()), float(subset.max()), len(subset))
