"""Read-only geometry measurements for liquid-container calibration."""
import json

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

stage = Usd.Stage.Open("assets/chemistry_lab/lab_001/lab_001.usd")
cache = UsdGeom.XformCache()
for root in ("/World/beaker1", "/World/beaker2", "/World/conical_bottle02"):
    print(root)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            matrix = cache.GetLocalToWorldTransform(prim)
            world = points @ np.asarray(matrix)[:3, :3] + np.asarray(matrix)[3, :3]
            # Signed tetrahedral integration of the original closed glass
            # shell, not its convex hull (which would fill the hollow vessel).
            # Shift near the origin to reduce cancellation for small objects.
            origin = world.mean(0)
            vertices = world - origin
            mesh = UsdGeom.Mesh(prim)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            volume, first_moment, cursor = 0.0, np.zeros(3), 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = vertices[indices[cursor:cursor + count]]
                cursor += count
                for index in range(1, count - 1):
                    a, b, c = face[0], face[index], face[index + 1]
                    tetrahedron = np.dot(a, np.cross(b, c)) / 6.0
                    volume += tetrahedron
                    first_moment += tetrahedron * (a + b + c) / 4.0
            shell_properties = {}
            if abs(volume) > 1e-12:
                center = origin + first_moment / volume
                shell_properties = {
                    "closed_shell_volume_m3": abs(volume),
                    "mass_at_uniform_density_2230_kg_m3": abs(volume) * 2230,
                    "closed_shell_com_world_m": center.tolist(),
                    "com_above_world_aabb_bottom_m": float(center[2] - world[:, 2].min()),
                }
            print(json.dumps({
                "path": str(prim.GetPath()), "vertices": len(points),
                "local_min": points.min(0).tolist(), "local_max": points.max(0).tolist(),
                "world_min": world.min(0).tolist(), "world_max": world.max(0).tolist(),
                "matrix": np.asarray(matrix).tolist(),
                "collision": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                "rigid": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                **shell_properties,
            }))
            height_axis = int(np.argmax(np.abs(np.asarray(matrix)[:3, 2])))
            radial_axes = [i for i in range(3) if i != height_axis]
            radii = np.linalg.norm(points[:, radial_axes], axis=1)
            for z in np.unique(np.round(points[:, height_axis], 3)):
                subset = radii[np.isclose(points[:, height_axis], z, atol=0.0006, rtol=0)]
                if len(subset) >= 8:
                    print("profile", float(z), float(subset.min()), float(subset.max()), len(subset))
