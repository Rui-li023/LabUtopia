"""Blender background mesh-clearance check across the complete hinge range."""

import json
import math
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree

root = Path(__file__).resolve().parents[2]
bpy.ops.wm.open_mainfile(filepath=str(root / "assets/chemistry_lab/rotary_evaporator/rotary_evaporator.blend"))
bpy.context.scene.frame_set(1)
rig = bpy.data.objects["tilt"]


def tree(label):
    obj = bpy.data.objects[label]
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    vs = [obj.matrix_world @ v.co for v in mesh.vertices]
    fs = [list(p.vertices) for p in mesh.polygons]
    t = BVHTree.FromPolygons(vs, fs)
    ev.to_mesh_clear()
    return t, vs


results = []
for deg in range(-20, 6):
    rig.rotation_euler = (0, math.radians(deg), 0)
    bpy.context.view_layer.update()
    for moving in ["Drive | ivory motor shell", "Drive | spindle nose"]:
        a, vs = tree(moving)
        for fixed in ["Controller | speed dial", "Controller | body", "Controller | dial scale surround"]:
            b, ws = tree(fixed)
            distance = min([b.find_nearest(v)[3] for v in vs] + [a.find_nearest(v)[3] for v in ws])
            results.append(
                {
                    "tilt_deg": deg,
                    "moving": moving,
                    "control": fixed,
                    "surface_gap_m": distance,
                    "overlap": bool(a.overlap(b)),
                }
            )
assert not any(r["overlap"] for r in results)
(root / "outputs/rotary_evaporator/refined/clearance.json").write_text(
    json.dumps(
        {
            "passed": True,
            "samples": len(results),
            "minimum_gap": min(results, key=lambda r: r["surface_gap_m"]),
            "checks": results,
        },
        indent=2,
    )
)
print("CLEARANCE", min(results, key=lambda r: r["surface_gap_m"]))
