"""Blender: cut original glass into small convex collision patches; no visual edits."""

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path

import bmesh
import bpy
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/chemistry_lab/lab_001/pour_liquid"
parser = argparse.ArgumentParser()
parser.add_argument("--vessel", choices=("beaker", "conical"), default="beaker")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
stage = Usd.Stage.Open(str(ROOT / "assets/chemistry_lab/lab_001/lab_001.usd"))
source_path = "/World/conical_bottle02/mesh" if args.vessel == "conical" else "/World/beaker2/mesh"
source = UsdGeom.Mesh(stage.GetPrimAtPath(source_path))
points = source.GetPointsAttr().Get()
indices = source.GetFaceVertexIndicesAttr().Get()
faces, cursor = [], 0
for count in source.GetFaceVertexCountsAttr().Get():
    faces.append(list(indices[cursor:cursor + count]))
    cursor += count
mesh = bpy.data.meshes.new("OriginalGlass")
mesh.from_pydata(points, [], faces)
volume_mesh = bmesh.new()
volume_mesh.from_mesh(mesh)
native_volume = abs(volume_mesh.calc_volume(signed=True))
volume_mesh.free()
patches = []
bands = [(-86.689, -83.688), (-83.688, -69.67), (-69.67, 67.12), (67.12, 94.114)]
sectors = 24
center = (53.346556, 0, 0)
if args.vessel == "conical":
    # Measured floor, rounded base, cone, shoulder, straight neck and rolled rim.
    # Keep the authored glass shell: never bridge across the bottle's interior.
    levels = [-0.001, 48.279, 115.77, 172.30, 1221.78, 1297.28, 1556.11, 1645.242]
    bands = list(pairwise(levels))
    sectors = 32
    center = (0, 0, 0)
for band, (low, high) in enumerate(bands):
    for sector in range(1 if band == 0 else sectors):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        planes = [((0, low, 0), (0, 1, 0)), ((0, high, 0), (0, -1, 0))]
        if band:
            a, b = 2 * math.pi * sector / sectors, 2 * math.pi * (sector + 1) / sectors
            planes += [(center, (-math.sin(a), 0, math.cos(a))), (center, (math.sin(b), 0, -math.cos(b)))]
        for point, normal in planes:
            bmesh.ops.bisect_plane(
                bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                plane_co=point, plane_no=normal, clear_inner=True, dist=1e-5,
            )
        hull = bmesh.new()
        for vertex in bm.verts:
            hull.verts.new(vertex.co)
        bm.free()
        bmesh.ops.remove_doubles(hull, verts=list(hull.verts), dist=1e-4)
        result = bmesh.ops.convex_hull(hull, input=list(hull.verts), use_existing_faces=False)
        unused = set(result["geom_interior"] + result["geom_unused"])
        unused = [v for v in unused if isinstance(v, bmesh.types.BMVert)]
        if unused:
            bmesh.ops.delete(hull, geom=unused, context="VERTS")
        bmesh.ops.recalc_face_normals(hull, faces=list(hull.faces))
        hull.verts.index_update()
        patches.append({
            "name": f"Glass_{band}_{sector:02d}",
            "points": [list(v.co) for v in hull.verts],
            "counts": [len(f.verts) for f in hull.faces],
            "indices": [v.index for f in hull.faces for v in f.verts],
        })
        hull.free()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{args.vessel}_collision.json").write_text(json.dumps({"version": 1, "native_glass_volume": native_volume, "patches": patches}, separators=(",", ":")))
print(f"Wrote {len(patches)} original-mesh collision patches to {OUT}")
