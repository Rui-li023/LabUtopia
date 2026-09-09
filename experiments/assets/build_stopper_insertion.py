"""Blender repair of the original stopper and hollow-neck collision geometry.

The original glass mesh, material and UVs remain untouched. Only the existing
stopper vertices are calibrated to a catalogued 36/30 x 30 mm cork taper; its
topology and UVs survive. Run with Blender --background --factory-startup.
"""

import json
import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/chemistry_lab/lab_002/stopper_insertion"
FLASK = "/World/round_bottom_flask_01/mesh_001"
STOPPER = "/World/rubber_stopper/mesh_003"
SEGMENTS = 32


def import_mesh(stage, path, scale, name):
    source = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    points = [(p[0] * scale, -p[2] * scale, p[1] * scale) for p in source.GetPointsAttr().Get()]
    indices = source.GetFaceVertexIndicesAttr().Get()
    faces, cursor = [], 0
    for count in source.GetFaceVertexCountsAttr().Get():
        faces.append(list(indices[cursor : cursor + count]))
        cursor += count
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    uv = UsdGeom.PrimvarsAPI(source).GetPrimvar("st")
    if uv and uv.HasValue():
        values = uv.ComputeFlattened()
        layer = mesh.uv_layers.new(name="st")
        for i, loop in enumerate(mesh.loops):
            index = i if uv.GetInterpolation() == "faceVarying" else loop.vertex_index
            layer.data[i].uv = values[index]
    return obj


def material(name, color, glass=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = 0.15 if glass else 0.8
    bsdf.inputs["Transmission Weight"].default_value = 1 if glass else 0
    bsdf.inputs["IOR"].default_value = 1.47
    return mat


def clipped_hull(source, low, high, sector=None):
    bm = bmesh.new()
    bm.from_mesh(source.data)
    planes = [((0, 0, low), (0, 0, 1)), ((0, 0, high), (0, 0, -1))]
    if sector is not None:
        a, b = 2 * math.pi * sector / SEGMENTS, 2 * math.pi * (sector + 1) / SEGMENTS
        planes += [((0, 0, 0), (-math.sin(a), math.cos(a), 0)), ((0, 0, 0), (math.sin(b), -math.cos(b), 0))]
    for point, normal in planes:
        bmesh.ops.bisect_plane(
            bm,
            geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
            plane_co=point,
            plane_no=normal,
            clear_inner=True,
            dist=1e-8,
        )
    points = [v.co.copy() for v in bm.verts]
    bm.free()
    hull = bmesh.new()
    for point in points:
        hull.verts.new(point)
    bmesh.ops.remove_doubles(hull, verts=list(hull.verts), dist=1e-7)
    result = bmesh.ops.convex_hull(hull, input=list(hull.verts), use_existing_faces=False)
    unused = [v for v in result["geom_interior"] + result["geom_unused"] if isinstance(v, bmesh.types.BMVert)]
    if unused:
        bmesh.ops.delete(hull, geom=list(set(unused)), context="VERTS")
    bmesh.ops.recalc_face_normals(hull, faces=list(hull.faces))
    return hull


def write_collider(stage, name, bm):
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    points = [(v.co.x / 0.000067, v.co.z / 0.000067, -v.co.y / 0.000067) for v in bm.verts]
    mesh = UsdGeom.Mesh.Define(stage, FLASK + "/" + name)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f.verts) for f in bm.faces])
    mesh.CreateFaceVertexIndicesAttr([v.index for f in bm.faces for v in f.verts])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    mesh.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    mesh.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0002)
    mesh.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    debug_mesh = bpy.data.meshes.new(name)
    bm.to_mesh(debug_mesh)
    obj = bpy.data.objects.new(name, debug_mesh)
    bpy.context.collection.objects.link(obj)
    obj.display_type = "WIRE"
    obj.hide_render = True
    bm.free()


def marker(stage, parent, name, point):
    # Parent is the scaled Y-up source mesh, point is metres in Z-up Blender.
    scale = 0.000067 if parent == FLASK else 0.00006
    root = UsdGeom.Xform.Define(stage, parent + "/" + name)
    root.AddTranslateOp().Set(Gf.Vec3d(point[0] / scale, point[2] / scale, -point[1] / scale))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    original = Usd.Stage.Open(str(ROOT / "assets/chemistry_lab/lab_002/lab_002.usd"))
    flask = import_mesh(original, FLASK, 0.000067, "Original_glass_reference")
    stopper = import_mesh(original, STOPPER, 0.00006, "Catalog_cork_36_30_30")
    flask.data.materials.append(material("ReferenceGlass", (0.75, 0.86, 0.86), glass=True))
    cork = material("Cork", (0.53, 0.34, 0.16))
    texture = cork.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(
        str(OUT.parent / "flask_support_collision/textures/cork001/Cork001_1K-PNG_Color.png")
    )
    cork.node_tree.links.new(texture.outputs["Color"], cork.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
    stopper.data.materials.append(cork)

    # Retain the original rounded ends/UVs rather than replace the object with a
    # primitive. Map its old gently tapered sidewall onto a real catalogue taper.
    height = max(v.co.z for v in stopper.data.vertices)
    for vertex in stopper.data.vertices:
        z = vertex.co.z
        old_radius = 0.01295 + (0.014089 - 0.01295) * (z - 0.00271) / (0.027335 - 0.00271)
        target_radius = 0.015 + 0.003 * z / height
        vertex.co.x *= target_radius / old_radius
        vertex.co.y *= target_radius / old_radius
        vertex.co.z = z * 0.030 / height
    stopper.data.update()

    stage = Usd.Stage.CreateNew(str(OUT / "lab_002_stopper_insertion.usda"))
    stage.GetRootLayer().subLayerPaths = ["../flask_support_collision/lab_002_flask_cork.usda"]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    flask_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(FLASK))
    UsdPhysics.CollisionAPI.Apply(flask_mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    # The bulb's exterior hull lies wholly below the insertion workspace. The
    # visible glass remains hollow; these colliders are for stopper manipulation,
    # not liquid or tool motion inside the bulb.
    write_collider(stage, "BulbExteriorCollision", clipped_hull(flask, -0.001, 0.076))
    for band, (low, high) in enumerate(((0.0755, 0.110), (0.110, 0.125), (0.125, 0.142))):
        for sector in range(SEGMENTS):
            write_collider(stage, f"OpenNeck_{band}_{sector:02d}", clipped_hull(flask, low, high, sector))
    # Real glass density applied to the original closed glass-shell volume.
    bm = bmesh.new()
    bm.from_mesh(flask.data)
    volume = abs(bm.calc_volume(signed=True))
    bm.free()
    UsdPhysics.MassAPI.Apply(flask_mesh.GetPrim()).CreateMassAttr(volume * 2230)
    UsdPhysics.MassAPI(flask_mesh.GetPrim()).CreateDensityAttr(0.0)

    target = UsdGeom.Mesh(stage.GetPrimAtPath(STOPPER))
    points = [(v.co.x / 0.00006, v.co.z / 0.00006, -v.co.y / 0.00006) for v in stopper.data.vertices]
    target.CreatePointsAttr(points)
    target.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    # Normals must follow the deformed surface, not stale authored normal vectors.
    target.GetNormalsAttr().Block()
    UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/World/FlaskCorkMaterials/Cork"))
    )
    body = UsdPhysics.RigidBodyAPI(target.GetPrim())
    body.CreateVelocityAttr(Gf.Vec3f(0))
    body.CreateAngularVelocityAttr(Gf.Vec3f(0))
    bm = bmesh.new()
    bm.from_mesh(stopper.data)
    cork_volume = abs(bm.calc_volume(signed=True))
    bm.free()
    UsdPhysics.MassAPI.Apply(target.GetPrim()).CreateMassAttr(cork_volume * 240)
    UsdPhysics.MassAPI(target.GetPrim()).CreateDensityAttr(0.0)
    target.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    target.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0002)
    target.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    top = max(v.co.z for v in flask.data.vertices)
    marker(stage, FLASK, "Mouth", (0, 0, top))
    marker(stage, STOPPER, "Tip", (0, 0, 0))
    marker(stage, STOPPER, "Grasp", (0, 0, 0.026))
    # Reuse the original cork ring and its actual contact-calibrated seat.
    # Its existing FlaskSeat refers to the capped flask's geometry centre;
    # subtract the difference in half-height for the uncapped original bottle.
    cache = UsdGeom.XformCache()
    bounds = UsdGeom.BBoxCache(0, ["default"])
    closed_height = (
        bounds.ComputeWorldBound(stage.GetPrimAtPath("/World/round_bottom_flask")).ComputeAlignedRange().GetSize()[2]
    )
    seat = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/cork_ring/FlaskSeat")).ExtractTranslation()
    seat[2] -= (closed_height - top) / 2
    local_seat = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/cork_ring")).GetInverse().Transform(seat)
    UsdGeom.Xform.Define(stage, "/World/cork_ring/OpenFlaskSeat").AddTranslateOp().Set(local_seat)
    for path in (FLASK, STOPPER):
        stage.GetPrimAtPath(path).CreateAttribute("reference:upAxis", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0, 1, 0))

    glass_bm = bmesh.new()
    glass_bm.from_mesh(flask.data)
    glass_tree = BVHTree.FromBMesh(glass_bm)
    first_collision = None
    for index in range(401):
        depth = index * 0.0001
        bm = bmesh.new()
        bm.from_mesh(stopper.data)
        bmesh.ops.translate(bm, verts=list(bm.verts), vec=Vector((0, 0, top - depth)))
        overlap = glass_tree.overlap(BVHTree.FromBMesh(bm))
        bm.free()
        if overlap:
            first_collision = depth
            break
    glass_bm.free()
    print(
        "INSERTION_ASSET "
        + json.dumps(
            {
                "flask_height_m": top,
                "flask_mass_kg": volume * 2230,
                "stopper_mass_kg": cork_volume * 240,
                "axial_first_surface_contact_depth_m": first_collision,
                "neck_convex_sectors": 3 * SEGMENTS,
                "visible_flask_changed": False,
            }
        )
    )
    stage.GetRootLayer().Save()
    stopper.location = (0.12, 0, 0)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "stopper_insertion.blend"))


if __name__ == "__main__":
    main()
