"""Build editable Level-2 glassware in Blender and a scoped USD scene layer.

Run: blender -b --python experiments/assets/build_level2_glassware.py
Geometry is in metres, Z up. The original lab scene is a read-only sublayer.
"""

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/chemistry_lab/lab_002/level2_glassware"


def material(name, color, metallic=0.0, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = 0.18 if transmission else 0.38
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = 1.47
    return mat


def mesh_object(name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for face in mesh.polygons:
        face.use_smooth = True
    return obj


def lathe(name, profile, mat, segments=96):
    vertices = [
        (r * math.cos(2 * math.pi * j / segments), r * math.sin(2 * math.pi * j / segments), z)
        for r, z in profile
        for j in range(segments)
    ]
    faces = []
    for i in range(len(profile)):
        following = (i + 1) % len(profile)
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((i * segments + j, i * segments + k, following * segments + k, following * segments + j))
    return mesh_object(name, vertices, faces, mat)


def cylinder_between(name, a, b, radius, mat):
    delta = Vector(b) - Vector(a)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=delta.length)
    obj = bpy.context.object
    obj.name = name
    obj.location = (Vector(a) + Vector(b)) / 2
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Rounded edges", "BEVEL")
    bevel.width = 0.0006
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    return obj


def shell_center_of_mass(obj):
    """Uniform-density shell centroid from signed tetrahedral volumes."""
    obj.data.calc_loop_triangles()
    volume = 0.0
    moment = Vector((0, 0, 0))
    for triangle in obj.data.loop_triangles:
        a, b, c = (obj.data.vertices[i].co for i in triangle.vertices)
        tetrahedron = a.dot(b.cross(c)) / 6.0
        volume += tetrahedron
        moment += (a + b + c) * (tetrahedron / 4.0)
    if abs(volume) < 1e-12:
        raise ValueError("Flask shell has no enclosed glass volume")
    return moment / volume


def write_mesh(stage, path, obj, collision=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    UsdGeom.Xformable(mesh).ClearXformOpOrder()
    points = [tuple(obj.matrix_world @ v.co) for v in obj.data.vertices]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(p.vertices) for p in obj.data.polygons])
    mesh.CreateFaceVertexIndicesAttr([v for p in obj.data.polygons for v in p.vertices])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))
    mat_name = obj.data.materials[0].name
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/World/Level2Materials/" + mat_name))
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(collision)
        mesh.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        mesh.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    elif mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        mesh.GetPrim().RemoveAPI(UsdPhysics.CollisionAPI)
        mesh.GetPrim().RemoveAPI(UsdPhysics.MeshCollisionAPI)
    return mesh


def replace_root(stage, name, dynamic=False):
    path = "/World/" + name
    prim = stage.GetPrimAtPath(path)
    for child in prim.GetChildren():
        stage.OverridePrim(child.GetPath()).SetActive(False)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    # Reuse an existing translate op, whose precision may differ by asset.
    attr = prim.GetAttribute("xformOp:translate")
    op = UsdGeom.XformOp(attr) if attr else xform.AddTranslateOp()
    xform.SetXformOpOrder([op])
    op.Set(Gf.Vec3d(0, 0, -2))
    if dynamic:
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.08)
        UsdPhysics.MassAPI(prim).CreateDensityAttr(0.0)
        body.CreateVelocityAttr(Gf.Vec3f(0))
        body.CreateAngularVelocityAttr(Gf.Vec3f(0))
    elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    glass = material("Borosilicate", (0.70, 0.88, 0.91), transmission=1.0)
    ceramic = material("Porcelain", (0.88, 0.90, 0.86))
    steel = material("SupportSteel", (0.24, 0.28, 0.30), metallic=0.8)
    cork = material("Cork", (0.40, 0.24, 0.10))
    plastic = material("PipetteBody", (0.82, 0.87, 0.90))
    blue = material("PipetteBlue", (0.025, 0.28, 0.55))

    # Closed glass shell with a reinforced bottom, open lip and smooth neck.
    outer = [(0.0001, 0.0)]
    for i in range(1, 39):
        theta = math.pi - (math.pi - 0.36) * i / 38
        outer.append((0.04 * math.sin(theta), 0.04 + 0.04 * math.cos(theta)))
    outer += [(0.014, 0.080), (0.014, 0.137), (0.0155, 0.140), (0.0155, 0.144), (0.013, 0.145), (0.0125, 0.143)]
    inner = [(0.0125, 0.080)]
    for i in range(39):
        theta = 0.40 + (math.pi - 0.40) * i / 38
        inner.append((max(0.0001, 0.036 * math.sin(theta)), 0.0435 + 0.036 * math.cos(theta)))
    flask = lathe("Flask_shell", outer + inner, glass)
    ring = lathe(
        "Cork_seat",
        [
            (0.027, 0),
            (0.046, 0),
            (0.048, 0.003),
            (0.048, 0.027),
            (0.046, 0.030),
            (0.029, 0.030),
            (0.026, 0.026),
            (0.026, 0.003),
        ],
        cork,
    )
    corners = [
        (0.065 * math.cos(a), 0.065 * math.sin(a), 0.085) for a in (math.pi / 2, math.pi * 7 / 6, math.pi * 11 / 6)
    ]
    stand = []
    for i, a in enumerate(corners):
        b = corners[(i + 1) % 3]
        stand.append(cylinder_between(f"Ceramic_{i}", a, b, 0.006, ceramic))
        foot = (a[0] * 1.18, a[1] * 1.18, 0.004)
        stand.append(cylinder_between(f"Leg_{i}", foot, a, 0.0035, steel))
    bpy.context.view_layer.update()
    pipette = lathe(
        "Pipette_body",
        [
            (0.0001, 0.10),
            (0.006, 0.10),
            (0.014, 0.17),
            (0.014, 0.295),
            (0.030, 0.295),
            (0.030, 0.307),
            (0.015, 0.307),
            (0.015, 0.35),
            (0.018, 0.35),
            (0.018, 0.365),
            (0.0001, 0.365),
        ],
        plastic,
        segments=64,
    )
    tip = lathe("Pipette_tip", [(0.0001, 0), (0.001, 0), (0.006, 0.10), (0.0001, 0.10)], blue, segments=48)
    rack = []
    for y in (-0.20, 0.20):
        rack.append(cylinder_between(f"RackPost_{len(rack)}", (0.04, y, 0.005), (0.04, y, 0.355), 0.007, ceramic))
        rack.append(cylinder_between(f"RackFoot_{len(rack)}", (-0.07, y, 0.006), (0.075, y, 0.006), 0.006, steel))
    rack.append(cylinder_between("RackRail", (0.04, -0.20, 0.35), (0.04, 0.20, 0.35), 0.008, blue))
    for slot in (-0.14, 0.0, 0.14):
        for side in (-0.024, 0.024):
            rack.append(
                cylinder_between(
                    f"RackFork_{len(rack)}", (-0.05, slot + side, 0.35), (0.04, slot + side, 0.35), 0.004, blue
                )
            )
    bpy.context.view_layer.update()

    scene_path = OUT / "lab_002_glassware.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    stage.GetRootLayer().subLayerPaths = ["../lab_002.usd"]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for mat in (glass, ceramic, steel, cork, plastic, blue):
        usd_mat = UsdShade.Material.Define(stage, "/World/Level2Materials/" + mat.name)
        shader = UsdShade.Shader.Define(stage, str(usd_mat.GetPath()) + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*mat.diffuse_color[:3]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15 if mat == glass else 0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8 if mat == steel else 0.0)
        if mat == glass:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.22)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.47)
        usd_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for name in ("round_bottom_flask", "round_bottom_flask_01"):
        path = replace_root(stage, name, dynamic=name == "round_bottom_flask")
        write_mesh(stage, path + "/GlassShell", flask)
        # The exterior sphere proxy must not determine the hollow vessel's COM.
        centroid = shell_center_of_mass(flask)
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)).CreateCenterOfMassAttr(Gf.Vec3f(*centroid))
        print(f"{name} glass centroid: {tuple(centroid)}")
        # A spherical bulb and separate convex neck sectors retain the opening.
        sphere = UsdGeom.Sphere.Define(stage, path + "/BulbCollision")
        sphere.CreateRadiusAttr(0.04)
        sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.04))
        sphere.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        for i in range(24):
            a, b = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24
            vertices = [
                (r * math.cos(t), r * math.sin(t), z) for z in (0.076, 0.144) for r in (0.0125, 0.014) for t in (a, b)
            ]
            piece = mesh_object(
                f"NeckCollision_{name}_{i}",
                vertices,
                [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)],
                glass,
            )
            mesh = write_mesh(stage, path + f"/NeckCollision_{i}", piece, "convexHull")
            mesh.CreateVisibilityAttr("invisible")
            bpy.data.objects.remove(piece, do_unlink=True)
    path = replace_root(stage, "cork_ring")
    write_mesh(stage, path + "/Seat", ring, "none")
    seat = UsdGeom.Xform.Define(stage, path + "/FlaskSeat")
    seat.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.026 + math.sqrt(0.04**2 - 0.026**2) + 0.0325))
    path = replace_root(stage, "pipeclay_triangle")
    for obj in stand:
        write_mesh(stage, path + "/" + obj.name, obj, "convexHull")
    seat = UsdGeom.Xform.Define(stage, path + "/FlaskSeat")
    seat.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.085 + math.sqrt(0.046**2 - 0.0325**2) + 0.0325))

    path = replace_root(stage, "pipette_01", dynamic=True)
    root = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    root.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
    write_mesh(stage, path + "/mesh_003", pipette).GetPrim().SetActive(True)
    write_mesh(stage, path + "/mesh_004", tip, "convexHull").GetPrim().SetActive(True)
    for name, radius, bottom, top in (
        ("Body", 0.014, 0.17, 0.295),
        ("Collar", 0.030, 0.295, 0.307),
        ("Head", 0.015, 0.307, 0.365),
    ):
        shape = UsdGeom.Cylinder.Define(stage, path + "/" + name + "Collision")
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(top - bottom)
        shape.AddTranslateOp().Set(Gf.Vec3d(0, 0, (top + bottom) / 2))
        shape.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        shape.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        shape.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    grip_material = UsdShade.Material.Define(stage, "/World/Level2Materials/PipetteGrip")
    physics = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
    physics.CreateStaticFrictionAttr(1.0)
    physics.CreateDynamicFrictionAttr(0.8)
    physics.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
        grip_material, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
    # Keep the existing observation paths, with an occupied reference fork.
    path = replace_root(stage, "pipette_rack")
    for obj in rack:
        write_mesh(stage, path + "/" + obj.name, obj, "convexHull")
    group = UsdGeom.Xform.Define(stage, path + "/group_000")
    group.GetPrim().SetActive(True)
    group.ClearXformOpOrder()
    # Collar underside (z=.295) rests on the fork top (z=.354).
    group.AddTranslateOp().Set(Gf.Vec3d(-0.015, -0.14, 0.059))
    write_mesh(stage, path + "/group_000/mesh_000", pipette, "none")
    write_mesh(stage, path + "/group_000/mesh_001", tip, "none")
    stage.OverridePrim(path + "/group_000/mesh_002").SetActive(False)
    stage.GetRootLayer().Save()

    # An editable assembly and a studio preview, separate from simulation poses.
    flask.location = (0, 0, 0.074)
    ring.location = (0.18, 0, 0)
    # Keep the pipette assembly in the blend without obscuring the flask preview.
    for obj in [pipette, tip, *rack]:
        obj.location.x += 0.60
    for obj in (pipette, tip):
        obj.location += Vector((-0.015, -0.14, 0.059))
    bpy.ops.mesh.primitive_plane_add(size=200)
    floor = bpy.context.object
    floor.name = "PreviewFloor"
    floor.data.materials.append(material("PreviewFloor", (0.15, 0.18, 0.20)))
    bpy.ops.object.camera_add(location=(0.38, -0.55, 0.34))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0.065, 0, 0.09)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.39
    bpy.context.scene.camera = camera
    for location, energy, size in [((0, -0.2, 0.5), 35, 0.4), ((0.2, 0.3, 0.35), 50, 0.3)]:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy, light.data.shape, light.data.size = energy, "DISK", size
        light.rotation_euler = (Vector((0, 0, 0.07)) - light.location).to_track_quat("-Z", "Y").to_euler()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 1000, 800
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(ROOT / "outputs/level2_mesh_validation/glassware_preview.png")
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "glassware.blend"))
    bpy.ops.render.render(write_still=True)
    camera.location = (0.13, -0.65, 0.53)
    camera.rotation_euler = (Vector((0.60, 0, 0.20)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = 0.64
    scene.render.filepath = str(ROOT / "outputs/level2_mesh_validation/pipette_preview.png")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
