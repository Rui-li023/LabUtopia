"""Reference-guided pipette restoration and acrylic stand, built in Blender.

Run with Blender 4.3: blender -b --python experiments/assets/refine_pipette_assets.py
See the generated asset directory's REFERENCES.md for measured/estimated sizes.
"""

import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--length-mm", type=float, default=220.0)
parser.add_argument("--skip-render", action="store_true")
parser.add_argument("--stand-profile", choices=("heathrow", "tall", "tall_wide", "tall_four"), default="heathrow")
parser.add_argument("--lower-slot-mm", type=float, default=None)
options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
if not 180 <= options.length_mm <= 320:
    parser.error("--length-mm must be within 180..320 mm")
BODY_SCALE = options.length_mm / 220.0
variant = "realistic_pipette" if options.length_mm == 220 else f"realistic_pipette_{options.length_mm:g}mm"
TALL = options.stand_profile != "heathrow"
WIDE = options.stand_profile == "tall_wide"
FOUR = options.stand_profile == "tall_four"
if TALL:
    variant += "_" + options.stand_profile
if options.lower_slot_mm is not None:
    if not 20 <= options.lower_slot_mm <= 36:
        parser.error("--lower-slot-mm must be within 20..36")
    variant += f"_guide{options.lower_slot_mm:g}"
OUT = ROOT / "assets/chemistry_lab/lab_002" / variant
PREVIEW = ROOT / "outputs" / ("level2_realism" if BODY_SCALE == 1 else f"pipette_resize_{options.length_mm:g}mm")
if TALL:
    PREVIEW = ROOT / "outputs/pipette_reliability/tall_asset"
STAND_HEIGHT = 0.285 if TALL else 0.206
STAND_WIDTH = 0.245 if TALL and not WIDE else 0.296
SLOT_PITCH = 0.038 if TALL and not WIDE else 0.045
SLOT_COUNT = 4 if FOUR else 6
if FOUR:
    STAND_WIDTH = 0.256
    SLOT_PITCH = 0.065
UPPER_SLOT = 0.033 if TALL else 0.028
LOWER_SLOT = 0.031 if TALL else 0.020
if options.lower_slot_mm is not None:
    LOWER_SLOT = options.lower_slot_mm / 1000
SOURCE = ROOT / "assets/chemistry_lab/lab_002/lab_002.usd"
TEXTURES = ROOT / "assets/chemistry_lab/lab_002/SubUSDs/Pipette/Materials/folder_0003"


def bitmap(name, pixels, non_color=False):
    height, width = pixels.shape[:2]
    image = bpy.data.images.new(name, width=width, height=height, alpha=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    rgba = np.ones((height, width, 4), dtype=np.float32)
    if pixels.ndim == 2:
        rgba[:, :, :3] = pixels[:, :, None]
    else:
        rgba[:, :, : pixels.shape[2]] = pixels
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = str(OUT / "textures" / f"{name}.png")
    image.file_format = "PNG"
    image.save()
    return image


def material(name, color, roughness, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*color, 1)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = 1.49
    return mat


def texture_node(mat, image, socket):
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    mat.node_tree.links.new(node.outputs["Color"], mat.node_tree.nodes.get("Principled BSDF").inputs[socket])
    return node


def materials():
    body = material("Pipette_ABS", (0.82, 0.84, 0.85), 0.32)
    base = bpy.data.images.load(str(TEXTURES / "image_0008.jpg"), check_existing=True)
    pixels = np.array(base.pixels[:], dtype=np.float32).reshape(base.size[1], base.size[0], 4)
    # Restore the mechanical counter's pale digits, not the legacy purple tint.
    purple = (
        (pixels[:, :, 2] > pixels[:, :, 1] * 1.8) & (pixels[:, :, 0] > pixels[:, :, 1] * 1.8) & (pixels[:, :, 0] > 0.08)
    )
    pixels[purple, :3] = (0.78, 0.79, 0.76)
    base = bitmap("pipette_basecolor", pixels)
    texture_node(body, base, "Base Color")
    gloss = bpy.data.images.load(str(TEXTURES / "image_0009.jpg"), check_existing=True)
    gloss.colorspace_settings.name = "Non-Color"
    values = np.asarray(gloss.pixels[:], dtype=np.float32).reshape(gloss.size[1], gloss.size[0], 4)
    # The legacy map was a reflection-glossiness input, not a roughness map.
    rough = bitmap("pipette_roughness", np.clip(1 - values[:, :, 0], 0.20, 0.52), True)
    texture_node(body, rough, "Roughness")
    normal = bpy.data.images.load(str(TEXTURES / "image_0011.jpg"), check_existing=True)
    normal.colorspace_settings.name = "Non-Color"
    _ = normal.pixels[0]
    normal.filepath_raw = str(OUT / "textures/pipette_normal.png")
    normal.file_format = "PNG"
    normal.save()
    tex = body.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = normal
    normal_node = body.node_tree.nodes.new("ShaderNodeNormalMap")
    normal_node.inputs["Strength"].default_value = 0.15
    body.node_tree.links.new(tex.outputs["Color"], normal_node.inputs["Color"])
    body.node_tree.links.new(
        normal_node.outputs["Normal"], body.node_tree.nodes.get("Principled BSDF").inputs["Normal"]
    )

    # Stand dimensions must not silently replace the requested transparent finish.
    acrylic = material("Clear_PMMA", (0.97, 0.99, 1.0), 0.07, 1.0)
    rng = np.random.default_rng(37)
    size = 2048
    heights = rng.normal(0, 0.002, (size, size)).astype(np.float32)
    for _ in range(150):
        x, y = rng.integers(0, size - 160, 2)
        length = int(rng.integers(12, 160))
        heights[y, x : x + length] += rng.uniform(0.01, 0.035)
    roughness = bitmap("acrylic_roughness", np.clip(0.07 + heights, 0.04, 0.35), True)
    texture_node(acrylic, roughness, "Roughness")
    dy, dx = np.gradient(heights)
    normals = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_image = bitmap("acrylic_normal", normals * 0.5 + 0.5, True)
    tex = acrylic.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = normal_image
    node = acrylic.node_tree.nodes.new("ShaderNodeNormalMap")
    acrylic.node_tree.links.new(tex.outputs["Color"], node.inputs["Color"])
    acrylic.node_tree.links.new(node.outputs["Normal"], acrylic.node_tree.nodes.get("Principled BSDF").inputs["Normal"])
    return body, acrylic, material("Silicone_Feet", (0.72, 0.75, 0.76), 0.65, 0.12)


def restore_pipette(mat):
    stage = Usd.Stage.Open(str(SOURCE))
    path = "/World/pipette_01/mesh_003"
    prim = stage.GetPrimAtPath(path)
    usd_mesh = UsdGeom.Mesh(prim)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    points = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in usd_mesh.GetPointsAttr().Get()])
    # Source asset is tilted on the table. PCA aligns its long body axis upright;
    # tip-side mesh identifies the sign without guessing from the USD rotation.
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points - center).T))
    direction = axes[:, -1]
    tip = stage.GetPrimAtPath("/World/pipette_01/mesh_004")
    box = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(tip).ComputeAlignedRange()
    tip_center = np.asarray(box.GetMidpoint())
    if np.dot(direction, center - tip_center) < 0:
        direction = -direction
    rotation = Vector(direction).rotation_difference(Vector((0, 0, 1)))
    points = np.array([rotation @ Vector(p - center) for p in points])
    points[:, 2] -= points[:, 2].min()
    points *= 0.220 / np.ptp(points[:, 2])
    points[:, :2] -= (points[:, :2].max(axis=0) + points[:, :2].min(axis=0)) / 2
    indices = list(usd_mesh.GetFaceVertexIndicesAttr().Get())
    faces, cursor = [], 0
    for count in usd_mesh.GetFaceVertexCountsAttr().Get():
        faces.append(indices[cursor : cursor + count])
        cursor += count
    mesh = bpy.data.meshes.new("Original_pipette_with_UV")
    mesh.from_pydata(points, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="st")
    coords = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").ComputeFlattened()
    if len(coords) != len(mesh.loops):
        raise ValueError("Original pipette UV topology does not match its faces")
    for loop, coord in zip(uv.data, coords):
        loop.uv = coord
    # The imported USD splits vertices at UV seams. Weld coincident geometry
    # while preserving per-loop UVs before smoothing the injection-moulded shell.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Manual_pipette_10_100uL", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new("Manufactured_edge_radius", "BEVEL")
    modifier.width = 0.00025
    modifier.segments = 3
    modifier.limit_method = "ANGLE"
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    subdivision = obj.modifiers.new("Moulded_shell_smoothing", "SUBSURF")
    subdivision.levels = 2
    bpy.ops.object.modifier_apply(modifier=subdivision.name)
    # Fine moulded grip ribs on the rotating plunger cap, visible in the photos.
    coords = np.array([v.co[:] for v in obj.data.vertices])
    top = float(coords[:, 2].max())
    cap = coords[coords[:, 2] > top - 0.005]
    center_xy = (cap[:, :2].min(axis=0) + cap[:, :2].max(axis=0)) / 2
    radius = float(np.ptp(cap[:, :2], axis=0).max() / 2)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.get("st")
    rings = []
    for z in (top - 0.0048, top - 0.0005):
        ring = []
        for i in range(160):
            angle = 2 * math.pi * i / 160
            r = radius + (0.00015 if i % 4 in (1, 2) else -0.00005)
            ring.append(bm.verts.new((center_xy[0] + r * math.cos(angle), center_xy[1] + r * math.sin(angle), z)))
        rings.append(ring)
    for i in range(160):
        following = (i + 1) % 160
        face = bm.faces.new((rings[0][i], rings[0][following], rings[1][following], rings[1][i]))
        face.smooth = True
        for loop in face.loops:
            loop[uv_layer].uv = (0.75, 0.90)
    bm.to_mesh(obj.data)
    bm.free()
    # Uniformly resize the finished visible mesh, preserving all proportions,
    # UVs and details. Export cooks collision from these same scaled vertices.
    for vertex in obj.data.vertices:
        vertex.co *= BODY_SCALE
    obj.select_set(False)
    return obj


def block(name, size, location, mat, bevel=0.0006):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("Polished_edge", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return obj


def slotted_shelf(name, z, front, width, mat):
    rear = 0.040 if TALL else 0.060
    obj = block(name, (rear - front, STAND_WIDTH, 0.005 if TALL else 0.004), ((front + rear) / 2, 0, z), mat, bevel=0)
    # Tall slot dimensions are engineering fit estimates, not manufacturer CAD.
    for y in (np.arange(SLOT_COUNT) - (SLOT_COUNT - 1) / 2) * SLOT_PITCH:
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=width / 2, depth=0.025, location=(front + 0.019, y, z))
        cutter = bpy.context.object
        box = block("Slot_mouth", (0.05, width, 0.025), (front - 0.006, y, z), mat, bevel=0)
        for part in (cutter, box):
            bpy.context.view_layer.objects.active = obj
            modifier = obj.modifiers.new("Open_slot", "BOOLEAN")
            modifier.operation = "DIFFERENCE"
            modifier.solver = "EXACT"
            modifier.object = part
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(part, do_unlink=True)
    bpy.context.view_layer.objects.active = obj
    bevel = obj.modifiers.new("Polished_slot_edges", "BEVEL")
    bevel.width = 0.0005
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    return obj


def build_stand(acrylic, rubber):
    depth = 0.115 if TALL else 0.172
    parts = [block("Base_plate", (depth, STAND_WIDTH, 0.004), (0, 0, 0.005), acrylic)]
    back = block(
        "Inclined_back",
        (0.005 if TALL else 0.004, STAND_WIDTH, 0.282 if TALL else 0.197),
        (0.005 if TALL else 0.054, 0, 0.145 if TALL else 0.105),
        acrylic,
    )
    back.rotation_euler[1] = math.radians(12 if TALL else -8)
    parts += [
        back,
        slotted_shelf(
            "Upper_33mm_slots" if TALL else "Upper_28mm_slots",
            STAND_HEIGHT - (0.0025 if TALL else 0.002),
            -0.038,
            UPPER_SLOT,
            acrylic,
        ),
        slotted_shelf(f"Lower_{LOWER_SLOT * 1000:g}mm_slots", 0.180 if TALL else 0.098, -0.055, LOWER_SLOT, acrylic),
    ]
    for x in (-depth / 2 + 0.012, depth / 2 - 0.012):
        for y in (-STAND_WIDTH / 2 + 0.013, STAND_WIDTH / 2 - 0.013):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1, location=(x, y, 0.002))
            obj = bpy.context.object
            obj.name = f"Foot_{len(parts)}"
            obj.scale = (0.005, 0.005, 0.002)
            obj.data.materials.append(rubber)
            parts.append(obj)
    for obj in parts:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
    return parts


def usd_material(stage, mat):
    path = "/World/RealisticPipetteMaterials/" + mat.name
    usd_mat = UsdShade.Material.Define(stage, path)
    preview_shader = UsdShade.Shader.Define(stage, path + "/Preview")
    preview_shader.CreateIdAttr("UsdPreviewSurface")
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    preview_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*mat.diffuse_color[:3]))
    preview_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(bsdf.inputs["Roughness"].default_value)
    usd_mat.CreateSurfaceOutput().ConnectToSource(preview_shader.ConnectableAPI(), "surface")
    uv = UsdShade.Shader.Define(stage, path + "/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    mdl = UsdShade.Shader.Define(stage, path + "/MDL")
    is_glass = mat.name == "Clear_PMMA"
    mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl" if is_glass else "OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniGlass" if is_glass else "OmniPBR", "mdl")
    for output in (
        usd_mat.CreateSurfaceOutput("mdl"),
        usd_mat.CreateVolumeOutput("mdl"),
        usd_mat.CreateDisplacementOutput("mdl"),
    ):
        output.ConnectToSource(mdl.ConnectableAPI(), "out")
    if is_glass:
        mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.49)
        mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        mdl.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(1.0)
        mdl.CreateInput("normal_map_strength", Sdf.ValueTypeNames.Float).Set(0.05)
        mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
        mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.07)
        preview_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.15)
        preview_shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.49)
        maps = [
            ("acrylic_roughness", "roughness", "roughness_texture"),
            ("acrylic_normal", "normal", "normal_map_texture"),
        ]
    else:
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*mat.diffuse_color[:3]))
        mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
            bsdf.inputs["Roughness"].default_value
        )
        maps = []
        if mat.name == "Pipette_ABS":
            maps = [
                ("pipette_basecolor", "diffuseColor", "diffuse_texture"),
                ("pipette_roughness", "roughness", "reflectionroughness_texture"),
                ("pipette_normal", "normal", "normalmap_texture"),
            ]
            mdl.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
            mdl.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.15)
    for name, socket, mdl_input in maps:
        asset = Sdf.AssetPath("./textures/" + name + ".png")
        mdl.CreateInput(mdl_input, Sdf.ValueTypeNames.Asset).Set(asset)
        tex = UsdShade.Shader.Define(stage, path + "/" + name)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(asset)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if socket == "diffuseColor" else "raw")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        if socket == "normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        kind = (
            Sdf.ValueTypeNames.Float
            if socket == "roughness"
            else Sdf.ValueTypeNames.Normal3f
            if socket == "normal"
            else Sdf.ValueTypeNames.Color3f
        )
        preview_shader.CreateInput(socket, kind).ConnectToSource(
            tex.ConnectableAPI(), "r" if socket == "roughness" else "rgb"
        )
    return usd_mat


def export_mesh(stage, path, obj, collision):
    mesh = obj.data
    mesh.calc_loop_triangles()
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    points = [tuple(obj.matrix_world @ v.co) for v in mesh.vertices]
    usd_mesh.CreatePointsAttr(points)
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.loop_triangles))
    usd_mesh.CreateFaceVertexIndicesAttr([i for t in mesh.loop_triangles for i in t.vertices])
    usd_mesh.CreateSubdivisionSchemeAttr("none")
    usd_mesh.CreateDoubleSidedAttr(True)
    usd_mesh.CreateExtentAttr(UsdGeom.PointBased(usd_mesh).ComputeExtent(points))
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    normals = [
        tuple((normal_matrix @ mesh.corner_normals[i].vector).normalized())
        for t in mesh.loop_triangles
        for i in t.loops
    ]
    usd_mesh.CreateNormalsAttr(normals)
    usd_mesh.SetNormalsInterpolation("faceVarying")
    coords = [tuple(mesh.uv_layers.active.data[i].uv) for t in mesh.loop_triangles for i in t.loops]
    UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(coords)
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath("/World/RealisticPipetteMaterials/" + obj.data.materials[0].name))
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
        usd_mesh.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim()).CreateApproximationAttr(collision)
        usd_mesh.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0005)
        usd_mesh.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    return usd_mesh


def root_override(stage, name, dynamic=False):
    prim = stage.GetPrimAtPath("/World/" + name)
    for child in prim.GetChildren():
        stage.OverridePrim(child.GetPath()).SetActive(False)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translation = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
    xform.SetXformOpOrder([translation])
    # Staged assets must not overlap during the world's initial physics warmup.
    translation.Set(Gf.Vec3d(-2 if dynamic else 2, 0, -2))
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(0.09)
        mass.CreateDensityAttr(0.0)
    return str(prim.GetPath())


def marker(stage, path, position):
    prim = UsdGeom.Xform.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*position))
    return prim


def export_scene(pipette, stand, mats):
    bpy.context.view_layer.update()
    stage = Usd.Stage.CreateNew(str(OUT / "lab_002_pipette.usda"))
    stage.GetRootLayer().subLayerPaths = ["../lab_002.usd"]
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    for mat in mats:
        usd_material(stage, mat)
    path = root_override(stage, "pipette_01", True)
    UsdGeom.Xformable(stage.GetPrimAtPath(path)).AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
    export_mesh(stage, path + "/Body", pipette, "convexDecomposition")
    body = stage.GetPrimAtPath(path + "/Body")
    body.CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.003)
    body.CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.001)
    body.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
    for name, kind, value in (
        ("errorPercentage", Sdf.ValueTypeNames.Float, 0.5),
        ("minThickness", Sdf.ValueTypeNames.Float, 0.0001),
        ("maxConvexHulls", Sdf.ValueTypeNames.Int, 64),
        ("shrinkWrap", Sdf.ValueTypeNames.Bool, True),
    ):
        body.CreateAttribute("physxConvexDecompositionCollision:" + name, kind).Set(value)
    marker(stage, path + "/AxisBase", (0, 0, 0.02 * BODY_SCALE))
    marker(stage, path + "/AxisHead", (0, 0, 0.19 * BODY_SCALE))
    marker(stage, path + "/Grasp", (-0.006 * BODY_SCALE, 0, 0.190 * BODY_SCALE))
    marker(stage, path + "/Reference", (0, 0, 0.110 * BODY_SCALE))
    contact_mat = UsdShade.Material.Define(stage, "/World/RealisticPipetteMaterials/PipetteContact")
    contact = UsdPhysics.MaterialAPI.Apply(contact_mat.GetPrim())
    contact.CreateStaticFrictionAttr(0.9)
    contact.CreateDynamicFrictionAttr(0.7)
    contact.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(body).Bind(contact_mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    root = stage.GetPrimAtPath(path)
    root.CreateAttribute("reference:estimatedLength", Sdf.ValueTypeNames.Float).Set(options.length_mm / 1000)
    rack = root_override(stage, "pipette_rack")
    for obj in stand:
        export_mesh(stage, rack + "/" + obj.name, obj, "none")
    # Seat pose is a fit estimate from the restored finger rest, not a new collar.
    # The stand is NOT scaled with the tool. This is the geometric finger-rest
    # contact target; a larger tool must independently pass a physical fit test.
    seat_height = STAND_HEIGHT - (0.206 - 0.126) * BODY_SCALE
    marker(stage, rack + "/PipetteSeat", (-0.033, (SLOT_COUNT - 1) / 2 * SLOT_PITCH, seat_height))
    for index in range(SLOT_COUNT):
        marker(stage, rack + f"/Seat{index}", (-0.033, (index - (SLOT_COUNT - 1) / 2) * SLOT_PITCH, seat_height))
    marker(stage, rack + "/AxisBase", (-0.019, 0.1125, 0.02))
    marker(stage, rack + "/AxisHead", (-0.019, 0.1125, 0.19))
    stage.GetPrimAtPath(rack).CreateAttribute("reference:slotWidths", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(UPPER_SLOT, LOWER_SLOT)
    )
    stage.GetPrimAtPath(rack).CreateAttribute("reference:slotPitch", Sdf.ValueTypeNames.Float).Set(SLOT_PITCH)
    stage.GetPrimAtPath(rack).CreateAttribute("reference:slotCount", Sdf.ValueTypeNames.Int).Set(SLOT_COUNT)
    stage.GetRootLayer().Save()


def preview(pipette, stand):
    floor_mat = material("Studio_surface", (0.13, 0.16, 0.18), 0.48)
    block("Studio_floor", (200, 200, 0.01), (0, 0, -0.009), floor_mat)
    pipette.location = (-0.26, 0, 0.006)
    for location, power, size in [
        ((-0.3, -0.4, 0.65), 30, 0.55),
        ((0.25, 0.35, 0.5), 45, 0.35),
        ((0.45, -0.2, 0.25), 15, 0.30),
    ]:
        bpy.ops.object.light_add(type="AREA", location=location)
        lamp = bpy.context.object
        lamp.data.energy, lamp.data.size = power, size
        lamp.rotation_euler = (Vector((-0.05, 0, 0.12)) - lamp.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.object.camera_add(location=(-0.64, -0.62, 0.43))
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector((-0.08, 0, 0.15 if TALL else 0.11)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 0.64 if TALL else 0.57
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1100
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(PREVIEW / "reference_guided_preview.png")
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "pipette_and_stand.blend"))
    bpy.ops.file.make_paths_relative()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "pipette_and_stand.blend"))
    if options.skip_render:
        return
    bpy.ops.render.render(write_still=True)
    camera.location = (-0.5, -0.25, 0.16)
    camera.rotation_euler = (
        (Vector((-0.26, 0, 0.125 * BODY_SCALE)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.ortho_scale = 0.25 * BODY_SCALE
    scene.render.resolution_x, scene.render.resolution_y = 1000, 1500
    scene.render.filepath = str(PREVIEW / "pipette_detail.png")
    bpy.ops.render.render(write_still=True)


def main():
    (OUT / "textures").mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    body, acrylic, rubber = materials()
    pipette = restore_pipette(body)
    stand = build_stand(acrylic, rubber)
    export_scene(pipette, stand, (body, acrylic, rubber))
    preview(pipette, stand)


if __name__ == "__main__":
    main()
