"""Build a reference-inspired articulated rotary evaporator for Blender and Isaac Sim.

Run: blender -b --factory-startup --python experiments/assets/build_rotary_evaporator.py
All dimensions are estimated from the reference photograph, in metres.
"""

import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/chemistry_lab/rotary_evaporator"
PREVIEWS = ROOT / "outputs/rotary_evaporator/refined"
sys.path.insert(0, str(ROOT))
from experiments.assets.rotary_evaporator_calibration import rpm_to_dial_angle  # noqa: E402
from experiments.assets.rotary_evaporator_displays import build_displays  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
PREVIEWS.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for item in list(bpy.data.collections):
    if item.name != "Collection":
        bpy.data.collections.remove(item)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.length_unit = "METERS"
main = bpy.data.collections.get("Collection")
main.name = "Rotary Evaporator"
collections = {}
for name in [
    "01 Stand and lift",
    "02 Drive assembly",
    "03 Condenser",
    "04 Receiving flask",
    "05 Evaporating flask",
    "06 Heating bath",
    "07 Controls and markings",
    "Studio",
]:
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    collections[name] = col
current = collections["01 Stand and lift"]


def finish(obj, name, mat=None, smooth=True):
    obj.name = name
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    current.objects.link(obj)
    if mat:
        obj.data.materials.append(mat)
    if smooth and obj.type == "MESH":
        for face in obj.data.polygons:
            face.use_smooth = True
    return obj


def material(name, color, roughness=0.35, metallic=0, transmission=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    p = mat.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = roughness
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Transmission Weight"].default_value = transmission
    p.inputs["IOR"].default_value = 1.46 if transmission else 1.5
    mat.diffuse_color = (*color, 1)
    return mat


white = material("Ivory | powder-coated housing", (0.72, 0.75, 0.75), 0.31)
black = material("Graphite | textured polymer", (0.002, 0.003, 0.004), 0.58)
rubber = material("Rubber | grip and feet", (0.008, 0.011, 0.014), 0.6)
steel = material("Stainless steel | polished bath", (0.66, 0.70, 0.72), 0.19, 0.95)
aluminum = material("Brushed aluminum | lift rail", (0.54, 0.59, 0.61), 0.29, 0.85)
blue = material("Soft blue | controls", (0.13, 0.49, 0.72), 0.32)
cyan = material("Cyan | membrane buttons", (0.10, 0.62, 0.68), 0.4)
orange = material("Warm orange | brand", (0.75, 0.24, 0.07), 0.42)
ink = material("Dark gray | silkscreen", (0.006, 0.008, 0.010), 0.85)
ink.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = 0
scale_plate = material("Scale | matte anodized plate", (0.025, 0.032, 0.039), 0.75)
glass = material("Borosilicate | clear glass", (0.965, 0.994, 0.985), 0.045, 0, 1)
frosted = material("Ground glass | joints", (0.88, 0.95, 0.94), 0.22, 0, 0.8)
screen = material("Display | plum filter", (0.10, 0.023, 0.055), 0.22)
digits = material("Display | illuminated digits", (0.88, 0.27, 0.46), 0.28)
black.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = 0.12
p = digits.node_tree.nodes.get("Principled BSDF")
p.inputs["Emission Color"].default_value = (0.72, 0.15, 0.28, 1)
p.inputs["Emission Strength"].default_value = 0.35
mark = material("Glass | white etched markings", (0.76, 0.83, 0.83), 0.5)


def bevel(obj, amount=0.004, segments=3):
    mod = obj.modifiers.new("Soft manufactured edges", "BEVEL")
    mod.width = amount
    mod.segments = segments
    mod = obj.modifiers.new("Weighted surface normals", "WEIGHTED_NORMAL")
    return obj


def box(name, loc, size, mat, radius=0.004):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    finish(obj, name, mat, False)
    if radius:
        bevel(obj, radius, 5)
    return obj


def cylinder(name, loc, radius, depth, mat, direction=(0, 0, 1), vertices=96, edge=0.001):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = finish(bpy.context.object, name, mat)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector(direction).to_track_quat("Z", "Y")
    if edge:
        bevel(obj, edge)
    return obj


def between(name, a, b, radius, mat, edge=0.001):
    a, b = Vector(a), Vector(b)
    return cylinder(name, (a + b) / 2, radius, (b - a).length, mat, b - a, edge=edge)


def torus(name, loc, radius, tube, mat, direction=(0, 0, 1)):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=radius, minor_radius=tube, major_segments=96, minor_segments=12, location=loc
    )
    obj = finish(bpy.context.object, name, mat)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector(direction).to_track_quat("Z", "Y")
    return obj


def lathe(name, profile, loc, mat, direction=(0, 0, 1), wall=0, segments=128):
    verts, faces = [], []
    for r, z in profile:
        for i in range(segments):
            a = i * 2 * math.pi / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for j in range(len(profile) - 1):
        for i in range(segments):
            a = j * segments + i
            b = j * segments + (i + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    current.objects.link(obj)
    obj.location = loc
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector(direction).to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    for face in mesh.polygons:
        face.use_smooth = True
    if wall:
        mod = obj.modifiers.new("Real glass wall thickness", "SOLIDIFY")
        mod.thickness = wall
        mod.offset = -1
        mod.use_even_offset = True
    return obj


def tube(name, points, radius, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 16
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    obj = bpy.data.objects.new(name, curve)
    current.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


FONT_REGULAR = bpy.data.fonts.load("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf")
FONT_BOLD = bpy.data.fonts.load("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf")
FONT_REGULAR.pack()
FONT_BOLD.pack()


def text(name, body, loc, size, mat, align="LEFT"):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.font = FONT_REGULAR
    curve.size = size
    curve.align_x = align
    curve.extrude = 0.000012
    curve.resolution_u = 6
    obj = bpy.data.objects.new(name, curve)
    current.objects.link(obj)
    obj.location = loc
    obj.rotation_euler = (math.pi / 2, 0, 0)
    curve.materials.append(mat)
    return obj


def grip(name, center, radius, length, direction=(0, 0, 1), ribs=48):
    center, axis = Vector(center), Vector(direction).normalized()
    cylinder(name, center, radius, length, black, axis)
    q = axis.to_track_quat("Z", "Y")
    for i in range(ribs):
        a = i * 2 * math.pi / ribs
        offset = q @ Vector((radius * math.cos(a), radius * math.sin(a), 0))
        cylinder(name + " | knurl", center + offset, 0.00125, length * 0.86, rubber, axis, vertices=8, edge=0.00035)


# Wide cast support foot: a curved U-shaped footprint with two projecting toes.
outline = [
    (-0.225, 0.11),
    (-0.205, 0.15),
    (0.075, 0.15),
    (0.113, 0.11),
    (0.118, -0.13),
    (0.096, -0.21),
    (0.053, -0.22),
    (0.020, -0.20),
    (0.019, -0.15),
    (0.032, -0.08),
    (0.016, -0.045),
    (-0.032, -0.036),
    (-0.091, -0.055),
    (-0.147, -0.116),
    (-0.21, -0.139),
    (-0.242, -0.124),
    (-0.246, -0.10),
    (-0.211, -0.073),
    (-0.126, -0.003),
    (-0.16, 0.026),
    (-0.209, 0.061),
]
verts = [(x, y, z) for z in [0.013, 0.031] for x, y in outline]
n = len(outline)
faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
faces += [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
mesh = bpy.data.meshes.new("Cast support foot mesh")
mesh.from_pydata(verts, [], faces)
obj = bpy.data.objects.new("Stand | broad curved foot", mesh)
current.objects.link(obj)
obj.data.materials.append(white)
bevel(obj, 0.009, 5)
for x, y in [(-0.211, 0.078), (-0.20, -0.103), (0.076, -0.176), (0.064, 0.107)]:
    cylinder("Stand | rubber foot", (x, y, 0.01), 0.016, 0.017, rubber)
box("Lift | black column", (0.035, 0.087, 0.342), (0.062, 0.063, 0.615), black, 0.012)
box("Lift | exposed aluminum rail", (0.072, 0.055, 0.369), (0.012, 0.029, 0.552), aluminum, 0.003)
box("Lift | narrow guide track", (0.083, 0.046, 0.372), (0.006, 0.011, 0.524), black, 0.001)
box("Lift | top end cap", (0.039, 0.087, 0.651), (0.073, 0.073, 0.024), black, 0.007)
box("Lift | sliding carriage", (0.024, 0.057, 0.472), (0.088, 0.089, 0.135), black, 0.012)
# A transverse pin and double-shear clevis provide a real tilt hinge.
for y in [0.064, 0.105]:
    box("Hinge | fixed clevis cheek", (-0.067, y, 0.513), (0.081, 0.012, 0.084), black, 0.014)
    between("Hinge | carriage bridge", (-0.067, y, 0.48), (0.027, y, 0.48), 0.017, black)
between("Hinge | transverse steel pin", (-0.067, 0.042, 0.513), (-0.067, 0.125, 0.513), 0.013, steel)
cylinder("Hinge | outer bearing", (-0.067, 0.119, 0.513), 0.033, 0.009, aluminum, (0, 1, 0))
grip("Hinge | tilt locking wheel", (-0.067, 0.132, 0.513), 0.030, 0.017, (0, 1, 0), 40)
for i in range(7):
    a = math.radians(-35 + i * 10)
    x = -0.067 + 0.041 * math.sin(a)
    z = 0.513 + 0.041 * math.cos(a)
    tick = box("Hinge | engraved angle tick", (x, 0.112, z), (0.0015, 0.0008, 0.005), mark, 0.0002)
    tick.rotation_euler[1] = a
between("Lift lock | shaft", (0.078, 0.071, 0.489), (0.118, 0.071, 0.489), 0.007, steel)
grip("Lift lock | thumbwheel", (0.12, 0.071, 0.489), 0.023, 0.018, (1, 0, 0), 32)
cylinder("Lift lock | center cap", (0.131, 0.071, 0.489), 0.008, 0.002, aluminum, (1, 0, 0))
between("Lift lever | stem", (0.035, 0.052, 0.101), (-0.004, -0.045, 0.077), 0.008, black)
lever = box("Lift lever | foot paddle", (-0.009, -0.064, 0.074), (0.041, 0.068, 0.024), black, 0.012)
lever.rotation_euler[0] = -0.35

# Inclined motor and the dark, ribbed rotary couplings.
current = collections["02 Drive assembly"]
axis = Vector((0.8660254, 0, -0.5))
center = Vector((-0.067, -0.012, 0.513))


def on(t):
    return center + axis * t


lathe(
    "Drive | ivory motor shell",
    [(0.044, -0.067), (0.052, -0.061), (0.064, -0.038), (0.067, 0.018), (0.06, 0.06), (0.049, 0.067)],
    center,
    white,
    axis,
)
cylinder("Drive | back sealing plate", on(-0.066), 0.046, 0.010, black, axis)
grip("Drive | glass joint retaining nut", on(-0.088), 0.047, 0.032, axis)
cylinder("Drive | front bearing collar", on(0.062), 0.051, 0.014, black, axis)
torus("Drive | blue release ring", on(0.072), 0.046, 0.004, blue, axis)
grip("Drive | rotation grip", on(0.083), 0.043, 0.020, axis, 64)
cylinder("Drive | spindle nose", on(0.106), 0.026, 0.028, white, axis)
lathe("Drive | hollow vapor shaft", [(0.015, 0), (0.015, 0.053)], on(0.117), frosted, axis, wall=0.002)
cylinder("Hinge | moving trunnion", (-0.067, 0.047, 0.513), 0.024, 0.02, aluminum, (0, 1, 0))
# Vent slots across the side of the motor cover.
for i in range(7):
    vent = box(
        "Drive | ventilation slot",
        (-0.065 + i * 0.006, -0.075, 0.518 - i * 0.0035),
        (0.003, 0.0015, 0.023),
        black,
        0.001,
    )
    vent.rotation_euler[1] = math.radians(30)

# Tall control head mounted above the rotary motor.
current = collections["07 Controls and markings"]
box("Controller | body", (-0.023, 0.006, 0.626), (0.112, 0.072, 0.147), white, 0.022)
box("Controller | face plate", (-0.043, -0.035, 0.626), (0.065, 0.005, 0.136), white, 0.009)
text("Controller | logo", "EYELA", (-0.070, -0.0385, 0.677), 0.015, orange)
text("Controller | model", "ROTARY EVAPORATOR", (-0.07, -0.0385, 0.665), 0.0037, ink)
for x, mat, label in [(-0.061, black, "POWER"), (-0.035, cyan, "RUN")]:
    cylinder("Controller | " + label, (x, -0.040, 0.649), 0.008, 0.003, mat, (0, -1, 0))
    text("Controller | " + label + " label", label, (x, -0.042, 0.647), 0.0028, white, "CENTER")
box("Controller | LCD bezel", (-0.049, -0.040, 0.622), (0.047, 0.003, 0.024), aluminum, 0.002)
box("Controller | LCD glass", (-0.049, -0.042, 0.622), (0.041, 0.001, 0.018), screen, 0.001)
text("Controller | RPM readout", "120", (-0.067, -0.043, 0.615), 0.018, digits)
text("Controller | RPM label", "rpm", (-0.026, -0.043, 0.612), 0.0045, ink)
text("Controller | speed label", "ROTATION", (-0.070, -0.039, 0.600), 0.0045, ink)
cylinder("Controller | knob bezel", (-0.049, -0.042, 0.578), 0.023, 0.006, aluminum, (0, -1, 0))
cylinder("Controller | speed dial", (-0.049, -0.049, 0.578), 0.020, 0.014, blue, (0, -1, 0))
cylinder("Controller | speed dial inset", (-0.049, -0.057, 0.578), 0.016, 0.001, blue, (0, -1, 0))
box("Controller | dial index", (-0.049, -0.058, 0.589), (0.0015, 0.0008, 0.0055), white, 0.0004)

# Leave clearance above the motor through the complete hinge range.
for obj in current.objects:
    obj.location.z += 0.065
box("Controller | support neck", (0.008, 0.035, 0.605), (0.042, 0.036, 0.110), white, 0.007)

# Condenser jacket with closed rounded shoulder and a visible helical coolant tube.
current = collections["03 Condenser"]
cx, cy = -0.205, 0.008
profile = [
    (0.019, 0.535),
    (0.021, 0.548),
    (0.037, 0.568),
    (0.045, 0.593),
    (0.047, 0.618),
    (0.047, 0.911),
    (0.046, 0.926),
    (0.04, 0.941),
    (0.029, 0.951),
    (0.016, 0.955),
    (0.016, 0.968),
]
lathe("Condenser | outer glass jacket", profile, (cx, cy, 0), glass, wall=0.0018)
lathe("Condenser | central vapor passage", [(0.011, 0.563), (0.011, 0.924)], (cx, cy, 0), glass, wall=0.0012)
coil = []
turns = 25
for i in range(3201):
    t = i / 3200
    a = t * turns * math.tau
    coil.append((cx + 0.030 * math.cos(a), cy + 0.030 * math.sin(a), 0.602 + t * 0.319))
tube("Condenser | continuous 25-turn cooling coil", coil, 0.0036, glass)
tube(
    "Condenser | coolant return",
    [
        (cx + 0.03, cy, 0.921),
        (cx + 0.026, cy + 0.012, 0.933),
        (cx + 0.012, cy + 0.025, 0.936),
        (cx, cy + 0.029, 0.931),
        (cx, cy + 0.032, 0.589),
    ],
    0.0036,
    glass,
)
grip("Condenser | top vacuum cap", (cx, cy, 0.974), 0.0175, 0.025, ribs=32)
torus("Condenser | top cap seal", (cx, cy, 0.958), 0.017, 0.0025, rubber)
# External hose barbs at the lower left of the condenser.
for z in [0.593, 0.626]:
    a = (cx - 0.034, cy + 0.008, z)
    b = (cx - 0.074, cy + 0.008, z + 0.024)
    between("Condenser | coolant sidearm", a, b, 0.008, glass)
    for t in [0.55, 0.73, 0.90]:
        pt = Vector(a).lerp(Vector(b), t)
        torus("Condenser | hose barb ridge", pt, 0.008, 0.0017, glass, Vector(b) - Vector(a))
# Main retaining band and the bracket joining it to the column.
lathe(
    "Condenser | support band",
    [(0.048, 0.835), (0.054, 0.835), (0.056, 0.839), (0.056, 0.86), (0.054, 0.864), (0.048, 0.864), (0.048, 0.835)],
    (cx, cy, 0),
    black,
)
for z in [0.839, 0.845, 0.851, 0.857, 0.862]:
    torus("Condenser | band rib", (cx, cy, z), 0.055, 0.0012, rubber)
tube(
    "Condenser | rear support arm",
    [(cx, 0.064, 0.849), (-0.15, 0.075, 0.849), (-0.15, 0.075, 0.544), (0.035, 0.095, 0.503)],
    0.006,
    aluminum,
)
box("Condenser | clamp block", (cx, 0.068, 0.849), (0.041, 0.024, 0.043), black, 0.005)
# Angled glass distribution manifold joins receiver, condenser, and motor.
start = Vector((-0.314, -0.012, 0.644))
end = on(-0.105)
direction = end - start
lathe(
    "Manifold | inclined glass body",
    [
        (0.014, 0),
        (0.014, 0.018),
        (0.025, 0.025),
        (0.028, 0.038),
        (0.028, direction.length - 0.027),
        (0.020, direction.length - 0.01),
        (0.020, direction.length),
    ],
    start,
    glass,
    direction,
    wall=0.0018,
)
lathe("Manifold | condenser lower junction", [(0.020, 0.532), (0.021, 0.58)], (cx, cy, 0), glass, wall=0.0016)
# Glass feed tap with a white T-handle.
between("Feed tap | glass inlet", (-0.313, -0.012, 0.644), (-0.343, -0.012, 0.661), 0.010, glass)
cylinder("Feed tap | PTFE plug", (-0.35, -0.012, 0.666), 0.009, 0.024, white, axis)
between("Feed tap | T crossbar", (-0.358, -0.012, 0.647), (-0.339, -0.012, 0.683), 0.0047, white)
between("Feed tap | drip outlet", (-0.297, -0.01, 0.627), (-0.309, -0.01, 0.601), 0.005, glass)

# Suspended receiving flask with a continuous hollow bulb and neck.
current = collections["04 Receiving flask"]
receiver = (-0.205, -0.012, 0.311)


def flask_profile(radius, neck_length):
    prof = [(0, -radius)]
    for i in range(60):
        a = math.pi - 0.025 - i * (math.pi - 0.39) / 59
        prof.append((radius * math.sin(a), radius * math.cos(a)))
    prof.extend(
        [
            (0.020, radius + 0.010),
            (0.0165, radius + 0.025),
            (0.0165, radius + neck_length),
            (0.021, radius + neck_length + 0.002),
            (0.021, radius + neck_length + 0.008),
            (0.0165, radius + neck_length + 0.009),
        ]
    )
    return prof


lathe("Receiver | round-bottom flask 1 L", flask_profile(0.084, 0.026), receiver, glass, wall=0.0016)
lathe(
    "Receiver | upper ground joint",
    [(0.0165, 0.415), (0.018, 0.427), (0.018, 0.48), (0.023, 0.495), (0.023, 0.535)],
    (-0.205, -0.012, 0),
    glass,
    wall=0.0018,
)
torus("Receiver | ground glass flange", (-0.205, -0.012, 0.451), 0.022, 0.004, frosted)
# Two-sided spring clamp with pivot and locking bolt.
for y in [-0.044, 0.022]:
    box("Receiver | metal joint clamp", (-0.205, y, 0.454), (0.079, 0.008, 0.008), steel, 0.002)
for x in [-0.245, -0.165]:
    between("Receiver | clamp pin", (x, -0.044, 0.451), (x, 0.022, 0.451), 0.003, steel)
    cylinder("Receiver | clamp screw", (x, -0.045, 0.463), 0.005, 0.02, aluminum)
# Move the coupling and its fasteners down with the shorter receiving neck.
for coupling in current.objects:
    if coupling.name.startswith(
        (
            "Receiver | ground glass flange",
            "Receiver | metal joint clamp",
            "Receiver | clamp pin",
            "Receiver | clamp screw",
        )
    ):
        coupling.location.z -= 0.026

text("Receiver | volume", "1000 mL", (-0.227, -0.089, 0.329), 0.009, mark)
text("Receiver | material label", "BOROSILICATE 3.3", (-0.239, -0.087, 0.313), 0.0042, mark)
for i in range(4):
    z = 0.347 + i * 0.008
    r = math.sqrt(max(0.084**2 - (z - 0.311) ** 2, 0))
    box(
        "Receiver | graduation",
        (-0.205, -0.012 - r - 0.0003, z),
        (0.012 if i % 2 == 0 else 0.007, 0.0004, 0.0007),
        mark,
        0.0001,
    )

# Rotating evaporation bulb: neck points diagonally back into the drive.
current = collections["05 Evaporating flask"]
flask_center = Vector((0.214, -0.012, 0.351))
flask_axis = -axis
lathe(
    "Evaporation | pear-shaped rotating flask",
    flask_profile(0.083, 0.062),
    flask_center,
    glass,
    flask_axis,
    wall=0.0016,
)
neck_center = flask_center + flask_axis * 0.132
lathe("Evaporation | ground neck", [(0.017, 0), (0.017, 0.026)], neck_center, frosted, flask_axis, wall=0.0015)
torus("Evaporation | blue joint clip", flask_center + flask_axis * 0.15, 0.021, 0.003, blue, flask_axis)
text("Evaporation | volume", "1000", (0.198, -0.091, 0.378), 0.009, mark)

# Heating bath, open steel well, rolled lip, and integral controller pedestal.
current = collections["06 Heating bath"]
bx, by = 0.204, -0.002
lathe(
    "Bath | pedestal",
    [
        (0, 0.016),
        (0.133, 0.016),
        (0.152, 0.022),
        (0.158, 0.038),
        (0.15, 0.108),
        (0.137, 0.129),
        (0.12, 0.135),
        (0, 0.135),
    ],
    (bx, by, 0),
    white,
)
lathe("Bath | black bottom seam", [(0.136, 0.017), (0.145, 0.019), (0.148, 0.023), (0.144, 0.027)], (bx, by, 0), black)
for x in [0.11, 0.30]:
    for y in [-0.088, 0.078]:
        cylinder("Bath | rubber foot", (x, y, 0.012), 0.017, 0.016, rubber)
# Bowl profile is a full continuous shell with an open top.
lathe(
    "Bath | white outer vessel",
    [
        (0, 0.128),
        (0.127, 0.128),
        (0.144, 0.136),
        (0.15, 0.149),
        (0.151, 0.258),
        (0.155, 0.283),
        (0.17, 0.296),
        (0.176, 0.308),
        (0.176, 0.320),
        (0.17, 0.330),
        (0.155, 0.331),
        (0.153, 0.319),
        (0.15, 0.308),
        (0.139, 0.286),
        (0.136, 0.164),
        (0.125, 0.151),
        (0, 0.151),
    ],
    (bx, by, 0),
    white,
)
lathe(
    "Bath | stainless inner well",
    [
        (0, 0.154),
        (0.107, 0.154),
        (0.129, 0.161),
        (0.137, 0.177),
        (0.138, 0.274),
        (0.147, 0.302),
        (0.156, 0.316),
        (0.158, 0.324),
        (0.155, 0.330),
        (0.148, 0.331),
        (0.143, 0.325),
        (0.14, 0.31),
        (0.131, 0.284),
        (0.13, 0.179),
        (0.12, 0.17),
        (0, 0.17),
    ],
    (bx, by, 0),
    steel,
)
torus("Bath | rolled polished rim", (bx, by, 0.328), 0.151, 0.0055, steel)
torus("Bath | outer lip seam", (bx, by, 0.306), 0.174, 0.0015, white)
torus("Bath | lower vessel seam", (bx, by, 0.142), 0.145, 0.0017, aluminum)
# The front controller sits on the base and faces forward.
bath_body_objects = set(current.objects)
box("Bath | front control fascia", (bx, -0.132, 0.075), (0.223, 0.033, 0.083), white, 0.013)
box("Bath | LCD surround", (0.162, -0.150, 0.089), (0.052, 0.003, 0.024), aluminum, 0.002)
box("Bath | temperature LCD", (0.162, -0.152, 0.089), (0.046, 0.001, 0.019), screen, 0.001)
text("Bath | temperature readout", "40.0", (0.142, -0.153, 0.082), 0.016, digits)
text("Bath | Celsius", "°C", (0.192, -0.151, 0.085), 0.0055, ink)
for i, label in enumerate(["M", "▼", "▲", "SET"]):
    x = 0.143 + i * 0.025
    cylinder("Bath | " + label + " key", (x, -0.152, 0.057), 0.0085, 0.003, blue if i < 3 else cyan, (0, -1, 0))
    text("Bath | " + label + " legend", label, (x, -0.154, 0.055), 0.004, white, "CENTER")
for z, label in [(0.098, "RUN"), (0.082, "HEATING")]:
    cylinder("Bath | status lamp", (0.22, -0.152, z), 0.002, 0.001, cyan, (0, -1, 0))
    text("Bath | status " + label, label, (0.227, -0.153, z - 0.002), 0.0045, ink)
text("Bath | logo", "EYELA", (0.232, -0.151, 0.042), 0.013, orange)
text("Bath | identification", "SB  /  WATER BATH", (0.23, -0.151, 0.058), 0.0035, ink)
box("Bath | power switch surround", (0.355, -0.035, 0.061), (0.007, 0.024, 0.037), black, 0.006)
box("Bath | power rocker", (0.360, -0.035, 0.063), (0.005, 0.016, 0.027), rubber, 0.003)
# Subtle housing fasteners on the heater panel.
for x in [0.108, 0.300]:
    cylinder("Bath | front fascia screw", (x, -0.150, 0.106), 0.002, 0.001, steel, (0, -1, 0), vertices=24)

# Raise the open well so the evaporation bulb sits partly inside the bath.
for obj in current.objects:
    obj.location.z *= 1.2
    obj.scale.z *= 1.2
    if obj not in bath_body_objects:
        obj.location.y -= 0.020
    if obj.name.startswith(
        ("Bath | white outer", "Bath | stainless", "Bath | rolled", "Bath | outer lip", "Bath | lower vessel")
    ):
        # Shorter bowl with a higher bottom, matching the reference silhouette.
        obj.location.z = 0.396 + (obj.location.z - 0.396) * 0.85
        obj.scale.z *= 0.85
    elif obj.name.startswith(("Bath | pedestal",)):
        obj.scale.z *= 1.10

# Editable UV-mapped screen faces for the Blender source as well as USD.
# Runtime USD screens below use the same PNGs plus live seven-segment digits.
for display_id, collection_name, center_screen, screen_size, readout in [
    ("rpm", "07 Controls and markings", (-0.049, -0.0426, 0.687), (0.040, 0.017), "Controller | RPM readout"),
    ("bath", "06 Heating bath", (0.162, -0.1727, 0.1068), (0.045, 0.021), "Bath | temperature readout"),
]:
    current = collections[collection_name]
    image_path = OUT / "textures" / (display_id + "_background.png")
    display_mat = material("Editable LCD | " + display_id, (1, 1, 1), 0.28)
    nodes = display_mat.node_tree.nodes
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(image_path), check_existing=True)
    image_node.image.filepath = "//textures/" + image_path.name
    display_mat.node_tree.links.new(image_node.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    w, h = screen_size
    mesh = bpy.data.meshes.new("Editable screen UV " + display_id)
    mesh.from_pydata(
        [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)], [], [(0, 1, 2, 3)]
    )
    mesh.uv_layers.new(name="UVMap")
    for uv, coord in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.uv = coord
    obj = bpy.data.objects.new("Controller | editable LCD " + display_id, mesh)
    current.objects.link(obj)
    obj.location = center_screen
    mesh.materials.append(display_mat)
    obj["exclude_usd"] = True
    if display_id == "rpm":
        bpy.data.objects[readout].data.body = "000"

# Instrument typography and scales. Keep labels as editable Blender text.
current = collections["07 Controls and markings"]
for label in ["Controller | POWER label", "Controller | RUN label"]:
    obj = bpy.data.objects[label]
    obj.data.font = FONT_BOLD
    obj.data.size = 0.0029
bpy.data.objects["Controller | RUN label"].data.materials[0] = ink
bpy.data.objects["Controller | model"].data.size = 0.006
bpy.data.objects["Controller | model"].data.body = "ROTAVAP"
bpy.data.objects["Controller | logo"].data.font = FONT_BOLD
for x, caption in [(-0.061, "POWER"), (-0.035, "RUN/STOP")]:
    text("Controller | " + caption, caption, (x, -0.039, 0.700), 0.0035, ink, "CENTER")
speed_title = bpy.data.objects["Controller | speed label"]
speed_title.data.body = "rpm"
speed_title.data.size = 0.0045
speed_title.location = (-0.066, -0.0423, 0.665)
lathe(
    "Controller | dial scale surround",
    [(0.021, -0.0005), (0.037, -0.0005), (0.037, 0.0005), (0.021, 0.0005), (0.021, -0.0005)],
    (-0.049, -0.0415, 0.643),
    white,
    (0, -1, 0),
)
# The existing dial rotates +Y from its top index through 300 degrees.
for i in range(25):
    angle = rpm_to_dial_angle(i * 5)
    long_tick = i % 4 == 0
    start_r, end_r = 0.0235, 0.0254 if long_tick else 0.0246
    a = (-0.049 + start_r * math.sin(angle), -0.0423, 0.643 + start_r * math.cos(angle))
    b = (-0.049 + end_r * math.sin(angle), -0.0423, 0.643 + end_r * math.cos(angle))
    tube("Controller | speed tick " + str(i * 5), [a, b], 0.00019 if long_tick else 0.00012, ink)
    if i % 8 == 0:
        obj = text(
            "Controller | speed value " + str(i * 5),
            str(i * 5),
            (-0.049 + 0.031 * math.sin(angle), -0.0423, 0.6405 + 0.031 * math.cos(angle)),
            0.0046,
            ink,
            "CENTER",
        )
# Short identification on the lower mounting neck, away from moving components.
text("Controller | instrument ID", "SPEED", (-0.008, 0.0165, 0.594), 0.0055, ink)
text("Controller | speed range", "0-120 rpm", (-0.008, 0.0165, 0.584), 0.0047, ink)

current = collections["06 Heating bath"]
for name, body, size in [("M", "HEAT", 0.0030), ("▼", "▼", 0.0044), ("▲", "▲", 0.0044), ("SET", "SET", 0.0036)]:
    obj = bpy.data.objects["Bath | " + name + " legend"]
    obj.data.body, obj.data.size, obj.data.font = body, size, FONT_BOLD
    obj.data.materials[0] = ink
text("Bath | display caption", "TEMPERATURE", (0.136, -0.171, 0.128), 0.0060, ink)
for x, caption in [(0.143, "HEAT"), (0.168, "TEMP -"), (0.193, "TEMP +"), (0.218, "ENTER")]:
    text("Bath | button caption " + caption, caption, (x, -0.170, 0.0495), 0.0044, ink, "CENTER")
bpy.data.objects["Bath | status RUN"].data.body = "POWER"
bpy.data.objects["Bath | status HEATING"].data.body = "HEAT"
bpy.data.objects["Bath | identification"].data.body = "WATER BATH"
bpy.data.objects["Bath | identification"].data.size = 0.0065
bpy.data.objects["Bath | logo"].data.size = 0.010
bpy.data.objects["Bath | logo"].data.font = FONT_BOLD
text("Bath | temperature range", "5-95 °C", (0.253, -0.171, 0.128), 0.0060, ink)
# Fixed labels by the side rocker; I/O on its cap remain on the moving link.
obj = text("Bath | mains label", "POWER", (0.359, -0.057, 0.106), 0.0065, ink, "CENTER")
obj.rotation_euler = (math.pi / 2, 0, math.pi / 2)

# A millimetre ruler on the lift rail; its pointer travels with the carriage.
current = collections["01 Stand and lift"]
box("Lift | narrow ruler plate", (0.079, 0.0396, 0.546), (0.023, 0.0004, 0.160), black, 0.0002)
for mm in range(0, 121, 5):
    z = 0.472 + mm / 1000
    major = mm % 20 == 0
    obj = box(
        "Lift | ruler tick " + str(mm),
        (0.074 if major else 0.073, 0.0391, z),
        (0.007 if major else 0.004, 0.00015, 0.00035 if major else 0.00022),
        mark,
        0,
    )
    obj["rig_link"] = "base"
    if major:
        obj = text("Lift | ruler value " + str(mm), str(mm), (0.078, 0.0389, z - 0.0023), 0.0048, mark)
        obj["rig_link"] = "base"
obj = text("Lift | ruler unit", "mm", (0.078, 0.0389, 0.610), 0.0045, mark)
obj["rig_link"] = "base"
obj = text("Lift | ruler title", "LIFT", (0.069, 0.0389, 0.618), 0.0048, mark)
obj["rig_link"] = "base"
obj = tube(
    "Lift | moving ruler index",
    [(0.103, 0.0384, 0.474), (0.091, 0.0384, 0.472), (0.103, 0.0384, 0.470)],
    0.0005,
    orange,
)
obj["rig_link"] = "lift"
bpy.data.objects["Lift | narrow ruler plate"]["rig_link"] = "base"

# Angular scale on the rear bearing, referenced to the real Y-axis hinge.
for obj in list(bpy.data.objects):
    if obj.name.startswith("Hinge | engraved angle tick"):
        bpy.data.objects.remove(obj, do_unlink=True)
current = collections["01 Stand and lift"]
lathe(
    "Hinge | engraved scale ring",
    [(0.037, -0.001), (0.069, -0.001), (0.069, 0.001), (0.037, 0.001), (0.037, -0.001)],
    (-0.067, 0.126, 0.513),
    scale_plate,
    (0, 1, 0),
)
for degrees in range(-20, 6):
    a = math.radians(degrees)
    major = degrees % 5 == 0
    points = [(-0.067 + r * math.sin(a), 0.1274, 0.513 + r * math.cos(a)) for r in (0.047, 0.052 if major else 0.0495)]
    tube("Hinge | degree tick " + str(degrees), points, 0.00022 if major else 0.00012, mark)
    if degrees in (-20, 0, 5):
        obj = text(
            "Hinge | degree value " + str(degrees),
            f"{degrees:+d}" if degrees else "0",
            (
                -0.067 + (0.064 if degrees == 5 else 0.057) * math.sin(a),
                0.1277,
                0.510 + (0.064 if degrees == 5 else 0.057) * math.cos(a),
            ),
            0.0042,
            mark,
            "CENTER",
        )
        obj.rotation_euler = (math.pi / 2, 0, math.pi)
obj = text("Hinge | degree unit", "TILT / °", (-0.067, 0.1277, 0.461), 0.0060, mark, "CENTER")
obj.rotation_euler = (math.pi / 2, 0, math.pi)
obj = tube(
    "Hinge | moving zero pointer",
    [(-0.0685, 0.1285, 0.556), (-0.067, 0.1285, 0.560), (-0.0655, 0.1285, 0.556)],
    0.0004,
    orange,
)
obj["rig_link"] = "tilt"


# Project labels onto the actual outside surface; no text embedded inside glass.
def on_surface(obj, target):
    modifier = obj.modifiers.new("Printed on instrument surface", "SHRINKWRAP")
    modifier.target = bpy.data.objects[target]
    modifier.wrap_method = "PROJECT"
    modifier.use_project_x = False
    modifier.use_project_y = False
    modifier.use_project_z = True  # Text's local Z points toward the front (-world Y).
    modifier.use_negative_direction = True
    modifier.use_positive_direction = False
    modifier.offset = 0.00035
    return obj


current = collections["04 Receiving flask"]
for obj in list(current.objects):
    if obj.name.startswith(("Receiver | volume", "Receiver | material label", "Receiver | graduation")):
        bpy.data.objects.remove(obj, do_unlink=True)
flask = "Receiver | round-bottom flask 1 L"
for name, body, z, size in [
    ("capacity", "1000 mL", 0.358, 0.011),
    ("material", "BORO 3.3", 0.345, 0.0060),
    ("approximate", "APPROX. mL", 0.262, 0.0060),
]:
    on_surface(text("Receiver | " + name, body, (-0.205, -0.125, z), size, mark, "CENTER"), flask)
# Nominal capacity fractions of the spherical bulb, for visual reference only.
for amount in (250, 500, 750):
    lo, hi = -1.0, 1.0
    for _ in range(40):
        middle = (lo + hi) / 2
        fraction = (2 + 3 * middle - middle**3) / 4
        if fraction < amount / 1000:
            lo = middle
        else:
            hi = middle
    z = 0.311 + 0.084 * (lo + hi) / 2
    on_surface(text("Receiver | scale " + str(amount), str(amount), (-0.188, -0.125, z - 0.0035), 0.0080, mark), flask)
    points = []
    for j in range(31):
        x = -0.222 + j * 0.025 / 30
        y = -0.012 - math.sqrt(0.084**2 - (z - 0.311) ** 2 - (x + 0.205) ** 2) - 0.0004
        points.append((x, y, z))
    tube("Receiver | curved tick " + str(amount), points, 0.00035, mark)
current = collections["05 Evaporating flask"]
bpy.data.objects.remove(bpy.data.objects["Evaporation | volume"], do_unlink=True)
for name, body, z, size in [
    ("capacity", "1000 mL", 0.387, 0.0105),
    ("label", "EVAP. FLASK", 0.375, 0.0055),
    ("glass", "BORO 3.3", 0.363, 0.0055),
]:
    on_surface(
        text("Evaporation | " + name, body, (0.214, -0.13, z), size, mark, "CENTER"),
        "Evaporation | pear-shaped rotating flask",
    )
current = collections["03 Condenser"]
for name, body, z, size in [
    ("identity", "CONDENSER", 0.808, 0.0075),
    ("glass", "BORO 3.3", 0.790, 0.0055),
    ("lower port", "IN", 0.611, 0.0065),
    ("upper port", "OUT", 0.646, 0.0065),
]:
    on_surface(
        text("Condenser | " + name, body, (-0.205, -0.08, z), size, mark, "CENTER"), "Condenser | outer glass jacket"
    )

# Each physical control becomes a separate link. Button legends move with caps.
CONTROL_SPECS = {}
for name, parent, control_center, objects, kind in [
    ("motor_power", "lift", (-0.061, -0.040, 0.714), ["Controller | POWER", "Controller | POWER label"], "button"),
    ("motor_run", "lift", (-0.035, -0.040, 0.714), ["Controller | RUN", "Controller | RUN label"], "button"),
    ("bath_mode", "base", (0.143, -0.172, 0.0684), ["Bath | M key", "Bath | M legend"], "button"),
    ("bath_down", "base", (0.168, -0.172, 0.0684), ["Bath | ▼ key", "Bath | ▼ legend"], "button"),
    ("bath_up", "base", (0.193, -0.172, 0.0684), ["Bath | ▲ key", "Bath | ▲ legend"], "button"),
    ("bath_set", "base", (0.218, -0.172, 0.0684), ["Bath | SET key", "Bath | SET legend"], "button"),
    (
        "speed_dial",
        "lift",
        (-0.049, -0.049, 0.643),
        ["Controller | speed dial", "Controller | speed dial inset", "Controller | dial index"],
        "dial",
    ),
    ("tilt_lever", "base", (0.035, 0.052, 0.101), ["Lift lever | stem", "Lift lever | foot paddle"], "lever"),
    ("mains_switch", "base", (0.360, -0.055, 0.0756), ["Bath | power rocker"], "rocker"),
]:
    CONTROL_SPECS[name] = {"parent": parent, "origin": control_center, "kind": kind, "joint": name + "_joint"}
    for label in objects:
        bpy.data.objects[label]["device_control"] = name
# The rocker has visible I/O markings on its right-facing surface.
current = collections["06 Heating bath"]
for label, z in [("I", 0.085), ("O", 0.066)]:
    obj = text("Switch | " + label, label, (0.3632, -0.055, z), 0.004, white, "CENTER")
    obj.rotation_euler = (math.pi / 2, 0, math.pi / 2)
    obj["device_control"] = "mains_switch"

# Strong, legible printing; keep display numerals at their fitted dimensions.
bpy.data.objects["Controller | POWER label"].data.body = "I/O"
for label in ("Controller | POWER label", "Controller | RUN label", "Bath | SET legend"):
    bpy.data.objects[label].data.size *= 1.5
for label in (
    "Bath | status RUN",
    "Bath | status HEATING",
    "Bath | Celsius",
    "Controller | RPM label",
    "Switch | I",
    "Switch | O",
):
    bpy.data.objects[label].data.size *= 1.5
for label_object in bpy.data.objects:
    if label_object.type == "FONT" and "readout" not in label_object.name:
        label_object.data.font = FONT_BOLD

current = collections["01 Stand and lift"]
lever_caption = text("Lift lever | function", "TILT", (-0.009, -0.075, 0.108), 0.0055, mark, "CENTER")
lever_caption.data.font = FONT_BOLD
lever_caption.rotation_euler = (-0.35, 0, 0)
on_surface(lever_caption, "Lift lever | foot paddle")
lever_caption["rig_link"] = "tilt_lever"

# Normalize imported OpenType cap height to the requested physical label sizes.
for label_object in bpy.data.objects:
    if label_object.type == "FONT":
        label_object.data.size *= 1.65
        if label_object.name.startswith(("Controller | RPM readout", "Bath | temperature readout")):
            label_object.data.size *= 1.25

# Mechanical hierarchy. Keep every mesh editable; joint pivots are named empties.
ORIGINS = {
    "base": Vector((0, 0, 0)),
    "lift": Vector((0.024, 0.057, 0.472)),
    "tilt": Vector((-0.067, 0.040, 0.513)),
    "rotor": on(0.117),
}
ORIGINS.update({name: Vector(spec["origin"]) for name, spec in CONTROL_SPECS.items()})
rig = {}
for name, origin in ORIGINS.items():
    empty = bpy.data.objects.new(name, None)
    main.objects.link(empty)
    empty.location = origin
    empty.empty_display_type = "ARROWS"
    empty.empty_display_size = 0.065
    rig[name] = empty
bpy.context.view_layer.update()
for child, parent in [("lift", "base"), ("tilt", "lift"), ("rotor", "tilt")] + [
    (name, spec["parent"]) for name, spec in CONTROL_SPECS.items()
]:
    matrix = rig[child].matrix_world.copy()
    rig[child].parent = rig[parent]
    rig[child].matrix_world = matrix
bpy.context.view_layer.update()
for colname, col in collections.items():
    if colname == "Studio":
        continue
    for obj in list(col.objects):
        n = obj.name
        if colname == "06 Heating bath" or n.startswith(
            ("Stand |", "Lift | black column", "Lift | exposed", "Lift | narrow", "Lift | top", "Lift lever")
        ):
            link = "base"
        elif (
            n.startswith(("Lift | sliding", "Lift lock", "Controller |", "Hinge |")) and n != "Hinge | moving trunnion"
        ):
            link = "lift"
        elif colname == "05 Evaporating flask" or n.startswith(("Drive | spindle nose", "Drive | hollow vapor shaft")):
            link = "rotor"
        else:
            link = "tilt"
        link = obj.get("rig_link", obj.get("device_control", link))
        world_matrix = obj.matrix_world.copy()
        obj.parent = rig[link]
        obj.matrix_world = world_matrix
        obj["physics_link"] = link
bpy.context.view_layer.update()
rig["lift"]["joint"] = "lift_joint: Z translation, 0 to 0.12 m"
rig["tilt"]["joint"] = "tilt_joint: Y hinge, -20 to +5 degrees relative to rest"
rig["rotor"]["joint"] = "spin_joint: continuous, along vapor shaft, velocity drive"
scene["Model notes"] = (
    "Reference-inspired articulated redesign; estimated geometry and mass. Export includes rigid bodies, contacts, limits and drives. No thermal/fluid simulation."
)
scene["Units"] = "metres"
reference_path = ROOT / "outputs/rotary_evaporator/reference.png"
if reference_path.exists():
    img = bpy.data.images.load(str(reference_path))
    img.pack()
    img.use_fake_user = True


def export_articulated_usd():
    """Export evaluated visual meshes and lightweight PhysX collision geometry."""
    stage = Usd.Stage.CreateNew(str(OUT / "rotary_evaporator.usdc"))
    root = UsdGeom.Xform.Define(stage, "/RotaryEvaporator")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root.GetPrim().SetCustomDataByKey(
        "description",
        "Reference-inspired rotary evaporator; three mechanism DOFs and eight physical controls; estimated physical properties.",
    )
    prefix = "/RotaryEvaporator"
    materials = {}
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        p = mat.node_tree.nodes.get("Principled BSDF")
        if p is None:
            continue
        path = prefix + "/Materials/" + re.sub(r"[^A-Za-z0-9_]", "_", mat.name)
        usdmat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Preview")
        shader.CreateIdAttr("UsdPreviewSurface")
        color = Gf.Vec3f(*p.inputs["Base Color"].default_value[:3])
        roughness = p.inputs["Roughness"].default_value
        metallic = p.inputs["Metallic"].default_value
        isglass = p.inputs["Transmission Weight"].default_value > 0.5
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        usdmat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mdl = UsdShade.Shader.Define(stage, path + "/MDL")
        mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl" if isglass else "OmniPBR.mdl"), "mdl")
        mdl.SetSourceAssetSubIdentifier("OmniGlass" if isglass else "OmniPBR", "mdl")
        mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
        usdmat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
        if isglass:
            mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(color)
            mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.46)
            mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
            mdl.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(1.0)
            mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.18)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.46)
        else:
            mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
            mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
            mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
        materials[mat.name] = usdmat
    masses = {"base": 8.5, "lift": 0.8, "tilt": 2.1, "rotor": 0.26}
    coms = {
        "base": (0.11, 0, 0.22),
        "lift": (-0.035, -0.015, 0.075),
        "tilt": (-0.095, -0.025, 0.10),
        "rotor": (0.145, 0, -0.084),
    }
    inertias = {
        "base": (0.25, 0.32, 0.15),
        "lift": (0.003, 0.003, 0.002),
        "tilt": (0.043, 0.047, 0.012),
        "rotor": (0.0010, 0.0011, 0.0008),
    }
    for name in CONTROL_SPECS:
        masses[name] = 0.025
        coms[name] = (0, 0, 0)
        inertias[name] = (0.000002, 0.000002, 0.000002)
    masses["tilt_lever"] = 0.06
    coms["tilt_lever"] = (-0.028, -0.075, -0.021)
    inertias["tilt_lever"] = (0.000045, 0.000014, 0.000050)
    for name, origin in ORIGINS.items():
        body = UsdGeom.Xform.Define(stage, prefix + "/" + name)
        body.AddTranslateOp().Set(Gf.Vec3d(*origin))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(masses[name])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*coms[name]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertias[name]))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
        UsdGeom.Scope.Define(stage, prefix + "/" + name + "/Visuals")
        UsdGeom.Scope.Define(stage, prefix + "/" + name + "/Collisions")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if "physics_link" not in obj or obj.get("exclude_usd") or obj.type not in {"MESH", "CURVE", "FONT"}:
            continue
        name = obj["physics_link"]
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        transform = Matrix.Translation(-ORIGINS[name]) @ obj.matrix_world
        points = [tuple(transform @ v.co) for v in mesh.vertices]
        path = prefix + "/" + name + "/Visuals/" + re.sub(r"[^A-Za-z0-9_]", "_", obj.name)
        usdmesh = UsdGeom.Mesh.Define(stage, path)
        usdmesh.CreatePointsAttr(points)
        usdmesh.CreateFaceVertexCountsAttr([len(f.vertices) for f in mesh.polygons])
        usdmesh.CreateFaceVertexIndicesAttr([v for f in mesh.polygons for v in f.vertices])
        usdmesh.CreateSubdivisionSchemeAttr("none")
        usdmesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        normalmatrix = transform.to_3x3().inverted().transposed()
        normals = [tuple((normalmatrix @ n.vector).normalized()) for n in mesh.corner_normals]
        usdmesh.CreateNormalsAttr(normals)
        usdmesh.SetNormalsInterpolation("faceVarying")
        if mesh.uv_layers.active:
            UsdGeom.PrimvarsAPI(usdmesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
                [tuple(uv.uv) for uv in mesh.uv_layers.active.data]
            )
        usdmesh.GetPrim().SetDisplayName(obj.name)
        if mesh.materials:
            UsdShade.MaterialBindingAPI.Apply(usdmesh.GetPrim()).Bind(materials[mesh.materials[0].name])
        evaluated.to_mesh_clear()
    contact_mat = UsdShade.Material.Define(stage, prefix + "/Materials/Contact")
    contact = UsdPhysics.MaterialAPI.Apply(contact_mat.GetPrim())
    contact.CreateStaticFrictionAttr(0.55)
    contact.CreateDynamicFrictionAttr(0.4)
    contact.CreateRestitutionAttr(0.05)

    def collision(name, label, kind, loc, size, rotation=None):
        path = prefix + "/" + name + "/Collisions/" + label
        if kind == "box":
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
        elif kind == "sphere":
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(size[0])
        elif kind == "cylinder":
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateRadiusAttr(size[0])
            shape.CreateHeightAttr(size[1])
            shape.CreateAxisAttr("Z")
        else:
            shape = UsdGeom.Capsule.Define(stage, path)
            shape.CreateRadiusAttr(size[0])
            shape.CreateHeightAttr(size[1])
            shape.CreateAxisAttr("Z")
        xf = UsdGeom.Xformable(shape)
        xf.AddTranslateOp().Set(Gf.Vec3d(*(Vector(loc) - ORIGINS[name])))
        if rotation:
            xf.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
        if kind == "box":
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.CreateVisibilityAttr("invisible")
        prim = shape.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(contact_mat, materialPurpose="physics")

    collision("base", "column", "box", (0.035, 0.087, 0.342), (0.062, 0.063, 0.615))
    collision("base", "foot_left", "box", (-0.10, 0.06, 0.021), (0.27, 0.12, 0.025))
    collision("base", "foot_front", "box", (0.055, -0.07, 0.021), (0.09, 0.29, 0.025))
    collision("base", "heater_base", "cylinder", (0.204, -0.002, 0.09), (0.143, 0.16))
    collision("base", "bath_floor", "cylinder", (0.204, -0.002, 0.221), (0.13, 0.020))
    # An actual open cavity: separate sidewall segments instead of a solid hull.
    for i in range(48):
        a = i * math.tau / 48
        collision(
            "base",
            f"bath_wall_{i:02d}",
            "box",
            (0.204 + 0.151 * math.cos(a), -0.002 + 0.151 * math.sin(a), 0.306),
            (0.039, 0.021, 0.17),
            Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(a)),
        )
    collision("lift", "carriage", "box", (0.024, 0.057, 0.472), (0.088, 0.089, 0.135))
    collision("lift", "controller", "box", (-0.023, 0.006, 0.691), (0.11, 0.07, 0.145))
    collision("tilt", "motor", "cylinder", center, (0.063, 0.13), Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*axis)))
    collision("tilt", "condenser", "capsule", (-0.205, 0.008, 0.750), (0.046, 0.355))
    collision("tilt", "receiver_bulb", "sphere", receiver, (0.083,))
    collision("tilt", "receiver_neck", "cylinder", (-0.205, -0.012, 0.422), (0.020, 0.12))
    collision("rotor", "flask_bulb", "sphere", flask_center, (0.082,))
    collision(
        "rotor",
        "flask_neck",
        "cylinder",
        flask_center - axis * 0.125,
        (0.018, 0.10),
        Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*axis)),
    )
    joints = prefix + "/Joints"
    UsdGeom.Scope.Define(stage, joints)
    fixed = UsdPhysics.FixedJoint.Define(stage, joints + "/fixed_base")
    fixed.CreateBody1Rel().SetTargets([prefix + "/base"])
    baseprim = stage.GetPrimAtPath(prefix + "/base")
    UsdPhysics.ArticulationRootAPI.Apply(baseprim)
    baseprim.AddAppliedSchema("PhysxArticulationAPI")
    baseprim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    baseprim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    baseprim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)

    def joint(label, kind, parent, child, pivot, axis_name, limits, kp, kd, effort, rotation=None):
        cls = UsdPhysics.PrismaticJoint if kind == "linear" else UsdPhysics.RevoluteJoint
        j = cls.Define(stage, joints + "/" + label)
        j.CreateBody0Rel().SetTargets([prefix + "/" + parent])
        j.CreateBody1Rel().SetTargets([prefix + "/" + child])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(Vector(pivot) - ORIGINS[parent])))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(Vector(pivot) - ORIGINS[child])))
        q = Gf.Quatf(rotation.GetQuat()) if rotation else Gf.Quatf(1)
        j.CreateLocalRot0Attr(q)
        j.CreateLocalRot1Attr(q)
        j.CreateAxisAttr(axis_name)
        j.CreateCollisionEnabledAttr(False)
        if limits:
            j.CreateLowerLimitAttr(limits[0])
            j.CreateUpperLimitAttr(limits[1])
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(kp)
        drive.CreateDampingAttr(kd)
        drive.CreateMaxForceAttr(effort)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        return j

    joint("lift_joint", "linear", "base", "lift", ORIGINS["lift"], "Z", (0, 0.12), 30000, 900, 400)
    joint("tilt_joint", "angular", "lift", "tilt", ORIGINS["tilt"], "Y", (-20, 5), 250, 25, 30)
    joint(
        "spin_joint",
        "angular",
        "tilt",
        "rotor",
        ORIGINS["rotor"],
        "X",
        None,
        0,
        0.12,
        1,
        Gf.Rotation(Gf.Vec3d(0, 1, 0), 30),
    )
    for name, spec in CONTROL_SPECS.items():
        kind = spec["kind"]
        if kind == "button":
            joint(spec["joint"], "linear", spec["parent"], name, ORIGINS[name], "Y", (0, 0.0022), 600, 4, 6)
            collision(
                name,
                "cap",
                "cylinder",
                ORIGINS[name],
                (0.008, 0.004),
                Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0)),
            )
        elif kind == "dial":
            joint(spec["joint"], "angular", spec["parent"], name, ORIGINS[name], "Y", (0, 300), 0.006, 0.00008, 0.1)
            collision(
                name,
                "knob",
                "cylinder",
                ORIGINS[name],
                (0.02, 0.017),
                Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0)),
            )
        elif kind == "lever":
            joint(spec["joint"], "angular", spec["parent"], name, ORIGINS[name], "X", (-20, 5), 0.05, 0.002, 2)
            collision(
                name,
                "paddle",
                "box",
                (-0.009, -0.064, 0.074),
                (0.041, 0.068, 0.024),
                Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(-0.35)),
            )
            stage.GetPrimAtPath(prefix + "/" + name).AddAppliedSchema("PhysxRigidBodyAPI")
            stage.GetPrimAtPath(prefix + "/" + name).CreateAttribute(
                "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool
            ).Set(True)
        else:
            j = joint(
                spec["joint"], "angular", spec["parent"], name, ORIGINS[name], "Y", (-12, 12), 0.0015, 0.0002, 0.1
            )
            UsdPhysics.DriveAPI(j.GetPrim(), "angular").CreateTargetPositionAttr(12)
            collision(name, "rocker", "box", ORIGINS[name], (0.006, 0.016, 0.032))
        stage.GetPrimAtPath(prefix + "/" + name).SetCustomDataByKey("control_id", name)
    (OUT / "control_manifest.json").write_text(json.dumps(CONTROL_SPECS, indent=2) + "\n")
    # Housing parts intentionally overlap the attached column at the hinge.
    # Preserve rotor-vs-bath and receiver-vs-base contact by filtering only this pair.
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(prefix + "/tilt")).CreateFilteredPairsRel().SetTargets(
        [prefix + "/lift"]
    )
    for path in ["lift/Visuals/Controller___RPM_readout", "base/Visuals/Bath___temperature_readout"]:
        UsdGeom.Imageable(stage.GetPrimAtPath(prefix + "/" + path)).CreateVisibilityAttr("invisible")
    build_displays(stage, prefix, ORIGINS)
    stage.GetRootLayer().Save()
    export_demo_scene()
    (OUT / "model_spec.json").write_text(
        json.dumps(
            {
                "units": "m, kg, s; USD angular drives in degrees; Isaac articulation API angles in radians",
                "bodies": masses,
                "inertias_kg_m2": inertias,
                "joint_limits": {"lift_joint": [0, 0.12], "tilt_joint_degrees": [-20, 5], "spin_joint": "continuous"},
                "rest_shaft_angle_degrees": 30,
                "estimated_from_single_image": True,
                "bath_collision": "48 wall boxes plus bottom cylinder; open cavity",
                "scope": "rigid mechanism only; no fluid, thermal, glass breakage or flask detachment",
            },
            indent=2,
        )
        + "\n"
    )


def export_demo_scene():
    """A standalone stage: open it in Isaac Sim and press Play."""
    stage = Usd.Stage.CreateNew(str(OUT / "demo.usda"))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    physics.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    physics.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)
    physics.GetPrim().CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    asset = UsdGeom.Xform.Define(stage, "/World/Evaporator")
    asset.GetPrim().GetReferences().AddReference("./rotary_evaporator.usdc")
    for name, kind, parameter, value in [
        ("lift_joint", "linear", "position", 0.09),
        ("tilt_joint", "angular", "position", -15),
        ("spin_joint", "angular", "velocity", 180),
    ]:
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/World/Evaporator/Joints/" + name), kind)
        if parameter == "position":
            drive.CreateTargetPositionAttr(value)
        else:
            drive.CreateTargetVelocityAttr(value)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(1)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.011))
    floor.AddScaleOp().Set(Gf.Vec3f(3, 3, 0.02))
    floor.CreateDisplayColorAttr([(0.22, 0.24, 0.25)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(350)
    for name, loc, intensity, width, height in [
        ("Key", (-0.7, -1, 1.6), 900, 0.8, 1.2),
        ("Rim", (0.6, 0.5, 1.3), 1200, 0.5, 1.0),
    ]:
        light = UsdLux.RectLight.Define(stage, "/World/" + name)
        light.CreateIntensityAttr(intensity)
        light.CreateWidthAttr(width)
        light.CreateHeightAttr(height)
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*loc), Gf.Vec3d(0, 0, 0.5), Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(light).AddTransformOp().Set(matrix)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(50)
    camera.CreateHorizontalApertureAttr(24)
    camera.CreateVerticalApertureAttr(30)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.8, -3, 1.1), Gf.Vec3d(0, 0, 0.56), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
    stage.GetRootLayer().Save()


def setup_motion_preview():
    """Keyframed demonstration; USD itself starts at rest and is physics driven."""
    lift_rest = rig["lift"].location.copy()
    for frame, height, angle, spin in [
        (1, 0, 0, 0),
        (45, 0.12, 0, 0),
        (90, 0.12, -20, 0),
        (135, 0.12, -20, 360),
        (180, 0, 0, 720),
    ]:
        rig["lift"].location = lift_rest + Vector((0, 0, height))
        rig["tilt"].rotation_mode = "XYZ"
        rig["tilt"].rotation_euler = (0, math.radians(angle), 0)
        rig["tilt_lever"].rotation_euler = (math.radians(angle), 0, 0)
        rig["tilt_lever"].keyframe_insert(data_path="rotation_euler", frame=frame)
        rig["rotor"].rotation_mode = "AXIS_ANGLE"
        rig["rotor"].rotation_axis_angle = (math.radians(spin), *axis)
        rig["lift"].keyframe_insert(data_path="location", frame=frame)
        rig["tilt"].keyframe_insert(data_path="rotation_euler", frame=frame)
        rig["rotor"].keyframe_insert(data_path="rotation_axis_angle", frame=frame)
    scene.frame_start = 1
    scene.frame_end = 180
    scene.render.fps = 30
    scene.frame_set(1)


# Neutral studio with long softbox reflections to reveal transparent glass.
current = collections["Studio"]
floor_mat = material("Studio | warm neutral gray", (0.19, 0.205, 0.216), 0.78)
box("Studio | floor", (0, 0, -0.023), (200, 200, 0.04), floor_mat, 0)
world = bpy.data.worlds.new("Soft gray studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.31, 0.34, 0.37, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
scene.world = world


def area(name, loc, energy, size, target, color=(1, 1, 1), size_y=None):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.color = color
    if size_y:
        data.shape = "RECTANGLE"
        data.size = size
        data.size_y = size_y
    else:
        data.shape = "DISK"
        data.size = size
    obj = bpy.data.objects.new(name, data)
    current.objects.link(obj)
    obj.location = loc
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


area("Softbox | key", (-0.8, -1.1, 1.6), 110, 0.65, (0, 0, 0.5), (1, 0.96, 0.91), 1.2)
area("Softbox | glass rim", (0.45, 0.65, 1.2), 140, 0.42, (-0.15, 0, 0.60), (0.89, 0.95, 1), 1.15)
area("Softbox | long left strip", (-0.65, 0.18, 0.70), 65, 0.16, (-0.15, 0, 0.6), (1, 1, 1), 0.9)
area("Softbox | front fill", (0.9, -0.8, 0.9), 45, 0.6, (0, 0, 0.45))
cam_data = bpy.data.cameras.new("Camera | product")
cam = bpy.data.objects.new("Camera | product", cam_data)
current.objects.link(cam)
cam.location = (0.80, -3.0, 1.10)
target = Vector((0.006, 0, 0.491))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.type = "ORTHO"
cam_data.ortho_scale = 1.15
cam_data.lens = 65
scene.camera = cam
scene.render.engine = "CYCLES"
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 16
scene.cycles.transmission_bounces = 12
scene.cycles.transparent_max_bounces = 12
scene.cycles.glossy_bounces = 8
prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
except Exception:
    scene.cycles.device = "CPU"
scene.render.resolution_x = 1800
scene.render.resolution_y = 2100
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "AgX"
scene.render.filepath = str(PREVIEWS / "preview.png")
scene.render.film_transparent = False
# Open the file in a clean material-colored model view.
bpy.ops.object.select_all(action="DESELECT")
for screen_data in bpy.data.screens:
    for a in screen_data.areas:
        if a.type == "VIEW_3D":
            a.spaces.active.region_3d.view_perspective = "CAMERA"
            a.spaces.active.clip_end = 100
            a.spaces.active.shading.type = "MATERIAL"
            a.spaces.active.overlay.show_overlays = False
(OUT / "markings_manifest.json").write_text(
    json.dumps(
        [
            {
                "object": obj.name,
                "text": obj.data.body,
                "link": obj.get("physics_link"),
                "font_size_m": obj.data.size,
                "surface_projected": any(mod.type == "SHRINKWRAP" for mod in obj.modifiers),
            }
            for obj in bpy.data.objects
            if obj.type == "FONT" and "physics_link" in obj
        ],
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
export_articulated_usd()
setup_motion_preview()
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "rotary_evaporator.blend"))
bpy.ops.render.render(write_still=True)
scene.frame_set(90)
scene.render.filepath = str(PREVIEWS / "raised_tilted.png")
bpy.ops.render.render(write_still=True)
scene.frame_set(1)
cam.location = (0.30, -1.3, 0.80)
cam.rotation_euler = (Vector((-0.030, -0.020, 0.630)) - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.ortho_scale = 0.34
scene.render.filepath = str(PREVIEWS / "controls_detail.png")
bpy.ops.render.render(write_still=True)
cam.location = (0.65, -0.9, 0.27)
cam.rotation_euler = (Vector((0.20, -0.055, 0.095)) - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.ortho_scale = 0.43
scene.render.filepath = str(PREVIEWS / "bath_controls.png")
bpy.ops.render.render(write_still=True)
for label, eye, aim, width in [
    ("controls_front", (-0.049, -1.3, 0.687), (-0.049, -0.020, 0.687), 0.21),
    ("receiver_markings", (-0.05, -1.0, 0.44), (-0.205, -0.012, 0.327), 0.21),
    ("hinge_markings", (-0.17, 0.8, 0.66), (-0.067, 0.126, 0.54), 0.18),
    ("lift_markings", (0.30, -0.7, 0.66), (0.078, 0.04, 0.548), 0.21),
    ("condenser_markings", (-0.03, -1.0, 0.78), (-0.205, 0.008, 0.756), 0.30),
]:
    cam.location = eye
    cam.rotation_euler = (Vector(aim) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.ortho_scale = width
    scene.render.filepath = str(PREVIEWS / (label + ".png"))
    bpy.ops.render.render(write_still=True)
print("MODEL_COMPLETE", len(bpy.data.objects), "objects")
