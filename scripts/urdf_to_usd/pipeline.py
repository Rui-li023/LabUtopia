"""URDF -> USD conversion and verification logic, free of CLI and app bootstrap.

Every function here assumes a ``SimulationApp`` is ALREADY running: importing this
module pulls in ``omni``/``pxr``, which only resolve after the app exists. Callers
(``convert.py``, ``inspect_usd.py``, ``batch.py``) construct the app first, then
import this module.

Splitting it out is what lets ``batch.py`` convert and verify a whole fleet inside a
single Isaac Sim process. Booting per robot would cost ~10 s each, and this repo
forbids running Isaac Sim processes concurrently, so one boot is the only fast path.
"""

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import omni.kit.commands
import omni.usd
from isaacsim.asset.importer.urdf import _urdf
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

DEG_PER_RAD = 57.29577951308232

# A mainstream arm spans somewhere between a desktop model and a large industrial
# reach. The unit trap this guards against (a DAE unit conversion landing in
# xformOp:scale) misses by 100x or 10000x, never by 10%, so a broad band catches it
# without needing a datasheet number per robot.
SANITY_MIN_EXTENT_M = 0.15
SANITY_MAX_EXTENT_M = 3.5


def say(message: str) -> None:
    """Print on stderr -- Kit installs a stdout sink that swallows plain print()."""
    print(message, file=sys.stderr, flush=True)


@dataclass
class CheckReport:
    """Outcome of verifying one converted asset."""

    name: str
    ok: bool = True
    joints: int = 0
    dof: int = 0
    materials: list[str] = field(default_factory=list)
    material_colors: int = 0
    extent_m: float = 0.0
    visual_meshes: int = 0
    unbound_meshes: int = 0
    failures: list[str] = field(default_factory=list)

    def record(self, ok: bool, label: str, detail: str) -> bool:
        say(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            self.failures.append(f"{label}: {detail}")
            self.ok = False
        return ok


# ---------------------------------------------------------------- conversion


def convert(
    urdf_path: Path,
    out_path: Path,
    *,
    fix_base: bool = True,
    merge_fixed_joints: bool = False,
    collision_from_visuals: bool = False,
) -> bool:
    """Import a plain URDF and write a USD robot asset. Returns True on success.

    Materials are NOT a knob: ``ImportConfig`` has no material option at all. What
    survives is whatever the source *visual* meshes carry -- DAE and OBJ bring their
    materials, STL has none and can only show the flat ``<color rgba>`` from the URDF.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Never let a failed import masquerade as success because a stale entry-point or
    # configuration layer happens to exist from an earlier run.
    stale_layers = [out_path]
    config_dir = out_path.parent / "configuration"
    if config_dir.is_dir():
        stale_layers.extend(config_dir.glob(f"{out_path.stem}_*.usd"))
    for stale in stale_layers:
        if stale.is_file():
            stale.unlink()

    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        say("  ERROR: URDFCreateImportConfig failed")
        return False

    # URDF is metres and the LabUtopia stage is metres -- keep 1:1.
    config.distance_scale = 1.0
    config.merge_fixed_joints = merge_fixed_joints
    config.fix_base = fix_base
    config.collision_from_visuals = collision_from_visuals
    config.make_default_prim = True
    # A reusable robot asset, not a scene: a baked-in physics scene would collide
    # with the one LabUtopia's stage already owns when this gets referenced.
    config.create_physics_scene = False
    config.self_collision = False
    config.import_inertia_tensor = True
    # Wants a UrdfJointTargetType enum -- passing an int raises TypeError.
    config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    config.parse_mimic = True

    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=config,
        dest_path=str(out_path),
    )
    if not out_path.is_file():
        return False

    stage = Usd.Stage.Open(str(out_path), Usd.Stage.LoadAll)
    return bool(stage and list(stage.GetPseudoRoot().GetChildren()))


def asset_layers(out_path: Path) -> list[Path]:
    """All files making up the asset.

    The importer emits a layered asset: a thin entry-point USD referencing a sibling
    ``configuration/`` directory that holds the geometry and physics layers. They must
    travel together -- copying only the entry point yields an empty shell.
    """
    config_dir = out_path.parent / "configuration"
    layers = [out_path]
    if config_dir.is_dir():
        layers += sorted(config_dir.glob(f"{out_path.stem}_*.usd"))
    return [p for p in layers if p.is_file()]


def asset_size_mb(out_path: Path) -> float:
    return sum(p.stat().st_size for p in asset_layers(out_path)) / 1024 / 1024


# ---------------------------------------------------------------- verification


def urdf_movable_joints(urdf_path: Path) -> dict[str, tuple[float, float] | None]:
    """Movable joint name -> (lower, upper) in radians/metres, or None if unbounded.

    ``continuous`` joints spin without limits and carry no lower/upper in the URDF --
    Kinova's arms use them heavily. They are still DOFs, so they must be counted as
    movable; only their limits are skipped when comparing against the USD.
    """
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, tuple[float, float] | None] = {}
    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type == "continuous":
            joints[joint.get("name")] = None
            continue
        if joint_type not in ("revolute", "prismatic"):
            continue

        # A mimic FOLLOWER's limits are recomputed by the importer from the mimic
        # relation rather than copied from the URDF, so comparing them reports a
        # false mismatch. Measured on widowx_vx300s: the leader (left_finger) comes
        # through exactly, while the follower (right_finger) is rewritten
        # [-0.057,-0.021] -> [-0.0642,-0.0138]. Counted as a DOF, limits not compared.
        if joint.find("mimic") is not None:
            joints[joint.get("name")] = None
            continue

        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            joints[joint.get("name")] = None
            continue
        joints[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return joints


def urdf_has_rich_meshes(urdf_path: Path) -> bool:
    """True when the URDF references DAE/OBJ visual meshes, which carry materials.

    Used to decide how many distinct materials to demand: an STL-only description
    legitimately produces exactly one flat colour, so failing it would be wrong.
    """
    text = urdf_path.read_text().lower()
    # GLB/glTF carry full PBR materials (often texture-driven) just like DAE and OBJ.
    return any(ext in text for ext in (".dae", ".obj", ".glb", ".gltf"))


def walk(stage: Usd.Stage):
    """Traverse the stage INCLUDING instance proxies.

    The importer marks each link's ``visuals``/``collisions`` scope instanceable, and
    a plain ``stage.Traverse()`` stops at an instanceable prim rather than descending
    into it -- so every mesh and material would be invisible to the checks.
    """
    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def count_usd_movable_joints(stage: Usd.Stage) -> int:
    """Movable joints present in the USD itself, for assets that ship no URDF."""
    return sum(1 for p in walk(stage) if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint))


def check_articulation(stage: Usd.Stage, report: CheckReport) -> None:
    roots = [p for p in walk(stage) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    report.record(len(roots) == 1, "articulation root", f"{len(roots)} found")


def check_joints(
    stage: Usd.Stage, expected: dict[str, tuple[float, float] | None], tol: float, report: CheckReport
) -> None:
    """Compare USD joint limits against the URDF.

    USD stores ANGULAR limits in degrees while URDF uses radians, so revolute values
    are converted back before comparing.
    """
    found: dict[str, tuple[float, float, bool]] = {}
    for prim in walk(stage):
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        if not (is_revolute or is_prismatic):
            continue
        joint = UsdPhysics.RevoluteJoint(prim) if is_revolute else UsdPhysics.PrismaticJoint(prim)
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            continue
        found[prim.GetName()] = (float(lower), float(upper), is_revolute)

    report.joints = len(found)
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    report.record(
        not missing and not extra,
        "joint set",
        f"{len(found)} movable vs {len(expected)} in URDF; missing={missing} unexpected={extra}",
    )

    mismatched = []
    compared = 0
    for name, want in expected.items():
        if name not in found or want is None:  # continuous joints have nothing to compare
            continue
        want_lo, want_hi = want
        got_lo, got_hi, is_revolute = found[name]
        if is_revolute:
            got_lo, got_hi = got_lo / DEG_PER_RAD, got_hi / DEG_PER_RAD
        compared += 1
        if abs(got_lo - want_lo) > tol or abs(got_hi - want_hi) > tol:
            mismatched.append(f"{name}: usd=[{got_lo:+.4f},{got_hi:+.4f}] urdf=[{want_lo:+.4f},{want_hi:+.4f}]")

    report.record(
        not mismatched,
        "joint limits match URDF",
        f"{compared} compared, {len(mismatched)} mismatched {mismatched[:3]}",
    )


# Shader input names differ per source format: UsdPreviewSurface uses diffuseColor,
# OmniPBR uses diffuse_color_constant, and glTF/GLB imports use gltf/pbr.mdl with
# base_color_factor -- missing that one made every ARX material read as "no colour"
# and collapsed 7 distinct materials into 1.
DIFFUSE_INPUTS = (
    "diffuseColor",
    "diffuse_color_constant",
    "base_color_constant",
    "base_color_factor",
)
TEXTURE_INPUTS = ("diffuse_texture", "file", "base_color_texture", "albedo_texture")


def material_signature(material_prim: Usd.Prim) -> str | None:
    """A material's visual identity: its diffuse colour, or its texture if driven by one.

    GLB/glTF materials usually drive base colour from a texture rather than a constant,
    so a colour-only probe reads None for every one of them and collapses genuinely
    different materials into "1 distinct". Falling back to the texture asset keeps them
    apart. Returns None only when neither is authored.
    """
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        for input_name in DIFFUSE_INPUTS:
            shader_input = shader.GetInput(input_name)
            if shader_input is not None and shader_input.Get() is not None:
                return str(tuple(round(float(c), 3) for c in shader_input.Get()))
        for input_name in TEXTURE_INPUTS:
            shader_input = shader.GetInput(input_name)
            if shader_input is not None and shader_input.Get() is not None:
                return f"tex:{shader_input.Get()}"
    return None


def bound_materials(prim: Usd.Prim) -> list[Usd.Prim]:
    """Material prims affecting a mesh, bound either to it or to its GeomSubsets.

    A DAE mesh using several materials arrives as ONE UsdGeom.Mesh with UsdGeom.Subset
    children, the material bound per subset. Asking the mesh itself returns nothing in
    that case, so subsets must be checked too.
    """
    materials: list[Usd.Prim] = []
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material and material.GetPrim().IsValid():
        materials.append(material.GetPrim())
    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset_material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()
        if subset_material and subset_material.GetPrim().IsValid():
            materials.append(subset_material.GetPrim())
    return materials


def check_materials(stage: Usd.Stage, min_materials: int, report: CheckReport) -> None:
    """Every visual mesh carries a material, and enough distinct materials survived.

    ``min_materials`` is 2 for DAE/OBJ sources (losing them collapses everything to one
    flat colour) and 1 for STL-only sources, where one colour is the honest ceiling.
    Collision meshes are excluded -- they are invisible proxies with placeholder materials.

    Distinctness is counted by DIFFUSE COLOUR, not by material name. Vendors reuse names:
    every material in Kinova's DAE files is called ``Material_001``, so name-based
    dedup collapsed 32 genuinely different materials (3 distinct greys) into "1" and
    reported a material loss that had not happened. Colour is what the check is
    actually about.
    """
    bound, unbound = [], []
    colors: set[str | None] = set()
    names: set[str] = set()

    for prim in walk(stage):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide or "collision" in path.lower():
            continue
        materials = bound_materials(prim)
        (bound if materials else unbound).append(path)
        for material_prim in materials:
            names.add(material_prim.GetName())
            colors.add(material_signature(material_prim))

    report.visual_meshes = len(bound) + len(unbound)
    report.unbound_meshes = len(unbound)
    report.materials = sorted(names)
    report.material_colors = len(colors)

    report.record(
        not unbound and bool(bound),
        "visual meshes carry a material",
        f"{len(bound)} bound, {len(unbound)} unbound",
    )
    report.record(
        len(colors) >= min_materials,
        f"distinct material colours (need >={min_materials})",
        f"{len(colors)} colours across {len(names)} name(s) {sorted(names)[:5]}",
    )


def check_scale(stage: Usd.Stage, report: CheckReport, datasheet_reach_m: float | None) -> None:
    """Measure the world-space bounding box and sanity-check its magnitude.

    Compares the LARGEST dimension, not height: an arm's URDF zero pose is usually a
    horizontal sprawl, so height says little while the longest span tracks reach.
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if bbox.IsEmpty():
        report.record(False, "bounding box", "empty -- no renderable geometry")
        return

    size = bbox.GetSize()
    extent = max(size) * UsdGeom.GetStageMetersPerUnit(stage)
    report.extent_m = extent

    detail = f"largest span={extent:.3f} m"
    if datasheet_reach_m:
        detail += f" (datasheet reach ~{datasheet_reach_m:.2f} m)"
    report.record(SANITY_MIN_EXTENT_M <= extent <= SANITY_MAX_EXTENT_M, "scale sane", detail)


def check_physics(usd_path: Path, expected_dof: int, report: CheckReport, *, drive_test: bool = True) -> None:
    """Load the asset as a live articulation and drive a joint.

    Structural checks only prove the file is well formed. This proves PhysX parses it
    into an articulation with the expected DOFs and that the position drives respond --
    the difference between "the file looks right" and "the robot works in a scene".
    """
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import create_new_stage
    from isaacsim.core.utils.types import ArticulationAction

    # World is a SINGLETON. Across a batch, a second World(...) hands back the previous
    # instance whose physics view PhysX has already invalidated, and the next
    # SingleArticulation dies with "'NoneType' object has no attribute 'link_names'".
    # A fresh stage plus clear_instance() (below) is what makes each robot independent.
    create_new_stage()
    World.clear_instance()

    world = World(stage_units_in_meters=1.0)
    add_robot_reference(omni.usd.get_context().get_stage().DefinePrim("/World/Robot", "Xform"), usd_path)
    articulation = SingleArticulation(prim_path="/World/Robot", name=f"test_{report.name}")
    world.scene.add(articulation)
    world.reset()

    report.dof = articulation.num_dof
    if drive_test:
        report.record(
            articulation.num_dof == expected_dof,
            "articulation DOF",
            f"{articulation.num_dof} (expected {expected_dof})",
        )
    else:
        # A closed loop legitimately resolves to fewer DOFs than it has joint prims:
        # Robotiq 2F-85 has 10 revolute joints but 8 DOF, the other two being
        # loop-closure constraints. Report it, do not assert equality.
        say(f"  [info] closed-loop component: {articulation.num_dof} DOF from {expected_dof} joint prim(s)")

    if not drive_test:
        say("  [info] closed-loop component: DOF checked, free-drive response not asserted")
        world.stop()
        World.clear_instance()
        return

    start = articulation.get_joint_positions().copy()
    target = start.copy()
    target[0] += 0.3  # rad -- small enough to stay inside every arm's first-joint limit
    articulation.apply_action(ArticulationAction(joint_positions=target))
    for _ in range(120):
        world.step(render=False)
    moved = float(articulation.get_joint_positions()[0] - start[0])

    report.record(abs(moved - 0.3) < 0.05, "joint drive responds", f"commanded +0.300 rad, moved {moved:+.3f} rad")

    world.stop()
    World.clear_instance()


def verify(
    usd_path: Path,
    urdf_path: Path | None,
    name: str,
    *,
    joint_tol: float,
    datasheet_reach_m: float | None,
    drive_test: bool = True,
    min_material_colors: int | None = None,
) -> CheckReport:
    """Run every structural check on an asset.

    ``urdf_path`` may be None for a downloaded official asset, which ships no URDF: joint
    limits then have nothing to compare against and the DOF count comes from the USD.

    ``drive_test`` should be False for a closed-loop component (a gripper), whose lead
    joint legitimately will not track a free position step because the linkage constrains it.
    """
    report = CheckReport(name=name)

    # Payloads are a separate load pass: NVIDIA's Robotiq assets keep their geometry in a
    # payloads/ sublayer, so a default Open() composes a stage with zero meshes.
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        report.record(False, "open USD", f"cannot open: {usd_path}")
        return report

    # A sublayer holding the actual geometry (Robotiq's payloads/*_base.usd) has content
    # but no defaultPrim -- that is normal for a layer meant to be referenced, so fall
    # back to the first top-level prim instead of rejecting the asset.
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            report.record(False, "open USD", f"stage has no prims: {usd_path}")
            return report
        root = children[0]
        say(f"  [info] no defaultPrim; using first top-level prim {root.GetPath()}")
        stage.SetDefaultPrim(root)

    check_articulation(stage, report)

    if urdf_path is not None:
        limits = urdf_movable_joints(urdf_path)
        check_joints(stage, limits, joint_tol, report)
        min_materials = (
            min_material_colors if min_material_colors is not None else (2 if urdf_has_rich_meshes(urdf_path) else 1)
        )
        expected_dof = len(limits)
    else:
        expected_dof = count_usd_movable_joints(stage)
        report.joints = expected_dof
        say(f"  [info] no URDF to compare against; {expected_dof} movable joint(s) in USD")
        min_materials = 1  # nothing to infer a richer expectation from

    check_materials(stage, min_materials, report)
    check_scale(stage, report, datasheet_reach_m)

    # Physics is the one check that can throw rather than merely fail (PhysX rejecting
    # a malformed articulation). In a batch that must not take down the other robots.
    try:
        check_physics(usd_path, expected_dof, report, drive_test=drive_test)
    except Exception as exc:
        report.record(False, "physics", f"{type(exc).__name__}: {exc}")
    return report


# ---------------------------------------------------------------- preview


# Roughly "showroom" joint angles, applied to the first DOFs and clamped to each
# joint's own limits. A URDF zero pose is a poor showroom pose for many arms: the
# Kinova Gen3 stands as a 1.19 m x 0.09 m pencil, the xArm folds back onto itself, and
# the Fanuc's forearm points straight at a fixed camera. Unfolding the elbow and wrist
# makes every arm read as an arm.
SHOWROOM_POSE_RAD = (0.0, -0.5, 0.5, 0.0, 0.8, 0.0, 0.3)


def source_default_prim(usd_path: Path) -> str | None:
    """Prim path to reference from a source USD.

    ``AddReference(file)`` targets ``<defaultPrim>``; a layer without one (Robotiq's
    payloads/*_base.usd) then composes to an unresolved reference and renders nothing.
    Naming the first top-level prim explicitly fixes it.
    """
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        return None
    if stage.GetDefaultPrim().IsValid():
        return None  # the default works; no explicit path needed
    children = list(stage.GetPseudoRoot().GetChildren())
    return str(children[0].GetPath()) if children else None


def add_robot_reference(prim, usd_path: Path) -> None:
    """Reference a source USD, naming its root prim when the file lacks a defaultPrim."""
    explicit = source_default_prim(usd_path)
    if explicit:
        prim.GetReferences().AddReference(str(usd_path), explicit)
    else:
        prim.GetReferences().AddReference(str(usd_path))


def pose_for_preview(usd_path: Path, pose: tuple[float, ...] | None = None) -> None:
    """Load the asset as an articulation and set a showroom pose on the current stage.

    Falls back to the zero pose if anything about the articulation is unexpected --
    a preview is never worth failing the run over.
    """
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    World.clear_instance()  # the singleton would otherwise hand back a dead physics view
    world = World(stage_units_in_meters=1.0)
    # No gravity for a preview. A fixed-base arm is anchored, but a mobile manipulator
    # (Ridgeback+Franka) free-falls during the settling steps and leaves the frame --
    # the render came out an empty background even though the asset was fine.
    world.get_physics_context().set_gravity(0.0)
    add_robot_reference(omni.usd.get_context().get_stage().DefinePrim("/World/Robot", "Xform"), usd_path)
    articulation = SingleArticulation(prim_path="/World/Robot", name="preview_robot")
    world.scene.add(articulation)
    world.reset()

    wanted_pose = pose or SHOWROOM_POSE_RAD
    lower, upper = articulation.dof_properties["lower"], articulation.dof_properties["upper"]
    target = np.zeros(articulation.num_dof, dtype=np.float32)
    for index in range(articulation.num_dof):
        wanted = wanted_pose[index] if index < len(wanted_pose) else 0.0
        target[index] = float(np.clip(wanted, lower[index], upper[index]))

    # Both are needed. set_joint_positions alone only moves the STATE; the position
    # drives still target zero, so the next physics steps drag the arm straight back to
    # the zero pose and the preview looks unposed.

    articulation.set_joint_positions(target)
    articulation.apply_action(ArticulationAction(joint_positions=target))
    for _ in range(60):
        world.step(render=False)

    # Deliberately no world.stop(): stopping restores the pre-play state, which undoes
    # the pose right before the render (and tears down the articulation view, so
    # get_joint_positions() then returns None). Leave the timeline running.


def render_preview(app, usd_path: Path, out_png: Path, pose: tuple[float, ...] | None = None) -> bool:
    """Render a three-quarter view of the asset. Returns True if a PNG was written."""
    import omni.replicator.core as rep
    from isaacsim.core.utils.stage import create_new_stage

    out_png.parent.mkdir(parents=True, exist_ok=True)
    create_new_stage()

    try:
        pose_for_preview(usd_path, pose)
    except Exception as exc:
        say(f"  [warn] could not pose for preview ({type(exc).__name__}); using zero pose")
        omni.usd.get_context().new_stage()
        add_robot_reference(omni.usd.get_context().get_stage().DefinePrim("/World/Robot", "Xform"), usd_path)

    stage = omni.usd.get_context().get_stage()
    robot = stage.GetPrimAtPath("/World/Robot")

    # Dark surround: most arms are white or light grey, and on a light background their
    # silhouettes disappear.
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(6000.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 35.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(0.10, 0.11, 0.14))

    # Frame from the asset's own bounds so any robot lands in shot, whatever its size
    # or zero pose.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(robot).ComputeAlignedRange()
    if bbox.IsEmpty():
        say("  [warn] nothing to render")
        return False
    center = bbox.GetMidpoint()
    size = bbox.GetSize()

    # Orbit so the arm's longest HORIZONTAL axis lies across the frame instead of
    # pointing at the lens. A fixed azimuth foreshortens any arm whose zero pose
    # happens to extend toward the camera -- the Fanuc LR Mate reads as a solid blob
    # from (1,-1,0.55) but as an obvious articulated arm viewed side-on.
    if size[0] >= size[1]:
        direction = Gf.Vec3d(0.35, -1.0, 0.45).GetNormalized()  # x is long -> look along -y
    else:
        direction = Gf.Vec3d(1.0, 0.35, 0.45).GetNormalized()  # y is long -> look along +x

    # Place the camera at a true distance along that direction. Scaling the offset
    # components directly puts the camera |(1,-1,0.55)| = 1.52x further out than
    # intended, which shrank tall arms to a speck in frame.
    distance = max(max(size) * 1.35, 0.4)
    eye = Gf.Vec3d(center) + direction * distance

    camera = rep.create.camera(
        position=(float(eye[0]), float(eye[1]), float(eye[2])),
        look_at=tuple(float(v) for v in center),
        # Replicator's default camera near-clips at ~1 m. Bench-scale arms sit well
        # inside that: framing a 0.55 m arm snugly puts the camera 0.75 m away and the
        # whole robot vanishes behind the near plane -- a blank grey frame. Widening the
        # range is what lets the camera come in close enough to fill the frame.
        clipping_range=(0.01, 1000.0),
    )
    product = rep.create.render_product(camera, (900, 900))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(out_png.parent), rgb=True)
    writer.attach([product])

    # RTX accumulates progressively; capturing early leaves colour speckle over large
    # flat-shaded areas (very visible on the Fanuc's single yellow body).
    for _ in range(150):
        app.update()
    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    writer.detach()
    product.destroy()

    produced = sorted(out_png.parent.glob("rgb_*.png"))
    if not produced:
        say("  [warn] renderer produced no PNG")
        return False
    produced[-1].replace(out_png)
    for leftover in produced[:-1]:
        leftover.unlink()
    return True
