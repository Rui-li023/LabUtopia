"""Verify a converted robot USD against its source URDF, and render a preview.

Checks, in order (each prints PASS/FAIL and the numbers behind it):

1. articulation -- exactly one prim carries UsdPhysics.ArticulationRootAPI
2. joints       -- movable joint names and lower/upper limits match the URDF
3. materials    -- every visual mesh carries a material, and more than one distinct
                   material exists (a single material means the source mesh materials
                   were lost and everything collapsed to a flat colour)
4. scale        -- world-space bounding box is the physically expected size,
                   guarding against a unit conversion sneaking in as xformOp:scale
5. physics      -- (``--physics``) PhysX parses it into an articulation with the
                   expected DOFs and the position drives respond
6. preview      -- (``--render``) a PNG for eyeball inspection

Run with the Isaac Sim interpreter (conda env ``isaacsim5.1``)::

    python scripts/urdf_to_usd/inspect_usd.py \
        --usd assets/robots/ur5e.usd \
        --urdf third_party/urdf/universal_robot/ur_description/urdf/ur5e_generated.urdf \
        --expect-extent 0.93 --extent-tol 0.10 --physics \
        --render outputs/urdf_to_usd/ur5e.png

Note: all output goes to stderr -- Kit installs its own stdout sink that swallows
plain ``print``.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim import SimulationApp

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--usd", required=True, help="Converted robot USD to verify")
_parser.add_argument("--urdf", required=True, help="Source URDF to check against")
_parser.add_argument("--joint-tol", type=float, default=1e-4, help="Joint limit tolerance (rad)")
_parser.add_argument(
    "--expect-extent",
    type=float,
    default=None,
    help="Expected largest bounding-box dimension in metres at the URDF zero pose "
    "(for an arm this tracks the datasheet reach)",
)
_parser.add_argument("--extent-tol", type=float, default=0.10, help="Allowed extent error (m)")
_parser.add_argument("--render", default=None, help="Write a preview PNG to this path")
_parser.add_argument(
    "--physics",
    action="store_true",
    help="Also load the asset as a live articulation and drive a joint",
)
_args = _parser.parse_args()

# SimulationApp must exist before any omni/pxr import. Rendering needs a real
# renderer, so headless-with-RTX rather than a null pipeline.
_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import omni.replicator.core as rep  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

DEG_PER_RAD = 57.29577951308232


def _say(message: str) -> None:
    """Print on stderr -- Kit's stdout sink swallows plain print()."""
    print(message, file=sys.stderr, flush=True)


def _result(ok: bool, label: str, detail: str) -> bool:
    _say(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def urdf_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """Movable joint name -> (lower, upper) in radians/metres, from the URDF."""
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def walk(stage: Usd.Stage):
    """Traverse the stage *including* instance proxies.

    The importer marks each link's ``visuals``/``collisions`` scope instanceable,
    and a plain ``stage.Traverse()`` stops at an instanceable prim instead of
    descending into it -- so every mesh and material would be invisible to the
    checks below.
    """
    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def check_articulation(stage: Usd.Stage) -> bool:
    roots = [p for p in walk(stage) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    paths = [str(p.GetPath()) for p in roots]
    return _result(len(roots) == 1, "articulation root", f"{len(roots)} found {paths}")


def check_joints(stage: Usd.Stage, expected: dict[str, tuple[float, float]], tol: float) -> bool:
    """Compare USD joint limits against the URDF.

    USD angular joint limits are in DEGREES while URDF is in radians, so the USD
    values are converted back before comparing.
    """
    found: dict[str, tuple[float, float, bool]] = {}
    for prim in walk(stage):
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        if not (is_revolute or is_prismatic):
            continue
        joint = UsdPhysics.RevoluteJoint(prim) if is_revolute else UsdPhysics.PrismaticJoint(prim)
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            continue
        found[prim.GetName()] = (float(lower), float(upper), is_revolute)

    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    ok = not missing and not extra
    _result(ok, "joint set", f"{len(found)} movable; missing={missing} unexpected={extra}")

    for name in sorted(expected):
        if name not in found:
            continue
        want_lo, want_hi = expected[name]
        got_lo, got_hi, is_revolute = found[name]
        if is_revolute:  # USD stores angular limits in degrees
            got_lo, got_hi = got_lo / DEG_PER_RAD, got_hi / DEG_PER_RAD
        match = abs(got_lo - want_lo) <= tol and abs(got_hi - want_hi) <= tol
        ok &= _result(
            match,
            f"  limits {name}",
            f"usd=[{got_lo:+.6f}, {got_hi:+.6f}] urdf=[{want_lo:+.6f}, {want_hi:+.6f}]",
        )
    return ok


def _bound_material_names(prim: Usd.Prim) -> list[str]:
    """Materials affecting a mesh, whether bound to the mesh or to its subsets.

    A DAE mesh that uses several materials arrives as one UsdGeom.Mesh carrying
    UsdGeom.Subset children, with the material bound per subset. Asking the mesh
    itself for its bound material returns nothing in that case, so the subsets
    have to be checked too.
    """
    names: list[str] = []

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if material and material.GetPrim().IsValid():
        names.append(material.GetPrim().GetName())

    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset_material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()
        if subset_material and subset_material.GetPrim().IsValid():
            names.append(subset_material.GetPrim().GetName())

    return names


def check_materials(stage: Usd.Stage) -> bool:
    """Every visual mesh carries a material, and more than one distinct material exists.

    Collision meshes are excluded: they are invisible proxies and legitimately
    render with a placeholder material.
    """
    bound: list[str] = []
    unbound: list[str] = []
    materials = set()

    for prim in walk(stage):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        purpose = UsdGeom.Imageable(prim).ComputePurpose()
        if purpose == UsdGeom.Tokens.guide or "collision" in path.lower():
            continue

        names = _bound_material_names(prim)
        if names:
            bound.append(path)
            materials.update(names)
        else:
            unbound.append(path)

    ok = _result(
        not unbound and bool(bound),
        "visual meshes carrying a material",
        f"{len(bound)} bound, {len(unbound)} unbound",
    )
    for path in unbound[:10]:
        _say(f"         unbound: {path}")

    ok &= _result(
        len(materials) > 1,
        "distinct materials",
        f"{len(materials)} -> {sorted(materials)}",
    )
    return ok


def check_scale(stage: Usd.Stage, expect_extent: float | None, tol: float) -> bool:
    """Measure the world-space bounding box.

    Catches the unit trap where a mesh-level xformOp:scale (DAE files often carry
    a 1e-2/1e-4 unit conversion) silently shrinks or inflates the robot.

    The comparison uses the LARGEST bbox dimension, not height: an arm's URDF zero
    pose is usually stretched out horizontally rather than standing upright, so
    height says little while the longest span tracks the datasheet reach.
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    if bbox.IsEmpty():
        return _result(False, "bounding box", "empty -- no renderable geometry found")

    size: Gf.Vec3d = bbox.GetSize()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    _say(
        f"  [info] bbox size = ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) "
        f"stage units, metersPerUnit={meters_per_unit}"
    )

    if expect_extent is None:
        return True

    extent_m = max(size) * meters_per_unit
    return _result(
        abs(extent_m - expect_extent) <= tol,
        "scale",
        f"largest span={extent_m:.3f} m, expected {expect_extent:.3f} +/- {tol:.3f} m",
    )


def check_physics(usd_path: Path, expected_dof: int) -> bool:
    """Load the asset as a live articulation and drive a joint.

    Structural checks on the USD only prove the file is well formed. This proves
    PhysX actually parses it into an articulation with the expected DOFs and that
    the position drives respond -- the difference between "the file looks right"
    and "the robot works in a scene".
    """
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(usd_path=str(usd_path), prim_path="/World/Robot")
    articulation = SingleArticulation(prim_path="/World/Robot", name="robot_under_test")
    world.scene.add(articulation)
    world.reset()

    num_dof = articulation.num_dof
    ok = _result(
        num_dof == expected_dof,
        "articulation DOF",
        f"{num_dof} (expected {expected_dof}); names={list(articulation.dof_names)}",
    )

    start = articulation.get_joint_positions().copy()
    target = start.copy()
    target[0] += 0.5  # rad, well inside every UR joint limit
    articulation.apply_action(ArticulationAction(joint_positions=target))
    for _ in range(120):
        world.step(render=False)
    moved = float(articulation.get_joint_positions()[0] - start[0])

    ok &= _result(
        abs(moved - 0.5) < 0.05,
        "joint drive responds",
        f"commanded +0.500 rad on {articulation.dof_names[0]}, moved {moved:+.3f} rad",
    )
    return ok


def render_preview(usd_path: Path, out_png: Path) -> None:
    """Render a three-quarter view of the asset for eyeball inspection."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    stage = omni.usd.get_context().get_stage()
    robot = stage.DefinePrim("/World/Robot", "Xform")
    robot.GetReferences().AddReference(str(usd_path))

    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(3000.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 35.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(700.0)

    # Frame from the asset's own bounds so any robot lands in shot, whatever its
    # size or zero pose (an arm's zero pose is often a horizontal sprawl, not upright).
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(robot).ComputeAlignedRange()
    center = bbox.GetMidpoint()
    distance = max(max(bbox.GetSize()) * 1.6, 0.5)
    camera = rep.create.camera(
        position=(center[0] + distance, center[1] - distance, center[2] + distance * 0.6),
        look_at=tuple(float(v) for v in center),
    )
    product = rep.create.render_product(camera, (1280, 960))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(out_png.parent), rgb=True)
    writer.attach([product])

    for _ in range(60):  # let RTX accumulate before capture
        _app.update()
    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()

    written = sorted(out_png.parent.glob("rgb_*.png"))
    if written:
        written[-1].replace(out_png)
        _say(f"  [info] preview -> {out_png}")
    else:
        _say("  [warn] renderer produced no PNG")


def main() -> int:
    usd_path = Path(_args.usd).resolve()
    urdf_path = Path(_args.urdf).resolve()
    for path in (usd_path, urdf_path):
        if not path.is_file():
            _say(f"ERROR: not found: {path}")
            return 1

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        _say(f"ERROR: USD has no valid default prim: {usd_path}")
        return 1

    _say(f"\n=== verifying {usd_path} ===")
    _say(f"    against {urdf_path}\n")

    ok = True
    _say("1. structure")
    ok &= check_articulation(stage)
    ok &= check_joints(stage, urdf_joint_limits(urdf_path), _args.joint_tol)

    _say("2. materials")
    ok &= check_materials(stage)

    _say("3. scale")
    ok &= check_scale(stage, _args.expect_extent, _args.extent_tol)

    if _args.physics:
        _say("4. physics")
        ok &= check_physics(usd_path, expected_dof=len(urdf_joint_limits(urdf_path)))

    if _args.render:
        _say("5. preview")
        render_preview(usd_path, Path(_args.render).resolve())

    _say(f"\n=== {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} ===\n")
    return 0 if ok else 1


if __name__ == "__main__":
    import omni.usd

    code = main()
    _app.close()
    sys.exit(code)
