"""Convert a plain URDF into a USD robot asset with Isaac Sim's URDF importer.

Run with the Isaac Sim interpreter (conda env ``isaacsim5.1``)::

    python scripts/urdf_to_usd/convert.py \
        --urdf third_party/urdf/universal_robot/ur_description/urdf/ur5e_generated.urdf \
        --out assets/robots/ur5e.usd

Materials are NOT a knob here: ``ImportConfig`` has no material option at all.
What comes out is whatever the source *visual* meshes carry -- DAE and OBJ bring
their materials along, STL has none and can only render the flat ``<color rgba>``
from the URDF. Pick the source repo accordingly; see ``prepare_urdf.py``.
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--urdf", required=True, help="Input plain .urdf (not xacro)")
_parser.add_argument("--out", required=True, help="Output .usd path")
_parser.add_argument(
    "--merge-fixed-joints",
    action="store_true",
    help="Collapse fixed joints. Off by default so frames like flange/tool0 survive "
    "-- they are needed later as the end-effector frame.",
)
_parser.add_argument(
    "--no-fix-base",
    action="store_true",
    help="Leave the base free-floating. Default is a fixed base (arm bolted down).",
)
_parser.add_argument(
    "--collision-from-visuals",
    action="store_true",
    help="Derive collision from visual meshes instead of the URDF collision meshes.",
)
_args = _parser.parse_args()

# SimulationApp must be constructed before any omni/isaacsim module is imported.
_app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
from isaacsim.asset.importer.urdf import _urdf  # noqa: E402
from pxr import Usd  # noqa: E402


def _say(message: str) -> None:
    """Print progress on stderr.

    Kit installs its own stdout sink once SimulationApp starts, which swallows
    plain ``print``. stderr survives, so all progress goes there.
    """
    print(f"[convert] {message}", file=sys.stderr, flush=True)


def main() -> int:
    urdf_path = Path(_args.urdf).resolve()
    out_path = Path(_args.out).resolve()

    if not urdf_path.is_file():
        print(f"ERROR: URDF not found: {urdf_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stale_layers = [out_path]
    config_dir = out_path.parent / "configuration"
    if config_dir.is_dir():
        stale_layers.extend(config_dir.glob(f"{out_path.stem}_*.usd"))
    for stale in stale_layers:
        if stale.is_file():
            stale.unlink()

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        print("ERROR: URDFCreateImportConfig failed", file=sys.stderr)
        return 1

    # URDF is metres and the LabUtopia stage is metres -- keep 1:1. Guards against
    # the scale trap where a unit conversion sneaks in as an xformOp:scale.
    import_config.distance_scale = 1.0
    import_config.merge_fixed_joints = _args.merge_fixed_joints
    import_config.fix_base = not _args.no_fix_base
    import_config.collision_from_visuals = _args.collision_from_visuals
    import_config.make_default_prim = True
    # This is a reusable robot asset, not a scene. A baked-in physics scene would
    # collide with the one LabUtopia's stage already owns when it gets referenced.
    import_config.create_physics_scene = False
    import_config.self_collision = False
    import_config.import_inertia_tensor = True
    # Takes a UrdfJointTargetType enum, not an int -- passing 1 raises TypeError.
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.parse_mimic = True

    _say(f"{urdf_path}")
    _say(f"  -> {out_path}")
    _say(
        f"  distance_scale=1.0 fix_base={import_config.fix_base} merge_fixed_joints={import_config.merge_fixed_joints}"
    )

    result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=str(out_path),
    )
    _say(f"importer returned: {result}")

    if not out_path.is_file():
        print(f"ERROR: no USD produced at {out_path}", file=sys.stderr)
        return 1

    stage = Usd.Stage.Open(str(out_path))
    if stage is None:
        print(f"ERROR: produced USD does not open: {out_path}", file=sys.stderr)
        return 1

    default_prim = stage.GetDefaultPrim()
    prim_count = sum(1 for _ in stage.Traverse())
    if not default_prim.IsValid() or prim_count == 0:
        print(f"ERROR: produced USD contains no valid default prim: {out_path}", file=sys.stderr)
        return 1

    # The importer emits a layered asset: a thin entry-point USD that references
    # a sibling configuration/ directory holding the geometry and physics layers.
    # Both must travel together -- report the real footprint, not the shell's.
    layers = [out_path] + (sorted(config_dir.glob(f"{out_path.stem}_*.usd")) if config_dir.is_dir() else [])
    total_mb = sum(p.stat().st_size for p in layers) / 1024 / 1024

    _say(f"OK  default_prim={default_prim.GetPath()}  prims={prim_count}  total={total_mb:.1f} MB")
    for layer in layers:
        _say(f"  layer {layer.relative_to(out_path.parent.parent)}  {layer.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    code = main()
    _app.close()
    sys.exit(code)
