"""Generate a Lula robot_descriptor.yaml (and RMPFlow config) from a URDF.

Every arm driven by the shared RMPFlow controllers needs three files: the URDF, a
``robot_descriptor.yaml`` naming the c-space joints, and an ``rmpflow_common.yaml`` of
policy gains. Only the descriptor is arm-specific, and everything in it is already in
the URDF -- writing it by hand is how Piper ended up with joint limits and gains that
did not match its own model.

The RMPFlow gains are copied from Isaac's Franka config rather than invented. Piper
shipped a hand-retuned copy whose ``target_rmp.min_metric_scalar`` was 1e-4 against
Franka's 2500, which effectively switched off task-space tracking: the arm stalled
part-way to every goal. Start from the known-good gains and only tune with evidence.

Usage::

    python scripts/urdf_to_usd/make_lula_descriptor.py \\
        --urdf third_party/urdf/arx/_generated/arx_x5.urdf \\
        --out-dir robots/arx_x5 --name arx_x5 \\
        --cspace joint1 joint2 joint3 joint4 joint5 joint6
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRANKA_RMPFLOW = (
    "isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/"
    "franka/rmpflow/franka_rmpflow_common.yaml"
)

# Per-joint defaults, scaled by how far down the chain a joint sits: proximal joints
# carry more inertia and are limited harder than the wrist. Mirrors the shape of the
# values Isaac ships for Franka.
ACCELERATION_LIMITS = [15.0, 10.0, 10.0, 12.5, 15.0, 20.0, 20.0]
JERK_LIMITS = [7500.0, 5000.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def find_franka_rmpflow() -> Path | None:
    for site in Path(sys.prefix, "lib").glob("python*/site-packages"):
        candidate = site / FRANKA_RMPFLOW
        if candidate.is_file():
            return candidate
    return None


def urdf_joints(urdf_path: Path) -> dict[str, dict]:
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        joints[joint.get("name")] = {
            "type": joint.get("type"),
            "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            "mimic": joint.find("mimic") is not None,
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
        }
    return joints


def root_link(urdf_path: Path) -> str:
    """The link that is never a child -- the base of the kinematic tree."""
    root = ET.parse(urdf_path).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    links = [link.get("name") for link in root.findall("link")]
    for name in links:
        if name not in children:
            return name
    return links[0]


def write_kinematic_urdf(source: Path, target: Path) -> None:
    """Write the standalone kinematic model consumed by Lula/RMPFlow.

    The full conversion URDF resolves meshes from the ignored ``third_party``
    source checkout. Lula only reads links, joints, limits, and mimic rules;
    visual geometry lives in the converted USD and RMPFlow collision geometry
    lives in ``robot_descriptor.yaml``. Removing visual/collision blocks keeps
    the committed runtime model portable after the source checkout is deleted.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
    for tag in ("gazebo", "transmission"):
        for element in root.findall(tag):
            root.remove(element)

    ET.indent(tree, space="  ")
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)


def auto_cspace(joints: dict[str, dict]) -> list[str]:
    """Arm joints: movable, not a mimic follower, and not a gripper finger.

    Gripper joints are excluded because RMPFlow steers the arm only -- the fingers are
    driven by the gripper controller and are pinned in the Lula model.
    """
    return [
        name
        for name, spec in joints.items()
        if spec["type"] in ("revolute", "continuous", "prismatic")
        and not spec["mimic"]
        and not any(tag in name.lower() for tag in ("finger", "gripper", "pad"))
    ]


def midpoint_default_q(joints: dict[str, dict], cspace: list[str]) -> list[float]:
    """Start each joint mid-range, clamped away from the limits.

    A default_q sitting on a limit makes the joint-limit RMP fight the task from step
    one; mid-range keeps the arm's nullspace attractor out of the way.
    """
    q = []
    for name in cspace:
        spec = joints[name]
        if spec["lower"] is None or spec["upper"] is None:
            q.append(0.0)
            continue
        q.append(round((spec["lower"] + spec["upper"]) / 2.0, 4))
    return q


def default_collision_links(urdf_path: Path) -> list[str]:
    """Wrist-and-beyond links, used as RMPFlow's collision control points.

    Only the far end of the chain matters for the self/obstacle avoidance spheres, and
    those are the links whose names differ most between arms.
    """
    root = ET.parse(urdf_path).getroot()
    names = [link.get("name") for link in root.findall("link")]
    return names[-4:] if len(names) > 4 else names


def write_rmpflow_config(source: Path, target: Path, cspace: list[str], collision_links: list[str]) -> None:
    """Copy Franka's gains but replace every arm-specific section.

    Copying the file wholesale is not enough: it carries Franka's own link names in
    ``body_collision_controllers`` and its base geometry in ``body_cylinders``, and Lula
    aborts with `"panda_link7" does not exist in kinematics` the moment another arm
    loads it. ``joint_limit_buffers`` is likewise per-joint.
    """
    import yaml

    config = yaml.safe_load(source.read_text())
    config["joint_limit_buffers"] = [0.01] * len(cspace)
    config["body_cylinders"] = []  # Franka's base cylinders; each arm would need its own
    config["body_collision_controllers"] = [{"name": name, "radius": 0.05} for name in collision_links]

    header = (
        "# RMPFlow gains copied from Isaac's Franka config -- a known-good baseline.\n"
        "# Arm-specific sections (joint_limit_buffers, body_cylinders,\n"
        "# body_collision_controllers) are regenerated for THIS arm: Franka's link names\n"
        "# make Lula abort on any other robot.\n"
        "# Generated by scripts/urdf_to_usd/make_lula_descriptor.py.\n"
    )
    target.write_text(header + yaml.safe_dump(config, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--out-dir", required=True, help="Robot package dir, e.g. robots/arx_x5")
    parser.add_argument("--name", required=True, help="Used for <name>_rmpflow_common.yaml")
    parser.add_argument("--cspace", nargs="*", default=None, help="Arm joints; auto-detected if omitted")
    parser.add_argument("--default-q", nargs="*", type=float, default=None)
    parser.add_argument(
        "--collision-link",
        action="append",
        default=[],
        help="Link used as an RMPFlow collision control point; repeatable",
    )
    parser.add_argument(
        "--fixed-joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Joint pinned in the Lula model (the gripper fingers); repeatable",
    )
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.is_file():
        say(f"ERROR: URDF not found: {urdf_path}")
        return 1

    joints = urdf_joints(urdf_path)
    cspace = args.cspace or auto_cspace(joints)
    missing = [j for j in cspace if j not in joints]
    if missing:
        say(f"ERROR: joints not in URDF: {missing}")
        return 1

    default_q = args.default_q or midpoint_default_q(joints, cspace)
    if len(default_q) != len(cspace):
        say(f"ERROR: default_q has {len(default_q)} values for {len(cspace)} joints")
        return 1

    out_dir = Path(args.out_dir).resolve()
    rmpflow_dir = out_dir / "rmpflow"
    rmpflow_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Generated by scripts/urdf_to_usd/make_lula_descriptor.py -- do not hand-tune",
        "# limits here; they are read from the URDF and must stay in sync with it.",
        "api_version: 1.0",
        "",
        "cspace:",
        *[f"    - {name}" for name in cspace],
        "",
        f"root_link: {root_link(urdf_path)}",
        "",
        "default_q: [",
        "    " + ", ".join(f"{v:.4f}" for v in default_q),
        "]",
        "",
        "# URDF limits, for reference:",
        *[f"#   {name}: [{joints[name]['lower']}, {joints[name]['upper']}]" for name in cspace],
        "",
        "acceleration_limits: [" + ", ".join(str(v) for v in ACCELERATION_LIMITS[: len(cspace)]) + "]",
        "jerk_limits: [" + ", ".join(str(v) for v in JERK_LIMITS[: len(cspace)]) + "]",
        "",
    ]

    if args.fixed_joint:
        lines.append("cspace_to_urdf_rules:")
        for entry in args.fixed_joint:
            name, _, value = entry.partition("=")
            lines += [f"    - name: {name}", "      rule: fixed", f"      value: {float(value)}"]
        lines.append("")

    descriptor = rmpflow_dir / "robot_descriptor.yaml"
    descriptor.write_text("\n".join(lines))
    say(f"[lula] wrote {descriptor.relative_to(REPO_ROOT)}  cspace={cspace}")
    say(f"[lula]   default_q={default_q}")

    franka_config = find_franka_rmpflow()
    target = rmpflow_dir / f"{args.name}_rmpflow_common.yaml"
    if franka_config is None:
        say("[lula] WARNING: Franka RMPFlow config not found; write the gains manually")
    elif target.exists():
        say(f"[lula] {target.name} already exists, left untouched")
    else:
        write_rmpflow_config(franka_config, target, cspace, args.collision_link or default_collision_links(urdf_path))
        say(f"[lula] wrote {target.relative_to(REPO_ROOT)} (Franka gains, this arm's links)")

    local_urdf = out_dir / urdf_path.name
    if not local_urdf.exists():
        write_kinematic_urdf(urdf_path, local_urdf)
        say(f"[lula] wrote kinematic URDF -> {local_urdf.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
