"""Expand a xacro robot description into a plain URDF that Isaac Sim can import.

Two things stand between a ROS ``*_description`` package and Isaac Sim's URDF
importer:

1. ROS ships xacro macros, not plain URDF. Expanding them needs ``xacro`` +
   ``rospkg``, which must NOT be installed into the ``isaacsim5.1`` env.
2. The expanded URDF references meshes as ``package://<pkg>/meshes/...``.
   Isaac Sim's importer does not resolve ROS package URIs -- NVIDIA's own
   bundled samples use relative ``../meshes/...`` paths instead. So the URIs
   are rewritten to paths relative to the output URDF.

Run this with the throwaway xacro venv, not with the Isaac Sim interpreter::

    /tmp/xacro_venv/bin/python scripts/urdf_to_usd/prepare_urdf.py \
        --repo third_party/urdf/universal_robot \
        --package ur_description \
        --xacro urdf/ur5e.xacro \
        --out third_party/urdf/universal_robot/ur_description/urdf/ur5e_generated.urdf

The output is written *inside* the package's ``urdf/`` directory on purpose, so
that ``../meshes/...`` resolves naturally -- the same layout NVIDIA uses for its
bundled ur10 sample.
"""

import argparse
import math
import os
import re
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

# ROS package names conventionally use letters, digits, and underscores, but
# real vendor descriptions also use directory-style package aliases containing
# hyphens (Mobile ALOHA references ``package://arx5-urdf/...``).  Match those
# aliases as well so unresolved meshes are rewritten or reported instead of
# silently passing through to Isaac's importer.
PACKAGE_URI_RE = re.compile(r"package://([A-Za-z0-9_.-]+)/([^\"'\s]+)")

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def install_ament_stub(package_roots: dict[str, Path]) -> None:
    """Teach xacro how to resolve ``$(find <pkg>)`` without a ROS installation.

    The ``xacro`` package on PyPI is the ROS 2 flavour: it resolves ``$(find)``
    via ``ament_index_python.packages.get_package_share_directory`` (a function-local
    import in ``xacro/substitution_args.py``), ignoring ROS_PACKAGE_PATH entirely.
    ``ament_index_python`` ships only with ROS 2, not PyPI.

    Since the import is function-local, injecting a stub into ``sys.modules``
    beforehand is enough -- and it keeps package resolution pointed at the cloned
    repo rather than at whatever ROS may be installed on the machine.
    """
    packages_mod = types.ModuleType("ament_index_python.packages")

    def get_package_share_directory(package_name: str) -> str:
        root = package_roots.get(package_name)
        if root is None:
            raise KeyError(f"package '{package_name}' not found in the cloned repo. Available: {sorted(package_roots)}")
        return str(root)

    packages_mod.get_package_share_directory = get_package_share_directory

    ament_mod = types.ModuleType("ament_index_python")
    ament_mod.packages = packages_mod

    sys.modules["ament_index_python"] = ament_mod
    sys.modules["ament_index_python.packages"] = packages_mod


def expand_xacro(package_roots: dict[str, Path], xacro_path: Path, mappings: dict[str, str]) -> str:
    """Expand a xacro file to URDF XML text.

    ``mappings`` supplies ``$(arg ...)`` values. Several vendors ship a single
    parameterised top-level xacro rather than one file per model -- xArm's
    ``xarm_device.urdf.xacro`` needs ``dof``/``robot_type``, Interbotix's arms
    need ``robot_model`` -- so the model is chosen through these.
    """
    install_ament_stub(package_roots)

    import xacro  # imported late: only available inside the xacro venv

    doc = xacro.process_file(str(xacro_path), mappings=mappings)
    return doc.toprettyxml(indent="  ")


def load_description(package_roots: dict[str, Path], source_path: Path, mappings: dict[str, str]) -> str:
    """Return URDF XML text from either a xacro or an already-plain URDF.

    Not every vendor ships xacro: Doosan's ``dsr_description`` has plain ``.urdf``
    files, which only need the ``package://`` rewrite.
    """
    if source_path.suffix == ".xacro" or source_path.name.endswith(".urdf.xacro"):
        print(f"[prepare] expanding xacro {source_path}")
        return expand_xacro(package_roots, source_path, mappings)

    print(f"[prepare] plain URDF (no xacro expansion) {source_path}")
    return source_path.read_text()


def parse_mappings(pairs: list[str]) -> dict[str, str]:
    """Parse ``--arg name:=value`` (ROS style) or ``name=value``."""
    mappings: dict[str, str] = {}
    for pair in pairs:
        if ":=" in pair:
            key, value = pair.split(":=", 1)
        elif "=" in pair:
            key, value = pair.split("=", 1)
        else:
            raise ValueError(f"--arg expects name:=value, got {pair!r}")
        mappings[key.strip()] = value.strip()
    return mappings


def find_package_roots(repo_root: Path) -> dict[str, Path]:
    """Map ROS package name -> package directory.

    Primarily by locating ``package.xml`` files. Some repos also reference a plain
    directory as if it were a package: mobile_aloha's URDF asks for
    ``package://tracer/tracer_description/meshes/...`` where ``tracer/`` is just a
    folder holding the real package. Those directory names are registered as a
    fallback, never overriding a genuine package.
    """
    roots: dict[str, Path] = {}
    fallback: dict[str, Path] = {}
    for manifest in repo_root.rglob("package.xml"):
        roots[manifest.parent.name] = manifest.parent
        parent = manifest.parent.parent
        if parent != repo_root:
            fallback.setdefault(parent.name, parent)
    for name, path in fallback.items():
        roots.setdefault(name, path)
    return roots


def rewrite_package_uris(urdf_text: str, package_roots: dict[str, Path], out_path: Path) -> tuple[str, list[str]]:
    """Rewrite ``package://`` mesh URIs to paths relative to ``out_path``.

    Returns the rewritten XML and a list of URIs that could not be resolved.
    """
    unresolved: list[str] = []
    out_dir = out_path.parent

    def _replace(match: re.Match) -> str:
        pkg, rel = match.group(1), match.group(2)
        pkg_root = package_roots.get(pkg)
        if pkg_root is None:
            unresolved.append(match.group(0))
            return match.group(0)

        target = (pkg_root / rel).resolve()
        if not target.exists():
            unresolved.append(f"{match.group(0)} -> {target} (missing)")
            return match.group(0)

        return os.path.relpath(target, out_dir)

    return PACKAGE_URI_RE.sub(_replace, urdf_text), unresolved


def rewrite_absolute_mesh_paths(urdf_text: str, out_path: Path) -> tuple[str, int]:
    """Make xacro-expanded ``file://$(find pkg)`` mesh paths portable.

    Xacro expands ``$(find pkg)`` before this script strips ``file://``, leaving an
    absolute path tied to the preparation machine. Package URIs do not have this
    problem because :func:`rewrite_package_uris` sees them before expansion. Parse the
    generated XML and rewrite only mesh filenames; plugin/library paths are unrelated.
    """
    root = ET.fromstring(urdf_text)
    rewritten = 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is None or not Path(filename).is_absolute():
            continue
        mesh.set("filename", os.path.relpath(filename, out_path.parent))
        rewritten += 1
    if not rewritten:
        return urdf_text, 0
    return ET.tostring(root, encoding="unicode"), rewritten


def sanitize_mesh_filenames(urdf_text: str, out_path: Path) -> tuple[str, int]:
    """Alias meshes whose stems cannot be used as USD prim identifiers.

    Isaac Sim's URDF importer uses the source mesh stem as a child prim name without
    always applying ``Tf.MakeValidIdentifier`` first. Techman's ``tm5-base.obj`` then
    becomes ``/visuals/link_0/tm5-base``, an invalid SdfPath, and the entire import
    aborts with ``Used null prim``. Keep the vendor files untouched and create a
    same-directory alias such as ``tm5_base.obj`` so OBJ sidecar MTL references still
    resolve naturally.

    This is opt-in because most vendor assets already import correctly, and copying
    large meshes unnecessarily would waste disk space in the ignored source clones.
    """
    root = ET.fromstring(urdf_text)
    rewritten = 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue

        source = Path(filename)
        if not source.is_absolute():
            source = (out_path.parent / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"mesh referenced by URDF does not exist: {source}")

        safe_stem = re.sub(r"[^A-Za-z0-9_]", "_", source.stem)
        if not safe_stem or safe_stem[0].isdigit():
            safe_stem = f"_{safe_stem}"
        if safe_stem == source.stem:
            continue

        alias = source.with_name(f"{safe_stem}{source.suffix}")
        if not alias.is_file() or alias.stat().st_size != source.stat().st_size:
            shutil.copy2(source, alias)
        mesh.set("filename", os.path.relpath(alias, out_path.parent))
        rewritten += 1

    if not rewritten:
        return urdf_text, 0
    return ET.tostring(root, encoding="unicode"), rewritten


def bound_continuous_joints(urdf_text: str, position_limit: float) -> tuple[str, int]:
    """Give continuous joints finite planning limits for Lula/RMPFlow.

    Lula requires every c-space coordinate to have position limits. Some vendor
    descriptions model multi-turn arm axes as ``continuous``, which is faithful to
    the hardware but cannot be loaded by Lula. For robots that opt in through the
    manifest, represent those axes as revolute joints over a symmetric planning
    interval. Existing effort and velocity limits are preserved.
    """
    if position_limit <= 0:
        raise ValueError("continuous joint position limit must be positive")

    root = ET.fromstring(urdf_text)
    rewritten = 0
    for joint in root.findall("joint"):
        if joint.get("type") != "continuous":
            continue
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", str(-position_limit))
        limit.set("upper", str(position_limit))
        rewritten += 1

    if not rewritten:
        return urdf_text, 0
    return ET.tostring(root, encoding="unicode"), rewritten


def _rpy_matrix(rpy: Vector3) -> Matrix3:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _multiply_matrix(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(left[row][inner] * right[inner][column] for inner in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matrix_rpy(matrix: Matrix3) -> Vector3:
    """Convert a rotation matrix back to URDF fixed-axis roll/pitch/yaw."""
    sin_pitch = max(-1.0, min(1.0, -matrix[2][0]))
    pitch = math.asin(sin_pitch)
    if abs(math.cos(pitch)) > 1e-10:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        # At gimbal lock choose yaw=0; this preserves the represented rotation.
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return roll, pitch, yaw


def _parse_vector(value: str, attribute: str) -> Vector3:
    parts = value.split()
    if len(parts) != 3:
        raise ValueError(f"{attribute} must contain three values, got {value!r}")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ValueError(f"{attribute} contains a non-numeric value: {value!r}") from exc


def _format_vector(vector: Vector3) -> str:
    cleaned = (0.0 if abs(value) < 1e-14 else value for value in vector)
    return " ".join(format(value, ".17g") for value in cleaned)


def rotate_visual_origins(urdf_text: str, correction_rpy: Vector3) -> tuple[str, int]:
    """Rotate every visual mesh in its local frame while preserving existing origins.

    Some glTF assets use Y-up coordinates even though their URDF is Z-up. The URDF
    importer does not infer this convention from GLB metadata, so the correction must
    be represented explicitly on each ``<visual>``. Existing link-to-visual rotation is
    composed with the correction on the right; its translation is left unchanged.
    Collision geometry is intentionally untouched.
    """
    root = ET.fromstring(urdf_text)
    correction = _rpy_matrix(correction_rpy)
    rewritten = 0
    for visual in root.iter("visual"):
        origin = visual.find("origin")
        if origin is None:
            origin = ET.Element("origin")
            visual.insert(0, origin)

        current_rpy = _parse_vector(origin.get("rpy", "0 0 0"), "visual origin rpy")
        composed = _multiply_matrix(_rpy_matrix(current_rpy), correction)
        origin.set("rpy", _format_vector(_matrix_rpy(composed)))
        if origin.get("xyz") is None:
            origin.set("xyz", "0 0 0")
        rewritten += 1

    return ET.tostring(root, encoding="unicode"), rewritten


def strip_visual_materials(urdf_text: str) -> tuple[str, int]:
    """Drop ``<material>`` from every ``<visual>`` so the MESH's own materials win.

    A URDF material declared inside ``<visual>`` overrides whatever the mesh file
    carries. AgileX's piper URDF paints every link with one flat colour that way,
    discarding the 13-15 materials each of its DAE files actually defines. Removing
    those tags lets the richer mesh materials through.

    Uses an XML parse rather than a regex: ``<material name=""><color rgba=".."/>
    </material>`` defeats a non-greedy text match, which stops at the inner ``/>``
    and leaves an orphaned ``</material>`` behind, corrupting the file.

    Only ``<visual>`` blocks are touched; top-level ``<material>`` declarations and
    collision geometry are left alone.
    """
    root = ET.fromstring(urdf_text)
    removed = 0
    for visual in root.iter("visual"):
        for material in visual.findall("material"):
            visual.remove(material)
            removed += 1
    return ET.tostring(root, encoding="unicode"), removed


def summarize_meshes(urdf_text: str) -> dict[str, int]:
    """Count referenced mesh files by extension -- the material-fidelity tell.

    STL carries no material data, so a description whose *visual* meshes are STL
    can only ever render as the flat ``<color rgba>`` from the URDF. DAE and OBJ
    carry materials that survive conversion.
    """
    counts: dict[str, int] = {}
    # Only <mesh filename="..."> counts. A bare filename= search also picks up Gazebo
    # plugin libraries (<plugin filename="libgazebo_ros_control.so">), which would show
    # up as a bogus "so" mesh format.
    for filename in re.findall(r'<mesh[^>]*\sfilename="([^"]+)"', urdf_text):
        ext = filename.rsplit(".", 1)[-1].lower()
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Cloned description repo root")
    parser.add_argument(
        "--extra-repo",
        action="append",
        default=[],
        metavar="PATH",
        help="Additional repo root to resolve packages from; repeatable. Some vendors "
        "split a robot across repos -- ARX's arm xacros live in robot-descriptions-arx "
        "but their meshes come from the component_models package in "
        "robot-descriptions-common.",
    )
    parser.add_argument("--package", required=True, help="ROS package containing the xacro")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--xacro",
        help="Path relative to the package of the .xacro (expanded) or plain .urdf (copied)",
    )
    source_group.add_argument(
        "--source-file",
        help="Repository-owned xacro/URDF overlay path. Includes still resolve through the cloned ROS packages.",
    )
    parser.add_argument("--out", required=True, help="Output .urdf path")
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME:=VALUE",
        help="xacro $(arg) value; repeatable (e.g. --arg dof:=6 --arg robot_type:=xarm)",
    )
    parser.add_argument(
        "--strip-visual-materials",
        action="store_true",
        help="Remove <material> from <visual> so the mesh file's own materials are used",
    )
    parser.add_argument(
        "--sanitize-mesh-filenames",
        action="store_true",
        help="Create aliases for mesh stems that are invalid USD prim identifiers",
    )
    parser.add_argument(
        "--continuous-joint-limit",
        type=float,
        default=None,
        metavar="RADIANS",
        help="Convert continuous joints to revolute joints with symmetric Lula planning limits",
    )
    parser.add_argument(
        "--visual-rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Compose this local-frame RPY rotation onto every <visual> origin",
    )
    args = parser.parse_args()

    try:
        mappings = parse_mappings(args.arg)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    repo_root = Path(args.repo).resolve()
    if not repo_root.is_dir():
        print(f"ERROR: repo not found: {repo_root}", file=sys.stderr)
        return 1

    package_roots = find_package_roots(repo_root)
    for extra in args.extra_repo:
        extra_root = Path(extra).resolve()
        if not extra_root.is_dir():
            print(f"ERROR: --extra-repo not found: {extra_root}", file=sys.stderr)
            return 1
        # The primary repo wins on name collisions.
        for name, path in find_package_roots(extra_root).items():
            package_roots.setdefault(name, path)
    if args.package not in package_roots:
        print(
            f"ERROR: package '{args.package}' not found under {repo_root}. Found: {sorted(package_roots)}",
            file=sys.stderr,
        )
        return 1

    source_path = Path(args.source_file).resolve() if args.source_file else package_roots[args.package] / args.xacro
    if not source_path.is_file():
        print(f"ERROR: description source not found: {source_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    urdf_text = load_description(package_roots, source_path, mappings)

    if args.visual_rpy is not None:
        try:
            visual_rpy = (args.visual_rpy[0], args.visual_rpy[1], args.visual_rpy[2])
            urdf_text, rotated = rotate_visual_origins(urdf_text, visual_rpy)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"[prepare] rotated {rotated} <visual> origin(s) by RPY {_format_vector(visual_rpy)}")

    if args.strip_visual_materials:
        urdf_text, removed = strip_visual_materials(urdf_text)
        print(f"[prepare] stripped {removed} <material> tag(s) from <visual> blocks")

    # Strip the file:// scheme some vendors emit (robotiq_description writes
    # file://$(find pkg)/...). The importer wants a plain path; the scheme prefix makes
    # it look for a file literally named "file:".
    stripped = urdf_text.count('filename="file://')
    if stripped:
        urdf_text = urdf_text.replace('filename="file://', 'filename="')
        print(f"[prepare] stripped file:// from {stripped} mesh path(s)")

    print("[prepare] rewriting package:// URIs")
    urdf_text, unresolved = rewrite_package_uris(urdf_text, package_roots, out_path)
    if unresolved:
        print(f"ERROR: {len(unresolved)} unresolved mesh reference(s):", file=sys.stderr)
        for uri in unresolved:
            print(f"  {uri}", file=sys.stderr)
        return 1

    if PACKAGE_URI_RE.search(urdf_text):
        print("ERROR: package:// URIs remain after rewrite", file=sys.stderr)
        return 1

    urdf_text, rewritten_absolute = rewrite_absolute_mesh_paths(urdf_text, out_path)
    if rewritten_absolute:
        print(f"[prepare] relativized {rewritten_absolute} absolute mesh path(s)")

    if args.sanitize_mesh_filenames:
        try:
            urdf_text, sanitized = sanitize_mesh_filenames(urdf_text, out_path)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"[prepare] sanitized {sanitized} mesh filename(s) for USD prim paths")

    if args.continuous_joint_limit is not None:
        try:
            urdf_text, bounded = bound_continuous_joints(urdf_text, args.continuous_joint_limit)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"[prepare] bounded {bounded} continuous joint(s) to +/-{args.continuous_joint_limit:g} rad for Lula")

    out_path.write_text(urdf_text)

    counts = summarize_meshes(urdf_text)
    print(f"[prepare] wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"[prepare] mesh references by extension: {counts}")
    if "dae" not in counts and "obj" not in counts:
        print(
            "[prepare] WARNING: no DAE/OBJ meshes referenced -- materials will be flat URDF colors only.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
