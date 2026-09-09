"""Transfer per-part colours from upstream COLLADA onto flat-shaded robot USDs.

The problem
-----------
Several arms were converted from their STL visual meshes. STL carries no material,
so each link collapsed to the single flat `<color rgba>` its URDF declares --
Kinova Gen3 renders in one grey, Unitree Z1 in one grey, Piper in one near-white.
The same vendors also ship DAE visual meshes that carry real per-part materials
(Piper's links are grey and white with black bands and red accents).

Why not just re-import the DAE geometry
---------------------------------------
Because it is risky for no extra benefit here. Each DAE splits a link into dozens
of sub-meshes with their own transforms, mixed with stray fasteners and decals
that sit outside the link's own extent, and the sub-mesh frames do not always
agree with the STL's (measured: Piper link1 needs an axis swap, link3 does not
align at all). Getting that wrong yields a visibly broken robot -- misplaced
visual geometry against correctly placed colliders.

What this does instead
----------------------
Geometry is never touched. The DAE is used only as a *colour field*: every face
of the existing mesh is assigned the material of the nearest DAE triangle, and
faces are then grouped into GeomSubsets. Different triangulations are fine --
only proximity matters -- and stray DAE parts are ignored because they are never
anyone's nearest neighbour.

Alignment between the two is solved, not assumed: the 48 axis permutation/sign
combinations are searched and the one minimising median nearest-neighbour
distance wins. A link is only rewritten if that distance is under
`--max-dist-ratio` of its own size, so a link that cannot be matched keeps
exactly what it has today.

Usage:
    python -m experiments.urdf_to_usd.transfer_dae_materials --dae-root <dir> --check
    python -m experiments.urdf_to_usd.transfer_dae_materials --dae-root <dir> --robot piper
"""

import argparse
import glob
import itertools
import os

import numpy as np
from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from scipy.spatial import cKDTree

from experiments.urdf_to_usd.upstream import DEFAULT_DAE_ROOT, require_vendor_tree

TRAVERSE = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
# Gripper links come from the Robotiq/native gripper packages, not the arm
# vendor's DAE set, so they are left alone.
GRIPPER_HINTS = ("robotiq", "finger", "knuckle", "pad", "coupler", "outer", "inner", "gripper_base")

# robot -> upstream vendor directory under --dae-root. Only arms whose links
# currently resolve to one or two flat colours are listed; the rest already carry
# a real per-part partition (UR, Techman, FR3, ARX, Yaskawa, Franka, ...).
ROBOTS = {
    "piper": "agilex_piper",
    # Gen3 is deliberately absent: ros_kortex ships DAE for the Robotiq gripper
    # only, so there is no arm-link colour to read anywhere upstream.
    "xarm6_robotiq85": "xarm_ros",
    "xarm7_robotiq85": "xarm_ros",
    # Left out after inspection, each for a different reason:
    #   jaco2      - DAE says one flat colour per link, which it already has, and
    #                its links carry several meshes this transfer would only part-cover
    #   unitree_z1 - the Z1 DAEs do not align; only unrelated quadruped parts match
    #   abb_gofa   - named DAEs exist but none align, they are other ABB models
    #   doosan     - the DAE confirms the white livery already in the asset
}

AXIS_TRANSFORMS = [
    (perm, signs) for perm in itertools.permutations(range(3)) for signs in itertools.product((1, -1), repeat=3)
]


def _diffuse_of(primitive) -> tuple[float, float, float] | None:
    """RGB of a bound COLLADA primitive's material, or None if it is textured."""
    material = getattr(primitive, "material", None)
    effect = getattr(material, "effect", None)
    diffuse = getattr(effect, "diffuse", None) if effect is not None else None
    if isinstance(diffuse, (list, tuple, np.ndarray)) and len(diffuse) >= 3:
        return tuple(round(float(c), 4) for c in diffuse[:3])
    return None


def load_dae(path: str):
    """Triangle centroids and per-triangle colour for one DAE, scene transforms applied."""
    import collada

    try:
        document = collada.Collada(path)
    except Exception as error:
        logger.debug(f"{path}: unreadable ({error})")
        return None
    centroids, colours = [], []
    for obj in document.scene.objects("geometry"):
        for primitive in obj.primitives():
            vertices = getattr(primitive, "vertex", None)
            indices = getattr(primitive, "vertex_index", None)
            if vertices is None or indices is None or not len(indices):
                continue
            colour = _diffuse_of(primitive)
            if colour is None:
                continue
            indices = np.asarray(indices).reshape(-1)
            # Vendors mix triangle sets and polylists; a polylist's index stream
            # is not a multiple of three, and quietly reshaping it would either
            # crash or silently scramble the geometry.
            if indices.size % 3:
                triangulated = getattr(primitive, "triangleset", None)
                if triangulated is None:
                    continue
                try:
                    primitive = triangulated()
                except Exception:
                    continue
                vertices = getattr(primitive, "vertex", None)
                indices = getattr(primitive, "vertex_index", None)
                if vertices is None or indices is None:
                    continue
                indices = np.asarray(indices).reshape(-1)
                if indices.size % 3:
                    continue
            triangles = np.asarray(vertices)[indices.reshape(-1, 3)]
            centroids.append(triangles.mean(axis=1))
            colours.extend([colour] * len(triangles))
    if not centroids:
        return None
    return np.vstack(centroids), colours


def arm_link_meshes(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    """One renderable mesh per arm link, keyed by link name."""
    meshes: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange.Stage(stage, TRAVERSE):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputePurpose() != UsdGeom.Tokens.default_:
            continue
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        parts = prim.GetPath().pathString.split("/")
        link = parts[2] if len(parts) > 2 else ""
        if not link or any(hint in link.lower() for hint in GRIPPER_HINTS):
            continue
        meshes.setdefault(link, prim)
    return meshes


def face_centroids(mesh: Usd.Prim) -> np.ndarray | None:
    """Centroid per triangular face, or None if the mesh is not all triangles."""
    geom = UsdGeom.Mesh(mesh)
    points = geom.GetPointsAttr().Get()
    indices = geom.GetFaceVertexIndicesAttr().Get()
    counts = geom.GetFaceVertexCountsAttr().Get()
    if points is None or indices is None or counts is None:
        return None
    counts = np.asarray(counts)
    if counts.size == 0 or not (counts == 3).all():
        return None
    return np.asarray(points)[np.asarray(indices).reshape(-1, 3)].mean(axis=1)


def robust_extents(points: np.ndarray) -> np.ndarray:
    """Sorted extents ignoring outliers.

    A vendor DAE routinely carries stray fasteners and decals parked well outside
    the link's own body -- Piper's link1 has screws a quarter of a metre away. Raw
    min/max would inflate the extent and make a correct match fail a shape gate,
    so the 1st..99th percentile is used instead.
    """
    low, high = np.percentile(points, [1, 99], axis=0)
    return np.sort(high - low)


def best_alignment(target: np.ndarray, source: np.ndarray) -> tuple[float, tuple, tuple]:
    """Axis permutation/sign taking *source* onto *target*, by median NN distance."""
    best = (float("inf"), (0, 1, 2), (1, 1, 1))
    for perm, signs in AXIS_TRANSFORMS:
        distances, _ = cKDTree(source[:, perm] * np.asarray(signs)).query(target)
        median = float(np.median(distances))
        if median < best[0]:
            best = (median, perm, signs)
    return best


def _same_part(dae_stem: str, link: str) -> bool:
    """Whether a DAE filename names this link, allowing vendor decoration.

    Vendors decorate the same part name differently on each side: Unitree ships
    `z1_Link00.dae` for a link the USD calls `link00`. Comparing the alphanumeric
    cores, and allowing either to be a suffix of the other, catches those without
    hand-written tables.
    """
    a = "".join(c for c in dae_stem.lower() if c.isalnum())
    b = "".join(c for c in link.lower() if c.isalnum())
    return bool(a) and bool(b) and (a == b or a.endswith(b) or b.endswith(a))


def dominant_colour(colours: list) -> tuple:
    """The colour covering the most triangles."""
    counts: dict[tuple, int] = {}
    for colour in colours:
        counts[colour] = counts.get(colour, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def current_colour(mesh: Usd.Prim) -> tuple | None:
    """The link's effective diffuse today, used to keep the vendor's variant."""
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    if not material:
        return None
    for prim in Usd.PrimRange(material.GetPrim(), TRAVERSE):
        if not prim.IsA(UsdShade.Shader):
            continue
        for shader_input in UsdShade.Shader(prim).GetInputs():
            name = shader_input.GetBaseName().lower()
            value = shader_input.Get()
            if ("diffuse" in name or "basecolor" in name) and value is not None and hasattr(value, "__len__"):
                return tuple(float(c) for c in value[:3])
    return None


def _material_name(colour: tuple[float, ...]) -> str:
    return "dae_" + "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in colour[:3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dae-root", default=DEFAULT_DAE_ROOT, help="root of the fetched upstream URDF trees")
    parser.add_argument("--robot", default=None, help="only process this robot")
    parser.add_argument(
        "--max-dist-ratio",
        type=float,
        default=0.02,
        help="reject a link whose median NN distance exceeds this fraction of its size",
    )
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    for robot, vendor in ROBOTS.items():
        if args.robot and args.robot != robot:
            continue
        usd_path = f"experiments/robots/assets/robots/{robot}.usd"
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            logger.error(f"{robot}: could not open {usd_path}")
            continue
        meshes = arm_link_meshes(stage)
        if not meshes:
            logger.warning(f"{robot}: no arm link meshes found")
            continue

        if require_vendor_tree(vendor, args.dae_root) is None:
            continue
        dae_paths = sorted(glob.glob(os.path.join(args.dae_root, vendor, "**", "*.dae"), recursive=True))
        if not dae_paths:
            logger.error(f"{robot}: {vendor} is present but contains no .dae; is it a different branch?")
            continue

        cache: dict[str, tuple[np.ndarray, list] | None] = {}
        logger.info(f"=== {robot}: {len(meshes)} arm link(s), {len(dae_paths)} candidate DAE(s)")

        assignments = []
        flat: list[tuple[Usd.Prim, tuple]] = []
        for link, mesh in meshes.items():
            target = face_centroids(mesh)
            if target is None:
                logger.warning(f"{robot}/{link}: not a pure triangle mesh; skipping")
                continue
            size = float(np.ptp(target, axis=0).max())
            target_extents = robust_extents(target)

            # A DAE named after the link is strong enough evidence on its own; the
            # shape gate is only needed when the vendor's filenames say nothing.
            named = [p for p in dae_paths if _same_part(os.path.splitext(os.path.basename(p))[0], link)]
            existing = current_colour(mesh)
            candidates: list[tuple] = []
            for path in named + dae_paths:
                if path not in cache:
                    cache[path] = load_dae(path)
                loaded = cache[path]
                if loaded is None:
                    continue
                source, colours = loaded
                # Cheap gate first: a DAE for this link must have the same shape,
                # up to axis order. Full alignment on every file would be far too
                # slow when a vendor ships a hundred of them.
                if path not in named and not np.allclose(robust_extents(source), target_extents, rtol=0.15, atol=4e-3):
                    continue
                median, perm, signs = best_alignment(target, source)
                candidates.append((median, perm, signs, path, source, colours))

            if not candidates:
                logger.info(f"    {link:26s} no DAE of matching shape; left as is")
                continue
            floor = min(c[0] for c in candidates)

            # Vendors ship the SAME geometry in several liveries -- Doosan has an
            # m0609_blue and an m0609_white whose meshes are identical, so
            # alignment cannot separate them and picking by distance alone is a
            # coin flip that can repaint the robot. Among candidates that align
            # equally well, keep the one whose dominant colour is nearest what the
            # link already is: the job is to refine the vendor's chosen livery
            # into per-part detail, not to re-choose it.
            def _rank(candidate: tuple) -> tuple:
                median = candidate[0]
                if existing is None:
                    return (median, 0.0)
                delta = float(np.linalg.norm(np.asarray(dominant_colour(candidate[5])[:3]) - np.asarray(existing)))
                return (0.0 if median <= floor * 1.05 + 1e-9 else median, delta)

            best = min(candidates, key=_rank)
            median, perm, signs, path, source, colours = best
            if median > size * args.max_dist_ratio:
                # Alignment failed, so faces cannot be partitioned -- but the DAE
                # still knows what colour this link mostly is, and that is a
                # link-level fact needing no alignment at all. Falling back to it
                # keeps the arm one consistent family instead of leaving a few
                # links in the flat URDF colour the others just moved away from.
                # The flat fallback must come from a DAE that NAMES this link.
                # Falling back to whatever happened to align best repaints the
                # link from an unrelated part -- that is how Unitree's link00
                # picked up a quadruped hip and ABB's base_link picked up link_5.
                # A generic filename is not enough on its own: every ABB model in
                # the vendor tree ships a `link_1.dae`, so a name match there can
                # hand back a different robot's part (measured: GoFa picking up
                # IRB orange). Require the shape to agree too.
                named_candidates = [
                    c
                    for c in candidates
                    if c[3] in named and np.allclose(robust_extents(c[4]), target_extents, rtol=0.15, atol=4e-3)
                ]
                if not named_candidates:
                    logger.info(
                        f"    {link:26s} best match {os.path.basename(path):24s} "
                        f"median={median:.5f} (> {size * args.max_dist_ratio:.5f}); left as is"
                    )
                    continue
                chosen = min(named_candidates, key=_rank)
                path, colours = chosen[3], chosen[5]
                dominant = dominant_colour(colours)
                share = 100 * sum(1 for c in colours if c == dominant) / len(colours)
                logger.info(
                    f"    {link:26s} no alignment (median={median:.5f}); "
                    f"flat fallback {dominant} ({share:.0f}% of {os.path.basename(path)})"
                )
                flat.append((mesh, dominant))
                continue

            _, nearest = cKDTree(source[:, perm] * np.asarray(signs)).query(target)
            face_colours = [colours[i] for i in nearest]
            distinct = sorted(set(face_colours))
            logger.info(
                f"    {link:26s} <- {os.path.basename(path):24s} median={median:.5f} "
                f"faces={len(target)} colours={len(distinct)} {distinct[:3]}"
            )
            if len(distinct) > 1:
                assignments.append((mesh, face_colours, distinct))
            else:
                # Aligned, but the DAE says this link is one colour. No partition
                # to build -- just adopt that colour so the link stops sitting on
                # the flat URDF value the rest of the arm has moved off.
                flat.append((mesh, distinct[0]))

        if args.check or not (assignments or flat):
            logger.success(
                f"{robot}: {len(assignments)} link(s) would gain a partition, {len(flat)} would take a flat DAE colour"
            )
            continue

        # Same two obstacles as fix_dae_subset_materials: subsets cannot be
        # authored on instance proxies, and a link-level strongerThanDescendants
        # binding would override them anyway.
        for mesh in [a[0] for a in assignments] + [f[0] for f in flat]:
            prim = mesh
            while prim and prim.GetPath() != Sdf.Path.absoluteRootPath:
                if prim.IsInstance():
                    prim.SetInstanceable(False)
                    break
                prim = prim.GetParent()
        for mesh in [a[0] for a in assignments] + [f[0] for f in flat]:
            prim = stage.GetPrimAtPath(mesh.GetPath()).GetParent()
            while prim and prim.GetPath() != Sdf.Path.absoluteRootPath:
                relationship = prim.GetRelationship("material:binding")
                if (
                    relationship
                    and relationship.GetMetadata("bindMaterialAs") == UsdShade.Tokens.strongerThanDescendants
                ):
                    relationship.SetMetadata("bindMaterialAs", UsdShade.Tokens.weakerThanDescendants)
                prim = prim.GetParent()

        looks = stage.GetDefaultPrim().GetPath().AppendChild("Looks")
        materials: dict[str, UsdShade.Material] = {}
        written = 0
        for mesh, face_colours, distinct in assignments:
            mesh_prim = stage.GetPrimAtPath(mesh.GetPath())
            if not mesh_prim.IsValid():
                logger.error(f"{robot}: {mesh.GetPath()} vanished after de-instancing")
                continue
            geom = UsdGeom.Mesh(mesh_prim)
            face_array = np.asarray(face_colours, dtype=object)
            for colour in distinct:
                key = _material_name(colour)
                if key not in materials:
                    path = looks.AppendChild(key)
                    material = UsdShade.Material.Define(stage, path)
                    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour[:3]))
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    materials[key] = material
                faces = np.where(np.asarray([c == colour for c in face_array]))[0]
                subset = UsdGeom.Subset.CreateGeomSubset(
                    geom,
                    f"dae_{key[4:]}",
                    UsdGeom.Tokens.face,
                    Vt.IntArray([int(v) for v in faces]),
                    "materialBind",
                    "partition",
                )
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(materials[key])
            written += 1

        for mesh, colour in flat:
            mesh_prim = stage.GetPrimAtPath(mesh.GetPath())
            if not mesh_prim.IsValid():
                continue
            key = _material_name(colour)
            if key not in materials:
                path = looks.AppendChild(key)
                material = UsdShade.Material.Define(stage, path)
                shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour[:3]))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials[key] = material
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(materials[key])

        stage.GetRootLayer().Save()
        logger.success(
            f"{robot}: partitioned {written} link(s), flat-recoloured {len(flat)}, {len(materials)} DAE colour(s) total"
        )


if __name__ == "__main__":
    main()
