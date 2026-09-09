"""Re-point collapsed GeomSubset material bindings at their upstream DAE colours.

The bug
-------
`yaskawa_hc10_robotiq85.usd` imported the HC10's geometry *perfectly*: every arm
link carries the same face partition the upstream COLLADA does, under the same
names -- `Yaskawa_white`, `Yaskawa_blue`, `metallic` -- with face counts that match
the DAE one for one (base_link white = 7584 in both, link_2_l blue = 5050 in both).

Only the material assignment collapsed. All 16 arm subsets point at a single
material, `material_2126BF` = (0.129, 0.149, 0.749). That colour is the one flat
`<color rgba>` in Yaskawa's URDF; it has red in it, which is why the arm renders
as uniform violet instead of the blue-and-white the real HC10 is.

The fix
-------
The partition is intact, so nothing needs re-importing: each subset just has to
be bound to the colour its own DAE symbol carries. Those colours are read from
the upstream DAE at fix time rather than typed in here.

COLLADA `<diffuse><color>` is linear by specification, so unlike a PNG texture
(see fix_widowx_materials.py) these values are used as `diffuseColor` directly.

Only the collapsed asset is listed in ROBOTS. A survey of all 24 arms found the
rest already bind their subsets to distinct materials -- UR to 6, ARX to 7, FR3
to 55 -- so they need nothing.

Usage:
    python -m experiments.urdf_to_usd.fix_dae_subset_materials --check
    python -m experiments.urdf_to_usd.fix_dae_subset_materials --dae-root <dir>
"""

import argparse
import os
import xml.etree.ElementTree as ET

from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from experiments.urdf_to_usd.upstream import DEFAULT_DAE_ROOT, require_vendor_tree

COLLADA = "{http://www.collada.org/2005/11/COLLADASchema}"
TRAVERSE = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)

ROBOTS = {
    "yaskawa_hc10_robotiq85": {
        "usd": "experiments/robots/assets/robots/yaskawa_hc10_robotiq85.usd",
        "looks": "/yaskawa_hc10_labutopia_robotiq_2f_85/Looks",
        # Relative to --dae-root. Link name -> DAE stem; here they are identical.
        "dae_subdir": "motoman/motoman_hc10_support/meshes/hc10/visual",
        "links": ["base_link", "link_1_s", "link_2_l", "link_3_u", "link_4_r", "link_5_b", "link_6_t"],
    },
}


def dae_symbol_colors(dae_path: str) -> dict[str, tuple[float, float, float]]:
    """Map each COLLADA material symbol to its diffuse colour."""
    root = ET.parse(dae_path).getroot()

    effects: dict[str, tuple[float, float, float]] = {}
    for effect in root.iter(COLLADA + "effect"):
        for diffuse in effect.iter(COLLADA + "diffuse"):
            colour = diffuse.find(COLLADA + "color")
            if colour is not None:
                effects[effect.get("id")] = tuple(float(v) for v in colour.text.split()[:3])
                break

    materials = {}
    for material in root.iter(COLLADA + "material"):
        instance = material.find(COLLADA + "instance_effect")
        if instance is not None:
            materials[material.get("id")] = instance.get("url", "").lstrip("#")

    symbols = {}
    for binding in root.iter(COLLADA + "instance_material"):
        target = binding.get("target", "").lstrip("#")
        colour = effects.get(materials.get(target, ""))
        if colour is not None:
            symbols[binding.get("symbol")] = colour
    return symbols


def _subset_key(name: str) -> str:
    """USD subset names drop COLLADA's `-material` suffix."""
    return name[: -len("-material")] if name.endswith("-material") else name


def _material_name(colour: tuple[float, float, float]) -> str:
    return "dae_" + "".join(f"{round(c * 255):02X}" for c in colour)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dae-root",
        default=DEFAULT_DAE_ROOT,
        help="root of the fetched upstream URDF trees",
    )
    parser.add_argument("--robot", default=None, help="only fix this robot")
    parser.add_argument("--check", action="store_true", help="report the edits without writing")
    args = parser.parse_args()

    for name, spec in ROBOTS.items():
        if args.robot and args.robot != name:
            continue
        vendor = spec["dae_subdir"].split("/")[0]
        if require_vendor_tree(vendor, args.dae_root) is None:
            continue
        dae_dir = os.path.join(args.dae_root, spec["dae_subdir"])
        if not os.path.isdir(dae_dir):
            logger.error(f"{name}: {vendor} is present but {dae_dir} is not; is it a different branch?")
            continue
        stage = Usd.Stage.Open(spec["usd"])
        if stage is None:
            logger.error(f"{name}: could not open {spec['usd']}")
            continue

        wanted: dict[str, dict[str, tuple[float, float, float]]] = {}
        for link in spec["links"]:
            dae_path = os.path.join(dae_dir, f"{link}.dae")
            if not os.path.isfile(dae_path):
                logger.warning(f"{name}: no DAE for link {link}; leaving it alone")
                continue
            wanted[link] = {_subset_key(k): v for k, v in dae_symbol_colors(dae_path).items()}

        plan: list[tuple[str, tuple[float, float, float], str]] = []
        for mesh in Usd.PrimRange.Stage(stage, TRAVERSE):
            if not mesh.IsA(UsdGeom.Mesh):
                continue
            if UsdGeom.Imageable(mesh).ComputePurpose() != UsdGeom.Tokens.default_:
                continue
            parts = mesh.GetPath().pathString.split("/")
            link = parts[2] if len(parts) > 2 else ""
            if link not in wanted:
                continue
            for subset in mesh.GetChildren():
                if not subset.IsA(UsdGeom.Subset):
                    continue
                colour = wanted[link].get(_subset_key(subset.GetName()))
                if colour is None:
                    logger.warning(f"{name}: {link}/{subset.GetName()} has no matching DAE symbol")
                    continue
                bound = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()[0]
                current = bound.GetPath().name if bound else "<unbound>"
                plan.append((subset.GetPath().pathString, colour, f"{link:12s} {subset.GetName():16s} {current}"))

        for _path, colour, label in plan:
            logger.info(f"    {label} -> {tuple(round(c, 3) for c in colour)}")
        if args.check:
            logger.success(f"{name}: checked {len(plan)} subset(s), nothing written")
            continue
        if not plan:
            logger.error(f"{name}: nothing to rebind")
            continue

        # Subsets live inside `<link>/visuals`, which the importer marked
        # instanceable, and USD refuses to author on an instance proxy. Unlike the
        # WidowX case there is no ancestor-level shortcut here: sibling subsets of
        # the SAME mesh need DIFFERENT colours, which a binding on their shared
        # ancestor cannot express. So the instances have to be opened up.
        #
        # That is free in this asset: it has 36 prototypes for 36 instances, i.e.
        # every instance has its own prototype and instancing is deduplicating
        # nothing. Only the ancestors actually being edited are touched.
        opened = set()
        for subset_path, _colour, _label in plan:
            prim = stage.GetPrimAtPath(subset_path)
            while prim and prim.GetPath() != Sdf.Path.absoluteRootPath:
                if prim.IsInstance():
                    if prim.GetPath().pathString not in opened:
                        prim.SetInstanceable(False)
                        opened.add(prim.GetPath().pathString)
                    break
                prim = prim.GetParent()
        if opened:
            logger.info(f"{name}: de-instanced {len(opened)} `visuals` prim(s) to allow subset binding")

        # The real root cause. The importer ALSO authored a link-level binding to
        # the single flat URDF colour, with `bindMaterialAs=strongerThanDescendants`
        # -- which steamrolls the per-subset bindings it imported alongside. Even a
        # correctly partitioned mesh therefore renders as one flat colour. Demote
        # those ancestor bindings so the subsets win; the ancestor stays as the
        # fallback for any face a subset does not cover.
        demoted = 0
        for subset_path, _colour, _label in plan:
            prim = stage.GetPrimAtPath(subset_path).GetParent()
            while prim and prim.GetPath() != Sdf.Path.absoluteRootPath:
                relationship = prim.GetRelationship("material:binding")
                if (
                    relationship
                    and relationship.GetMetadata("bindMaterialAs") == UsdShade.Tokens.strongerThanDescendants
                ):
                    relationship.SetMetadata("bindMaterialAs", UsdShade.Tokens.weakerThanDescendants)
                    demoted += 1
                prim = prim.GetParent()
        if demoted:
            logger.info(f"{name}: demoted {demoted} ancestor binding(s) from strongerThanDescendants")

        materials: dict[str, UsdShade.Material] = {}
        for _path, colour, _label in plan:
            key = _material_name(colour)
            if key in materials:
                continue
            path = f"{spec['looks']}/{key}"
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials[key] = material

        for subset_path, colour, _label in plan:
            subset = stage.GetPrimAtPath(subset_path)
            if not subset.IsValid():
                logger.error(f"{name}: {subset_path} vanished after de-instancing")
                continue
            UsdShade.MaterialBindingAPI.Apply(subset).Bind(materials[_material_name(colour)])

        stage.GetRootLayer().Save()
        logger.success(f"{name}: rebound {len(plan)} subset(s) to {len(materials)} DAE colour(s)")


if __name__ == "__main__":
    main()
