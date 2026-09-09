"""Restore the WidowX VX300s' upstream link colour.

The bug
-------
Interbotix's `vx300s.urdf.xacro` paints every one of its 12 visual links with the
same material, `interbotix_black`. That material is declared as a **texture**:

    <material name="interbotix_black">
      <texture filename=".../meshes/interbotix_black.png" />
    </material>

The URDF->USD importer only carries `<color rgba>`, not `<texture>`, so none of
that reached the asset. All 11 renderable meshes ended up on `DefaultMaterial`
with colours the converter invented: 0.627 grey on six links, pure white on the
five gripper parts, and a stray green (0.267, 0.588, 0.282) on the lower forearm.
A real WidowX is black throughout.

The fix
-------
`interbotix_black.png` is a 64x85 tile of near-uniform dark grey (91 distinct
values, all grain around the mean), and the STL-derived meshes carry no UVs --
only `primvars:displayColor`. There is nothing for a texture lookup to do here,
so the faithful translation is a flat material at the texture's mean value. That
value is read from the PNG at fix time rather than hard-coded, so the colour
traces back to the upstream asset rather than to a guess.

The mean is sRGB-decoded to linear before use as `diffuseColor`; see
texture_mean_rgb for why that differs from how the importer treats
every `<color rgba>` in this repo.

Usage:
    python -m experiments.urdf_to_usd.fix_widowx_materials --check
    python -m experiments.urdf_to_usd.fix_widowx_materials --texture <path/to/png>
"""

import argparse
import os
import shutil

import cv2
from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from experiments.urdf_to_usd.upstream import clone_hint

USD_PATH = "assets/robots/widowx_vx300s.usd"
# Where the upstream texture is vendored to, so the asset's provenance lives in
# the repo instead of in a fetched tree.
TEXTURE_DEST = "assets/robots/textures/interbotix_black.png"
MATERIAL_PATH = "/vx300s/Looks/interbotix_black"
UPSTREAM_TEXTURE = (
    "third_party/urdf/interbotix/interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/interbotix_black.png"
)
TRAVERSE = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)


def _srgb_to_linear(channel: float) -> float:
    """Decode one sRGB-encoded 0-1 channel to linear."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def texture_mean_rgb(texture_path: str) -> tuple[float, float, float]:
    """Mean colour of the upstream texture, as LINEAR 0-1 RGB.

    The sRGB decode is not optional. A PNG is sRGB-encoded by definition and
    `UsdPreviewSurface.diffuseColor` is defined in linear space, so passing the
    raw 8-bit mean straight through inflates 0.214 into an albedo it never had:
    under the publication light rig this "black" texture rendered as mid-grey.
    Decoding lands it at 0.038, which reads as the black the real arm is.

    (This is why the value is NOT handled like the `<color rgba>` fields
    elsewhere in this repo, which the importer passes through untouched -- those
    are authored numbers with no declared colour space, a PNG is not.)
    """
    image = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read texture: {texture_path}")
    blue, green, red = image.reshape(-1, 3).mean(axis=0) / 255.0
    return tuple(_srgb_to_linear(float(c)) for c in (red, green, blue))


def renderable_visual_meshes(stage: Usd.Stage) -> list[Usd.Prim]:
    """Meshes that actually render: default purpose, not hidden, not collision."""
    meshes = []
    for prim in Usd.PrimRange.Stage(stage, TRAVERSE):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputePurpose() != UsdGeom.Tokens.default_:
            continue
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        meshes.append(prim)
    return meshes


def binding_targets(stage: Usd.Stage, meshes: list[Usd.Prim]) -> list[Usd.Prim]:
    """The authorable prims to bind, one per mesh.

    Each `<link>/visuals` is an *instance*, so its meshes are instance proxies and
    USD refuses to author on them. The instance prim itself is authorable though,
    and a binding there resolves down into the prototype -- so that is where the
    material goes. Walking up from the mesh (rather than assuming a fixed depth)
    keeps this correct if a link nests its geometry differently.
    """
    targets = []
    for mesh in meshes:
        prim = mesh
        while prim and not prim.IsInstance():
            prim = prim.GetParent()
        # A mesh outside any instance can be bound directly.
        targets.append(prim if prim and prim.IsValid() else mesh)
    # dict.fromkeys keeps first-seen order while removing the duplicates that
    # arise when one `visuals` instance holds several meshes.
    return list(dict.fromkeys(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--texture",
        default=TEXTURE_DEST,
        help=f"upstream interbotix_black.png (default: {TEXTURE_DEST})",
    )
    parser.add_argument("--check", action="store_true", help="report the edits without writing")
    args = parser.parse_args()

    texture_source = args.texture
    if not os.path.isfile(texture_source):
        logger.error(f"Texture not found: {texture_source}")
        logger.error(f"The upstream copy lives at {UPSTREAM_TEXTURE} once Interbotix is cloned:")
        logger.error(f"    {clone_hint('interbotix')}")
        logger.error(f"A copy is normally vendored at {TEXTURE_DEST}; pass --texture to point elsewhere.")
        return
    rgb = texture_mean_rgb(texture_source)
    logger.info(f"{texture_source} -> mean RGB {tuple(round(c, 4) for c in rgb)}")

    stage = Usd.Stage.Open(USD_PATH)
    if stage is None:
        logger.error(f"Could not open {USD_PATH}")
        return
    meshes = renderable_visual_meshes(stage)
    if not meshes:
        logger.error(f"{USD_PATH}: found no renderable meshes")
        return

    for mesh in meshes:
        bound = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
        current = bound.GetPath().name if bound else "<unbound>"
        logger.info(f"    {mesh.GetPath().pathString.split('/')[2]:30s} {current} -> interbotix_black")
    if args.check:
        logger.success(f"Checked {len(meshes)} mesh(es), nothing written")
        return

    if os.path.abspath(texture_source) != os.path.abspath(TEXTURE_DEST):
        os.makedirs(os.path.dirname(TEXTURE_DEST), exist_ok=True)
        shutil.copy2(texture_source, TEXTURE_DEST)
        logger.info(f"Vendored texture to {TEXTURE_DEST}")

    material = UsdShade.Material.Define(stage, MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, MATERIAL_PATH + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    # Interbotix shells are matte plastic: no metallic, and rough enough that the
    # studio rig does not put a hot specular streak down every link.
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    targets = binding_targets(stage, meshes)
    for target in targets:
        binding = UsdShade.MaterialBindingAPI.Apply(target)
        # The meshes inside the prototype carry their own DIRECT binding to
        # DefaultMaterial, and a direct binding normally beats one inherited from
        # an ancestor. strongerThanDescendants inverts that, which is the only way
        # to override a prototype's bindings without de-instancing the asset.
        binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    stage.GetRootLayer().Save()
    logger.success(
        f"Bound {len(targets)} instance(s) covering {len(meshes)} mesh(es) to "
        f"{MATERIAL_PATH} at RGB {tuple(round(c, 4) for c in rgb)}"
    )


if __name__ == "__main__":
    main()
