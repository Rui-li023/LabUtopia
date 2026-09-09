"""Restore the original packed clear material without regenerating any mesh.

Run with Blender --background --python this_file -- --backup-dir <new directory>.
Only the two active 260 mm guide36 .blend assets are written. USD material
bindings are maintained separately; this tool does not export geometry.
"""

import argparse
import hashlib
import json
import shutil
import sys
from array import array
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/chemistry_lab/lab_002"
REFERENCE = ASSETS / "realistic_pipette_260mm/pipette_and_stand.blend"
VARIANTS = ("tall_four", "tall_wide")


def geometry_digest() -> str:
    """Include transforms, vertices, topology, material indices and all UVs."""
    digest = hashlib.sha256()
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        digest.update(repr((obj.name, obj.type, tuple(tuple(row) for row in obj.matrix_world))).encode())
        if obj.type != "MESH":
            continue
        mesh = obj.data
        for values, attribute, width, kind in (
            (mesh.vertices, "co", 3, "f"),
            (mesh.loops, "vertex_index", 1, "i"),
            (mesh.polygons, "loop_start", 1, "i"),
            (mesh.polygons, "loop_total", 1, "i"),
            (mesh.polygons, "material_index", 1, "i"),
            *((layer.data, "uv", 2, "f") for layer in mesh.uv_layers),
        ):
            buffer = array(kind, [0]) * (len(values) * width)
            values.foreach_get(attribute, buffer)
            digest.update(buffer.tobytes())
    return digest.hexdigest()


def restore(variant: str, backup_dir: Path) -> dict:
    path = ASSETS / f"realistic_pipette_260mm_{variant}_guide36/pipette_and_stand.blend"
    bpy.ops.wm.open_mainfile(filepath=str(path))
    before = geometry_digest()
    white = bpy.data.materials.get("White_PMMA")
    if white is None:
        raise ValueError(f"Expected the original white rack material in {path}")
    slots = [slot for obj in bpy.data.objects for slot in obj.material_slots if slot.material == white]
    if len(slots) != 4 or bpy.data.materials.get("Clear_PMMA") is not None:
        raise ValueError("Expected four white rack panels and no pre-existing clear replacement")
    with bpy.data.libraries.load(str(REFERENCE), link=False) as (available, selected):
        if "Clear_PMMA" not in available.materials:
            raise ValueError("Reference blend has no original transparent acrylic material")
        selected.materials = ["Clear_PMMA"]
    clear = selected.materials[0]
    bsdf = clear.node_tree.nodes.get("Principled BSDF")
    if bsdf is None or bsdf.inputs["Transmission Weight"].default_value != 1.0:
        raise ValueError("Reference is not the original fully transmitting material")
    images = [node.image for node in clear.node_tree.nodes if node.type == "TEX_IMAGE" and node.image is not None]
    if len(images) != 2 or any(image.packed_file is None for image in images):
        raise ValueError("Original roughness and normal maps must both be packed")
    for slot in slots:
        slot.material = clear
    bpy.data.materials.remove(white)
    after = geometry_digest()
    if before != after:
        raise RuntimeError("Material-only restoration unexpectedly changed geometry")
    backup = backup_dir / f"{variant}_before_clear.blend"
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(path, backup)
    # The exact previous file is backed up above; do not overwrite older .blend1.
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.ops.wm.open_mainfile(filepath=str(path))
    if geometry_digest() != before:
        raise RuntimeError("Saved Blender geometry does not match the original")
    return {
        "variant": variant,
        "asset": str(path),
        "backup": str(backup),
        "panels": len(slots),
        "geometry_sha256": before,
        "saved_geometry_unchanged": True,
        "restored_material": "Clear_PMMA",
        "packed_maps": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.backup_dir.mkdir(parents=True, exist_ok=False)
    for variant in VARIANTS:
        print("MATERIAL_RESTORE " + json.dumps(restore(variant, args.backup_dir)), flush=True)


if __name__ == "__main__":
    main()
