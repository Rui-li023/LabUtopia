"""Tile the per-robot preview PNGs into one labelled contact sheet.

Plain Python + Pillow, so it runs in any env with Pillow -- no Isaac Sim needed::

    python scripts/urdf_to_usd/contact_sheet.py
    python scripts/urdf_to_usd/contact_sheet.py --columns 4 --out /tmp/arms.png
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_ROOT = REPO_ROOT / "outputs" / "urdf_to_usd"

LABEL_HEIGHT = 34
PADDING = 6
BACKGROUND = (28, 28, 32)
LABEL_COLOR = (238, 238, 242)
DETAIL_COLOR = (150, 152, 160)


def load_font(size: int) -> ImageFont.ImageFont:
    """Prefer a real TTF; fall back to Pillow's bitmap font if none is installed."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews", default=str(PREVIEW_ROOT), help="Directory of <robot>.png files")
    parser.add_argument(
        "--summary", default=None, help="batch_summary.json (defaults to <previews>/batch_summary.json)"
    )
    parser.add_argument("--out", default=None, help="Output PNG (defaults to <previews>/contact_sheet.png)")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=460)
    args = parser.parse_args()

    preview_dir = Path(args.previews).resolve()
    out_path = Path(args.out) if args.out else preview_dir / "contact_sheet.png"
    summary_path = Path(args.summary) if args.summary else preview_dir / "batch_summary.json"

    details: dict[str, str] = {}
    if summary_path.is_file():
        for entry in json.loads(summary_path.read_text()):
            details[entry["name"]] = (
                f"{entry.get('vendor', '')} · {entry.get('dof', 0)} DOF · "
                f"{entry.get('n_colors', entry.get('n_materials', 0))} colours · {entry.get('extent_m', 0):.2f} m"
            )

    pngs = sorted(p for p in preview_dir.glob("*.png") if p.name not in {"contact_sheet.png"})
    if not pngs:
        print(f"ERROR: no preview PNGs in {preview_dir}", file=sys.stderr)
        return 1

    columns = max(1, min(args.columns, len(pngs)))
    rows = (len(pngs) + columns - 1) // columns

    with Image.open(pngs[0]) as probe:
        aspect = probe.height / probe.width
    tile_w = args.tile_width
    tile_h = int(tile_w * aspect)

    cell_w = tile_w + PADDING * 2
    cell_h = tile_h + LABEL_HEIGHT + PADDING * 2
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    name_font, detail_font = load_font(19), load_font(14)

    for index, png in enumerate(pngs):
        col, row = index % columns, index // columns
        x, y = col * cell_w + PADDING, row * cell_h + PADDING

        with Image.open(png) as img:
            sheet.paste(img.convert("RGB").resize((tile_w, tile_h), Image.LANCZOS), (x, y))

        name = png.stem
        draw.text((x + 2, y + tile_h + 3), name, fill=LABEL_COLOR, font=name_font)
        if name in details:
            draw.text((x + 2, y + tile_h + 20), details[name], fill=DETAIL_COLOR, font=detail_font)

    sheet.save(out_path)
    print(f"contact sheet: {out_path}  ({len(pngs)} tiles, {columns}x{rows}, {sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
