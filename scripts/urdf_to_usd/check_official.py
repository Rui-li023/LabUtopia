"""Verify and render the downloaded official NVIDIA assets.

They ship no URDF, so joint limits have nothing to compare against; everything else
(articulation root, material bindings, scale sanity, PhysX drive response) still runs.
The point is to see them and judge whether each one is usable as-is, rather than assume
"official" means "better" -- for arms whose upstream description is STL with flat URDF
colours, the official asset is exactly as plain as a local conversion.

Boots Isaac Sim once for the whole set::

    python scripts/urdf_to_usd/check_official.py
    python scripts/urdf_to_usd/check_official.py --only ur5e xarm6 --no-render
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from isaacsim import SimulationApp

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "robots.yaml"
OFFICIAL_ROOT = REPO_ROOT / "assets" / "robots" / "official"
PREVIEW_ROOT = REPO_ROOT / "outputs" / "urdf_to_usd" / "official"

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--manifest", default=str(MANIFEST))
_parser.add_argument("--only", nargs="*", default=None)
_parser.add_argument("--no-render", action="store_true")
_args = _parser.parse_args()

_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import pipeline  # noqa: E402


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


_MODEL_BY_NAME: dict[str, str] = {}
_ENTRY_BY_NAME: dict[str, str] = {}


def model_of(name: str) -> str:
    return _MODEL_BY_NAME.get(name, name)


def entry_usd(name: str) -> Path | None:
    """The asset's entry-point USD.

    Vendors don't agree on naming (``ur5e/ur5e.usd`` vs ``franka_fr3/fr3.usd``), so the
    top-level ``.usd`` sitting beside ``configuration/`` is taken as the entry point.
    """
    directory = OFFICIAL_ROOT / name
    if not directory.is_dir():
        return None

    # Explicit override for assets whose entry point is not a top-level file at all --
    # Robotiq/2F-85 keeps its only geometry under payloads/, and the top-level USD
    # composes to an empty stage.
    override = _ENTRY_BY_NAME.get(name)
    if override:
        candidate = directory / override
        return candidate if candidate.is_file() else None
    candidates = [p for p in directory.glob("*.usd") if p.is_file()]
    if not candidates:
        return None

    # Vendors ship several USDs per model and neither alphabetical order nor "largest
    # file" finds the entry point. Robotiq/2F-140 has five: alphabetical lands on an
    # empty *_edit.usd, and largest-file lands on *_physics_edit.usd -- a physics
    # override layer whose 10 joints include the four-bar linkage's passive ones.
    # Prefer an explicit instanceable/base build, then a name matching the model.
    def rank(path: Path) -> tuple[int, int]:
        stem = path.stem.lower()
        if any(tag in stem for tag in ("_physics_edit", "_edit", "_config", "_controller")):
            return (3, -path.stat().st_size)
        if "instanceable" in stem:
            return (0, -path.stat().st_size)
        if stem == model_of(name).lower().replace("-", "_") or stem.endswith("_base"):
            return (1, -path.stat().st_size)
        return (2, -path.stat().st_size)

    return min(candidates, key=rank)


def main() -> int:
    data = yaml.safe_load(Path(_args.manifest).read_text())
    entries = data.get("official") or []
    if _args.only:
        wanted = set(_args.only)
        entries = [e for e in entries if e["name"] in wanted]

    _MODEL_BY_NAME.update({e["name"]: e["model"] for e in entries})
    _ENTRY_BY_NAME.update({e["name"]: e["entry"] for e in entries if e.get("entry")})
    say(f"\n{'=' * 78}\nOfficial NVIDIA assets: {len(entries)}\n{'=' * 78}")

    results = []
    for entry in entries:
        name = entry["name"]
        usd_path = entry_usd(name)
        say(f"\n[{name}] {entry['vendor']}/{entry['model']}")
        if usd_path is None:
            say("  MISSING: not downloaded (run fetch_official.py)")
            results.append({"name": name, "ok": False, "failures": ["not downloaded"]})
            continue

        # A gripper is a closed-loop four-bar linkage, not a serial arm: driving each of
        # its joints independently pulls the mechanism apart (it renders as a splayed,
        # broken hinge), and its lead joint legitimately will not track a free +0.3 rad
        # step because the loop constrains it. Components skip both.
        try:
            report = pipeline.verify(
                usd_path,
                None,
                name,
                joint_tol=1e-4,
                datasheet_reach_m=None,
                drive_test=entry.get("kind") not in {"gripper", "component"},
            )
        except Exception as exc:
            say(f"  VERIFY RAISED: {type(exc).__name__}: {exc}")
            results.append({"name": name, "ok": False, "failures": [str(exc)]})
            continue

        results.append(
            {
                "name": name,
                "vendor": entry["vendor"],
                "ok": report.ok,
                "dof": report.dof,
                "joints": report.joints,
                "n_colors": report.material_colors,
                "n_materials": len(report.materials),
                "materials": report.materials,
                "visual_meshes": report.visual_meshes,
                "extent_m": round(report.extent_m, 3),
                "size_mb": round(pipeline.asset_size_mb(usd_path), 1),
                "usd": str(usd_path.relative_to(REPO_ROOT)),
                "failures": report.failures,
            }
        )

    if not _args.no_render:
        say(f"\n--- previews ({len(results)}) ---")
        for entry in results:
            usd_path = entry_usd(entry["name"])
            if usd_path is None:
                continue
            png = PREVIEW_ROOT / f"{entry['name']}.png"
            kind = next((e.get("kind") for e in entries if e["name"] == entry["name"]), None)
            pose = () if kind in {"gripper", "component"} else None
            if pipeline.render_preview(_app, usd_path, png, pose):
                say(f"  [render] {entry['name']}")

    # Merge, don't overwrite: a --only run must not drop the entries it did not touch,
    # or the contact-sheet captions lose them.
    summary = PREVIEW_ROOT / "official_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    if summary.is_file():
        merged = {e["name"]: e for e in json.loads(summary.read_text())}
    merged.update({e["name"]: e for e in results})
    summary.write_text(json.dumps(sorted(merged.values(), key=lambda e: e["name"]), indent=2))

    ok = [r for r in results if r["ok"]]
    say(f"\n{'=' * 78}")
    say(f"{'asset':<28}{'vendor':<18}{'dof':>4}{'colours':>8}{'span m':>9}{'MB':>7}  status")
    say("-" * 78)
    for entry in sorted(results, key=lambda r: (not r["ok"], r["name"])):
        status = "OK" if entry["ok"] else f"FAIL ({entry.get('failures', ['?'])[0][:28]})"
        say(
            f"{entry['name']:<28}{entry.get('vendor', ''):<18}{entry.get('dof', 0):>4}"
            f"{entry.get('n_colors', 0):>8}{entry.get('extent_m', 0):>9.3f}"
            f"{entry.get('size_mb', 0):>7.1f}  {status}"
        )
    say("-" * 78)
    say(f"{len(ok)}/{len(results)} usable   summary -> {summary.relative_to(REPO_ROOT)}")

    _app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
