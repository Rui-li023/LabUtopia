"""Convert and verify a whole fleet of arms from ``robots.yaml``.

Two stages, because they need different Python environments:

1. PREPARE -- runs ``prepare_urdf.py`` under the throwaway xacro venv (one subprocess
   per robot; xacro and rospkg must not be installed into the Isaac Sim env).
2. CONVERT + VERIFY -- boots Isaac Sim ONCE and loops over every robot. Booting per
   robot would cost ~10 s each, and this repo forbids concurrent Isaac Sim processes,
   so a single boot is the only fast path.

Usage (Isaac Sim interpreter; it shells out to the xacro venv itself)::

    python scripts/urdf_to_usd/batch.py                     # everything in robots.yaml
    python scripts/urdf_to_usd/batch.py --only ur5e panda   # a subset
    python scripts/urdf_to_usd/batch.py --skip-prepare      # reuse expanded URDFs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from isaacsim import SimulationApp

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "robots.yaml"
SRC_ROOT = REPO_ROOT / "third_party" / "urdf"
USD_ROOT = REPO_ROOT / "assets" / "robots"
PREVIEW_ROOT = REPO_ROOT / "outputs" / "urdf_to_usd"
XACRO_PYTHON = Path("/tmp/xacro_venv/bin/python")

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--manifest", default=str(MANIFEST), help="robots.yaml path")
_parser.add_argument("--only", nargs="*", default=None, help="Convert only these robot names")
_parser.add_argument("--skip-prepare", action="store_true", help="Reuse already-expanded URDFs")
_parser.add_argument("--no-render", action="store_true", help="Skip preview PNGs")
_parser.add_argument(
    "--render-only",
    action="store_true",
    help="Re-render previews from already-converted USDs, skipping convert and verify",
)
_parser.add_argument("--joint-tol", type=float, default=1e-4, help="Joint limit tolerance (rad)")
_args = _parser.parse_args()


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    data = yaml.safe_load(path.read_text())
    robots = data["robots"]
    if _args.only:
        wanted = set(_args.only)
        robots = [r for r in robots if r["name"] in wanted]
        unknown = wanted - {r["name"] for r in robots}
        if unknown:
            say(f"WARNING: unknown robot name(s) in --only: {sorted(unknown)}")
    return data["repos"], robots


def prepared_urdf_path(robot: dict) -> Path:
    """Where the expanded URDF goes.

    It lands beside the source description inside the package's own directory so that
    the rewritten relative mesh paths (``../meshes/...``) resolve naturally.
    """
    return SRC_ROOT / robot["repo"] / "_generated" / f"{robot['name']}.urdf"


def prepare(robot: dict) -> bool:
    """Expand xacro (or copy a plain URDF) and rewrite package:// URIs."""
    repo_dir = SRC_ROOT / robot["repo"]
    if not repo_dir.is_dir():
        say(f"  SKIP: repo not cloned: {repo_dir}")
        return False

    out_path = prepared_urdf_path(robot)
    cmd = [
        str(XACRO_PYTHON),
        str(Path(__file__).resolve().parent / "prepare_urdf.py"),
        "--repo",
        str(repo_dir),
        "--package",
        robot["package"],
        "--out",
        str(out_path),
    ]
    if robot.get("source_file"):
        cmd += ["--source-file", str(REPO_ROOT / robot["source_file"])]
    else:
        cmd += ["--xacro", robot["source"]]
    if robot.get("strip_visual_materials"):
        cmd.append("--strip-visual-materials")
    if robot.get("sanitize_mesh_filenames"):
        cmd.append("--sanitize-mesh-filenames")
    if robot.get("continuous_joint_limit") is not None:
        cmd += ["--continuous-joint-limit", str(robot["continuous_joint_limit"])]
    if robot.get("visual_rpy") is not None:
        cmd += ["--visual-rpy", *(str(value) for value in robot["visual_rpy"])]
    for extra in robot.get("extra_repos") or []:
        cmd += ["--extra-repo", str(SRC_ROOT / extra)]
    for key, value in (robot.get("args") or {}).items():
        cmd += ["--arg", f"{key}:={value}"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        say(f"  PREPARE FAILED: {tail[-1] if tail else 'unknown error'}")
        return False

    for line in result.stdout.splitlines():
        if "mesh references" in line:
            say(f"  {line.strip()}")

    collider_preset = robot.get("finger_collider_preset")
    if collider_preset:
        collider_cmd = [
            str(XACRO_PYTHON),
            str(Path(__file__).resolve().parent / "add_finger_colliders.py"),
            "--urdf",
            str(out_path),
            "--preset",
            str(collider_preset),
        ]
        if robot.get("finger_collider_force"):
            collider_cmd.append("--force")
        collider_result = subprocess.run(collider_cmd, capture_output=True, text=True)
        if collider_result.returncode != 0:
            tail = (collider_result.stderr or collider_result.stdout).strip().splitlines()
            say(f"  COLLIDER PREPARE FAILED: {tail[-1] if tail else 'unknown error'}")
            return False
        for line in collider_result.stderr.splitlines():
            if "added box collider" in line:
                say(f"  {line.strip()}")
    return True


def main() -> int:
    repos, robots = load_manifest(Path(_args.manifest))
    say(f"\n{'=' * 78}\nURDF -> USD batch: {len(robots)} arm(s)\n{'=' * 78}")

    # --- stage 1: prepare (xacro venv subprocesses, no Isaac Sim yet) -------------
    prepared: list[dict] = []
    if _args.skip_prepare:
        prepared = [r for r in robots if prepared_urdf_path(r).is_file()]
        say(f"\n[prepare] skipped; reusing {len(prepared)} existing URDF(s)")
    else:
        if not XACRO_PYTHON.is_file():
            say(f"ERROR: xacro venv missing at {XACRO_PYTHON}. See scripts/urdf_to_usd/README.md")
            return 1
        say(f"\n--- stage 1/3: expand descriptions ({len(robots)}) ---")
        for robot in robots:
            say(f"[prepare] {robot['name']}")
            if prepare(robot):
                prepared.append(robot)
        say(f"[prepare] {len(prepared)}/{len(robots)} succeeded")

    if not prepared:
        say("nothing prepared; aborting")
        return 1

    # --- stage 2 + 3: one Isaac Sim boot for convert + verify ---------------------
    say(f"\n--- stage 2/3: convert + verify ({len(prepared)}) — single Isaac Sim boot ---")
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

    import pipeline

    results = []
    if _args.render_only:
        # Reuse the previous run's verdicts so the contact-sheet captions stay accurate.
        previous = {}
        summary_path = PREVIEW_ROOT / "batch_summary.json"
        if summary_path.is_file():
            previous = {entry["name"]: entry for entry in json.loads(summary_path.read_text())}
        for robot in prepared:
            entry = previous.get(robot["name"])
            if entry and (USD_ROOT / f"{robot['name']}.usd").is_file():
                results.append(entry)
        say(f"[render-only] reusing {len(results)} converted asset(s)")

    for robot in prepared if not _args.render_only else []:
        name = robot["name"]
        urdf_path = prepared_urdf_path(robot)
        usd_path = USD_ROOT / f"{name}.usd"
        say(f"\n[{name}] {robot.get('vendor', '')}")

        # One bad description must not abort a 15-robot run.
        try:
            converted = pipeline.convert(urdf_path, usd_path)
        except Exception as exc:
            say(f"  CONVERT RAISED: {type(exc).__name__}: {exc}")
            converted = False

        if not converted:
            say("  CONVERT FAILED")
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "stage": "convert",
                    "vendor": robot.get("vendor", ""),
                    "failures": ["conversion produced no USD"],
                }
            )
            continue

        report = pipeline.verify(
            usd_path,
            urdf_path,
            name,
            joint_tol=_args.joint_tol,
            datasheet_reach_m=robot.get("reach_m"),
            min_material_colors=robot.get("min_material_colors"),
        )
        results.append(
            {
                "name": name,
                "vendor": robot.get("vendor", ""),
                "ok": report.ok,
                "stage": "verify",
                "joints": report.joints,
                "dof": report.dof,
                "materials": report.materials,
                "n_materials": len(report.materials),
                # Distinctness is counted by colour, not name -- vendors reuse names
                # (every Kinova material is "Material_001"), so the name count
                # understates how much material variety actually survived.
                "n_colors": report.material_colors,
                "visual_meshes": report.visual_meshes,
                "extent_m": round(report.extent_m, 3),
                "reach_m": robot.get("reach_m"),
                "size_mb": round(pipeline.asset_size_mb(usd_path), 1),
                "source_meshes": robot.get("visual_meshes", "?"),
                "failures": report.failures,
            }
        )

    # --- previews: rendering rebuilds the stage, so do it after all verification ---
    if not _args.no_render:
        say(f"\n--- stage 3/3: previews ({len(results)}) ---")
        # Per-robot showroom pose overrides, for arms whose joint limits clamp the
        # generic pattern back into a folded home pose (xArm's joint3 tops out at
        # ~0.19 rad, so the generic +0.5 leaves it folded).
        preview_poses = {r["name"]: tuple(r["preview_pose"]) for r in prepared if r.get("preview_pose")}
        for entry in results:
            if entry.get("stage") != "verify":
                continue
            usd_path = USD_ROOT / f"{entry['name']}.usd"
            png = PREVIEW_ROOT / f"{entry['name']}.png"
            pose = preview_poses.get(entry["name"])
            if pipeline.render_preview(app, usd_path, png, pose):
                say(f"  [render] {entry['name']} -> {png.relative_to(REPO_ROOT)}")

    # Merge rather than overwrite: a --only run must not wipe the entries for robots it
    # did not touch, or the contact sheet loses their captions.
    summary_path = PREVIEW_ROOT / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    if summary_path.is_file():
        merged = {entry["name"]: entry for entry in json.loads(summary_path.read_text())}
    merged.update({entry["name"]: entry for entry in results})
    summary_path.write_text(json.dumps(sorted(merged.values(), key=lambda e: e["name"]), indent=2))

    ok = [r for r in results if r["ok"]]
    say(f"\n{'=' * 78}")
    say(f"{'robot':<20}{'vendor':<20}{'dof':>4}{'colours':>7}{'span m':>9}{'MB':>7}  status")
    say("-" * 78)
    for entry in sorted(results, key=lambda r: (not r["ok"], r["name"])):
        status = "OK" if entry["ok"] else f"FAIL ({entry.get('failures', ['?'])[0][:34]})"
        say(
            f"{entry['name']:<20}{entry.get('vendor', ''):<20}{entry.get('dof', 0):>4}"
            f"{entry.get('n_colors', 0):>7}{entry.get('extent_m', 0):>9.3f}"
            f"{entry.get('size_mb', 0):>7.1f}  {status}"
        )
    say("-" * 78)
    say(f"{len(ok)}/{len(results)} passed all checks   summary -> {summary_path.relative_to(REPO_ROOT)}")
    say("=" * 78)

    app.close()
    return 0 if len(ok) >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
