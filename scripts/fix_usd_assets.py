#!/usr/bin/env python3
"""
USD Asset Health Check & Fix Script
====================================
1. Absolute paths  → relative paths        (auto-fix)
2. Missing textures/materials               (report or clean)
3. Old absolute paths in doc strings        (auto-clean)

Usage:
  conda run -n isaacsim5.1 python3 scripts/fix_usd_assets.py                        # Dry-run
  conda run -n isaacsim5.1 python3 scripts/fix_usd_assets.py --fix                  # Apply fixes
  conda run -n isaacsim5.1 python3 scripts/fix_usd_assets.py --fix --clean-missing   # Also remove dangling refs
  conda run -n isaacsim5.1 python3 scripts/fix_usd_assets.py --fix --backup          # Fix with .bak
"""

import os
import re
import sys
import glob
import shutil
import argparse
from collections import defaultdict

# ---------------------------------------------------------------------------
# NVIDIA runtime MDLs — shipped with Isaac Sim, never stored in project tree
# ---------------------------------------------------------------------------
NVIDIA_RUNTIME_MDLS = {
    "OmniPBR.mdl",
    "OmniGlass.mdl",
    "OmniSurfacePresets.mdl",
    "OmniSurface.mdl",
    "gltf/pbr.mdl",
    "nvidia/support_definitions.mdl",
}


def is_nvidia_runtime(ref: str) -> bool:
    """Return True if *ref* is a material provided by Isaac Sim at runtime."""
    return ref in NVIDIA_RUNTIME_MDLS or any(
        ref.endswith("/" + m) or ref == m for m in NVIDIA_RUNTIME_MDLS
    )


def abs_to_rel(abs_path: str, from_dir: str) -> str:
    """Convert *abs_path* to a relative path seen from *from_dir*."""
    rel = os.path.relpath(abs_path, from_dir)
    if not rel.startswith("."):
        rel = "./" + rel
    return rel


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def _save_layer(usd_file: str, new_text: str, Sdf):
    """Write modified USDA text back to the original USD file."""
    new_layer = Sdf.Layer.CreateAnonymous(".usd")
    success = new_layer.ImportFromString(new_text)
    if success:
        new_layer.Export(usd_file)
    else:
        # Fallback: write USDA text, reopen, export as USDC
        tmp_usda = usd_file + ".tmp.usda"
        with open(tmp_usda, "w") as f:
            f.write(new_text)
        tmp_layer = Sdf.Layer.FindOrOpen(tmp_usda)
        if tmp_layer:
            tmp_layer.Export(usd_file)
        os.remove(tmp_usda)


def process_file(
    usd_file: str,
    assets_dir: str,
    project_root: str,
    *,
    do_fix: bool,
    do_backup: bool,
    clean_missing: bool,
    Sdf,
):
    """Analyse (and optionally fix) a single USD file.

    Returns:
        dict with keys: abs_fixed, docs_cleaned, missing_cleaned, missing list[(ref, expected)]
    """
    rel_usd = os.path.relpath(usd_file, assets_dir)
    usd_dir = os.path.dirname(usd_file)

    result = {
        "file": rel_usd,
        "abs_fixed": 0,
        "docs_cleaned": 0,
        "missing_cleaned": 0,
        "missing": [],
    }

    try:
        layer = Sdf.Layer.FindOrOpen(usd_file)
        if not layer:
            print(f"  WARN: cannot open {rel_usd}")
            return result
    except Exception as e:
        print(f"  ERROR: {rel_usd}: {e}")
        return result

    usda_text = layer.ExportToString()
    new_text = usda_text

    # ------------------------------------------------------------------
    # 1. Fix absolute asset-path references  @/home/…@  →  @./rel/…@
    # ------------------------------------------------------------------
    abs_prefixes = ("/home/", "/root/", "/tmp/", "/opt/", "/usr/local/")

    def _fix_asset_ref(match):
        path = match.group(1)
        if not any(path.startswith(p) for p in abs_prefixes):
            return match.group(0)

        # Path inside the project tree → convert to relative
        if path.startswith(project_root):
            rel = abs_to_rel(path, usd_dir)
            result["abs_fixed"] += 1
            return f"@{rel}@"

        # External absolute path → cannot auto-fix, leave as-is
        return match.group(0)

    new_text = re.sub(r"@(/[^@\s]+)@", _fix_asset_ref, new_text)

    # ------------------------------------------------------------------
    # 2. Clean doc = """…""" metadata that embeds old absolute paths
    # ------------------------------------------------------------------
    def _clean_doc(match):
        body = match.group(1)
        if re.search(r"/home/|/root/|/tmp/", body):
            result["docs_cleaned"] += 1
            return 'doc = """"""'  # empty doc
        return match.group(0)

    new_text = re.sub(r'doc\s*=\s*"""([\s\S]*?)"""', _clean_doc, new_text)

    # ------------------------------------------------------------------
    # 3. Collect missing texture / material references (after fixes)
    # ------------------------------------------------------------------
    for ref in sorted(set(re.findall(r"@([^@\s]+)@", new_text))):
        if ref.startswith("http") or ref.startswith("omniverse:"):
            continue
        if is_nvidia_runtime(ref):
            continue

        if ref.startswith("/"):
            full_path = ref
        else:
            full_path = os.path.normpath(os.path.join(usd_dir, ref))

        if not os.path.exists(full_path):
            result["missing"].append((ref, full_path))

    # ------------------------------------------------------------------
    # 4. Clean missing references: @missing_file@ → @@
    # ------------------------------------------------------------------
    if clean_missing and result["missing"]:
        for ref, _ in result["missing"]:
            escaped = re.escape(ref)
            count = len(re.findall(f"@{escaped}@", new_text))
            if count:
                new_text = new_text.replace(f"@{ref}@", "@@")
                result["missing_cleaned"] += count

    # ------------------------------------------------------------------
    # 5. Save if anything changed
    # ------------------------------------------------------------------
    changed = (
        result["abs_fixed"] > 0
        or result["docs_cleaned"] > 0
        or result["missing_cleaned"] > 0
    )
    if changed and do_fix:
        if do_backup:
            shutil.copy2(usd_file, usd_file + ".bak")
        try:
            _save_layer(usd_file, new_text, Sdf)
        except Exception as e:
            print(f"    SAVE FAILED {rel_usd}: {e}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="USD Asset Health Check & Fix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fix", action="store_true", help="Apply fixes (default: dry-run report)")
    parser.add_argument("--clean-missing", action="store_true", help="Remove dangling refs to missing files (set to empty @@)")
    parser.add_argument("--backup", action="store_true", help="Create .bak before modifying")
    parser.add_argument("--assets-dir", default=None, help="Override assets directory")
    args = parser.parse_args()

    # --- Bootstrap Isaac Sim (needed for pxr) ---
    # Suppress Isaac Sim startup noise
    import logging
    logging.disable(logging.WARNING)
    os.environ.setdefault("CARB_LOG_LEVEL", "error")

    import isaacsim
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    from pxr import Sdf  # noqa: E402  (available only after SimulationApp)

    # Restore logging
    logging.disable(logging.NOTSET)

    # --- Resolve paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.assets_dir:
        assets_dir = os.path.abspath(args.assets_dir)
    else:
        assets_dir = os.path.normpath(os.path.join(script_dir, "..", "assets"))
    project_root = os.path.dirname(assets_dir)

    usd_files = sorted(glob.glob(os.path.join(assets_dir, "**/*.usd"), recursive=True))

    # --- Output helper: write to both console and report file ---
    report_path = os.path.join(project_root, "usd_check_report.txt")
    _report_fh = open(report_path, "w")

    def log(msg=""):
        print(msg)
        _report_fh.write(msg + "\n")
        _report_fh.flush()

    mode = "FIXING" if args.fix else "DRY-RUN"
    log(f"{'=' * 70}")
    log(f"  USD Asset Health Check  [{mode}]")
    log(f"  Assets : {assets_dir}")
    log(f"  Files  : {len(usd_files)}")
    log(f"{'=' * 70}")
    log()

    # --- Process each file ---
    total_abs = 0
    total_doc = 0
    total_dangling = 0
    all_missing = defaultdict(list)  # ref → [usd_file, …]
    modified_files = []

    for usd_file in usd_files:
        r = process_file(
            usd_file,
            assets_dir,
            project_root,
            do_fix=args.fix,
            do_backup=args.backup,
            clean_missing=args.clean_missing,
            Sdf=Sdf,
        )

        if r["abs_fixed"] or r["docs_cleaned"] or r["missing_cleaned"]:
            tag = "FIXED" if args.fix else "WOULD FIX"
            parts = []
            if r["abs_fixed"]:
                parts.append(f"abs_paths={r['abs_fixed']}")
            if r["docs_cleaned"]:
                parts.append(f"docs={r['docs_cleaned']}")
            if r["missing_cleaned"]:
                parts.append(f"dangling_refs={r['missing_cleaned']}")
            log(f"  [{tag}] {r['file']}  ({', '.join(parts)})")
            total_abs += r["abs_fixed"]
            total_doc += r["docs_cleaned"]
            total_dangling += r["missing_cleaned"]
            modified_files.append(r["file"])

        for ref, expected in r["missing"]:
            all_missing[ref].append(r["file"])

    # --- Print summary ---
    log()
    log(f"{'=' * 70}")
    log("  SUMMARY")
    log(f"{'=' * 70}")
    log(f"  Files scanned        : {len(usd_files)}")
    log(f"  Files modified       : {len(modified_files)}")
    log(f"  Absolute paths fixed : {total_abs}")
    log(f"  Doc strings cleaned  : {total_doc}")
    log(f"  Dangling refs cleaned: {total_dangling}")
    log(f"  Missing references   : {len(all_missing)} unique files")

    # --- Missing textures report (separate file) ---
    if all_missing:
        missing_path = os.path.join(project_root, "missing_assets.txt")
        with open(missing_path, "w") as f:
            f.write("# Missing Asset References\n")
            f.write(f"# Generated by fix_usd_assets.py\n")
            f.write(f"# {len(all_missing)} unique missing files\n")
            f.write("#\n")
            f.write("# Format:  MISSING_FILE  <-  [referenced by USD files]\n\n")

            for ref in sorted(all_missing.keys()):
                usds = all_missing[ref]
                f.write(f"{ref}\n")
                for u in usds:
                    f.write(f"    <- {u}\n")
                f.write("\n")

        log(f"\n  Missing assets list saved to: {missing_path}")

        log(f"\n{'=' * 70}")
        log("  MISSING TEXTURES / MATERIALS  (need manual action)")
        log(f"{'=' * 70}")
        for ref in sorted(all_missing.keys()):
            usds = all_missing[ref]
            log(f"\n  {ref}")
            for u in usds:
                log(f"      <- {u}")

    # --- Footer ---
    log()
    log(f"{'=' * 70}")
    if not args.fix:
        log("  DRY RUN complete. Re-run with --fix to apply changes.")
    else:
        log("  All fixes applied.")
    log(f"{'=' * 70}")

    _report_fh.close()
    # Print location last so it's easy to find even in noisy output
    print(f"\n>>> Full report saved to: {report_path}")

    app.close()


if __name__ == "__main__":
    main()
