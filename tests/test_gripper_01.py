#!/usr/bin/env python3
"""
Test all collect-mode configs: run simulation, check gripper state is 0/1.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate labutopia
    cd ~/LabUtopia
    python tests/test_gripper_01.py --max-episodes 2 --timeout 12 2>&1 | tee test_outputs/gripper_01/full_log.txt
"""

import os
import sys
import yaml
import subprocess
import glob
import argparse
import time
import re
import numpy as np
from pathlib import Path

SKIP_CONFIGS = {
    "level5_Navigation",
    "level5_Mobile_manipulation",
    "grasp_profiles",
}

BASE = Path(os.path.expanduser("~/LabUtopia"))
CONFIG_DIR = BASE / "config"
TEST_OUTPUT_DIR = BASE / "test_outputs" / "gripper_01"


def log(msg):
    """Print with flush so output is immediately visible in tee/logs."""
    print(msg, flush=True)


def get_all_configs():
    configs = []
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        name = f.stem
        if name in SKIP_CONFIGS:
            continue
        if "infer" in name.lower():
            continue
        configs.append(name)
    return configs


def make_temp_config(config_name, max_episodes=2):
    """Create temp config IN config/ dir so Hydra relative path works."""
    src = CONFIG_DIR / f"{config_name}.yaml"
    with open(src) as f:
        cfg = yaml.safe_load(f)

    cfg["max_episodes"] = max_episodes
    cfg["mode"] = "collect"

    temp_name = f"_test_{config_name}"
    temp_path = CONFIG_DIR / f"{temp_name}.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return temp_name, temp_path


def run_config(config_name, max_episodes=2, timeout_minutes=10):
    """Run a single config, return (success, log_path, error_msg)."""
    temp_name, temp_path = make_temp_config(config_name, max_episodes)
    log_path = TEST_OUTPUT_DIR / f"{config_name}.log"

    cmd = [
        sys.executable, "main.py",
        "--headless",
        "--no-video",
        "--config-name", temp_name,
    ]

    log(f"  CMD: {' '.join(cmd)}")

    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=timeout_minutes * 60,
                cwd=str(BASE),
            )

        with open(log_path) as f:
            log_content = f.read()

        # Check for fatal Python exceptions from our code
        if "Traceback (most recent call last):" in log_content:
            parts = log_content.split("Traceback (most recent call last):")
            last_tb = parts[-1][:600]
            if "File \"/home/user/LabUtopia/" in last_tb:
                for err_type in ["TypeError", "ValueError", "HydraException",
                                 "AttributeError", "KeyError", "NameError"]:
                    if err_type in last_tb:
                        return False, log_path, f"{err_type}:\n{last_tb[-300:]}"

        # Print success stats if found
        success_matches = re.findall(r"Episode Stats: Success Rate = (\d+)/(\d+)", log_content)
        if success_matches:
            last = success_matches[-1]
            log(f"  Stats: {last[0]}/{last[1]} successes")

        return True, log_path, ""

    except subprocess.TimeoutExpired:
        return False, log_path, f"Timeout after {timeout_minutes} min"
    except Exception as e:
        return False, None, str(e)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def find_latest_h5_files(config_name, since_time=None):
    """Find h5 files from the most recent run matching this config."""
    yaml_path = CONFIG_DIR / f"{config_name}.yaml"
    dir_name = None
    if yaml_path.exists():
        with open(yaml_path) as f:
            for line in f:
                m = re.match(r'^name:\s*(.+)', line)
                if m:
                    dir_name = m.group(1).strip().strip('"').strip("'")
                    break
    if not dir_name:
        dir_name = config_name

    collect_dir = BASE / "outputs" / "collect"
    if not collect_dir.exists():
        return []

    matching_dirs = []
    for date_dir in sorted(collect_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            if dir_name in run_dir.name:
                if since_time and run_dir.stat().st_mtime < since_time:
                    continue
                matching_dirs.append(run_dir)

    if not matching_dirs:
        return []

    latest_dir = matching_dirs[0]
    return sorted(glob.glob(str(latest_dir / "dataset" / "**" / "*.h5"), recursive=True))


def check_gripper_states(h5_files, max_check=10):
    """Check gripper states in h5 files are only 0.0 or 1.0."""
    try:
        import h5py
    except ImportError:
        return False, "h5py not available"

    if not h5_files:
        return False, "No h5 files found"

    details = []
    all_valid = True

    for h5_path in h5_files[:max_check]:
        fname = os.path.basename(os.path.dirname(h5_path)) + "/" + os.path.basename(h5_path)
        try:
            with h5py.File(h5_path, "r") as f:
                if "actions" in f:
                    actions = f["actions"][:]
                elif "action" in f:
                    actions = f["action"][:]
                else:
                    details.append(f"  {fname}: NO action key (keys: {list(f.keys())})")
                    all_valid = False
                    continue

                if actions.shape[1] < 8:
                    details.append(f"  {fname}: shape={actions.shape}, need >=8 cols")
                    all_valid = False
                    continue

                gripper = actions[:, 7]
                unique_vals = np.unique(gripper)
                valid = set(unique_vals).issubset({0.0, 1.0})

                if valid:
                    details.append(f"  {fname}: OK shape={actions.shape} gripper={unique_vals}")
                else:
                    details.append(f"  {fname}: FAIL shape={actions.shape} gripper={unique_vals}")
                    all_valid = False

        except Exception as e:
            details.append(f"  {fname}: ERROR: {e}")
            all_valid = False

    return all_valid, "\n".join(details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--check-only", action="store_true",
                        help="Only check existing h5 files, skip running")
    args = parser.parse_args()

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [args.config] if args.config else get_all_configs()

    log(f"{'=' * 70}")
    log(f"Gripper 0/1 Test - {len(configs)} configs")
    log(f"  max_episodes={args.max_episodes}, timeout={args.timeout}min")
    log(f"{'=' * 70}")

    start_time = time.time()
    results = {}

    for i, cfg_name in enumerate(configs):
        log(f"\n[{i+1}/{len(configs)}] {cfg_name}")
        log("-" * 50)

        run_start = time.time()

        if not args.check_only:
            ok, log_path, err = run_config(cfg_name, args.max_episodes, args.timeout)
            if not ok:
                log(f"  RUN FAILED: {err[:300]}")
                results[cfg_name] = {"run": False, "gripper": False, "error": err[:200]}
                continue
            elapsed = time.time() - run_start
            log(f"  Run OK ({elapsed:.0f}s)")

        h5_files = find_latest_h5_files(
            cfg_name,
            since_time=run_start if not args.check_only else None,
        )
        if h5_files:
            valid, details = check_gripper_states(h5_files)
            status = "PASS" if valid else "FAIL"
            log(f"  Gripper: {status} ({len(h5_files)} h5 files)")
            log(details)
            results[cfg_name] = {"run": True, "gripper": valid, "h5_count": len(h5_files)}
        else:
            log(f"  No h5 (0 successes in {args.max_episodes} episodes)")
            results[cfg_name] = {"run": True, "gripper": None, "note": "no h5"}

    # ── Summary ──
    log(f"\n{'=' * 70}")
    log(f"SUMMARY  ({time.time() - start_time:.0f}s)")
    log(f"{'=' * 70}")

    pass_count = fail_count = no_data = crash_count = 0
    for cfg_name, r in results.items():
        if not r.get("run"):
            s = "CRASH"; crash_count += 1
        elif r.get("gripper") is True:
            s = "PASS"; pass_count += 1
        elif r.get("gripper") is None:
            s = "NO_DATA"; no_data += 1
        else:
            s = "FAIL"; fail_count += 1

        extra = f" ({r['h5_count']} ep)" if r.get("h5_count") else ""
        extra = extra or (f" ({r.get('error','')[:60]})" if r.get("error") else "")
        extra = extra or (f" ({r.get('note','')})" if r.get("note") else "")
        log(f"  [{s:7s}] {cfg_name}{extra}")

    log(f"\nPassed: {pass_count}, Failed: {fail_count}, Crash: {crash_count}, No data: {no_data}")
    log(f"Total: {len(results)} configs")

    # Save structured report
    report_path = TEST_OUTPUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        for cfg_name, r in results.items():
            f.write(f"{cfg_name}: {r}\n")
    log(f"Report: {report_path}")

    return 0 if (fail_count == 0 and crash_count == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
