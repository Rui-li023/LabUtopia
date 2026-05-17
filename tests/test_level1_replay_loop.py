#!/usr/bin/env python3
"""Drive collect+replay for every level1 task, sequentially.

For each config:
  Phase A — write a temp config with max_episodes=5, collector.type=default,
            mode=collect, and a deterministic run_dir; run main.py until done.
            Locate the produced dataset dir (outputs/.../dataset).
  Phase B — write a temp config with mode=replay and replay.dataset_path=<dataset>;
            run main.py; parse the final "Success Rate = X/Y (Z%)" line.

Results are appended to test_outputs/level1_replay/RESULTS.md after every task,
plus a per-config record (collected count, replay success/total, init poses).

CLAUDE.md: never run two main.py in parallel. This driver is strictly sequential.
"""
import os
import re
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

import yaml
import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "test_outputs" / "level1_replay"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LEVEL1_CONFIGS = [
    "level1_pick",
    "level1_place",
    "level1_pour",
    "level1_press",
    "level1_shake",
    "level1_stir",
    "level1_open_door",
    "level1_close_door",
    "level1_open_drawer",
    "level1_close_drawer",
    "level1_CloseCentrifuge",
]

COLLECT_EPISODES = 10
COLLECT_TIMEOUT = 60 * 25   # 25 min hard cap per collect
REPLAY_TIMEOUT = 60 * 25    # 25 min hard cap per replay
PYTHON = os.environ.get("LABUTOPIA_PY", sys.executable)

SUCCESS_RE = re.compile(r"Success Rate\s*=\s*(\d+)/(\d+)\s*\(([\d.]+)%\)")


def write_temp_config(src: Path, dst: Path, mutate):
    cfg = yaml.safe_load(src.read_text())
    mutate(cfg)
    dst.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))


def run_main(config_name: str, config_dir: Path, log_path: Path, timeout: int) -> int:
    # Hydra requires a relative config_path
    try:
        rel_cfg_dir = config_dir.resolve().relative_to(REPO)
    except ValueError:
        rel_cfg_dir = config_dir
    cmd = [PYTHON, "main.py", "--config-name", config_name,
           "--config-dir", str(rel_cfg_dir), "--no-video", "--headless"]
    with log_path.open("w") as lf:
        lf.write(f"$ {' '.join(cmd)}\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=timeout, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            return proc.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n[DRIVER] TIMEOUT after {timeout}s\n")
            return -1


def parse_last_success_rate(log_path: Path):
    last = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = SUCCESS_RE.search(line)
        if m:
            last = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    return last


def find_dataset_dir(collect_run_dir: Path):
    candidate = collect_run_dir / "dataset"
    if candidate.exists() and any(candidate.glob("episode_*/episode_*.h5")):
        return candidate
    return None


def dump_init_poses(dataset_dir: Path):
    info = []
    for h5_path in sorted(dataset_dir.glob("episode_*/episode_*.h5")):
        try:
            with h5py.File(h5_path, "r") as f:
                init = f.get("init_state")
                if init is None:
                    continue
                paths = init.get("object_pose_paths")
                positions = init.get("object_pose_positions")
                orientations = init.get("object_pose_orientations")
                if paths is None or positions is None:
                    continue
                paths = [p.decode() if isinstance(p, bytes) else str(p) for p in paths[()]]
                positions = np.asarray(positions[()]).tolist()
                orientations = np.asarray(orientations[()]).tolist() if orientations is not None else None
                info.append({"episode": h5_path.parent.name, "paths": paths,
                             "positions": positions, "orientations": orientations})
        except Exception as e:
            info.append({"episode": h5_path.parent.name, "error": str(e)})
    return info


def append_results(line: str):
    results_md = OUT_ROOT / "RESULTS.md"
    with results_md.open("a") as f:
        f.write(line + "\n")
        f.flush()


def run_one(config_name: str, attempt: int):
    cfg_src = REPO / "config" / f"{config_name}.yaml"
    if not cfg_src.exists():
        append_results(f"- **{config_name}**: missing config")
        return None

    work_dir = OUT_ROOT / config_name / f"attempt_{attempt:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    collect_run_dir = work_dir / "collect_run"
    replay_run_dir = work_dir / "replay_run"

    # --- Phase A: collect ---
    collect_cfg_dir = work_dir / "cfg_collect"
    collect_cfg_dir.mkdir(exist_ok=True)
    temp_name = f"_tmp_{config_name}_collect"
    temp_path = collect_cfg_dir / f"{temp_name}.yaml"

    def mutate_collect(cfg):
        cfg["max_episodes"] = COLLECT_EPISODES
        cfg["mode"] = "collect"
        cfg.setdefault("collector", {})
        cfg["collector"]["type"] = "default"
        cfg["collector"].setdefault("compression", "gzip")
        cfg.setdefault("hydra", {}).setdefault("run", {})
        cfg["hydra"]["run"]["dir"] = str(collect_run_dir)
        cfg.setdefault("multi_run", {})
        cfg["multi_run"]["run_dir"] = str(collect_run_dir)

    write_temp_config(cfg_src, temp_path, mutate_collect)
    collect_log = work_dir / "collect.log"
    print(f"[{config_name}] collect → {collect_log}", flush=True)
    rc = run_main(temp_name, collect_cfg_dir, collect_log, COLLECT_TIMEOUT)
    dataset_dir = find_dataset_dir(collect_run_dir)
    collect_rate = parse_last_success_rate(collect_log)

    if dataset_dir is None:
        append_results(f"- **{config_name}** attempt {attempt}: collect FAILED (rc={rc}, no dataset). rate={collect_rate}")
        return {"config": config_name, "attempt": attempt, "phase": "collect", "rc": rc,
                "dataset": None, "collect_rate": collect_rate}

    n_episodes = len(list(dataset_dir.glob("episode_*/episode_*.h5")))
    poses = dump_init_poses(dataset_dir)
    (work_dir / "init_poses.json").write_text(json.dumps(poses, indent=2))

    # --- Phase B: replay ---
    replay_cfg_dir = work_dir / "cfg_replay"
    replay_cfg_dir.mkdir(exist_ok=True)
    replay_temp_name = f"_tmp_{config_name}_replay"
    replay_temp_path = replay_cfg_dir / f"{replay_temp_name}.yaml"

    def mutate_replay(cfg):
        cfg["mode"] = "replay"
        cfg.setdefault("replay", {})
        cfg["replay"]["dataset_path"] = str(dataset_dir)
        cfg["max_episodes"] = n_episodes
        cfg.setdefault("hydra", {}).setdefault("run", {})
        cfg["hydra"]["run"]["dir"] = str(replay_run_dir)
        cfg.setdefault("multi_run", {})
        cfg["multi_run"]["run_dir"] = str(replay_run_dir)

    write_temp_config(cfg_src, replay_temp_path, mutate_replay)
    replay_log = work_dir / "replay.log"
    print(f"[{config_name}] replay ({n_episodes} eps) → {replay_log}", flush=True)
    rc_r = run_main(replay_temp_name, replay_cfg_dir, replay_log, REPLAY_TIMEOUT)
    replay_rate = parse_last_success_rate(replay_log)

    summary = {
        "config": config_name, "attempt": attempt,
        "collected_episodes": n_episodes,
        "collect_log": str(collect_log),
        "replay_log": str(replay_log),
        "collect_rate": collect_rate,
        "replay_rate": replay_rate,
        "dataset": str(dataset_dir),
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if replay_rate is not None:
        succ, tot, pct = replay_rate
        flag = "✅" if succ == tot and tot > 0 else "❌"
        append_results(f"- {flag} **{config_name}** attempt {attempt}: collect {n_episodes}/{COLLECT_EPISODES}, "
                       f"replay {succ}/{tot} ({pct:.1f}%). logs: {collect_log.name}, {replay_log.name}")
    else:
        append_results(f"- ❓ **{config_name}** attempt {attempt}: collect {n_episodes}/{COLLECT_EPISODES}, "
                       f"replay rate UNPARSED (rc={rc_r})")
    return summary


def _is_pass(summary):
    if summary is None:
        return False
    rr = summary.get("replay_rate")
    if rr is None:
        return False
    succ, tot, _ = rr
    return tot > 0 and succ == tot


MAX_ATTEMPTS = 3


def main():
    only = sys.argv[1:] or LEVEL1_CONFIGS
    append_results(f"\n## Run {time.strftime('%Y-%m-%d %H:%M:%S')}  configs={only}\n")

    final: dict[str, dict | None] = {}
    for cfg in only:
        summary = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            summary = run_one(cfg, attempt)
            if _is_pass(summary):
                break
        final[cfg] = summary

    append_results("\n### Summary table")
    append_results("| Config | Collected | Replay | Status |")
    append_results("|---|---|---|---|")
    for cfg, s in final.items():
        if s is None or s.get("replay_rate") is None:
            append_results(f"| {cfg} | – | – | ❓ unparsed |")
            continue
        rr = s["replay_rate"]
        succ, tot, pct = rr
        st = "✅ PASS" if succ == tot and tot > 0 else "❌ FAIL"
        append_results(f"| {cfg} | {s['collected_episodes']}/{COLLECT_EPISODES} | {succ}/{tot} ({pct:.1f}%) | {st} |")


if __name__ == "__main__":
    main()
