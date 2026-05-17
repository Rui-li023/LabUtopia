"""Collect 100 episodes per level1 task using the real DataCollector.

For each config:
  - Write a temp YAML with max_episodes=100, mode=collect, collector.type=default
  - Point hydra/multi_run run_dir to test_outputs/level1_collect100/<task>/
  - Run main.py sequentially (CLAUDE.md: never two main.py in parallel)
  - Log results to RESULTS.md

Pass arguments to limit which configs run, e.g.:
    python tests/collect_level1_100.py level1_pick level1_place
"""
import os
import re
import sys
import time
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "test_outputs" / "level1_collect100"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LEVEL1_CONFIGS = [
    "level1_pick", "level1_place", "level1_pour", "level1_press",
    "level1_shake", "level1_stir", "level1_open_door", "level1_close_door",
    "level1_open_drawer", "level1_close_drawer", "level1_CloseCentrifuge",
]

COLLECT_EPISODES = 100
COLLECT_TIMEOUT = 60 * 90  # 90 min hard cap per task

PYTHON = os.environ.get("LABUTOPIA_PY", sys.executable)
SUCCESS_RE = re.compile(r"Success Rate\s*=\s*(\d+)/(\d+)\s*\(([\d.]+)%\)")


def append_results(line: str):
    with (OUT_ROOT / "RESULTS.md").open("a") as f:
        f.write(line + "\n")
        f.flush()


def write_temp_config(src: Path, dst: Path, run_dir: Path):
    cfg = yaml.safe_load(src.read_text())
    cfg["max_episodes"] = COLLECT_EPISODES
    cfg["mode"] = "collect"
    cfg.setdefault("collector", {})
    cfg["collector"]["type"] = "default"
    cfg["collector"].setdefault("compression", "gzip")
    cfg.setdefault("hydra", {}).setdefault("run", {})
    cfg["hydra"]["run"]["dir"] = str(run_dir)
    cfg.setdefault("multi_run", {})
    cfg["multi_run"]["run_dir"] = str(run_dir)
    dst.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))


def run_main(config_name: str, config_dir: Path, log_path: Path) -> int:
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
                                  timeout=COLLECT_TIMEOUT,
                                  env={**os.environ, "PYTHONUNBUFFERED": "1"})
            return proc.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n[DRIVER] TIMEOUT after {COLLECT_TIMEOUT}s\n")
            return -1


def parse_last_success_rate(log_path: Path):
    last = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = SUCCESS_RE.search(line)
        if m:
            last = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    return last


def count_episodes(dataset_dir: Path) -> int:
    if not dataset_dir.exists():
        return 0
    return len(list(dataset_dir.glob("episode_*/episode_*.h5")))


def run_one(config_name: str):
    cfg_src = REPO / "config" / f"{config_name}.yaml"
    if not cfg_src.exists():
        append_results(f"- ❓ **{config_name}**: missing config")
        return

    work_dir = OUT_ROOT / config_name
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir = work_dir / "run"
    cfg_dir = work_dir / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    temp_name = f"_tmp_{config_name}_collect100"
    temp_path = cfg_dir / f"{temp_name}.yaml"

    write_temp_config(cfg_src, temp_path, run_dir)
    log_path = work_dir / "collect.log"
    print(f"[{config_name}] collect → {log_path}", flush=True)
    t0 = time.time()
    rc = run_main(temp_name, cfg_dir, log_path)
    elapsed = time.time() - t0

    dataset_dir = run_dir / "dataset"
    n_eps = count_episodes(dataset_dir)
    rate = parse_last_success_rate(log_path)
    rate_str = f"{rate[0]}/{rate[1]} ({rate[2]:.1f}%)" if rate else "n/a"

    flag = "✅" if n_eps >= COLLECT_EPISODES else "❌"
    append_results(
        f"- {flag} **{config_name}**: saved {n_eps}/{COLLECT_EPISODES} eps in "
        f"{elapsed/60:.1f} min, last rate {rate_str}, rc={rc}, "
        f"dataset={dataset_dir.relative_to(REPO)}"
    )


def main():
    only = sys.argv[1:] or LEVEL1_CONFIGS
    append_results(f"\n## Run {time.strftime('%Y-%m-%d %H:%M:%S')}  configs={only}\n")
    for cfg in only:
        run_one(cfg)
    append_results("\n### Done")


if __name__ == "__main__":
    main()
