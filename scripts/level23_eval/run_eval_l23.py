#!/usr/bin/env python3
"""LabUtopia Level 1-4 inference evaluation runner.

Loops over the tasks of one level for one model, derives isolated temporary
inference configs from the checked-in task baselines, runs main.py per task,
parses the cumulative
"Episode Stats: Success Rate = N/M" line (emitted by BaseController.reset),
and writes a per-model summary JSON.

Prereqs:
- An SSH -L tunnel from 127.0.0.1:<model_port> to the model's serve pod
  (see scripts/level23_eval/tunnel.sh).
- Isaac Sim env available (the main.py launcher).

Usage:
  python3 scripts/level23_eval/run_eval_l23.py --level 3 --model openpi --episodes 5
  python3 scripts/level23_eval/run_eval_l23.py --level 2 --model lingbot --tasks pour_liquid stir_glassrod
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO / "config"

TASKS_BY_LEVEL: dict[int, list[str]] = {
    1: [
        "close_centrifuge",
        "close_door",
        "close_drawer",
        "open_door",
        "open_drawer",
        "pick",
        "place",
        "pour",
        "press",
        "shake",
        "stir",
    ],
    2: [
        "flask_to_cork",
        "flask_to_triangle",
        "heat_liquid",
        "open_close",
        "pipette_to_rack",
        "pour_liquid",
        "shake_beaker",
        "stir_glassrod",
        "stopper_to_flask",
        "transport_beaker",
    ],
    3: [
        "heat_liquid",
        "open",
        "pick",
        "pour_liquid",
        "press",
        "transport_beaker",
    ],
    4: [
        "liquid_mixing",
        "clean_beaker",
        "open_transport_pour",
        "device_operation",
    ],
}


@dataclass
class ModelSpec:
    name: str
    local_port: int
    action_chunk_len: int
    image_color: str = "bgr"
    obs_names: dict[str, str] = field(default_factory=dict)


OPENPI_OBS = {
    "image": "observation/image",
    "image_2": "observation/wrist_image",
    "wrist_image": "observation/wrist_image_2",
}
SMOLVLA_OBS = {
    "image": "observation/image",
    "image_2": "observation/image_2",
    "wrist_image": "observation/wrist_image",
}

MODELS: dict[str, ModelSpec] = {
    "openpi": ModelSpec("openpi", 18081, action_chunk_len=10, obs_names=OPENPI_OBS),
    "lingbot": ModelSpec("lingbot", 18082, action_chunk_len=50, obs_names=OPENPI_OBS),
    "smolvla": ModelSpec("smolvla", 18083, action_chunk_len=50, obs_names=SMOLVLA_OBS),
    "gr00t": ModelSpec("gr00t", 18084, action_chunk_len=40, obs_names=OPENPI_OBS),
}

STATS_RE = re.compile(r"Success Rate = (\d+)/(\d+)")
EPISODE_SUCC_RE = re.compile(r"Episode (\d+) succeeded")
EPISODE_FAIL_RE = re.compile(r"Episode (\d+) failed")


@dataclass
class TaskResult:
    task: str
    succeeded: int = 0
    failed: int = 0
    error: str | None = None
    duration_s: float = 0.0
    log_path: str = ""

    @property
    def total(self) -> int:
        return self.succeeded + self.failed

    @property
    def rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


def patch_config_for_model(
    source_path: Path,
    output_path: Path,
    port: int,
    chunk: int,
    image_color: str,
    max_episodes: int,
    obs_names: dict[str, str],
) -> None:
    cfg = OmegaConf.load(source_path)
    cfg.mode = "infer"
    cfg.max_episodes = max_episodes
    if "infer" not in cfg:
        cfg.infer = {}
    cfg.infer.type = "remote"
    cfg.infer.host = "127.0.0.1"
    cfg.infer.port = port
    cfg.infer.action_chunk_len = chunk
    cfg.infer.image_color = image_color
    cfg.infer.obs_names = obs_names
    cfg.infer.n_obs_steps = 1
    cfg.infer.timeout = int(getattr(cfg.infer, "timeout", 30))
    cfg.infer.max_retries = int(getattr(cfg.infer, "max_retries", 3))
    OmegaConf.save(cfg, output_path)


def run_one_task(
    level: int,
    model: ModelSpec,
    task: str,
    config_dir: Path,
    log_dir: Path,
    timeout_s: int,
    video: bool = False,
) -> TaskResult:
    log_path = log_dir / f"{model.name}_{task}.log"
    cfg_name = f"level{level}_{task}"
    # Video records episode_{N}.mp4 (failures tagged _failure) under the run's
    # outputs/infer/<date>/<task>/video/ dir; works in headless.
    cmd = [
        sys.executable,
        "main.py",
        "--config-name",
        cfg_name,
        "--config-dir",
        str(config_dir.relative_to(REPO)),
        "--headless",
    ]
    if not video:
        cmd.append("--no-video")
    print(f"\n>>> [{model.name}/L{level}/{task}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = TaskResult(task=task, log_path=str(log_path))
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_s, check=False
            )
        if proc.returncode != 0:
            res.error = f"main.py exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        res.error = f"timeout {timeout_s}s"
    except Exception as e:
        res.error = repr(e)
    res.duration_s = round(time.time() - t0, 1)

    try:
        log_text = log_path.read_text()
    except Exception:
        log_text = ""
    stats = STATS_RE.findall(log_text)
    if stats:
        succ, total = stats[-1]
        res.succeeded = int(succ)
        res.failed = int(total) - int(succ)
    else:
        res.succeeded = len(EPISODE_SUCC_RE.findall(log_text))
        res.failed = len(EPISODE_FAIL_RE.findall(log_text))
    print(
        f"<<< [{model.name}/L{level}/{task}] succ={res.succeeded}/{res.total} "
        f"dur={res.duration_s}s err={res.error or '-'}",
        flush=True,
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4], required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks (default: all tasks of the level)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=3600, help="Per-task timeout in seconds (covers all episodes)")
    ap.add_argument("--output-dir", default="outputs/infer_eval")
    ap.add_argument(
        "--video", action="store_true", help="record per-episode mp4s (slower; under outputs/infer/<date>/)"
    )
    args = ap.parse_args()

    model = MODELS[args.model]
    tasks = args.tasks or TASKS_BY_LEVEL[args.level]

    out_dir = REPO / args.output_dir / f"l{args.level}_{model.name}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(
        f"=== Eval L{args.level} {model.name} → port={model.local_port} "
        f"chunk={model.action_chunk_len} episodes={args.episodes} ==="
    )
    temp_config = tempfile.TemporaryDirectory(prefix="_eval_", dir=CONFIG_DIR)
    temp_config_dir = Path(temp_config.name)
    for task in tasks:
        source = CONFIG_DIR / f"level{args.level}_{task}.yaml"
        patch_config_for_model(
            source,
            temp_config_dir / source.name,
            model.local_port,
            model.action_chunk_len,
            model.image_color,
            args.episodes,
            model.obs_names,
        )

    results: list[TaskResult] = []
    for task in tasks:
        r = run_one_task(
            args.level,
            model,
            task,
            temp_config_dir,
            log_dir,
            timeout_s=args.timeout,
            video=args.video,
        )
        # Retry once on a crash that produced no episodes (transient WS-connect
        # TimeoutError -> Isaac dirty-shutdown segfault, exit -11). The tunnel is
        # typically still healthy; a 15s pause lets a stale server-side connection
        # from the prior task's process exit clear before reconnecting.
        if r.error and r.total == 0:
            print(f"  [retry] {task} crashed with 0 episodes ({r.error}); retry in 15s", flush=True)
            time.sleep(15)
            r = run_one_task(
                args.level,
                model,
                task,
                temp_config_dir,
                log_dir,
                timeout_s=args.timeout,
                video=args.video,
            )
        results.append(r)
        summary = {
            "level": args.level,
            "model": model.name,
            "local_port": model.local_port,
            "action_chunk_len": model.action_chunk_len,
            "image_color": model.image_color,
            "episodes_per_task": args.episodes,
            "results": [
                {
                    "task": x.task,
                    "succeeded": x.succeeded,
                    "failed": x.failed,
                    "total": x.total,
                    "rate": x.rate,
                    "error": x.error,
                    "duration_s": x.duration_s,
                    "log_path": x.log_path,
                }
                for x in results
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    temp_config.cleanup()
    total_succ = sum(r.succeeded for r in results)
    total_run = sum(r.total for r in results)
    print(f"\n=== L{args.level} {model.name} done ===")
    print(f"Overall: {total_succ}/{total_run} ({total_succ / total_run if total_run else 0:.1%})")
    print(f"Saved → {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
