"""Fixed-seed, paired collect/replay evaluation; failures remain in the denominator.

Run with the Isaac Python environment. Each seed is one attempt, never a retry
until success. Use development seeds while tuning, then a fresh held-out set.
The JSON manifest records config, logs, exit codes and data/video paths.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_ACCEPTANCE_THRESHOLD = 0.90


def run_attempt(
    cfg,
    mode: str,
    seed: int,
    directory: Path,
    timeout: int,
    attempts: int = 1,
    resolution: int = 256,
    observe_grip: bool = False,
    *,
    save_video: bool = False,
    render_height: int | None = None,
) -> dict:
    if render_height is not None and (not isinstance(render_height, int) or render_height <= 0):
        raise ValueError("Replay render height must be a positive integer")
    directory.mkdir(parents=True, exist_ok=False)
    cfg.mode = mode
    cfg.max_episodes = attempts
    cfg.multi_run.run_dir = str(directory / "run")
    cfg.task.replay_restore_orientation = True
    cfg_path = directory / "attempt.yaml"
    OmegaConf.save(cfg, cfg_path)
    command = [
        sys.executable,
        "experiments/main.py",
        "--config-dir",
        os.path.relpath(directory, ROOT),
        "--config-name",
        "attempt",
        "--headless",
        "--max-episodes",
        str(attempts),
        "--max-attempts",
        str(attempts),
        "--seed",
        str(seed),
        "--single-camera",
        "0",
        "--render-width",
        str(resolution),
        "--render-height",
        str(resolution if render_height is None else render_height),
        "--save-png-dir",
        str(directory / "frames"),
        "--save-png-on-failure",
    ]
    if not save_video:
        command.append("--no-video")
    trace_path = directory / "grip.jsonl"
    if observe_grip:
        command = [
            sys.executable,
            "-m",
            "experiments.pipette.observe_pipette_grip",
            "--output",
            str(trace_path),
            "--",
            *command[2:],
        ]
    log_path = directory / "simulation.log"
    with log_path.open("w") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = -999
    log_text = ANSI.sub("", log_path.read_text(errors="replace"))
    stats = re.findall(r"Episode Stats: Success Rate = (\d+)/(\d+)", log_text)
    valid_run = exit_code == 0 and bool(stats) and int(stats[-1][1]) == attempts
    success_count = int(stats[-1][0]) if valid_run else 0
    h5_files = list((directory / "run").rglob("episode_*.h5"))
    if mode == "collect":
        success_count = min(success_count, len(h5_files))
    success = success_count == attempts
    failures = re.findall(r"Failure Reason: (.+)", log_text)
    return {
        "seed": seed,
        "mode": mode,
        "success": success,
        "success_count": success_count,
        "attempt_count": attempts,
        "valid_run": valid_run,
        "exit_code": exit_code,
        "failure": failures[-1] if failures else "",
        "config": str(cfg_path),
        "log": str(log_path),
        "dataset": str(directory / "run/dataset"),
        "trajectories": [str(path) for path in h5_files],
        "command": command,
        "grip_trace": str(trace_path) if observe_grip else None,
        "rendered_videos": [str(path) for path in sorted((directory / "run/video").glob("*.mp4"))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=("level2_pipette_to_rack", "level4_three_pipettes_to_rack", "level4_four_pipettes_to_rack"),
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable Hydra override; evaluate a candidate without editing default configs",
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="Square observation resolution; task default is 256"
    )
    parser.add_argument("--min-episodes", type=int, default=30, help="Minimum independent attempts for acceptance")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ACCEPTANCE_THRESHOLD,
        help="Strict paired success-rate threshold for both levels (default: 0.90)",
    )
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=1,
        help="Attempts in each seeded RNG stream; avoids restarting Isaac every episode",
    )
    parser.add_argument(
        "--collect-only", action="store_true", help="Development diagnostics; cannot satisfy paired acceptance"
    )
    parser.add_argument(
        "--observe-grip", action="store_true", help="Save read-only physical hand/object poses for every attempt"
    )
    args = parser.parse_args()
    if args.min_episodes < 1 or args.episodes_per_seed < 1 or args.resolution < 64:
        parser.error("Episode counts must be positive")
    if not 0 < args.threshold < 1:
        parser.error("Acceptance threshold must be finite and strictly between zero and one")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        base = compose(config_name=args.config, overrides=args.override)
    OmegaConf.set_struct(base, False)
    asset = ROOT / str(base.usd_path)
    config_fingerprint = OmegaConf.to_container(base, resolve=True)
    for key in ("multi_run", "hydra", "max_episodes", "replay", "mode", "seed"):
        config_fingerprint.pop(key, None)
    manifest = {
        "config": args.config,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "observation_resolution": [args.resolution, args.resolution],
        "asset": str(asset),
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "task_config_sha256": hashlib.sha256(json.dumps(config_fingerprint, sort_keys=True).encode()).hexdigest(),
        "threshold": args.threshold,
        "minimum_episodes": args.min_episodes,
        "read_only_grip_observer": args.observe_grip,
        "code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "experiments/main.py",
                ROOT / "robots/franka/franka.py",
                ROOT / "robots/base_robot.py",
                ROOT / "utils/gripper_contact.py",
                ROOT / "utils/franka_mimic_force.py",
                ROOT / "experiments/support/physics_scene_steps.py",
                ROOT / "utils/pipette_rack_material.py",
                ROOT / "utils/object_utils.py",
                ROOT / "utils/replay_data_loader.py",
                ROOT / "experiments/support/grasp_telemetry.py",
                ROOT / "tasks/base_task.py",
                ROOT / "data_collectors/data_collector.py",
                ROOT / "controllers/robot_controllers/trajectory_controller.py",
                ROOT / "robots/franka/rmpflow_controller.py",
                ROOT / "robots/franka/rmpflow/franka_rmpflow_common.yaml",
                ROOT / "robots/franka/rmpflow/robot_descriptor.yaml",
                ROOT / "robots/franka/lula_franka_gen.urdf",
                ROOT / "experiments/pipette/evaluate_pipette_pairs.py",
                ROOT / "experiments/pipette/observe_pipette_grip.py",
                ROOT / "controllers/base_controller.py",
                ROOT / "controllers/atomic_actions/atomic_base_controller.py",
                *sorted((ROOT / "controllers").glob("*pipette*.py")),
                ROOT / "controllers/flask_to_cork_controller.py",
                ROOT / "controllers/atomic_actions/pick_controller.py",
                ROOT / "controllers/atomic_actions/place_controller.py",
                *sorted((ROOT / "tasks").glob("*pipette*.py")),
            ]
        },
        "attempts": [],
        "accepted": False,
    }
    for seed in args.seeds:
        collect_cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        collected = run_attempt(
            collect_cfg,
            "collect",
            seed,
            out / f"seed_{seed}_collect",
            args.timeout,
            args.episodes_per_seed,
            args.resolution,
            args.observe_grip,
        )
        pair = {"collect": collected, "replay": None}
        manifest["attempts"].append(pair)
        if collected["success_count"] and not args.collect_only:
            replay_cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
            replay_cfg.replay.dataset_path = collected["dataset"]
            replay_cfg.replay.episode_indices = list(range(collected["success_count"]))
            pair["replay"] = run_attempt(
                replay_cfg,
                "replay",
                seed,
                out / f"seed_{seed}_replay",
                args.timeout,
                collected["success_count"],
                args.resolution,
                args.observe_grip,
            )
        attempts = manifest["attempts"]
        n_collect = sum(p["collect"]["success_count"] for p in attempts)
        n_replay = sum(p["replay"]["success_count"] if p["replay"] else 0 for p in attempts)
        n_attempts = sum(p["collect"]["attempt_count"] for p in attempts)
        manifest["rates"] = {
            "collect": [n_collect, n_attempts],
            "replay_conditional": [n_replay, n_collect],
            "paired_end_to_end": [n_replay, n_attempts],
        }
        manifest["inputs_unchanged"] = hashlib.sha256(asset.read_bytes()).hexdigest() == manifest[
            "asset_sha256"
        ] and all(
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
            for path, digest in manifest["code_sha256"].items()
        )
        manifest["accepted"] = (
            len(attempts) == len(args.seeds)
            and n_attempts >= args.min_episodes
            and manifest["inputs_unchanged"]
            and not args.collect_only
            and all(p["collect"]["valid_run"] and p["replay"] and p["replay"]["valid_run"] for p in attempts)
            and n_replay / n_attempts > manifest["threshold"]
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"seed": seed, "rates": manifest["rates"], "accepted": manifest["accepted"]}), flush=True)


if __name__ == "__main__":
    main()
