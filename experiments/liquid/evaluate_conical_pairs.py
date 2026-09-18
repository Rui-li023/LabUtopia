"""Bounded fresh collection + strict replay, with every seed counted once.

Run in the Isaac Python environment. This script itself starts no simulator;
each child gets a fresh process, original cameras and a one-attempt limit.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import h5py
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]


def summarize_run(directory: Path, *, component: bool, expected_actions: int | None = None) -> dict:
    """Fail closed on incomplete runs, attachments, scope mismatch or short replay."""
    telemetry = directory / "run/conical_telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()] if telemetry.is_file() else []
    final = rows[-1] if rows else {}
    completed = bool(final.get("done"))
    correct_scope = bool(rows) and all(row.get("component_only") is component for row in rows)
    no_attachment = bool(rows) and all(row.get("kinematic_attachment", "missing") is None for row in rows)
    actions_consumed = final.get("replay_actions_consumed")
    full_replay = expected_actions is None or actions_consumed == expected_actions
    return {
        "directory": str(directory),
        "completed": completed,
        "success": completed and correct_scope and no_attachment and full_replay and bool(final.get("success")),
        "scope_matches": correct_scope,
        "no_kinematic_attachment": no_attachment,
        "expected_actions": expected_actions,
        "actions_consumed": actions_consumed,
        "all_actions_executed": full_replay if expected_actions is not None else None,
        "final_frame": final.get("frame"),
        "failure": final.get("failure", "incomplete_run"),
        "liquid": final.get("liquid", {}),
        "videos": [str(p) for p in sorted((directory / "run/video").glob("*.mp4"))],
    }


def counts(rows: list[dict]) -> dict:
    collected = sum(bool(row.get("collect", {}).get("success")) for row in rows)
    replayed = sum("replay" in row for row in rows)
    passed = sum(bool(row.get("replay", {}).get("success")) for row in rows)
    return {
        "collect_successes": collected,
        "collect_attempts": len(rows),
        "replay_successes": passed,
        "replay_attempts": replayed,
        "paired_successes": passed,
        "paired_attempts": len(rows),
        "collect_rate": collected / len(rows) if rows else None,
        "replay_conditional_rate": passed / replayed if replayed else None,
        "paired_rate": passed / len(rows) if rows else None,
    }


def dataset_for_replay(directory: Path) -> tuple[Path, int]:
    files = list((directory / "run/dataset").glob("episode_*/episode_*.h5"))
    if len(files) != 1:
        raise ValueError(f"Expected one fresh successful dataset, found {len(files)}")
    with h5py.File(files[0], "r") as file:
        extra = json.loads(file["init_state/init_extra_json"][()])
        if extra.get("robot_rendering", {}).get("representation") != "noninstanced":
            raise ValueError("Fresh dataset must record the robot rendering fix")
        if "liquid" not in extra or "physics_pre_warmup" not in extra:
            raise ValueError("Fresh dataset must record liquid and physical initial state")
        actions = file["replay_actions"]
        if actions.dtype.kind != "f" or actions.dtype.itemsize != 8 or len(actions) != len(file["actions"]):
            raise ValueError("Strict replay requires a full-length float64 action sequence")
        length = len(actions)
    return files[0].parent.parent, length


def run_child(cfg, directory: Path, *, component: bool, timeout: int) -> dict:
    directory.mkdir(parents=True)
    cfg.multi_run.run_dir = str(directory / "run")
    cfg.hydra.run.dir = cfg.multi_run.run_dir
    OmegaConf.save(cfg, directory / "config.yaml")
    environment = dict(os.environ)
    environment.pop("CONICAL_PHYSICS_ONLY", None)
    environment["CONICAL_ONLY"] = "1" if component else "0"
    command = [
        sys.executable,
        "-u",
        "-m",
        "experiments.liquid.run_conical_diagnostic",
        "--config-dir",
        os.path.relpath(directory, REPO),
        "--config-name",
        "config",
        "--headless",
        "--max-attempts",
        "1",
        "--max-episodes",
        "1",
    ]
    timed_out = False
    with (directory / "run.log").open("w") as log:
        child = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            child.send_signal(signal.SIGINT)
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=30)
            if isinstance(error, KeyboardInterrupt):
                raise
            timed_out = True
    return {"returncode": code, "timed_out": timed_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--scope", choices=("component", "full_level4"), default="component")
    parser.add_argument("--timeout", type=int, default=2400, help="Seconds per collect/replay process")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.timeout < 1:
        parser.error("Use unique seeds and a positive timeout")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a fresh output directory; previous attempts are never overwritten")
    output.mkdir(parents=True)
    component = args.scope == "component"
    base = OmegaConf.load(REPO / "config/level4_open_transport_pour.yaml")
    base.max_episodes = 1
    base.collector.record_precise_replay_actions = True
    base.action_randomization = False
    if component:
        base.task.max_steps = 5000
    sources = [
        REPO / name
        for name in (
            "config/level4_open_transport_pour.yaml",
            "main.py",
            "utils/pour_liquid.py",
            "utils/particle_fluid.py",
            "utils/liquid_particles.py",
            "utils/robot_rendering.py",
            "utils/robot_mass_properties.py",
            "utils/conical_grasp.py",
            "utils/gripper_contact.py",
            "utils/franka_mimic_force.py",
            "tasks/liquid_task_mixin.py",
            "tasks/open_transport_pour_task.py",
            "experiments/liquid/run_conical_diagnostic.py",
            "experiments/liquid/grasp_telemetry.py",
            "experiments/liquid/arm_telemetry.py",
            "controllers/open_transport_pour_controller.py",
            "controllers/atomic_actions/liquid_pour_controller.py",
            "controllers/atomic_actions/pick_controller.py",
            "controllers/atomic_actions/place_controller.py",
        )
    ]
    hashes = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = {
        "scope": args.scope,
        "planned_seeds": args.seeds,
        "source_hashes": hashes,
        "trained_policy_test": False,
        "rows": [],
        "status": "running",
    }

    def save() -> None:
        report["counts"] = counts(report["rows"])
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def verify_sources() -> None:
        for path in sources:
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(REPO))]:
                raise RuntimeError(f"Source changed during evaluation: {path}; do not mix versions")

    save()
    try:
        for seed in args.seeds:
            row = {"seed": seed}
            report["rows"].append(row)
            verify_sources()
            cfg = OmegaConf.create(OmegaConf.to_container(base))
            cfg.seed, cfg.mode = seed, "collect"
            directory = output / f"seed_{seed}/collect"
            print(f"seed={seed} collect started: {directory}", flush=True)
            process = run_child(cfg, directory, component=component, timeout=args.timeout)
            row["collect"] = {**summarize_run(directory, component=component), **process}
            if process["timed_out"] or process["returncode"] != 0:
                row["collect"]["success"] = False
            save()
            if not row["collect"]["success"]:
                continue
            dataset, length = dataset_for_replay(directory)
            verify_sources()
            cfg.mode, cfg.replay.dataset_path, cfg.replay.max_episodes = "replay", str(dataset), 1
            directory = output / f"seed_{seed}/replay"
            print(f"seed={seed} replay started: {directory}", flush=True)
            process = run_child(cfg, directory, component=component, timeout=args.timeout)
            row["replay"] = {**summarize_run(directory, component=component, expected_actions=length), **process}
            if process["timed_out"] or process["returncode"] != 0:
                row["replay"]["success"] = False
            save()
        verify_sources()
        report["status"] = "complete"
    except BaseException as error:
        report["status"], report["error"] = "interrupted", str(error)
        raise
    finally:
        save()
        print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
