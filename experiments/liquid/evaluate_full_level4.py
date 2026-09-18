"""Bounded full Level-4 collection evaluation; one attempt per seed, no replay.

Reuse the production configuration and the existing read-only diagnostic runner.
Only the seed, mode, episode limit and output paths differ between children.
"""

import argparse
import ast
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from experiments.liquid.evaluate_conical_pairs import REPO, run_child, summarize_run

PHASES = {"opening", "picking1", "transporting", "picking2", "pouring", "transporting2"}
EXTRA_SOURCES = (
    "experiments/liquid/evaluate_full_level4.py",
    "experiments/liquid/evaluate_conical_pairs.py",
    "controllers/atomic_actions/open_controller.py",
    "controllers/atomic_actions/atomic_base_controller.py",
    "controllers/atomic_actions/place_controller.py",
    "controllers/robot_controllers/grapper_manager.py",
    "tasks/base_task.py",
    "utils/liquid_metrics.py",
    "utils/liquid_transfers.py",
    "utils/object_utils.py",
    "utils/task_utils.py",
    "utils/video_recorder.py",
    "factories/task_factory.py",
    "factories/controller_factory.py",
    "factories/robot_factory.py",
    "factories/collector_factory.py",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_counts(report: dict) -> None:
    rows = report["rows"]
    finished = [row for row in rows if row["status"] == "complete"]
    passed = sum(bool(row.get("collect", {}).get("success")) for row in finished)
    planned = len(report["planned_seeds"])
    report["counts"] = {
        "planned": planned,
        "started": len(rows),
        "finished": len(finished),
        "successes": passed,
        "failures": len(finished) - passed,
        "unfinished": planned - len(finished),
        "completed_success_rate": passed / len(finished) if finished else None,
        "planned_success_fraction": passed / planned,
    }
    report["accepted"] = bool(
        report["status"] == "complete" and len(finished) == planned and passed >= report["required_successes"]
    )


def summarize_full_run(directory: Path) -> dict:
    result = summarize_run(directory, component=False)
    telemetry = directory / "run/conical_telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry.read_text().splitlines()] if telemetry.exists() else []
    observed = list(dict.fromkeys(row["phase"] for row in rows))
    log_path = directory / "run.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    sequences = re.findall(r"Generated random task sequence: (\[[^\n]*\])", log)
    sequence = ast.literal_eval(sequences[-1]) if sequences else []
    failure_lines = [line for line in log.splitlines() if "Failure Reason:" in line]
    all_phases = PHASES.issubset(observed) and len(sequence) == 6 and set(sequence) == PHASES
    result.update(
        phase_sequence=sequence,
        observed_phases=observed,
        all_six_phases_observed=all_phases,
        failure_log=failure_lines[-1] if failure_lines else "",
        ended_on_step_limit="ended on max_steps=" in log,
    )
    result["success"] = bool(result["success"] and all_phases)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path, help="Previous verified batch results.json")
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--required-successes", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not 1 <= args.required_successes <= len(args.seeds):
        parser.error("Use distinct seeds and a success threshold between one and the number of seeds")
    if args.timeout < 1:
        parser.error("Timeout must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a fresh output directory; attempts are never overwritten")

    baseline = json.loads(args.baseline.read_text())
    known = dict(baseline["source_hashes"])
    extra_manifest = args.baseline.parent / "additional_source_hashes.sha256"
    for line in extra_manifest.read_text().splitlines():
        digest, path = line.split(None, 1)
        known[path] = digest
    for path, digest in known.items():
        if sha256(REPO / path) != digest:
            raise RuntimeError(f"Verified component version has changed: {path}")
    hashes = dict(known)
    hashes.update({path: sha256(REPO / path) for path in EXTRA_SOURCES})

    base = OmegaConf.load(REPO / "config/level4_open_transport_pour.yaml")
    if base.task_type != "open_transport_pour" or base.controller_type != "open_transport_pour":
        raise ValueError("Expected the full open_transport_pour task/controller")
    if not base.liquid.enabled or bool(base.task.get("fixed_phase_sequence", False)):
        raise ValueError("Keep liquid enabled and the original randomized phase groups")
    base.mode = "collect"
    base.max_episodes = 1
    output.mkdir(parents=True)
    report = {
        "scope": "full_level4",
        "mode": "collect",
        "replay_test": False,
        "trained_policy_test": False,
        "planned_seeds": args.seeds,
        "required_successes": args.required_successes,
        "attempt_limit_per_seed": 1,
        "task_max_steps": int(base.task.max_steps),
        "physics_dt": float(base.liquid.physics_dt),
        "child_timeout_seconds": args.timeout,
        "source_hashes": hashes,
        "baseline": str(args.baseline.resolve()),
        "started_at": utc_now(),
        "rows": [],
        "status": "running",
    }

    def save() -> None:
        update_counts(report)
        report["updated_at"] = utc_now()
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def verify_sources() -> None:
        for path, digest in hashes.items():
            if sha256(REPO / path) != digest:
                raise RuntimeError(f"Source changed during evaluation: {path}")

    save()
    try:
        for seed in args.seeds:
            verify_sources()
            row = {"seed": seed, "status": "running", "started_at": utc_now()}
            report["rows"].append(row)
            save()
            cfg = OmegaConf.create(OmegaConf.to_container(base))
            cfg.seed = seed
            directory = output / f"seed_{seed}/collect"
            print(f"seed={seed} full_level4 collect started: {directory}", flush=True)
            process = run_child(cfg, directory, component=False, timeout=args.timeout)
            row["collect"] = {**summarize_full_run(directory), **process}
            row["collect"]["success"] = bool(
                row["collect"]["success"] and process["returncode"] == 0 and not process["timed_out"]
            )
            row.update(status="complete", finished_at=utc_now())
            save()
            print(json.dumps({"seed": seed, "result": row["collect"], "counts": report["counts"]}), flush=True)
        verify_sources()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        if report["rows"] and report["rows"][-1]["status"] == "running":
            report["rows"][-1]["status"] = "interrupted"
        raise
    finally:
        report["finished_at"] = utc_now()
        save()
        print(
            json.dumps({"status": report["status"], "accepted": report["accepted"], "counts": report["counts"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
