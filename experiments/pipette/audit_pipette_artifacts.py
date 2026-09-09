"""Read-only artifact checks for completed pipette collect/replay batches.

This does not run a policy, alter success criteria, or turn a small development
batch into acceptance. Angular peaks remain diagnostics; missing references
remain null. Use alongside the paired manifest, never instead of it.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from experiments.pipette.summarize_pipette_grip import summarize_trace


def audit_trace(records: list[dict], run: dict, tool_paths: list[str], hashes: dict) -> dict:
    report = summarize_trace(records)
    terminals = report["terminal_samples"]
    episode_ids = [sample["episode"] for sample in terminals]
    successful = [sample["episode"] for sample in terminals if sample["success"]]
    sources = records[0].get("source_sha256", {}) if records else {}
    checks = {
        "valid_process": bool(run["valid_run"] and run["exit_code"] == 0),
        "complete_trace": report["trace_complete"] and not report["continuity_errors"],
        "terminal_counts_match": len(terminals) == run["attempt_count"]
        and len(successful) == run["success_count"]
        and len(set(episode_ids)) == len(episode_ids),
        "observer_hashes_match": bool(sources) and all(hashes.get(name) == digest for name, digest in sources.items()),
        "read_only_observer": bool(records and records[0].get("read_only_observer")),
        "successful_tools_released": all(
            set(tool_paths).issubset(
                {
                    segment["object_path"]
                    for segment in report["segments"]
                    if segment["episode"] == episode and segment["release_observed"]
                }
            )
            for episode in successful
        ),
    }
    return {"checks": checks, "checks_passed": all(checks.values()), "grasp_report": report}


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr.strip():
        raise ValueError(f"Video decoding reported errors for {path}: {result.stderr.strip()}")
    stream = json.loads(result.stdout)["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]), "nb_frames": int(stream["nb_read_frames"])}


def audit_trajectory(path: Path, resolution: list[int]) -> dict:
    with h5py.File(path, "r") as dataset:
        actions, poses = dataset["actions"][()], dataset["agent_pose"][()]
        initial_positions = dataset["init_state/object_pose_positions"][()]
        initial_orientations = dataset["init_state/object_pose_orientations"][()]
        checks = {
            "finite_matching_8d_rows": actions.ndim == 2
            and actions.shape[1] == 8
            and actions.shape[0] > 0
            and actions.shape == poses.shape
            and bool(np.isfinite(actions).all() and np.isfinite(poses).all()),
            "binary_gripper_commands": actions.ndim == 2
            and actions.shape[1] == 8
            and bool(np.isin(actions[:, -1], [0, 1]).all()),
            "finite_initial_poses": bool(
                np.isfinite(initial_positions).all() and np.isfinite(initial_orientations).all()
            ),
        }
        frame_count = int(actions.shape[0])
        pose_hash = hashlib.sha256(initial_positions.tobytes()).hexdigest()
    videos = {str(video): probe_video(video) for video in sorted(path.parent.glob("*.mp4"))}
    checks["videos_match_actions"] = bool(videos) and all(
        info["nb_frames"] == frame_count and [info["width"], info["height"]] == resolution for info in videos.values()
    )
    return {
        "path": str(path),
        "checks": checks,
        "checks_passed": all(checks.values()),
        "action_rows": frame_count,
        "initial_positions_sha256": pose_hash,
        "videos": videos,
    }


def audit_manifest(manifest: dict) -> dict:
    pairs = []
    all_trajectories = []
    for pair in manifest["attempts"]:
        cfg = OmegaConf.load(pair["collect"]["config"])
        tool_paths = list(cfg.task.get("pipette_paths", [cfg.task.obj_paths[0].path]))
        traces = {}
        for mode in ("collect", "replay"):
            run = pair[mode]
            if run is None or not run.get("grip_trace"):
                traces[mode] = {"checks_passed": False, "error": "Missing run or physical grasp trace"}
                continue
            records = [json.loads(line) for line in Path(run["grip_trace"]).read_text().splitlines() if line.strip()]
            traces[mode] = audit_trace(records, run, tool_paths, manifest["code_sha256"])
        trajectories = [
            audit_trajectory(Path(path), manifest["observation_resolution"]) for path in pair["collect"]["trajectories"]
        ]
        all_trajectories.extend(trajectories)
        pairs.append(
            {
                "seed": pair["collect"]["seed"],
                "traces": traces,
                "trajectories": trajectories,
                "trajectory_count_matches": len(trajectories) == pair["collect"]["success_count"],
            }
        )
    initial_hashes = [item["initial_positions_sha256"] for item in all_trajectories]
    distinct = len(set(initial_hashes)) == len(initial_hashes)
    return {
        "artifact_checks_passed": bool(pairs)
        and distinct
        and all(
            pair["trajectory_count_matches"]
            and all(trace["checks_passed"] for trace in pair["traces"].values())
            and all(trajectory["checks_passed"] for trajectory in pair["trajectories"])
            for pair in pairs
        ),
        "distinct_initial_positions": distinct,
        "trajectory_count": len(all_trajectories),
        "action_rows": sum(item["action_rows"] for item in all_trajectories),
        "pairs": pairs,
        "note": "Artifact consistency only, not success-rate acceptance or a zero-rotation guarantee.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_manifest(json.loads(args.manifest.read_text()))
    result["manifest"] = str(args.manifest.resolve())
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
