"""Summarize a read-only grasp sidecar without interpreting missing data as success."""

import argparse
import json
from pathlib import Path

from experiments.support.grasp_telemetry import summarize_grasp_samples


def summarize_trace(records: list[dict]) -> dict:
    samples = [record for record in records if record.get("type") == "sample"]
    report = summarize_grasp_samples(samples)
    continuity_errors = []
    previous = None
    for sample in samples:
        if previous is not None and sample["mode"] == "replay" and sample["episode"] == previous["episode"]:
            # The replay action clock stops incrementing during final settling.
            delta = sample["replay_action_step"] - previous["replay_action_step"]
            if delta not in (0, 1):
                continuity_errors.append({"episode": sample["episode"], "step": sample["replay_action_step"]})
        previous = sample
    report.update(
        trace_complete=bool(
            records
            and records[0].get("type") == "metadata"
            and any(r.get("type") == "shutdown_requested" and r.get("last_step_terminal") for r in records[-2:])
        ),
        sample_count=len(samples),
        continuity_errors=continuity_errors,
        terminal_samples=[
            {"episode": sample["episode"], "success": sample["success"]} for sample in samples if sample.get("done")
        ],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    print(json.dumps(summarize_trace(records), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
