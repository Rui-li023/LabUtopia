"""Combine disjoint seeded evaluation batches, never repeated attempts."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(manifests, minimum_attempts=30):
    if not manifests:
        raise ValueError("No evaluation manifests")
    common = ("config", "asset_sha256", "task_config_sha256", "code_sha256", "observation_resolution", "threshold")
    reference = manifests[0]
    for manifest in manifests:
        if any(key not in manifest or manifest[key] != reference.get(key) for key in common):
            raise ValueError("Batches have different or missing frozen input fingerprints")
    seeds = [pair["collect"]["seed"] for manifest in manifests for pair in manifest["attempts"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated RNG streams cannot count as independent attempts")
    pairs = [pair for manifest in manifests for pair in manifest["attempts"]]
    attempts = sum(pair["collect"]["attempt_count"] for pair in pairs)
    collected = sum(pair["collect"]["success_count"] for pair in pairs)
    replayed = sum(pair["replay"]["success_count"] if pair["replay"] else 0 for pair in pairs)
    complete = all(pair["collect"]["valid_run"] and pair["replay"] and pair["replay"]["valid_run"] for pair in pairs)
    unchanged = all(manifest["inputs_unchanged"] for manifest in manifests)
    return {
        **{key: reference[key] for key in common},
        "seeds": seeds,
        "rates": {
            "collect": [collected, attempts],
            "replay_conditional": [replayed, collected],
            "paired_end_to_end": [replayed, attempts],
        },
        "minimum_episodes": minimum_attempts,
        "inputs_unchanged": unchanged,
        "all_runs_complete": complete,
        "accepted": bool(
            complete and unchanged and attempts >= minimum_attempts and replayed / attempts > reference["threshold"]
        ),
        "attempts": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads(path.read_text()) for path in args.manifests]
    result = summarize(manifests)
    result["source_manifests"] = [str(path.resolve()) for path in args.manifests]
    result["current_inputs_match"] = all(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
        for path, digest in result["code_sha256"].items()
    ) and all(hashlib.sha256(Path(m["asset"]).read_bytes()).hexdigest() == m["asset_sha256"] for m in manifests)
    result["accepted"] = result["accepted"] and result["current_inputs_match"]
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({key: result[key] for key in ("rates", "accepted")}, indent=2))


if __name__ == "__main__":
    main()
