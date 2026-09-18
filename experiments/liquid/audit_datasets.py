"""Read-only inventory of pour episodes; never upgrades legacy data in place."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import yaml

POUR_TYPES = {"pour", "pick_pour", "clean_beaker", "liquid_mixing", "open_transport_pour", "mobile_pour"}


def inspect_episode(path: Path) -> dict:
    result = {"path": str(path), "status": "legacy_no_liquid"}
    try:
        with h5py.File(path, "r") as episode:
            result["actions"] = int(episode["actions"].shape[0]) if "actions" in episode else 0
            result["precise_actions"] = "replay_actions" in episode
            key = "init_state/init_extra_json"
            extra = json.loads(episode[key][()]) if key in episode else {}
            liquid = extra.get("liquid")
            if liquid is None:
                return result
            positions = np.asarray(liquid.get("positions", []))
            velocities = np.asarray(liquid.get("velocities", []))
            valid = (
                positions.ndim == 2 and positions.shape[1:] == (3,) and len(positions) > 0
                and positions.shape == velocities.shape
                and np.isfinite(positions).all() and np.isfinite(velocities).all()
                and "physics_pre_warmup" in extra and bool(liquid.get("collision_hash"))
                and liquid.get("physics_dt", 0) > 0
            )
            result.update(status="liquid_snapshot_present" if valid else "invalid_liquid_snapshot",
                          particles=len(positions), version=liquid.get("version"))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(status="read_error", error=str(error))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/pour_liquid/all_tasks_audit.json")
    args = parser.parse_args()
    files = subprocess.check_output(
        ["rg", "--files", "--hidden", "--no-ignore", "-g", "*.h5", "-g", "!policy/**",
         "-g", "!assets/**", "-g", "!**/.git/**"], text=True).splitlines()
    configs = {}
    rows = []
    for name in sorted(files):
        path = Path(name)
        config = None
        for parent in path.parents:
            candidate = parent / "config.yaml"
            if candidate not in configs:
                configs[candidate] = yaml.safe_load(candidate.read_text()) if candidate.is_file() else None
            if configs[candidate] is not None:
                config = configs[candidate]
                break
        kind = (config or {}).get("controller_type", "")
        # Include orphaned historical exports, but label the identification method.
        if kind not in POUR_TYPES and not any(token in name.lower() for token in ("pour", "clean_beaker", "liquid_mixing")):
            continue
        row = inspect_episode(path)
        row.update(controller_type=kind or "unknown", identified_by="config" if kind in POUR_TYPES else "path_only")
        rows.append(row)
    report = {"scanned_h5": len(files), "pour_h5": len(rows),
              "statuses": dict(Counter(row["status"] for row in rows)),
              "by_controller": {kind: dict(Counter(r["status"] for r in rows if r["controller_type"] == kind))
                                for kind in sorted({row["controller_type"] for row in rows})},
              "note": "Metadata audit only, not a physical replay or policy success rate.", "episodes": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
