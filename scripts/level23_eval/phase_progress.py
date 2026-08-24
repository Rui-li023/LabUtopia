#!/usr/bin/env python3
"""Sub-goal (phase) progress metrics for L4 inference eval logs.

Binary task success (all phases) is ~0% for these long-horizon L4 tasks, so we
report PARTIAL CREDIT: how many of each task's ordered sub-goals the policy
completes per episode, plus a per-phase pass-rate funnel. Phase completion is
read from the controllers' oracle markers (`Inference: <phase> success!` and
`[cleanbeaker infer] step <N> success`) — the same world-state sub-goal
detection that drives infer-mode instruction advancement.

Usage:
  python3 scripts/level23_eval/phase_progress.py --model gr00t   # newest l4_gr00t_* dir
  python3 scripts/level23_eval/phase_progress.py --dir outputs/infer_eval/l4_gr00t_1782567113
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# canonical ordered sub-goals per task (phase .value strings / step indices)
TASK_PHASES: dict[str, list[str]] = {
    "liquid_mixing": [
        "picking1",
        "pouring1",
        "placing1",
        "picking2",
        "pouring2",
        "placing2",
        "picking3",
        "pouring3",
        "placing3",
        "press",
    ],
    "open_transport_pour": [
        "opening",
        "picking1",
        "transporting",
        "picking2",
        "pouring",
        "transporting2",
    ],  # NB: per-episode order is randomized
    "clean_beaker": [f"step{i}" for i in range(1, 8)],
    "device_operation": [
        "opening_door",
        "move_higher",
        "picking_beaker",
        "placing_beaker",
        "picking_beaker3",
        "placing_beaker3",
        "pressing_button",
    ],
}
RANDOMIZED = {"open_transport_pour"}

SR_RE = re.compile(r"Success Rate = (\d+)/(\d+)")
INF_RE = re.compile(r"Inference: ([a-z0-9_]+) success!")
CB_RE = re.compile(r"cleanbeaker infer\] step (\d+) success")


def parse_log(task: str, log: str) -> tuple[int, int, dict[str, int]]:
    """Return (binary_succ, episodes, {phase: completed_count})."""
    text = Path(log).read_text(errors="ignore")
    srs = SR_RE.findall(text)
    succ, eps = (int(srs[-1][0]), int(srs[-1][1])) if srs else (0, 0)
    counts: dict[str, int] = {p: 0 for p in TASK_PHASES[task]}
    if task == "clean_beaker":
        for n in CB_RE.findall(text):
            k = f"step{n}"
            if k in counts:
                counts[k] += 1
    else:
        for ph in INF_RE.findall(text):
            if ph in counts:
                counts[ph] += 1
    return succ, eps, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="eval output dir (l4_<model>_<ts>)")
    ap.add_argument("--model", default=None, help="resolve newest l4_<model>_* dir")
    args = ap.parse_args()

    if args.dir:
        d = Path(args.dir)
    elif args.model:
        cands = sorted(glob.glob(str(REPO / "outputs/infer_eval" / f"l4_{args.model}_*")), key=os.path.getmtime)
        if not cands:
            print(f"no l4_{args.model}_* dir found")
            return 1
        d = Path(cands[-1])
    else:
        print("need --dir or --model")
        return 1

    model = d.name.split("_")[1]
    print(f"=== L4 sub-goal progress — model={model}  dir={d.name} ===\n")
    rows = []
    for task, phases in TASK_PHASES.items():
        log = d / "logs" / f"{model}_{task}.log"
        if not log.exists():
            print(f"## {task}: (not started)\n")
            continue
        succ, eps, counts = parse_log(task, str(log))
        n = len(phases)
        total_adv = sum(counts.values())
        mean_phases = total_adv / eps if eps else 0.0
        prog = mean_phases / n if n else 0.0
        note = "  [order randomized per-episode]" if task in RANDOMIZED else ""
        print(f"## {task}  (sub-goals={n}, episodes={eps}){note}")
        print(f"   binary success : {succ}/{eps} ({(succ / eps if eps else 0):.0%})")
        print(f"   mean sub-goals : {mean_phases:.2f}/{n}   -> progress score {prog:.0%}")
        print("   per-phase pass rate (funnel):")
        for p in phases:
            c = counts[p]
            bar = "#" * round(20 * (c / eps if eps else 0))
            print(f"     {p:<16} {c:>3}/{eps}  {(c / eps if eps else 0):>4.0%} {bar}")
        print()
        rows.append((task, eps, succ, mean_phases, n, prog))

    if rows:
        print("=== SUMMARY ===")
        print(f"{'task':<22}{'eps':>4}{'binary':>9}{'mean_subgoals':>15}{'progress':>10}")
        tot_prog = 0.0
        for task, eps, succ, mp, n, prog in rows:
            print(f"{task:<22}{eps:>4}{f'{succ}/{eps}':>9}{f'{mp:.2f}/{n}':>15}{prog:>9.0%}")
            tot_prog += prog
        print(f"{'MEAN':<22}{'':>4}{'':>9}{'':>15}{tot_prog / len(rows):>9.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
