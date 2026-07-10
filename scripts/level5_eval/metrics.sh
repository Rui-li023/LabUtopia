#!/bin/bash
# Extract the two L5 metrics (nav progress + grasp success) from a run log.
# Usage: bash metrics.sh <main.py stdout log>
set -euo pipefail
LOG="${1:?run log path}"

echo "== grasp success =="
grep -E "Success Rate" "$LOG" | sed -E 's/\x1b\[[0-9;]*m//g' | tail -1

echo "== nav progress (per-episode fraction of spawn->dock distance closed) =="
grep -oE "NAV-PROGRESS\] ep[0-9]+: [0-9.]+" "$LOG" | grep -oE "[0-9.]+$" \
  | sort -n | awk '{a[NR]=$1; s+=$1}
      END{if(NR>0) printf "n=%d mean=%.3f min=%.2f max=%.2f\n", NR, s/NR, a[1], a[NR];
          else print "no [NAV-PROGRESS] lines (infer mode + mobile controller required)"}'
