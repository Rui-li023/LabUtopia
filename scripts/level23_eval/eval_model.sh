#!/bin/bash
# One-shot per-model eval: wait for the serve pod to bind, start the SSH tunnel,
# wait for the WS handshake, then run the full L2/L3 eval for that model.
# Assumes the serve rjob for (model, level) has already been submitted.
#
# Usage: bash eval_model.sh <level> <model> [episodes] [timeout_per_task]
#   bash eval_model.sh 3 lingbot 10
set -uo pipefail
LEVEL="${1:?level}"; MODEL="${2:?model}"; EP="${3:-20}"; TMO="${4:-2400}"
VIDEO="${VIDEO:-1}"   # default ON: save per-episode mp4s (user standing pref)
DIR="${LABUTOPIA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ISAAC="${LABUTOPIA_PYTHON:-python}"
cd "$DIR"

case "$MODEL" in
  openpi)  LP=18081;; lingbot) LP=18082;;
  smolvla) LP=18083;; gr00t)   LP=18084;;
  *) echo "unknown model $MODEL"; exit 2;;
esac

echo "===== eval_model L$LEVEL $MODEL ep=$EP ====="

# 1. Wait for the pod IP to surface in the serve log, then bring up the tunnel.
echo "[1] waiting for pod ip + starting tunnel..."
ok=0
for i in $(seq 1 60); do   # up to ~10 min
  bash scripts/level23_eval/tunnel.sh start "$MODEL" "$LEVEL" >/tmp/tun_${MODEL}.log 2>&1
  if grep -q "tunnel up" /tmp/tun_${MODEL}.log; then ok=1; cat /tmp/tun_${MODEL}.log; break; fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "[1] FAIL: tunnel never came up"; cat /tmp/tun_${MODEL}.log; exit 1; }

# 2. Do NOT pre-handshake here. A dangling readiness connection poisons
#    single-client servers (lingbot/gr00t): their next handshake then times out
#    and every task crashes (exit -11). main.py is the first/only WS client and
#    retries connect 15x/10s, which (plus ~80s Isaac boot) rides out the server's
#    model load. A short grace lets a freshly-submitted server start loading.
echo "[2] grace before launch (no pre-handshake; main.py retries connect) ..."
sleep 20

# 3. Run the full eval for this model+level.
echo "[3] running run_eval_l23 (video=$VIDEO) ..."
$ISAAC scripts/level23_eval/run_eval_l23.py --level "$LEVEL" --model "$MODEL" \
    --episodes "$EP" --timeout "$TMO" ${VIDEO:+--video}

echo "===== eval_model L$LEVEL $MODEL DONE ====="
