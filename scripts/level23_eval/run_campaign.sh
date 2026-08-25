#!/bin/bash
# Sequentially evaluate multiple models for one level, one Isaac Sim at a time.
# For each model: ensure serve rjob Running -> eval_model (tunnel+wait+eval) ->
# stop tunnel + serve (free the GPU). Continues to the next model on failure.
#
# Usage: bash run_campaign.sh <level> "<m1 m2 ...>" [episodes] [timeout_per_task]
#   bash run_campaign.sh 3 "lingbot smolvla gr00t" 10
#
# IMPORTANT: do NOT run this while any other main.py / Isaac Sim is active
# (single-GPU saturate rule). It runs its own evals strictly sequentially.
set -uo pipefail
LEVEL="${1:?level}"; MODELS="${2:?models}"; EP="${3:-20}"; TMO="${4:-3000}"
VIDEO="${VIDEO:-1}"; export VIDEO   # default ON (user standing pref); eval_model reads it
DIR="${LABUTOPIA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$DIR"
SK=scripts/level23_eval

for M in $MODELS; do
  echo ""
  echo "########## CAMPAIGN L$LEVEL $M (ep=$EP) ##########"

  # 1. ensure the serve rjob is Running (submit if not).
  if bash $SK/serve.sh status "$M" "$LEVEL" 2>/dev/null | grep -q ': Running'; then
    echo "[campaign] serve $M L$LEVEL already Running"
  else
    echo "[campaign] submitting serve $M L$LEVEL"
    bash $SK/serve.sh submit "$M" "$LEVEL" || echo "[campaign] submit returned nonzero (may already exist)"
    sleep 25
  fi

  # 2. wait-for-bind + tunnel + full eval (eval_model handles all three).
  bash $SK/eval_model.sh "$LEVEL" "$M" "$EP" "$TMO" || echo "[campaign] eval_model $M FAILED (continuing)"

  # 3. free the GPU for the next model.
  bash $SK/tunnel.sh stop "$M" 2>/dev/null || true
  bash $SK/serve.sh stop "$M" "$LEVEL" 2>/dev/null || true
  echo "########## CAMPAIGN L$LEVEL $M DONE ##########"
done

echo ""
echo "================= CAMPAIGN COMPLETE: L$LEVEL [$MODELS] ================="
