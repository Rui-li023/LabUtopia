#!/bin/bash
# Serial per-task L5 eval driver (1 GPU): for each model, submit its serve job,
# wait until listening, tunnel, run one Isaac eval (20 ep), record 2 metrics,
# then stop. Never runs two Isaac processes at once.
# Usage: bash eval_task.sh <close_pick|close_shake|close_pick_place|close_pour> [model...]
set -uo pipefail
TASK="${1:?task}"; shift || true
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(openpi lingbot gr00t smolvla)

ROOT="${LABUTOPIA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${LABUTOPIA_PYTHON:-python}"
cd "$ROOT"
RES="$ROOT/outputs/eval_logs/RESULTS.txt"
mkdir -p "$(dirname "$RES")"
WS="lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn"
ENV_INIT='unset http_proxy https_proxy; [ -f /etc/profile.d/ssh-init.sh ] && source /etc/profile.d/ssh-init.sh; . <(echo "export $(sudo strings /proc/1/environ 2>/dev/null | grep -v HOME | grep -v LS_COLORS | grep -v TERM | tr "\n" " ")") 2>/dev/null'
abbr() { case "$1" in close_pick) echo cp;; close_shake) echo csh;; close_pick_place) echo cpp;; close_pour) echo cpo;; esac; }
lport() { case "$1" in openpi) echo 18081;; lingbot) echo 18082;; gr00t) echo 18084;; smolvla) echo 18085;; esac; }
READY='listening on|serve_forever|WebsocketPolicyServer|Uvicorn running|Application startup complete|server ready|Model initialized'

for m in "${MODELS[@]}"; do
  job="labutopia-$m-l5$(abbr "$TASK")-serve"
  echo "########## $TASK / $m ##########"
  bash scripts/level5_eval/serve.sh submit "$m" "$TASK" >/dev/null 2>&1
  # wait ready (up to ~15 min)
  ready=""
  for i in $(seq 1 120); do
    st=$(ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "$ENV_INIT
      log=\$(rjob logs job $job 2>/dev/null)
      r=\$(echo \"\$log\" | grep -ciE '$READY')
      e=\$(echo \"\$log\" | grep -ciE 'Traceback|CUDA out of memory|Killed|FileExistsError|JSONDecodeError')
      echo \"r\$r e\$e\"" 2>/dev/null | grep -oE 'r[0-9]+ e[0-9]+' | tail -1)
    echo "[$m wait $i] $st"
    case "$st" in r[1-9]*) ready=1; break;; esac
    case "$st" in *e[1-9]*) echo "[$m] SERVE ERROR"; break;; esac
    sleep 30
  done
  if [ -z "$ready" ]; then echo "$TASK $m SERVE_FAILED" | tee -a "$RES"; bash scripts/level5_eval/serve.sh stop "$m" "$TASK" >/dev/null 2>&1; continue; fi

  bash scripts/level5_eval/tunnel.sh start "$m" "$TASK" >/dev/null 2>&1
  sleep 3
  LOG="$ROOT/outputs/eval_logs/${TASK}_${m}_$(date +%H%M%S).log"
  cfg="level5_${TASK}_${m}"
  EP=$(grep -E '^max_episodes:' "config/$cfg.yaml" | grep -oE '[0-9]+' | head -1); EP=${EP:-20}
  novid="--no-video"; [ -n "${VIDEO:-}" ] && novid=""
  nohup "$PYTHON_BIN" main.py --config-name="$cfg" --headless $novid > "$LOG" 2>&1 &
  echo "[$m] eval launched (ep=$EP video=${VIDEO:-0}) -> $LOG"
  # wait until EP episodes logged (or 50 min cap), then kill isaac
  done_ok=""
  for i in $(seq 1 150); do
    line=$(grep -oE "Success Rate = [0-9]+/$EP " "$LOG" 2>/dev/null | tail -1)
    [ -n "$line" ] && { done_ok=1; break; }
    # crashed early (process gone, no completion) -> stop waiting
    pgrep -f "$cfg" >/dev/null 2>&1 || { echo "[$m] process exited early"; break; }
    sleep 20
  done
  sleep 3
  sr=$(grep -E "Success Rate" "$LOG" | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE "[0-9]+/$EP \([0-9.]+%\)" | tail -1)
  nav=$(grep -oE "NAV-PROGRESS\] ep[0-9]+: [0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.3f",s/n; else printf "NA"}')
  ft=$(grep -cE "Traceback|Connection refused|CUDA out of memory" "$LOG")
  echo "$TASK $m grasp=${sr:-NA} nav=$nav fatal=$ft" | tee -a "$RES"
  if [ -n "${VIDEO:-}" ]; then
    # let Isaac finish encoding the last mp4 + shut down on its own before forcing
    for k in $(seq 1 18); do pgrep -f "$cfg" >/dev/null 2>&1 || break; sleep 10; done
  fi
  pkill -9 -f "$cfg" 2>/dev/null; sleep 2
  bash scripts/level5_eval/tunnel.sh stop "$m" >/dev/null 2>&1
  bash scripts/level5_eval/serve.sh stop "$m" "$TASK" >/dev/null 2>&1
  sleep 3
done
echo "========== $TASK ALL DONE =========="
