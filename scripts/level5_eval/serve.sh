#!/bin/bash
# Submit / stop / status the Level-5 serve rjobs (1 GPU each, ma4), parameterized
# by TASK so the same 4 model repos serve any of the 4 close tasks off their
# _0718 20k checkpoints. Each model repo carries its own container entry:
#   code/<repo>/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh
# which reads CKPT / CKPT_DIR / CONFIG / PORT from the env (defaults = close_pick).
# Usage:
#   bash serve.sh submit <openpi|lingbot|smolvla|gr00t> [task]
#   bash serve.sh stop   <model> [task]
#   bash serve.sh status [task]
# task in {close_pick(default) close_shake close_pick_place close_pour}
set -uo pipefail
ACT="${1:?action}"; MODEL="${2:-all}"; TASK="${3:-close_pick}"

WS="lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn"
OUT=/mnt/shared-storage-gpfs2/labutopia-shared/lirui/labutopia/outputs
SK=/mnt/shared-storage-gpfs2/labutopia-shared/code
ENV_INIT='unset http_proxy https_proxy; [ -f /etc/profile.d/ssh-init.sh ] && source /etc/profile.d/ssh-init.sh; . <(echo "export $(sudo strings /proc/1/environ 2>/dev/null | grep -v HOME | grep -v LS_COLORS | grep -v TERM | tr "\n" " ")") 2>/dev/null'

abbr() { case "$1" in
  close_pick) echo cp;; close_shake) echo csh;; close_pick_place) echo cpp;; close_pour) echo cpo;;
  *) return 1;; esac; }
entry() { case "$1" in
  openpi)  echo "$SK/openpi/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  lingbot) echo "$SK/lingbot-vla/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  smolvla) echo "$SK/lerobot/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  gr00t)   echo "$SK/Isaac-GR00T/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  *) return 1;; esac; }
port() { case "$1" in openpi) echo 9081;; lingbot) echo 9082;; gr00t) echo 9084;; smolvla) echo 9085;; esac; }
# Per-(model,task) checkpoint env override for the _0718 20k checkpoints.
ckpt_env() { local m="$1" t="$2"; case "$m" in
  openpi)  echo "CKPT_DIR=$OUT/pi05_labutopia_level5_$t/pi05_labutopia_level5_${t}_20k_0718/19999 CONFIG=pi05_labutopia_level5_$t";;
  lingbot) echo "CKPT=$OUT/labutopia_lingbot_level5_${t}_0718/checkpoints/global_step_20000/hf_ckpt";;
  gr00t)   echo "CKPT=$OUT/labutopia_gr00t_level5_${t}_0718";;
  smolvla) echo "CKPT=$OUT/labutopia_smolvla_level5_${t}_0718/checkpoints/020000/pretrained_model";;
esac; }
name() { echo "labutopia-$1-l5$(abbr "$2")-serve"; }

submit_one() {
  local m="$1" t="$2" e n p env; e=$(entry "$m") || { echo "unknown model $m"; return 2; }
  abbr "$t" >/dev/null || { echo "unknown task $t"; return 2; }
  n=$(name "$m" "$t"); p=$(port "$m"); env=$(ckpt_env "$m" "$t")
  ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
    $ENV_INIT
    rjob delete job '$n' 2>/dev/null; sleep 2
    rjob submit --name='$n' --gpu=1 --memory=120000 --cpu=12 \
      --namespace=ailab-ma4science --charged-group=ma4science_gpu --private-machine=group \
      --custom-resources brainpp.cn/fuse=1 --share-host-shm=true \
      --mount=gpfs://gpfs1/lirui:/mnt/shared-storage-user/lirui \
      --mount=gpfs://gpfs2/labutopia-shared:/mnt/shared-storage-gpfs2/labutopia-shared \
      --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
      --image=registry.h.pjlab.org.cn/ailab-ma4science-ma4science_cpu/lirui-workspace:20260115001617 \
      -- bash -c 'PORT=$p $env bash $e'
  " 2>&1 | grep -vE "client_global|Warning|ssh-init|bad signature"
  echo "[$m/$t] submitted as $n (port $p)"
}

case "$ACT" in
  submit)
    if [ "$MODEL" = all ]; then for m in openpi lingbot gr00t smolvla; do submit_one "$m" "$TASK"; done
    else submit_one "$MODEL" "$TASK"; fi;;
  stop)
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      rjob stop job '$(name "$MODEL" "$TASK")' 2>/dev/null; rjob delete job '$(name "$MODEL" "$TASK")' 2>/dev/null
    " 2>&1 | grep -vE "client_global|Warning|ssh-init|bad signature";;
  status)
    T="${2:-}"; pat="l5.*-serve"; [ -n "$T" ] && pat="l5$(abbr "$T")-serve"
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      rjob list 2>/dev/null | grep -E '$pat'
    " 2>&1 | grep -vE "client_global|Warning|ssh-init|bad signature";;
  *) echo "usage: $0 {submit <model> [task]|stop <model> [task]|status [task]}"; exit 2;;
esac
