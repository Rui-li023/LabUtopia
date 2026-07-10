#!/bin/bash
# Submit / stop / status the Level-5 close_pick serve rjobs (1 GPU each, ma4).
# Each model repo carries its own container entry:
#   code/<repo>/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh
# Usage:
#   bash serve.sh submit <openpi|lingbot|smolvla|gr00t>
#   bash serve.sh stop   <model>
#   bash serve.sh status [model]
set -uo pipefail
ACT="${1:?action}"; MODEL="${2:-all}"

WS="lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn"
SK=/mnt/shared-storage-gpfs2/labutopia-shared/code
ENV_INIT='unset http_proxy https_proxy; [ -f /etc/profile.d/ssh-init.sh ] && source /etc/profile.d/ssh-init.sh; . <(echo "export $(sudo strings /proc/1/environ 2>/dev/null | grep -v HOME | grep -v LS_COLORS | grep -v TERM | tr "\n" " ")") 2>/dev/null'

entry() { case "$1" in
  openpi)  echo "$SK/openpi/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  lingbot) echo "$SK/lingbot-vla/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  smolvla) echo "$SK/lerobot/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  gr00t)   echo "$SK/Isaac-GR00T/.claude/skills/labutopia/scripts/serve_labutopia_level5.sh";;
  *) return 1;; esac; }
port() { case "$1" in openpi) echo 9081;; lingbot) echo 9082;; gr00t) echo 9084;; smolvla) echo 9085;; esac; }
name() { echo "labutopia-$1-l5cp-serve"; }

submit_one() {
  local m="$1" e n p; e=$(entry "$m") || { echo "unknown model $m"; return 2; }
  n=$(name "$m"); p=$(port "$m")
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
      -- bash -c 'PORT=$p bash $e'
  " 2>&1 | grep -vE "client_global|Warning|ssh-init|bad signature"
}

case "$ACT" in
  submit) submit_one "$MODEL";;
  stop)
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      rjob stop job '$(name "$MODEL")' 2>/dev/null; rjob delete job '$(name "$MODEL")' 2>/dev/null
    " 2>&1 | grep -vE "client_global|Warning|ssh-init|bad signature";;
  status)
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      rjob list 2>/dev/null | grep -E 'l5cp-serve'
    " 2>&1 | grep -vE "client_global|Warning|ssh-init|bad signature";;
  *) echo "usage: $0 {submit <model>|stop <model>|status}"; exit 2;;
esac
