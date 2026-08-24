#!/bin/bash
# Submit / stop / status the L2/L3 serve rjobs (handles kubebrain env-init).
# Usage:
#   bash serve.sh submit <model> <level>
#   bash serve.sh stop   <model> <level>
#   bash serve.sh status <model> <level>
set -uo pipefail
ACT="${1:?action}"; MODEL="${2:?model}"; LEVEL="${3:?level}"

WS_MA="lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn"
WS_AI="lirui-ai4chem.lirui.ailab-ai4chem.ws@h.pjlab.org.cn"
SK=/mnt/shared-storage-gpfs2/labutopia-shared/code
ENV_INIT='unset http_proxy https_proxy; [ -f /etc/profile.d/ssh-init.sh ] && source /etc/profile.d/ssh-init.sh; . <(echo "export $(sudo strings /proc/1/environ 2>/dev/null | grep -v HOME | grep -v LS_COLORS | grep -v TERM | tr "\n" " ")") 2>/dev/null'

case "$MODEL" in
  openpi)  WS="$WS_MA"; PORT=9081; NAME="pi05-labutopia-l${LEVEL}-all-serve";       SUB="$SK/openpi/.claude/skills/labutopia/scripts/submit_serve_labutopia_levelN.sh"; NAMEVAR=JOB_NAME;;
  lingbot) WS="$WS_MA"; PORT=9082; NAME="lingbot-labutopia-l${LEVEL}-all-serve";    SUB="$SK/lingbot-vla/.claude/skills/labutopia/scripts/submit_serve_labutopia_levelN.sh"; NAMEVAR=JOB_NAME;;
  smolvla) WS="$WS_AI"; PORT=9083; NAME="labutopia-smolvla-level${LEVEL}-all-serve"; SUB="$SK/lerobot/.claude/skills/labutopia/scripts/submit_serve_labutopia_levelN.sh"; NAMEVAR=NAME;;
  gr00t)   WS="$WS_MA"; PORT=9084; NAME="labutopia-gr00t-l${LEVEL}-all-serve";       SUB="$SK/Isaac-GR00T/.claude/skills/labutopia/scripts/submit_serve_labutopia_levelN.sh"; NAMEVAR=NAME;;
  *) echo "unknown model $MODEL"; exit 2;;
esac

# The smolvla entries default to the ai4chem workspace, but that workspace's view
# of labutopia-shared does not contain code/lerobot (the pour retrain lives on the
# ma4science side). SERVE_WS lets a caller pin the workspace without changing the
# defaults the L2/L3 campaign relies on.
WS="${SERVE_WS:-$WS}"

case "$ACT" in
  submit)
    # Delete any pre-existing job of this name first (stale jobs from prior runs
    # block submit with AlreadyExists). Prefix match covers ai4chem's suffix.
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      for j in \$(rjob list 2>/dev/null | grep -oE '${NAME}[a-z0-9-]*' | sort -u); do
        rjob delete \"\$j\" 2>/dev/null || true
      done
      sleep 2
      LEVEL=$LEVEL PORT=$PORT $NAMEVAR=$NAME ${SERVE_NAMESPACE:+NAMESPACE=$SERVE_NAMESPACE} ${SERVE_CHARGED_GROUP:+CHARGED_GROUP=$SERVE_CHARGED_GROUP} bash '$SUB'
    " 2>&1 | grep -vE "client_global|Warning|ssh-init";;
  stop)
    # Resolve by prefix — ai4chem appends a unique suffix to the job name.
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      for j in \$(rjob list 2>/dev/null | grep -oE '${NAME}[a-z0-9-]*' | sort -u); do
        rjob stop \"\$j\" 2>&1 | grep -vE 'Warning|ssh-init' || true
        rjob delete \"\$j\" 2>&1 | grep -vE 'Warning|ssh-init' || true
      done
    " 2>&1 | grep -vE "client_global";;
  status)
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$WS" "
      $ENV_INIT
      rjob list 2>&1 | grep -vE 'Warning|ssh-init' | grep -E '${NAME}' | grep -iE 'Running|Starting|Pending|Failed|Succeed'
    " 2>&1 | grep -vE "client_global";;
  *) echo "usage: $0 {submit|stop|status} <model> <level>"; exit 2;;
esac
