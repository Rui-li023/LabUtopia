#!/bin/bash
# SSH -L tunnels from this machine to each L5 serve pod.
# The pod gets a cluster-routable 100.x IP (discovered from the serve log);
# we forward through the ma4 WORKSPACE host. Do NOT tunnel through the
# `ma4science` scheduler alias — its RemoteForward triggers a bad-signature
# error that kills the connection.
# Usage:
#   bash tunnel.sh start <openpi|lingbot|smolvla|gr00t>
#   bash tunnel.sh stop  [model|all]
#   bash tunnel.sh status
set -uo pipefail
WS="lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn"
ENV_INIT='unset http_proxy https_proxy; [ -f /etc/profile.d/ssh-init.sh ] && source /etc/profile.d/ssh-init.sh; . <(echo "export $(sudo strings /proc/1/environ 2>/dev/null | grep -v HOME | grep -v LS_COLORS | grep -v TERM | tr "\n" " ")") 2>/dev/null'
PIDDIR=/tmp/labutopia_tunnels_l5; mkdir -p "$PIDDIR"

lport() { case "$1" in openpi) echo 18081;; lingbot) echo 18082;; gr00t) echo 18084;; smolvla) echo 18085;; esac; }
rport() { case "$1" in openpi) echo 9081;; lingbot) echo 9082;; gr00t) echo 9084;; smolvla) echo 9085;; esac; }

podip() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$WS" "
    $ENV_INIT
    rjob logs job labutopia-$1-l5cp-serve 2>/dev/null | grep -oE 'POD_IP=[0-9.]+' | tail -1 | cut -d= -f2
  " 2>/dev/null | tr -d '\r' | tail -1
}

start_one() {
  local m="$1" lp rp ip; lp=$(lport "$m"); rp=$(rport "$m"); ip=$(podip "$m")
  [ -z "$ip" ] && { echo "[$m] FAIL: no POD_IP (serve running?)"; return 1; }
  pkill -f "ssh -N -L $lp:" 2>/dev/null; sleep 1
  setsid ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -N -L "$lp:$ip:$rp" "$WS" \
      >"$PIDDIR/$m.log" 2>&1 < /dev/null &
  echo $! > "$PIDDIR/$m.pid"; sleep 3
  if kill -0 "$(cat "$PIDDIR/$m.pid")" 2>/dev/null; then
    echo "[$m] tunnel up: 127.0.0.1:$lp -> $ip:$rp"
  else echo "[$m] FAIL: ssh died ($(tail -1 "$PIDDIR/$m.log" 2>/dev/null))"; return 1; fi
}

case "${1:-status}" in
  start) start_one "${2:?model}";;
  stop)
    m="${2:-all}"
    if [ "$m" = all ]; then for x in openpi lingbot smolvla gr00t; do pkill -f "ssh -N -L $(lport "$x"):" 2>/dev/null; done
    else pkill -f "ssh -N -L $(lport "$m"):" 2>/dev/null; fi; echo stopped;;
  status)
    for m in openpi lingbot smolvla gr00t; do
      lp=$(lport "$m")
      l=$(ss -tln 2>/dev/null | grep -c ":$lp ")
      echo "[$m] 127.0.0.1:$lp $([ "$l" -gt 0 ] && echo LISTENING || echo down)"
    done;;
  *) echo "usage: $0 {start <model>|stop [model]|status}"; exit 2;;
esac
