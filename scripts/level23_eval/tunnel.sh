#!/bin/bash
# SSH -L tunnels from local laptop to each model's serve POD for L2/L3 eval.
#
# Post-May cluster policy forbids --host-network for 1-GPU jobs, so serve pods
# are NOT on the GPU host's public port anymore. Instead the pod gets a
# cluster-routable IP (100.x.x.x) that the workspace can reach directly, so we
# forward straight to it:
#     local 127.0.0.1:<local_port> --(ssh -L)--> workspace --> <pod_ip>:<remote_port>
# No socat hop needed. The pod IP is discovered from the serve log each time
# (it changes per submission).
#
# Usage:
#   bash tunnel.sh start <model> <level>   # discover pod ip + start local -L
#   bash tunnel.sh stop  <model>
#   bash tunnel.sh status
#   bash tunnel.sh podip <model> <level>   # just print the discovered pod ip
#
# model -> local_port: openpi 18081, lingbot 18082, smolvla 18083, gr00t 18084
set -uo pipefail

WS_MA="lirui.lirui.ailab-ma4science.ws@h.pjlab.org.cn"
WS_AI="lirui-ai4chem.lirui.ailab-ai4chem.ws@h.pjlab.org.cn"

ws_for()      { if [ -n "${SERVE_WS:-}" ]; then echo "$SERVE_WS"; return; fi
                case "$1" in smolvla) echo "$WS_AI";; *) echo "$WS_MA";; esac; }
local_port()  { case "$1" in openpi) echo 18081;; lingbot) echo 18082;; smolvla) echo 18083;; gr00t) echo 18084;; esac; }
remote_port() { case "$1" in openpi) echo 9081;; lingbot) echo 9082;; smolvla) echo 9083;; gr00t) echo 9084;; esac; }
rjob_name()   { local m="$1" L="$2"; case "$m" in
                  openpi)  echo "pi05-labutopia-l${L}-all-serve";;
                  lingbot) echo "lingbot-labutopia-l${L}-all-serve";;
                  smolvla) echo "labutopia-smolvla-level${L}-all-serve";;
                  gr00t)   echo "labutopia-gr00t-l${L}-all-serve";; esac; }

ENV_INIT='unset http_proxy https_proxy; [ -f /etc/profile.d/ssh-init.sh ] && source /etc/profile.d/ssh-init.sh; . <(echo "export $(sudo strings /proc/1/environ 2>/dev/null | grep -v HOME | grep -v LS_COLORS | grep -v TERM | tr "\n" " ")") 2>/dev/null'

PIDDIR=/tmp/labutopia_tunnels_l23
mkdir -p "$PIDDIR"

discover_podip() {
    # Resolve the REAL rjob name by prefix first — some namespaces (ai4chem) append
    # a unique suffix (e.g. ...-serve-2368470), so an exact-name lookup fails.
    local m="$1" L="$2" ws base
    ws=$(ws_for "$m"); base=$(rjob_name "$m" "$L")
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$ws" "
        $ENV_INIT
        real=\$(rjob list 2>/dev/null | grep -E ': Running' | grep -oE '${base}[a-z0-9-]*' | head -1)
        [ -z \"\$real\" ] && real=\$(rjob list 2>/dev/null | grep -oE '${base}[a-z0-9-]*' | head -1)
        [ -z \"\$real\" ] && real='$base'
        rjob logs job \"\$real\" 2>/dev/null | grep -oE 'POD_IP=[0-9.]+|ip: [0-9.]+' | tail -1 | grep -oE '[0-9.]+'
    " 2>/dev/null | tr -d '\r' | tail -1
}

start_one() {
    local m="$1" L="$2" ws lp rp ip
    ws=$(ws_for "$m"); lp=$(local_port "$m"); rp=$(remote_port "$m")
    ip=$(discover_podip "$m" "$L")
    if [ -z "$ip" ]; then echo "[$m] FAIL: could not discover pod ip (is the serve job RUNNING + bound?)"; return 1; fi
    echo "[$m] pod_ip=$ip remote_port=$rp local_port=$lp ws=$ws"
    # kill existing
    if [ -f "$PIDDIR/$m.pid" ]; then kill "$(cat "$PIDDIR/$m.pid")" 2>/dev/null; rm -f "$PIDDIR/$m.pid"; fi
    pkill -f "ssh -N -L $lp:$ip:$rp" 2>/dev/null
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -N -L "$lp:$ip:$rp" "$ws" &
    echo $! > "$PIDDIR/$m.pid"
    sleep 3
    if kill -0 "$(cat "$PIDDIR/$m.pid")" 2>/dev/null; then
        echo "[$m] tunnel up: 127.0.0.1:$lp -> $ip:$rp (pid=$(cat "$PIDDIR/$m.pid"))"
    else
        echo "[$m] FAIL: ssh tunnel died"; return 1
    fi
}

stop_one() {
    local m="$1" lp; lp=$(local_port "$m")
    if [ -f "$PIDDIR/$m.pid" ]; then kill "$(cat "$PIDDIR/$m.pid")" 2>/dev/null; rm -f "$PIDDIR/$m.pid"; fi
    pkill -f "ssh -N -L $lp:" 2>/dev/null
    echo "[$m] stopped"
}

status_all() {
    for m in openpi lingbot smolvla gr00t; do
        local lp pidfile alive listen
        lp=$(local_port "$m"); pidfile="$PIDDIR/$m.pid"; alive="dead"
        [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null && alive="alive(pid=$(cat "$pidfile"))"
        listen=$(ss -tlnp 2>/dev/null | grep ":$lp " | head -1)
        echo "[$m] lport=$lp tunnel=$alive listen='${listen:-none}'"
    done
}

cmd="${1:-status}"
case "$cmd" in
    start)  start_one "${2:?model}" "${3:?level}";;
    stop)   if [ "${2:-all}" = all ]; then for m in openpi lingbot smolvla gr00t; do stop_one "$m"; done; else stop_one "$2"; fi;;
    status) status_all;;
    podip)  discover_podip "${2:?model}" "${3:?level}";;
    *) echo "usage: $0 {start <model> <level>|stop [model]|status|podip <model> <level>}"; exit 2;;
esac
