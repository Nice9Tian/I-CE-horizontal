#!/usr/bin/env bash
# Pod watchdog: stops the pod when it is idle, so a lost session cannot burn credit.
#   nohup bash watchdog.sh > /workspace/iceh/watchdog.log 2>&1 &
# Idle = no experiment process (train / eval / cache build / baseline / setup / run script) for IDLE_MIN minutes.
# A hard deadline (MAX_HOURS) applies whatever is running.  Touch /workspace/iceh/KEEPALIVE to reset the idle clock
# from outside (e.g. while uploading).  Stopping: `runpodctl stop pod` when the image provides it with credentials,
# otherwise the container's PID 1 is killed, which puts the pod into EXITED and ends GPU billing (the container disk
# is lost, so copy results out first: run scripts write them under /workspace/iceh/results).
IDLE_MIN="${IDLE_MIN:-30}"
MAX_HOURS="${MAX_HOURS:-14}"
ROOT="${ROOT:-/workspace/iceh}"
PATTERN='train\.py|eval_run\.py|build_draws\.py|build_pool_draw\.py|baseline_ice\.py|embed_tags\.py|setup\.sh|run_2048\.sh|join_full_pool|hf download'
start=$(date +%s)
idle_since=""
log() { echo "[$(date '+%F %T')] $*"; }
log "watchdog up: idle limit ${IDLE_MIN} min, hard limit ${MAX_HOURS} h, pod ${RUNPOD_POD_ID:-?}"
while true; do
  now=$(date +%s)
  busy=$(pgrep -f "$PATTERN" | grep -v "^$$\$" | wc -l)
  keep=0; [ -f "$ROOT/KEEPALIVE" ] && keep=$(( now - $(stat -c %Y "$ROOT/KEEPALIVE") < IDLE_MIN * 60 ? 1 : 0 ))
  if [ "$busy" -gt 0 ] || [ "$keep" = "1" ]; then
    idle_since=""
  else
    [ -z "$idle_since" ] && idle_since=$now && log "idle: no experiment process"
  fi
  reason=""
  [ -n "$idle_since" ] && [ $(( now - idle_since )) -ge $(( IDLE_MIN * 60 )) ] && reason="idle for ${IDLE_MIN} min"
  [ $(( now - start )) -ge $(( MAX_HOURS * 3600 )) ] && reason="hard limit ${MAX_HOURS} h reached"
  if [ -n "$reason" ]; then
    log "stopping the pod: $reason"
    sync
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ] && runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
      log "runpodctl stop pod issued"
    else
      log "no runpodctl credentials: killing the container (PID 1)"
      kill -TERM 1 2>/dev/null; sleep 20; kill -KILL 1 2>/dev/null
    fi
    exit 0
  fi
  sleep 300
done
