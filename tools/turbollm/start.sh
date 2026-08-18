#!/usr/bin/env bash
# Start TurboLLM (if needed) and apply the hybrid 1660/1080 engines + profiles.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=/dev/null
source "$HERE/common.sh"

BASE="${TURBOLLM_URL:-http://127.0.0.1:6996}"
TURBOLLM_HOME="${TURBOLLM_HOME:-$HOME/src/TurboLLM}"
ADDR="${TURBOLLM_ADDR:-0.0.0.0:6996}"
LOG="${TURBOLLM_LOG:-/tmp/turbollm.log}"

if [[ "$ADDR" == 0.0.0.0:* ]]; then
    HEALTH="http://127.0.0.1:${ADDR##*:}"
else
    HEALTH="http://${ADDR}"
fi
export TURBOLLM_URL="${TURBOLLM_URL:-$HEALTH}"

ready() {
    curl -sf "$HEALTH/api/v1/status" >/dev/null 2>&1
}

if ! ready; then
    export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
    # shellcheck source=/dev/null
    [[ -s "$NVM_DIR/nvm.sh" ]] && . "$NVM_DIR/nvm.sh"
    if ! command -v turbollm >/dev/null 2>&1; then
        echo "start: turbollm CLI not found. Install Node 22 and: npm install -g turbollm" >&2
        exit 127
    fi
    if [[ -x "$TURBOLLM_HOME/start-local.sh" ]]; then
        # start-local.sh already waits until the daemon answers.
        TURBOLLM_SKIP_HYBRID_SETUP=1 TURBOLLM_ADDR="$ADDR" TURBOLLM_LOG="$LOG" "$TURBOLLM_HOME/start-local.sh"
    else
        turbollm --stop 2>/dev/null || true
        sleep 1
        nohup turbollm --no-open --no-telemetry --addr "$ADDR" >"$LOG" 2>&1 &
        echo "start: TurboLLM pid $!  log $LOG"
        for _ in $(seq 1 40); do
            ready && break
            sleep 0.5
        done
        ready || { echo "start: daemon did not become ready; see $LOG" >&2; exit 1; }
    fi
fi

exec "$HERE/setup.sh"
