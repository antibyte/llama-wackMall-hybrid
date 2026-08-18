#!/usr/bin/env bash
# Register Hybrid GTX 1660 Ti + Hybrid GTX 1080 engines in TurboLLM and
# write the matching LoadProfiles. Safe to re-run.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=/dev/null
source "$HERE/common.sh"

BASE="${TURBOLLM_URL:-http://127.0.0.1:6996}"
export HYBRID_ROOT
HYBRID_ROOT="$(hybrid_root)"
SOURCE_REPO="${HYBRID_SOURCE_REPO:-https://github.com/antibyte/llama-wackMall-hybrid}"

curl_api() {
    curl -sf "$@"
}

wait_ready() {
    local i
    for i in $(seq 1 40); do
        if curl_api "$BASE/api/v1/status" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

if ! wait_ready; then
    echo "setup: TurboLLM is not reachable at $BASE" >&2
    echo "setup: start it first (tools/turbollm/start.sh or ~/src/TurboLLM/start-local.sh)." >&2
    exit 1
fi

engines_json="$(curl_api "$BASE/api/v1/engines")"

find_engine_id() {
    local gpu="$1"
    python3 - "$engines_json" "$gpu" <<'PY'
import json, sys
blob, gpu = sys.argv[1], sys.argv[2]
d = json.loads(blob)
engines = d if isinstance(d, list) else d.get("engines", [])
want = "Hybrid GTX 1660 Ti" if gpu == "1660" else "Hybrid GTX 1080"
legacy = ("sm75", "1660") if gpu == "1660" else ("1080", "sm61")
markers = ("gtx1660", "hybrid-wrapper", "llama-server-hybrid-wrapper") if gpu == "1660" else ("gtx1080",)
for e in engines:
    if (e.get("name") or "") == want:
        print(e["id"]); raise SystemExit
for e in engines:
    name = (e.get("name") or "").lower()
    path = (e.get("binPath") or "").lower()
    if any(s in name for s in legacy) or any(s in path for s in markers):
        print(e["id"]); raise SystemExit
PY
}

register_or_update() {
    local gpu="$1"
    local name wrapper id payload code
    name="$(engine_name "$gpu")"
    wrapper="$HERE/llama-server-gtx${gpu}.sh"
    if [[ ! -x "$wrapper" ]]; then
        echo "setup: wrapper not executable: $wrapper" >&2
        return 1
    fi
    if ! resolve_server "$gpu" >/dev/null; then
        echo "setup: skip engine $name (no sm$([[ $gpu == 1660 ]] && echo 75 || echo 61) llama-server build)"
        return 0
    fi
    id="$(find_engine_id "$gpu" || true)"
    if [[ -n "$id" ]]; then
        curl_api -X PUT "$BASE/api/v1/engines/${id}" \
            -H 'Content-Type: application/json' \
            -d "$(python3 -c "import json,sys; print(json.dumps({'name': sys.argv[1], 'sourceRepo': sys.argv[2]}))" "$name" "$SOURCE_REPO")" \
            >/dev/null
        echo "setup: reused engine $name ($id)"
        return 0
    fi
    payload="$(python3 -c "import json,sys; print(json.dumps({'name': sys.argv[1], 'binPath': sys.argv[2], 'sourceRepo': sys.argv[3]}))" \
        "$name" "$wrapper" "$SOURCE_REPO")"
    code="$(curl -s -o /tmp/tl-engine-resp.json -w '%{http_code}' \
        -X POST "$BASE/api/v1/engines" \
        -H 'Content-Type: application/json' \
        -d "$payload")"
    if [[ "$code" != "201" && "$code" != "200" ]]; then
        echo "setup: failed to register $name (HTTP $code)" >&2
        cat /tmp/tl-engine-resp.json >&2 || true
        return 1
    fi
    echo "setup: registered $name"
    engines_json="$(curl_api "$BASE/api/v1/engines")"
}

detected="$(detect_gpu)"
echo "setup: detected GPU=$detected  hybrid=$HYBRID_ROOT"

for gpu in 1660 1080; do
    register_or_update "$gpu" || true
done
engines_json="$(curl_api "$BASE/api/v1/engines")"

model="$(resolve_model || true)"
dflash="$(resolve_dflash || true)"
if [[ -n "$model" ]]; then
    model_dir="$(dirname -- "$model")"
    curl -sf -X POST "$BASE/api/v1/modeldirs" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json,sys; print(json.dumps({'dir': sys.argv[1]}))" "$model_dir")" \
        >/dev/null || true
fi
echo "setup: model=${model:-missing}  dflash=${dflash:-missing}"

"$HERE/apply-profiles.sh"

if [[ "$detected" == 1660 || "$detected" == 1080 ]]; then
    want_id="$(find_engine_id "$detected" || true)"
    active_id="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('activeEngineId') or '')" "$engines_json")"
    if [[ -n "$want_id" && "$active_id" == "$want_id" ]]; then
        echo "setup: already active $(engine_name "$detected")"
    elif [[ -n "$want_id" ]]; then
        curl -sf -X POST "$BASE/api/v1/engine/stop" >/dev/null || true
        sleep 0.3
        if curl -sf -X POST "$BASE/api/v1/engines/${want_id}/activate" >/dev/null; then
            echo "setup: activated $(engine_name "$detected")"
        else
            echo "setup: could not activate $(engine_name "$detected") (engine busy?)"
        fi
    fi
fi

echo
echo "TurboLLM hybrid profiles:"
echo "  Engines -> Hybrid GTX 1660 Ti  |  Hybrid GTX 1080"
echo "  Active on this machine: $(engine_name "$detected" 2>/dev/null || echo "none (unknown GPU)")"
if llama_server_running; then
    echo "  GPU is already held by another llama-server -- stop it before Load."
else
    echo "  Models -> pick the MTP Qwen3.6-35B-A3B GGUF -> Load"
fi
