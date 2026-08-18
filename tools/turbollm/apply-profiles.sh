#!/usr/bin/env bash
# Write TurboLLM LoadProfiles for the GTX 1660 Ti and GTX 1080 engines.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=/dev/null
source "$HERE/common.sh"

BASE="${TURBOLLM_URL:-http://127.0.0.1:6996}"
RELOAD="${1:-}"

dflash="$(resolve_dflash || true)"
if [[ -z "$dflash" ]]; then
    echo "apply-profiles: DFlash GGUF not found (set HYBRID_DFLASH)." >&2
    exit 1
fi

curl_api() {
    curl -sf "$@"
}

engines_json="$(curl_api "$BASE/api/v1/engines")" || {
    echo "apply-profiles: TurboLLM not reachable at $BASE" >&2
    exit 1
}

find_engine_id() {
    local gpu="$1"
    python3 - "$engines_json" "$gpu" <<'PY'
import json, sys
blob, gpu = sys.argv[1], sys.argv[2]
d = json.loads(blob)
engines = d if isinstance(d, list) else d.get("engines", [])
want = "Hybrid GTX 1660 Ti" if gpu == "1660" else "Hybrid GTX 1080"
legacy = ("sm75", "1660") if gpu == "1660" else ("1080", "sm61")
markers = ("gtx1660", "hybrid-wrapper") if gpu == "1660" else ("gtx1080",)
for e in engines:
    name = (e.get("name") or "")
    path = (e.get("binPath") or "")
    if name == want:
        print(e["id"]); raise SystemExit
for e in engines:
    name = (e.get("name") or "").lower()
    path = (e.get("binPath") or "").lower()
    if any(s in name for s in legacy) or any(s in path for s in markers):
        print(e["id"]); raise SystemExit
PY
}

target_keys() {
    curl_api "$BASE/api/v1/models" | python3 -c '
import sys, json
d = json.load(sys.stdin)
models = d if isinstance(d, list) else d.get("models", d.get("entries", []))
seen = set()
for m in models:
    k = m.get("key") or ""
    kl = k.lower()
    if "qwen3.6-35b-a3b" not in kl and "qwen3.6 35b a3b" not in kl:
        continue
    if "dflash" in kl:
        continue
    if k not in seen:
        seen.add(k)
        print(k)
'
}

apply_one() {
    local gpu="$1" engine_id="$2"
    local json_file
    json_file="$(mktemp)"
    python3 "$HERE/profile_json.py" "$gpu" --dflash "$dflash" >"$json_file"
    local key enc code
    while IFS= read -r key; do
        [[ -n "$key" ]] || continue
        enc="$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$key")"
        code="$(curl -s -o /tmp/tl-profile-resp.json -w '%{http_code}' \
            -X PUT "$BASE/api/v1/models/${enc}/profile?engine=${engine_id}" \
            -H 'Content-Type: application/json' \
            --data-binary @"$json_file")"
        if [[ "$code" != "200" ]]; then
            echo "FAIL profile gpu=$gpu key=$key HTTP $code" >&2
            cat /tmp/tl-profile-resp.json >&2 || true
            rm -f "$json_file"
            return 1
        fi
        echo "  OK $gpu  $key"
    done < <(target_keys)
    rm -f "$json_file"
}

echo "Applying TurboLLM profiles (DFlash $dflash)"
any=0
for gpu in 1660 1080; do
    id="$(find_engine_id "$gpu" || true)"
    if [[ -z "$id" ]]; then
        echo "  skip $gpu: engine not registered"
        continue
    fi
    echo "  engine $gpu = $id"
    apply_one "$gpu" "$id"
    any=1
done
if [[ "$any" -eq 0 ]]; then
    echo "apply-profiles: no hybrid engines registered. Run setup.sh first." >&2
    exit 1
fi

ctx_default=24576
case "$(detect_gpu)" in
    1080) ctx_default=32768 ;;
esac
curl_api -X PATCH "$BASE/api/v1/settings" \
    -H 'Content-Type: application/json' \
    -d "{\"modelDefaults\":{\"ctx\":${ctx_default},\"ngl\":99,\"imageMaxTokens\":0,\"maxTokens\":0}}" \
    >/dev/null && echo "  OK modelDefaults ctx=$ctx_default"

if [[ "$RELOAD" == "--reload" || "$RELOAD" == "reload" ]]; then
    if llama_server_running; then
        echo "  skip reload: another llama-server is already using the GPU"
        exit 0
    fi
    preferred="$(target_keys | python3 -c '
import sys
keys = [l.strip() for l in sys.stdin if l.strip()]
want = "qwen3.6-35b-a3b|Q4_K_M|22663387424"
print(want if want in keys else (keys[0] if keys else ""))
')"
    if [[ -z "$preferred" ]]; then
        echo "  skip reload: no target Qwen model" >&2
        exit 1
    fi
    echo "  Reloading $preferred"
    curl_api -X POST "$BASE/api/v1/engine/start" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json,sys; print(json.dumps({\"modelKey\": sys.argv[1]}))" "$preferred")" \
        >/dev/null
    echo "  load requested"
fi

echo "Done."
