#!/usr/bin/env bash
# gpu_output_check.sh LABEL MODEL
#
# Startet llama-server mit dem Modell, fragt eine Greedy-Completion ab und
# schreibt Text und SHA nach stdout. Dient dazu, die Ausgabe zweier Builds
# (z.B. unterschiedliche MMQ-Kachel) zu vergleichen: der MMQ-Kern addiert die
# Teil-Summen in einer anderen Reihenfolge, wenn sich die Kachel aendert.
#
# Aufruf:
#   tools/gpu_output_check.sh j128 ~/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf
set -uo pipefail
cd "$(dirname "$0")/.."

label=$1
model=$2
port=18093
log=/tmp/llama-server-$label.log

build-main-sm75/bin/llama-server -m "$model" --port $port --host 127.0.0.1 -ngl 99 \
    -c 4096 --no-warmup --offline > "$log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null' EXIT

for _ in $(seq 1 180); do
    if curl -s -o /dev/null "http://127.0.0.1:$port/health"; then
        break
    fi
    sleep 1
done
if ! curl -s -o /dev/null "http://127.0.0.1:$port/health"; then
    echo "server nicht bereit, Log:"
    tail -5 "$log"
    exit 1
fi

# /health antwortet, bevor das Modell geladen ist: die Completion erst
# auswerten, wenn sie nicht mehr "Loading model" meldet.
body='{"prompt":"The user asks in German: Erkläre in drei Sätzen, warum ein Puffermöwe den Ozean bevorzugt.","temperature":0,"top_k":1,"seed":42,"n_predict":96,"cache_prompt":false}'
resp=""
for _ in $(seq 1 180); do
    resp=$(curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' -d "$body")
    if [[ "$resp" != *'Loading model'* ]]; then
        break
    fi
    sleep 1
done
printf '%s' "$resp" | python3 -c "
import json,sys,hashlib
raw=sys.stdin.read()
try:
    d=json.loads(raw)
except Exception:
    print('label   :', sys.argv[1]); print('raw     :', raw[:400]); raise SystemExit(0)
text=d.get('content','')
print('label   :', sys.argv[1])
if not text:
    print('keys    :', list(d.keys()))
    print('raw     :', raw[:400])
print('sha     :', hashlib.sha256(text.encode()).hexdigest()[:16])
print('tokens  :', d.get('tokens_predicted'))
print('text    :', ' '.join(text.split())[:300])
" "$label"
