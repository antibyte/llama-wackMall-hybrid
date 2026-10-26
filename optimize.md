# Kolibri-1-DSpark vom 2026-10-26

**Ergebnis: Der DSpark-Draft `intuit/Kolibri-1-dspark` ist installiert und funktioniert, macht Kolibri-1 auf dieser Maschine aber nicht schneller (greedy +-0, mit dem Sampling von agollm -9.5 %). Er bleibt opt-in (`KOLIBRI_DSPARK=1` in `start-kolibri-1.sh`); das agollm-Rezept ist unveraendert.**

| agollm-Rezept (64k, S=19), A/B/A/B, 3 Thinking-Prompts x 384 Token | greedy | Sampling (temp 1.0, top_k 128, top_p 0.97) |
| --- | ---: | ---: |
| ohne Draft | 15.40 | 16.71 |
| DSpark n_max 4, p_min 0.25 | 15.37 (+-0) | 15.12 (-9.5 %) |
| DSpark n_max 3, p_min 0.25 | 15.28 (-1 %) | - |
| DSpark n_max 2, p_min 0 (eigener Lauf, Baseline 15.94) | 13.72 (-14 %) | - |
| DSpark n_max 7, p_min 0.85 (eigener Lauf, Baseline 15.94) | 13.05 (-18 %) | - |

## Ursachen

- Der Verify ist teuer: ein Schritt mit k Token rechnet bis zu 6k verschiedene kalte Experten je Layer (384 Experten, top-6, aus RAM und SSD). Gemessen: 1 Token ~52 ms, 3 Token ~130 ms, 4 Token ~160 ms, 8 Token ~280 ms. Mit 3 Token braucht der Break-even ~2.5 akzeptierte Token je Schritt; dazu kommen ~9 ms Draft und Inject je Schritt, auch wenn p_min nichts durchlaesst.
- Die Akzeptanz passt zur Modellkarte (MT-Bench 2.98 Token je Schritt mit FP8 und 7 Draft-Token): Essay 2.6-3.0, Code 2.1-2.3, der deutsche Go-Prompt 1.5-2.2 (Draft auf englischen Daten trainiert). Nur sehr repetitiver Text gewinnt: Zaehlen 1..300 77 % Akzeptanz, 32-38 statt ~19 Token/s.
- Der Confidence-Head sagt 1 - TV voraus (Akzeptanz unter Rejection Sampling) und ist fuer greedy konservativ: ab p_min 0.5 kommen kaum Drafts durch (die dann zu 97-100 % passen), 0.25 war der beste Wert.

## Installation

- `download-kolibri-1-dspark.sh`: laedt Revision a59d588 (SHA-256-Pruefung), uebersetzt die speculators-Config ins DFlash-Format und schreibt `~/models/kolibri-1/Kolibri-1-DSpark-Q8_0.gguf` (443 MB). `aux_hidden_state_layer_ids` [4, 14, 24, 34, 44] zaehlen wie HF-`hidden_states` (k = Eingang von Layer k), deshalb `target_layer_ids` minus 1. `sliding_window` mit `sliding_window_non_causal: false` macht die Block-Attention kausal (`causal_head`). Der Tokenizer kommt aus dem Kolibri-GGUF. Braucht ein Python mit torch, safetensors und numpy (`KOLIBRI_PYTHON`).
- `src/models/kolibri1.cpp`: der Graph setzt `res->t_layer_inp[il]`, damit DFlash/DSpark die Layer-Eingaenge bekommt (ohne Draft keine Wirkung).
- `start-kolibri-1.sh`, `KOLIBRI_DSPARK=1`: n_max 4, p_min 0.25, kombinierter Inject, Experten-Reserve 1200 statt 500 MiB. Der Auto-Fit laeuft vor dem Laden des Drafts und kennt ihn nicht; mit 500 MiB lief der Draft bei 8k Kontext in einen OOM. Mit 1200: 8k Kontext, S=22, Peak 4.9 GB.
- Gegengeprueft: Layer-Offset +1 (Ausgang von Layer 4..44) gibt 13.0 statt 14.4 % Akzeptanz; kausal statt bidirektional 15.5 statt 14.4 %; Sliding Window gegen volle Attention identisch auf kurzen Prompts; kombinierter gegen separaten Inject identisch; Q8_0 trifft wie BF16 (Zaehlen 77 gegen 80 %); BF16 laeuft in einen OOM (Draft-Speicher nicht gemessen).
- Greedy mit Draft weicht in den Token von greedy ohne Draft ab (Multi-Spalten-Kernel im Verify runden anders als der 1-Spalten-Decode); alle Ausgaben sind kohaerent.

Artefakte (gitignored): `benchmark-results/kolibri-dspark-20261025/` (`tools/`, `runs/`, `sweep1.txt`, `sweep2.txt`, `abab.txt`, `abab2.txt`).

# MiniCPM-V-4.7-Optimierung vom 2026-10-25

**Ergebnis: MiniCPM-V 4.7 35B-A3B decodiert mit +29 % (31.8 -> 41.0 Token/s, temp 1.0, ABBA); das Ziel von +10 % ist erreicht.** Die Scheduler-, Tiering- und MMVQ-Aenderungen sind im Hauptbuild und in `workers/llama` installiert. Sechs weitere getierte agollm-Modelle nutzen die neue Ueberlappung (+4 bis +20 %). Kontextgroesse (262144), q4_0-KV und Modellquantisierung bleiben erhalten. Kein Commit oder Push.

| Vergleich (GTX 1660 Ti, Ryzen 7 4800H, Performance) | Referenz | Neu | Gewinn |
| --- | ---: | ---: | ---: |
| MiniCPM-V 4.7, agollm-Rezept, temp 1.0, A/B/B/A (je 3 Prompts x 2, 384 Token) | 31.77 | 40.99 | +29.0 % |
| MiniCPM-V 4.7, greedy, Go/Code 256 Token, Referenz ohne Spec | 31.93 | 44.5 | +39 % |
| MiniCPM-V 4.7, Prompt mit 9098 Token | - | pp 371, tg 40.1 | kein OOM, Peak 5.38 GiB |
| MiniCPM-V 4.7 live ueber agollm (2184-Token-Systemprompt) | - | tg 40.3, pp 297 | |

Referenz ist das bisherige Rezept: Experten komplett auf der CPU (`-no-cmoe -ncmoe 99`) plus DSpark-Draft. Die Benchmark-Prompts (Go-Architektur, Python-Code, Essay) sind nicht im Heatprofil enthalten. Die Ausgaben aendern sich gegenueber der Referenz nur durch Rundung (heisse Experten rechnen auf der GPU), Stichproben auf Deutsch und Englisch sind inhaltlich korrekt.

## Ursachen

- Pro Decode-Token liegen ~15.7 ms in den CPU-Experten (8 x 1.7 MB je Layer, ~35 GB/s von maximal ~40 GB/s) und ~12 ms auf der GPU. Die CPU-Kernel sind nahe an der Speicherbandbreite; der Hebel ist, Experten von der CPU wegzunehmen und GPU- und CPU-Arbeit zu ueberlappen.
- Expert-Tiering war fuer MiniCPM nie mit Heatprofil gemessen: ohne `LLAMA_EXPERT_HOT` und mit ADAPT=0 legt der Tier keine Stores an (irrefuehrende Meldung "VRAM full"). Die Nutzung ist stark schief: 16 Experten je Layer decken auf fremden Prompts 16-46 % der Aufrufe ab, 40 Experten ~45-60 %.
- Mit Tiering liefen die heissen GPU-Experten seriell vor den CPU-Experten (gleicher Stream vor dem D2H) und kosteten ~2.5 ms je Token; bei 16 % Abdeckung brachte Tiering deshalb nichts. `LLAMA_EXPERT_CPU_ASYNC` lief fuer qwen35moe nie asynchron, weil im CPU-Split eine leere RESHAPE-View neben `MOE_COLD` steht; echtes Async (Koordinator-Thread) ist ausserdem langsamer.
- DSpark kostet je Schritt ~8 ms Draft (LM-Head mit 5 Spalten, 6 Markov-Matvecs mit K=256, Radix-Sorts fuer top-k(10)) und der Verify mit 2-4 Token ist auf Turing teuer: MMVQ mit 2-8 Spalten braucht 1.6-2.5x der Zeit von einer Spalte (Block-Reduktion und 16-Bit-Loads dominieren, DRAM nur ~52 %). Sein Draft-Kontext belegt ~1.6 GiB VRAM.

## Aenderungen im Code

- `ggml-backend.cpp`, `GGML_SCHED_DEFER_HOST_INPUTS=1` (opt-in): CPU-erzeugte Aktivierungen werden erst vor ihrem ersten Verbraucher in einen GPU-Split kopiert. Laeuft ein CPU-Split synchron und sind alle Eingaenge des folgenden GPU-Splits verzoegerbar, startet der Scheduler dessen unabhaengigen Kopf vor dem CPU-Split. So laufen heisse Experten und Shared Expert parallel zu den kalten CPU-Experten.
- `llama-graph.cpp`/`llama-expert-tier.*`, `LLAMA_EXPERT_COLD_FIRST=1` (opt-in): der kalte CPU-Knoten wird direkt nach dem Router gebaut (bisher nur mit CPU_ASYNC). `llm_graph_context::moe_overlap` laesst `build_moe_ffn` unabhaengige Arbeit direkt danach einreihen; qwen35moe baut dafuer den Shared Expert vorher (Graph ohne Cold-first unveraendert).
- `ggml-cpu.cpp`: der Async-Check fuer reine `MOE_COLD`-Splits ignoriert leere View-Ops.
- `mmvq.cu`, nur Turing (sm_75), standardmaessig aktiv: fuer 2..8 Spalten ein Warp je Block mit 8 Zeilen (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS) bzw. 4 Zeilen (Q2_K..Q6_K); fuer kleine K bei einer Spalte ein Warp mit 8 Zeilen statt einem aktiven von vier Warps. `GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS` hat jetzt Default 0 (Tabelle). Beispiele: Q8_0 K=2048 N=8192 bei 3 Spalten 101 -> 68 us, LM-Head 248144x2048 bei 5 Spalten 3.46 -> 2.33 ms, Markov-Head K=256 452 -> 281 us, IQ4_XS K=512 29 -> 16 us. Sweep ueber 11 Typen in `sweep-types.txt`.

Patch nur dieser Aenderungen: [source-changes.patch](benchmark-results/minicpm-v47-opt-20261025/source-changes.patch). Die uebrigen Arbeitsbaumaenderungen stammen aus frueheren Sessions.

## MiniCPM-Rezept

`config/llama/minicpm-v4.7-35b-a3b.yaml`: Tiering an (`LLAMA_ARG_NO_CPU_MOE` entfernt), statisches Heatprofil `workers/llama/assets/minicpm-v47-chat.csv` aus 44 Prompts (12 aus `scripts/hybrid_profile_corpus.jsonl`, 32 neue in `profile-corpus-extra.jsonl`), `LLAMA_EXPERT_S=40` (der Tier klemmt auf freien VRAM minus 512 MiB), ADAPT=0, keine Warm-Slots, Cold-first plus Defer, Phasen-Batching Prefill 2048 / Decode 64, DSpark aus.

- DSpark aus: mit Tiering und S=16 (mehr passt neben dem Draft-Kontext nicht) lag DSpark bei temp 1.0 unter Plain-Decode (35.8 vs 36.8 Token/s); ohne Draft passen S=40 (44.5 statt ~38 Token/s greedy). Mit den neuen MMVQ-Kerneln sank der Draft von 8.1 auf 6.8 ms, das reicht nicht.
- S-Kurve ohne Draft (greedy Go/Code): S=20 40.7, S=28 42.3, S=36 43.8, S=40 44.8. S=32 plus 8 Warm-Slots: 44.1. Peak-VRAM S=40: 5.27 GiB, mit 9k-Prompt 5.38 GiB.
- Verworfen: echtes CPU-Async (31.7 statt 34.4 Token/s), REUSE_ROWS/ACT_PARALLEL/DOWN_PREFETCH im Verify (+-0), Prefill-Ubatch unter 2048 (halbiert Prefill langer Prompts wegen der Expertenkopien ueber PCIe), `cuStreamWaitValue32` fuer einen vorab eingereihten GPU-Rest (Treiber meldet `CAN_USE_STREAM_MEM_OPS=0`).

## Andere Modelle

Die Ueberlappung (`LLAMA_EXPERT_COLD_FIRST=1`, `GGML_SCHED_DEFER_HOST_INPUTS=1`, ggf. `GGML_CUDA_ASYNC_HOST_COPY=1`) ist bitgleich; A/B mit dem neuen Build, greedy, Go/Code 256 Token:

| Rezept | ohne | mit | Gewinn |
| --- | ---: | ---: | ---: |
| nex-n2.5-mini-i1 | 35.36 | 41.98 | +18.7 % |
| mellum2.1-12b-a2.5b-thinking | 42.53 | 50.96 | +19.8 % (andere Rundung durch die Knotenreihenfolge, schon mit Cold-first allein) |
| flint | 37.01 | 40.18 | +8.6 % |
| qwen3.6-35b-a3b-hybrid | 38.21 | 39.31 | +2.9 % Plain; mit DFlash und Default-Sampling 3x3 Laeufe gegen das alte Paket: 44.22 -> 45.86 (+3.7 %) |
| hy-mt2-30b | 27.11 | 28.34 | +4.5 % |
| lfm2.5-24b-a2b | 36.02 | 37.50 | +4.1 % |
| holo4-35b-a3b | 40.27 | 39.23 | -2.6 %, nicht aktiviert (60 % Abdeckung plus Warm-Slots) |

Diese sechs Rezepte enthalten die Schalter. Kolibri wurde nicht gemessen und bleibt unveraendert. Die MMVQ-Aenderung wirkt fuer alle Turing-Rezepte: Qwen-Plain 37.47 -> 38.21 (kleines K bei heissen Experten), Qwen-DFlash-Draft 5.9 -> 5.25 ms. Speculative-Laeufe mit unterschiedlichen Tokenfolgen sind wegen der Akzeptanz nur ueber mehrere Wiederholungen vergleichbar.

## Pruefungen und Installation

- `test-backend-ops` CUDA0: MUL_MAT 1177/1177, MUL_MAT_ID 863/863, TOP_K, ARGMAX, ARGSORT bestanden. Eigener CUDA/CPU-Vergleich mit ungeraden Zeilenzahlen (10 Typen, K 256/512/2048, 1-8 Spalten): neue und alte Bibliothek mit identischem schlechtestem NMSE 5.6e-4.
- Defer/Prelaunch aendert keine Tokens (MiniCPM, Qwen, Nex, Hy-MT2, Holo4, Flint, LFM jeweils identische Hashes).
- Installiert mit `install.py` (Backups, atomarer Austausch, `ai-routerd -check`, `go test ./internal/config/`): `libggml-base`, `libggml-cpu`, `libggml-cuda` (RUNPATH `$ORIGIN`), `libllama`, `ORIGIN.txt`, Heatprofil, 7 Rezepte und die angepassten MiniCPM-Pruefungen in `config_test.go`. `go test` scheiterte schon vorher an `flint routing`; sonst keine Fehler. Rollback: `python3 benchmark-results/minicpm-v47-opt-20261025/install.py --rollback` bei gestopptem Dienst.
- agollm laeuft wieder; Live-Anfrage an MiniCPM-V 4.7 lud die Bibliotheken aus `workers/llama`, danach wurde ling-tiny vorgeladen.
- Startskripte: neues `start-minicpm-v47-35b.sh` (spiegelt das agollm-Rezept, Profil `profiles/minicpm-v47-chat.csv`, gleiche Tokens wie das Rezept). Cold-first und Defer stehen mit Default-, Validierungs-, Export- und Anzeigelogik in `start1660.sh`, `start-nex-n25-mini-i1.sh`, `start-nex-n25-mini.sh`, `start-hy-mt2-30b.sh`, `start-ornith.sh`, in `start-flint.sh`, im Qwen-Preset von `router1660.sh` und in `tools/turbollm/env/gtx1660.env`. Zusaetzlich ueber die Startskripte gemessen (greedy, DFlash wo konfiguriert): Nex ohne i1 32.28 -> 35.35 (+9.5 %), Ornith 42.89 -> 44.84 (+4.5 %), Gemma4 26B 33.56 -> 33.18 (-1.1 %, nicht aktiviert). Mit DFlash aendern sich die Tokens schon durch Cold-first allein (Knotenreihenfolge); Defer aendert sie nicht. Kolibri und die Pascal-Skripte sind unveraendert.
- agollm-Tests: zwei veraltete Flint-Erwartungen (Routing 40/114, Fused Gate/Up an) und ein Test mit dem geloeschten Rezept `spark-x2.5-4b` sind nachgezogen; `go test ./...` besteht.
- Benchmarks mit kaltem Page-Cache (nach anderen grossen Modellen) sind deutlich langsamer (36 statt 45 Token/s); alle Zahlen oben mit warmem Cache.

Nachweise: [Laeufe](benchmark-results/minicpm-v47-opt-20261025/runs/), [Messwerkzeug mbench.py](benchmark-results/minicpm-v47-opt-20261025/tools/), [MMVQ-Sweeps](benchmark-results/minicpm-v47-opt-20261025/sweep-types.txt), [Installation](benchmark-results/minicpm-v47-opt-20261025/deployment.json), [Profile](benchmark-results/minicpm-v47-opt-20261025/profiles/).

---

# Flint-Optimierung vom 2026-10-25

**Ergebnis: rund +1.6 bis +1.7 Token/s; das Ziel von +5 Token/s ist nicht erreicht.** Die geprueften CPU-/CUDA-Aenderungen sind im Hauptbuild, in den gemeinsamen agollm-Bibliotheken und im Flint-Startprofil installiert. Kontextgroesse, F16-KV und Modellquantisierung bleiben erhalten.

| Kontrollierter Vergleich | Referenz Token/s | Optimiert Token/s | Gewinn |
| --- | ---: | ---: | ---: |
| Go, bisheriger Prefill, A/B/B/A | 38.101 | 39.817 | +1.716 / +4.50% |
| Go, CPU-Experten-Prefill, A/B | 38.463 | 40.036 | +1.572 / +4.09% |
| Cafe, CPU-Experten-Prefill, A/B | 37.830 | 39.516 | +1.686 / +4.46% |

GTX 1660 Ti Mobile und Ryzen 7 4800H, 64512 Kontext, Batch/UBatch 512, acht CPU-Threads, F16-KV, unveraendertes Heatprofil und effektiv 15 Hot-Slots. Je Prozess zwei Warmups mit 256 Tokens, danach drei Messungen mit 256 Tokens, Greedy/Seed 42, Prompt-Cache aus. A/B verwendet Prozessmediane; A/B/B/A den Mittelwert der beiden Prozessmediane je Variante. Alle gemessenen Flint-Anfragen hatten null physische Reads. Die beiden neuen A/B-Paare sind keine breite Workload-Studie.

Innerhalb jeder Vergleichsreihe stimmen die Tokenarrays exakt ueberein. CPU-Prefill erzeugt beim Go-Prompt eine andere Sequenz als der bisherige GPU-Experten-Prefill; diese beiden Arbeitslasten werden nicht gegeneinander als tokenidentischer Gewinn gerechnet. Die kurzen Ausgabesequenzen ersetzen keine umfassende Qualitaetsbewertung. Der regulaere Produktionsbuild wurde separat geprueft: Go 40.139, Cafe 39.430 Token/s als Mediane, alle sechs Antworten tokenidentisch zur jeweiligen CPU-Prefill-Referenz. Diese spaeteren Werte sind Validierung, kein zusaetzlicher A/B-Gewinn.

## Uebernommene Aenderungen

- IQ4_XS auf AVX2: Nibble-Indizes werden vor dem 256-Bit-Shuffle zusammengefasst. Die Float-Akkumulation bleibt unveraendert. Die direkten Dot-Product-Mikrotests verbessern sich um etwa 6-8%; dieser Wert wird nicht zum Modellgewinn addiert.
- Q8_0-MMVQ mit Fusion und einer Token-Spalte: optional vier Ausgabezeilen je Block auf Turing. IDs, Small-K und unpassende Restzeilen behalten ihren bisherigen Pfad. `GGML_CUDA_MMVQ_Q8_FUSED_NCOLS1_ROWS` ist standardmaessig aus; Flint setzt `4`.
- Quantisierte dichte MMVQ/GLU-Fusion darf mit `GGML_CUDA_MMVQ_INPUT_ALIAS=1` ihren zuvor vollstaendig in Q8-Zwischenspeicher kopierten F32-Eingang als Ausgabespeicher wiederverwenden. Die Ausnahme gilt nur fuer diese bereits gepruefte Drei-Operator-Fusion. Andere Alias-Pruefungen bleiben aktiv; der Schalter ist standardmaessig aus.
- Flints statisches Heatprofil aktiviert die getestete Kombination aus gemeinsamen Hot-IDs, entfallender statischer Synchronisation, asynchronem D2H, Down-Prefetch 4, paralleler Aktivierung und der bereits typgeprueften CPU-Gate/Up-Fusion. Der Gesamtgewinn misst diese Kombination; nicht jeder einzelne Optionsbeitrag ist isoliert nachgewiesen.

Scheduler-Splitting, GPU-Shared-FFN, CPU-Chunk 128, sechs Threads und weitere MMVQ-Varianten brachten keinen ausreichenden Modellgewinn. Q8-KV aenderte die Ausgabetokens und wurde nicht uebernommen. Fehlerhafte Heat-Trainings wurden verworfen; es wurde kein daraus abgeleitetes Heatprofil installiert.

## Vorhandener Prefill-Fehler

Der unveraenderte Ausgangsbuild erzeugte beim Cafe-Prompt und bei weiteren kurzen Anfragen nur wiederholte `!` (Token 0). Diese Laeufe sind als Geschwindigkeitsnachweis ausgeschlossen. `GGML_OP_OFFLOAD_MIN_BATCH=1024` umgeht den Fehler bei UBatch 512, indem der Scheduler die CPU-Expertenoperationen nicht zusaetzlich auf CUDA auslagert. Andere GPU-Berechnungen einschliesslich Flash Attention bleiben aktiv. Der genaue fehlerhafte Kernel ist nicht nachgewiesen; dies ist ein Konfigurations-Workaround. Bei UBatch >= 1024 kann der problematische Offload wieder greifen.

Ein kurzer Kontrollprompt antwortete damit wieder sinnvoll; seine Prefill-Zeit sank von etwa 2.2 auf 0.59 Sekunden. Die beiden gueltigen A/B-Reihen verwenden den Workaround auf beiden Seiten. Die Auswertung trennt Prompt/Prefill-Referenzen und verwirft kollabierte Ausgaben, unvollstaendige Tokenfolgen sowie Diagnose-/Trainingslaeufe.

## Build, Tests und Integration

`ninja -C build-main-sm75 -j2 ggml-cpu ggml-cuda` hat den regulaeren Build aktualisiert; danach gibt es keine ausstehenden Schritte fuer diese Targets. Die gemeinsamen agollm-Bibliotheken wurden aus denselben Objekten mit `RUNPATH=$ORIGIN` verlinkt. Die vier direkt auf den Hauptbuild und die 14 auf `workers/llama` verweisenden Rezepte erhalten damit die verfuegbaren Bibliotheksverbesserungen. Nur Flint aktiviert die neuen CUDA-Schalter. Alte ungenutzte Builds sowie Pascal/Vulkan wurden nicht umgebaut.

Hauptbuild und agollm-Paket bestehen jeweils 12856 bitgleiche IQ4-Vergleiche und die zehn CUDA-Alias-/Replay-Faelle gegen die Referenz. Der CPU-MoE-Test mit mehreren Threadzahlen besteht ebenfalls. Compute Sanitizer meldet am Hauptbuild null Speicherfehler; die separat geprueften, erwarteten `cudaGraphExecUpdateFailure`-Rueckfaelle wurden bei diesem Speicherlauf von der API-Fehleranzeige ausgenommen. Die vorherigen MMVQ-Tests umfassen 149 Shapes mit Typ-, Fusions- und Restzeilenkontrollen. Ein Qwen-Regressionstest mit dem neuen agollm-Paket erzeugt dieselben 512 Token-IDs wie die alte Referenz.

`start-flint.sh` ist mit/ohne Heat und geerbten Tuningvariablen geprueft. Seine Voreinstellung bleibt 8192 Kontext; die Geschwindigkeitszahlen oben gelten fuer das agollm-Profil mit 64512 Kontext. agollm-Konfigurationspruefungen vor/nach dem atomaren Austausch bestehen. Das Flint-Rezept enthaelt die getesteten Optionen und aktualisierte Schaetzwerte von 40 Decode-/114 Prefill-Token/s. Bibliotheken, Rezept und ORIGIN sind gesichert; der Installer bietet einen geprueften Rollback. Kein Commit oder Push.

agollm laeuft wieder mit Flint. Die Anfrage nach `2 + 2` ueber den Router antwortet mit `4`. Router, LLM und Embedding melden HTTP 200. Prozessumgebung, gemappte Bibliothekspfade und SHA256 stimmen mit dem geprueften Produktionsbuild ueberein; es werden keine Benchmarkbibliotheken geladen. Die bestehenden `free_vega`-Regeln duerfen andere unbeschaeftigte Worker weiterhin pausieren.

Nachweise: [Vergleiche und Tokenpruefungen](benchmark-results/flint-opt-20261025/cpu-prefill-comparison.json), [ABBA](benchmark-results/flint-opt-20261025/final-abba.json), [ausgeschlossene Laeufe](benchmark-results/flint-opt-20261025/model-results-rejected.json), [Produktions-Modelltests](benchmark-results/flint-opt-20261025/final/model-validation.json), [Operatoren und Memcheck](benchmark-results/flint-opt-20261025/final/main-operators.json), [Qwen-Regression](benchmark-results/flint-opt-20261025/final/qwen-regression.json), [Installation und Backups](benchmark-results/flint-opt-20261025/final/deployment.json), [Live-Pruefung](benchmark-results/flint-opt-20261025/final/live-validation.json).

---

# Integration in agollm und Startskripte vom 2026-10-25

**Die getesteten CUDA-Verbesserungen sind jetzt produktiv integriert.** `build-main-sm75` wurde mit dem regulaeren Ninja-Target `ggml-cuda` aktualisiert: nur `mmvq.cu.o`, die CUDA-Bibliothek und ihre Symlinks wurden neu erzeugt. Ein erneuter Ninja-Aufruf meldet keinen ausstehenden Build. agollm verwendet den aktualisierten Hauptbuild direkt fuer Dory1, Flint, Nemotron Bolt und Spark 2.6; die anderen 14 LLM-Rezepte verwenden die ebenfalls aktualisierte Bibliothek unter `workers/llama`.

Die agollm-Bibliothek wurde aus denselben regulaeren Objekten mit `RUNPATH=$ORIGIN` verlinkt und als echte Datei installiert. Sie hat keine Abhaengigkeit von den Benchmark-Bibliotheken. Die bisherigen Dateien sind gesichert, `workers/llama/ORIGIN.txt` dokumentiert die Aktualisierung. Im Qwen-Rezept stehen die beiden neuen `ROWS=4`-Werte ausdruecklich; bei den anderen Rezepten greifen die neuen Bibliotheksdefaults. Der unsichere CPU-Kandidat wurde nicht installiert.

14 Turing-Startskripte, die drei Presets von `router1660.sh` und das TurboLLM-Profil `gtx1660.env` enthalten die neuen Schalter inklusive der jeweils vorhandenen Export-, Validierungs- und Anzeigelogik. `start-flint.sh` bereinigt die neuen Variablen wie die vorhandenen Overrides. Weitere Starter ohne expliziten Tuningblock verwenden denselben aktualisierten Hauptbuild. Insgesamt wurden 17 Skript-/Profildateien angepasst. `start.sh`, `start1080.sh` und Ling1080 verwenden den Pascal-Pfad und bleiben unveraendert. Nicht produktiv referenzierte alte Buildvarianten sowie Vulkan-/CPU-Builds wurden nicht mit einer fremden CUDA-Bibliothek vermischt.

Pruefungen nach dem regulaeren Neubau und der Installation:

- Hauptbuild und vorbereitetes agollm-Paket: jeweils 76 Operatorfaelle bitgleich zur Referenz; 64 weitere Compute-Sanitizer-Faelle ohne Fehler. Ein Qwen-Test mit dem Installationspaket erzeugte dieselben 512 Token-IDs wie die Referenz.
- 32 Shell-Dateien bestehen `bash -n`; simulierte Starts pruefen Turing `4/4`, getrennte Pascal-Einstellungen und Flints Umgebungsbereinigung. `git diff --check` ist sauber.
- agollm-Konfigurationspruefung vor/nach der Installation erfolgreich. Das Audit von 153 Dateien bestaetigt ausschliesslich die erwarteten CUDA-/Qwen-Aenderungen; alle 18 Assets und alle anderen LLM-Laufzeitbibliotheken bleiben unveraendert. CUDA-SONAME und Abhaengigkeiten bleiben kompatibel.
- Live-Anfragen ueber agollm: Qwen antwortet mit `1 2 3 4 5 6 7 8 9 10`; beide neuen MMVQ-Pfade melden vier Zeilen im Laufzeitprotokoll. Danach antwortet das Standardmodell Ling mit `OK`. Die geladenen CUDA-Dateien stimmen nach Pfad, Inode und SHA256 mit der Installation ueberein; kein Benchmarkverzeichnis wird geladen.

agollm ist wieder gestartet und verwendet zum Abschluss das normale Standardmodell `ling-tiny`. Router, LLM und Embedding melden HTTP 200. Qwens bestehende `free_vega`-Regel hat die unbeschaeftigten ASR-/Decision-/TTS-/Vision-Worker planmaessig pausiert; sie starten beim naechsten entsprechenden Auftrag wieder. Backups, Build-/Installationsmanifest, Skriptpatches und Validierung: [mmvq-integration-20261025](benchmark-results/mmvq-integration-20261025/), insbesondere [deployment.json](benchmark-results/mmvq-integration-20261025/deployment.json), [agollm-integration-audit.json](benchmark-results/mmvq-integration-20261025/agollm-integration-audit.json), [live-validation.json](benchmark-results/mmvq-integration-20261025/live-validation.json) und [final-runtime.json](benchmark-results/mmvq-integration-20261025/final-runtime.json). Kein Commit oder Push.

---

# Kandidatenvergleich und CUDA-Verbesserung vom 2026-10-25

**Ergebnis:** Zwei CUDA-MMVQ-Pfade wurden implementiert, getestet und in den Quellcode uebernommen. Vier Ausgabezeilen pro Block steigern den Durchsatz der betroffenen Operatoren um rund 9-23%. Im kontrollierten Qwen-Modellvergleich steigt Decode von 45.539 auf 46.085 Tokens/s, also um 0.546 Tokens/s beziehungsweise 1.20%. Der CPU-Vierzeilenversuch bleibt separat: Sein positiver Nex-Modellwert ist wegen gegenlaeufiger Mikrotests und unterschiedlicher Temperaturen noch nicht belastbar.

Artefakte, Referenz-/Kandidatenquellen, Buildkommandos, Rohdaten und Auswertungsskripte liegen lokal unter [benchmark-results/perf-candidates-20261025](benchmark-results/perf-candidates-20261025/). Das Verzeichnis ist ignoriert. Dauerhaft geaendert wurden in dieser Fortsetzung nur `ggml/src/ggml-cuda/mmvq.cu` und diese Dokumentation; vorhandene fremde Aenderungen bleiben erhalten. Der getestete finale Build liegt in `final-bin`. Regulaere Build- und agollm-Binaries wurden nicht ersetzt.

## Plan und Auswahlkriterien

1. Q8_0 mit fuenf Spalten und Q6_K mit vier Spalten auf realen Qwen-/DFlash-Matrixgroessen pruefen. Eine, zwei und vier Ausgabezeilen pro Block vergleichen; Warpzahl, Quantisierung und Reduktionsreihenfolge unveraendert lassen.
2. Im CPU-Kaltpfad vorhandene Q4_K-/Q6_K-Vierzeilenroutinen fuer die Down-Projektion bei genau einer Token-Spalte pro Experte ausprobieren. Chunkgrenzen, Prefetch-Adressen, Akkumulation und skalare Restzeilen erhalten.
3. Zuerst Referenzwerte, bitweise Ausgaben und Randfaelle pruefen. Danach residente Operatoren mit Warmup und wechselnder Reihenfolge messen, anschliessend vollstaendige Modelle in A/B/B/A-Reihenfolge.
4. Nur nachvollziehbare Verbesserungen uebernehmen. Kandidaten mit widerspruechlichen Messungen separat lassen; der finale Neubuild muss mit den neuen Defaults und den Rueckfallschaltern erneut bestehen.

## Uebernommene CUDA-Aenderung

Die vorhandenen MMVQ-Templates verarbeiten nun im Turing-Pfad standardmaessig vier Ausgabezeilen pro Block fuer dichtes Q8_0/ncols=5 und Q6_K/ncols=4. Q8 behaelt zwei Warps, Q6 vier Warps. IDs werden ausgeschlossen; der Q6-Pfad schliesst Fusion aus, das bestehende Fusion-Verhalten von Q8 bleibt erhalten. Bei einer Zeilenzahl, die nicht durch vier teilbar ist, bleibt der bisherige Tabellenpfad aktiv. Andere Typ-/Spaltenkombinationen bleiben unveraendert.

Die neuen Schalter `GGML_CUDA_MMVQ_Q8_NCOLS5_ROWS` und `GGML_CUDA_MMVQ_Q6_K_NCOLS4_ROWS` haben jeweils Default `4`. Mit `0` wird der bisherige Tabellenpfad mit zwei Zeilen verwendet; `1` ist fuer Vergleichstests verfuegbar. Die gemessene Einzeilenvariante ist auf dieser GPU langsamer. Die A/B-Kandidaten aktivierten `4` ausdruecklich; der finale Build wurde danach ohne diese beiden Umgebungsvariablen geprueft. Patch gegen den Ausgangszustand: [mmvq-final.patch](benchmark-results/perf-candidates-20261025/mmvq-final.patch).

## CUDA-Operatorbenchmarks

GTX 1660 Ti Mobile, SM75, identische Release-/CUDA-Compileoptionen und uebrige Objekte. Acht Prozesse: Referenz, eine Zeile, vier Zeilen, deaktivierter Kandidat, danach umgekehrte Reihenfolge. Pro Shape 500 ms Warmup und sieben Stichproben mit je 500 Berechnungen. CUDA-Graphen waren fuer diese Messung abgeschaltet. Gewichtserzeugung, Upload und Ausgabekopien liegen ausserhalb der Messzeit. Gemessen wird der residente Backend-Graph einschliesslich Aktivierungsquantisierung und Aufrufaufwand, keine isolierte CUDA-Kernelzeit.

Werte sind Mediane der beiden Prozessmediane; K ist die Eingabedimension, M die Ausgabezeilenzahl.

| Typ / Spalten | K | M | Referenz us | Vier Zeilen us | Durchsatzgewinn |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 / 5 | 2048 | 512 | 9.829 | 9.027 | +8.9% |
| Q8_0 / 5 | 2048 | 4096 | 61.552 | 50.759 | +21.3% |
| Q8_0 / 5 | 2048 | 8192 | 119.560 | 97.199 | +23.0% |
| Q8_0 / 5 | 512 | 2048 | 12.343 | 11.010 | +12.1% |
| Q8_0 / 5 | 4096 | 2048 | 59.203 | 48.321 | +22.5% |
| Q6_K / 4 | 2048 | 248320 | 3446.130 | 2926.989 | +17.7% |
| Q6_K / 4 | 2048 | 1024 | 17.691 | 14.577 | +21.4% |
| Q6_K / 4 | 6144 | 2048 | 78.740 | 67.563 | +16.5% |

Auf allen acht Zielgroessen ist auch der langsamere Vierzeilenprozess schneller als der schnellere Prozess des deaktivierten Kandidaten. Eine Zeile verliert auf allen Zielgroessen. Die unveraenderten Q8-Kontrollen mit drei/sechs Spalten liegen gegen den deaktivierten Kandidaten innerhalb 0.5%. Die Q6-Kontrollen schwanken staerker; der unveraenderte Sechsspaltenpfad zeigt sogar rund 7.7% mehr Laufzeit. Kleine Operatorunterschiede sind deshalb nicht als belastbare Codeeffekte zu werten. Zwei Prozesse pro Variante liefern keine enge statistische Absicherung. Einzelwerte und Kontrollen: [cuda-perf-analysis.csv](benchmark-results/perf-candidates-20261025/cuda-perf-analysis.csv).

## Modellvergleich

Je Modell vier neue Serverprozesse in A/B/B/A-Reihenfolge, zwei Warmups mit je 256 Tokens und drei Messanfragen pro Prozess. Gleicher lokaler Go-Prompt, Chatvorlage, Greedy und Seed 42. Qwen: 507 Prompt-/512 Ausgabetokens, S20/W8, `LLAMA_DFLASH_ADAPTIVE=0` fuer eine feste Draft-Obergrenze. Nex: 509 Prompt-/256 Ausgabetokens, unveraendertes Rezept. Nur die jeweils untersuchte CPU- oder CUDA-Bibliothek wurde ausgetauscht; geladene Bibliotheken und Hashes sind dokumentiert.

| Versuch | Referenz-Prozessmediane TPS | Kandidaten-Prozessmediane TPS | Medianvergleich | Entscheidung |
| --- | --- | --- | --- | --- |
| Qwen, CUDA vier Zeilen | 45.581 / 45.497 | 46.099 / 46.072 | 45.539 -> 46.085, +1.20% | CUDA-Aenderung uebernommen |
| Nex, CPU Down vier Zeilen | 40.308 / 40.751 | 41.617 / 41.548 | 40.529 -> 41.582, nominell +2.60% | Noch nicht uebernommen |

Alle 24 Messanfragen erzeugten die vollstaendige angeforderte Tokenzahl. Token-IDs und Text-Hashes stimmen fuer denselben Anfrageindex ueber alle vier Prozesse exakt ueberein. Qwen erzeugt zwischen Anfrageindex 1/2/3 unterschiedliche Tokenfolgen; verglichen werden deshalb nur entsprechende Indizes. Qwen hatte waehrend der Messanfragen keine physischen Reads. CPU-Temperaturen lagen dort als Stichproben bei 90.25-96 C, GPU bei 59-66 C. Die Aussage bezieht sich auf diesen Prompt und die feste Draft-Obergrenze, nicht auf beliebige Modelle, Prompts oder adaptive DFlash-Entscheidungen.

Der finale CUDA-Build wurde mit unveraenderter fester Draft-Obergrenze, aber ohne die beiden neuen Zeilenschalter erneut mit drei Qwen-Anfragen validiert. Alle drei Tokenfolgen stimmen mit der Referenz ueberein. Seine 46.262 / 46.954 / 47.967 Tokens/s sind ein separater Validierungslauf und werden nicht in den A/B/B/A-Gewinn eingerechnet. Details: [model-analysis.json](benchmark-results/perf-candidates-20261025/model-analysis.json) und [final-validation.json](benchmark-results/perf-candidates-20261025/final-validation.json).

## CPU-Kandidat bleibt experimentell

Der CPU-Versuch nutzt vorhandene Q4_K-/Q6_K-Vierzeilenroutinen nur fuer `cne1 == 1` auf AVX2. Die Mikrobenchmarks messen den vollstaendigen MOE_COLD-Graph mit persistentem Acht-Thread-Pool, Shape 2048/512/2048 und 256 Experten. Der wechselnde Expertenpool umfasst 432 beziehungsweise 498 MiB; vier/acht aktive Experten werden rotiert. Ein separater Wiederholungsmodus haelt die IDs fest, garantiert aber keine L3-Residenz. Je Fall A/B/B/A, sieben Stichproben mit 256 Graphberechnungen pro Prozess; Vorbereitung ausserhalb des Timers.

| Down-Typ | Aktive Experten | Wechselnde Experten: Durchsatzaenderung | Wiederholte Experten: Durchsatzaenderung |
| --- | ---: | ---: | ---: |
| Q4_K | 4 | +1.36% | -1.57% |
| Q4_K | 8 | -1.01% | -2.16% |
| Q6_K | 4 | -2.94% | -3.73% |
| Q6_K | 8 | -0.74% | -4.41% |

Die Ausgaben und Routingzaehler sind in allen 32 Prozessen identisch. Der positive Nex-Modellwert passt nicht zu diesen ueberwiegend negativen Operatorwerten. Zusaetzlich hatte die erste Nex-Referenz 446464 physisch gelesene Bytes, und die letzte Referenz war mit 93.25-95 C deutlich waermer als die Kandidaten mit etwa 87-91 C. Die letzte Referenz schwankte zwischen 39.697 und 41.751 Tokens/s. Daraus wird kein gesicherter CPU-Speedup abgeleitet. Vor einer Uebernahme waere eine erneute Modellmessung mit vergleichbaren CPU-Temperaturen/Takten und anschliessender Phasenmessung erforderlich. Patch und Rohdaten bleiben unter `ggml-cpu-downrows4.patch` und [cpu-perf-analysis.csv](benchmark-results/perf-candidates-20261025/cpu-perf-analysis.csv) erhalten; die Produkt-CPU-Quelle ist unveraendert.

## Korrektheit und Abschluss

- CUDA: 76 Faelle je anfaenglicher Variante, darunter reale Shapes, Zeilenreste, gepaddete Eingaben, Bias und unveraenderte Kontrollpfade. Alle 228 Vergleiche gegen die Referenz sind bitgleich. Zusaetzlich Stichproben gegen CPU-dequantisierte Skalarprodukte; maximale NMSE 0.000393565 bei Grenze 0.0005. Alle 84 Vergleiche der Performance-Ausgaben sind ebenfalls bitgleich.
- Compute Sanitizer: je 64 Randfaelle fuer Referenz, eine Zeile und vier Zeilen ohne Speicherfehler. Die lokale Debian-Installation benoetigt einen gemeinsamen Suchpfad fuer Libraries und TreeLauncher-Hilfsprogramme; lokale Symlinks beheben den Startfehler ohne Systemaenderung.
- CPU: 140 Faelle bei jeweils 1/2/4/8 Threads pro Variante, zusammen 1120 Graphlaeufe. Q4/Q5/Q6, 1/3/9 Tokens, kalte/gemischte/heisse Auswahl, Optimierungsschalter, Prefetch und Restzeilen; Ausgaben bitgleich.
- Finaler CUDA-Neubuild: 76 Faelle mit Defaults, 76 mit aktivierten CUDA-Graphen und 76 mit Rueckfallschaltern, jeweils bitgleich. Weitere 64 Sanitizer-Faelle ohne Fehler und drei vollstaendige tokenidentische Qwen-Anfragen. Keine Compilerwarnungen; `git diff --check` sauber.

Alle eigenen Benchmarkserver sind beendet. agollm bleibt gestoppt; die Abschlusskontrolle zeigt 1 MiB GPU-Belegung, 5748 MiB frei, 0% Auslastung und keine CUDA-Prozesse. Kein Commit, Push oder Deployment. Der Quellcode nutzt die neuen Defaults beim naechsten Build; der gepruefte separate Build ist bereits in `benchmark-results/perf-candidates-20261025/final-bin` verfuegbar.

---

# Optimierungsrunde vom 2026-10-25

**Ergebnis:** Kein belastbarer allgemeiner Modell-Speedup nachgewiesen. Ein kleiner CPU-Fix beschleunigt den leeren Expertenpfad, verbessert Nex aber nicht und wurde zurueckgenommen. Wiederverwendete Threadpools zeigen eine kleine positive Tendenz, die bei der beobachteten Streuung noch keine Produktionsaenderung rechtfertigt. Der aktuelle Trace grenzt zwei bisher nicht abgestimmte CUDA-MMVQ-Pfade als naechste konkrete Kandidaten ein.

Die Experimente und eingefrorenen Binaries liegen lokal unter [benchmark-results/perf-review-20261025](benchmark-results/perf-review-20261025/). Dieses Verzeichnis ist ignoriert und wird nicht automatisch mit dieser Dokumentation versioniert. Bestehende fremde Arbeitsbaumaenderungen wurden erhalten. Kein Commit, Push oder Deployment; die regulaeren Build- und agollm-Binaries wurden nicht ersetzt.

## Messgrundlage dieser Runde

- Ryzen 7 4800H, 8 Kerne/16 Threads, GTX 1660 Ti Mobile mit 6 GiB, Treiber 595.84. agollm wurde vor den Messungen per systemd gestoppt; seine Argumente, relevanten Umgebungsvariablen und geladenen Bibliotheken sind in `agollm-before.json` gesichert.
- Ausgangspunkt sind die aktuellen lokalen Binaries aus `build-main-sm75/bin` und die agollm-Rezepte. Laufzeitdateien wurden nach `frozen-bin` kopiert und gehasht. CPU-Referenz und Kandidat verwenden identische Compile-/Linkoptionen und identische uebrige Objekte; der Referenzquelltext von `ggml-cpu.c` entsprach vor der Runde HEAD. Geladene Bibliotheken werden pro Modelllauf dokumentiert.
- Vorhandener Rezept-Runner aus der Oktober-Runde, gleicher lokaler Go-Prompt mit jeweiliger Chatvorlage. Greedy, Seed 42, kein Promptcache-Reuse. In den normalen Serien zwei Warmups und drei Messanfragen pro Prozess. Nex/Ling jeweils 256 generierte Tokens, Qwen 512. Warmup: Nex/Qwen 256, Ling 128 Tokens.
- Nex: 509 Prompt-Tokens, S24/W0, acht Threads, keine Spekulation. Ling: 575 Prompt-Tokens, bestehendes GPU-Rezept. Qwen: 507 Prompt-Tokens, S20/W8, DFlash; fuer die Kontrollmessung `LLAMA_DFLASH_ADAPTIVE=0`. Beim Qwen-Kontrolllauf war zusaetzlich der JSON-Statistikexport angefordert; W8 erforderte bereits die zaehlerbasierte Tier-Aktualisierung. Phasen-Timing war dabei aus.
- Temperatur-/Taktsnapshots und physische Prozess-Reads sind gespeichert. CPU-Temperatur und Takte sind nicht fest geregelt. Einzelmessungen ueberlappen; wenige aufeinanderfolgende Wiederholungen sind kein statistisch abgesicherter Leistungsnachweis.

## Gemessener Modell-Durchsatz

Alle Werte sind Decode-Tokens/s aus den Server-Timings, ohne Promptverarbeitung. Die Zeilen nennen jeweils den Median eines Prozesses.

| Modell / Versuch | Median Tokens/s | Einzelwerte Tokens/s | Bewertung |
| --- | ---: | --- | --- |
| Ling, bestehendes Rezept | 133.506 | 133.506 / 133.568 / 132.936 | Referenz |
| Ling, ngram-Spekulation aus | 133.176 | 133.176 / 133.191 / 133.098 | Kein Vorteil; identische Tokenfolge |
| Nex, Referenz A | 41.865 | 41.003 / 41.877 / 41.865 | Vor dem CPU-Fix |
| Nex, leerer CPU-Pfad mit Fruehausstieg | 40.188 | 40.167 / 40.188 / 41.954 | Kein Modellgewinn |
| Nex, Referenz B | 41.034 | 41.034 / 41.032 / 41.811 | Rueckvergleich zum CPU-Fix |
| Nex, OpenMP auf physische Kernplaetze gebunden | 41.047 | 41.092 / 39.520 / 41.047 | Kein belegter Gewinn |
| Nex, persistenter Pool mit OpenMP | 42.103 | 42.103 / 41.918 / 42.131 | Kleine positive Tendenz |
| Nex, persistenter Pool mit nativem CPU-Executor | 41.993 | 41.993 / 42.142 / 39.821 | Kein klarer Vorteil gegen OpenMP |
| Nex, Referenz C | 41.182 | 41.996 / 41.182 / 40.576 | Rueckvergleich nach den Pool-Versuchen |
| Qwen, feste DFlash-Obergrenze | 44.802 | 44.585 / 44.802 / 46.108 | Kontrollmessung, kein A/B-Codegewinn |

Alle 21 vollstaendigen Nex-Messanfragen dieser Vergleichsserien erzeugten exakt dieselben 256 Token-IDs. Auch Ling ist mit/ohne Spekulation tokenidentisch. Die Qwen-Anfragen erzeugen untereinander unterschiedliche Tokenfolgen trotz identischem Sampling und deaktivierter zeitbasierter Draft-Anpassung; daraus folgt keine tokenkontrollierte Vergleichsgrundlage fuer einen spaeteren Kandidaten.

Die Referenz-Promptgeschwindigkeiten lagen im Median bei rund 1046 Tokens/s fuer Ling, 152 fuer Nex und 135 fuer Qwen. Rohdaten inklusive I/O, Hashes, Temperaturen und vollstaendiger Einstellungen: [model-results.json](benchmark-results/perf-review-20261025/model-results.json) und die jeweiligen Laufverzeichnisse.

Ein erster Nex-Versuch mit angeforderten 512 Tokens stoppte wegen Wiederholungen nach 394 Tokens. Er ist als `nex-baseline-a` erhalten, aber aus dem Codevergleich ausgeschlossen. CPU-Phasenmessungen und Nsight-Laeufe sind ebenfalls keine normalen Durchsatzreferenzen.

## Kleine CPU-Aenderung getestet und verworfen

`MOE_COLD` wurde nach seiner ersten Barriere beendet, wenn kein ausgewaehlter Experte auf der CPU berechnet werden musste. Nullausgabe, Routingzaehler, Hooks und die gemeinsame Threadentscheidung waren dabei erhalten. Die ersten Gate/Up- und Down-Schleifen sowie weitere Barrieren entfielen.

Der isolierte Mikrobenchmark nutzte persistente Pools, 128 Experten, acht ausgewaehlte IDs, 1/3 Tokens und 1/8 Threads. Vier Prozesse in A/B/B/A-Reihenfolge, jeweils 200 Warmups und drei Stichproben zu 2000 Graphberechnungen je Fall. Bei acht Threads sank der leere Pfad mit einem Token von 5.367 auf 4.300 us, mit drei Tokens von 5.851 auf 3.959 us. Ueber alle leeren Faelle lag die Latenzersparnis bei etwa 20-39%. Outputs und Zaehler-Hashes stimmen in allen Faellen ueberein.

Das ist ein Spezialfall von etwa 1-2 us Ersparnis, keine entsprechende Modellbeschleunigung. Der Kontrollfall mit einem kalten Experten zeigte teilweise 2.5-3% hoehere gepoolte Mediane und schwankende Wiederholungen. Im Nex-A/B/A gab es keinen Gewinn. Deshalb sind Produktcode und Testaenderung zurueckgenommen; [allhot-experiment.patch](benchmark-results/perf-review-20261025/allhot-experiment.patch) und [allhot-summary.json](benchmark-results/perf-review-20261025/allhot-summary.json) erhalten den Versuch.

Vor dem Verwerfen bestanden 256 vollstaendige MOE-Graphpruefungen mit/ohne Read-ahead: kalte/gemischte/heisse Auswahl, nicht ausgewaehlte kalte Experten, mehrere Token-/Threadzahlen, exakte Routingzaehler, Nullausgaben und Profilzaehler. Der Mikrobenchmark ist kein direkter Ersatz fuer das Serververhalten, weil der unveraenderte Server keinen persistenten Pool anhaengt.

## CPU-Platzierung und Pool-Lebensdauer

`OMP_PLACES=cores`, `OMP_PROC_BIND=spread`, `OMP_DYNAMIC=false` verteilten die acht OpenMP-Worker tatsaechlich auf die acht Kernplaetze. Beobachtete Masken waren `0-1`, `2-3`, ..., `14-15`. Allerdings erbten viele Hilfsthreads den ersten Kernplatz. Im Modellversuch ergab sich kein Geschwindigkeitsvorteil; das Rezept wurde nicht geaendert.

Der Server ruft derzeit `llama_attach_threadpool` nicht auf. `ggml_graph_compute` erzeugt deshalb pro CPU-Graph eine temporaere Pool-Verwaltung. OpenMP selbst kann seine Worker weiterverwenden. Ein blosses Abschalten von OpenMP waere keine sinnvolle Optimierung: Ohne angehaengten Pool wuerde der native Pfad immer wieder Threads erzeugen und beenden.

Ein separater LD_PRELOAD-Test haengte ueber die bestehende API persistente Pools an und gab sie erst nach Synchronisierung und Kontextabbau frei. Getestet wurden unveraendertes OpenMP und ein isolierter nativer CPU-Graph-Executor. Letzterer kompiliert `ggml-cpu.c` und `ggml-cpu.cpp` ohne OpenMP; die uebrigen Bibliotheken bleiben unveraendert. Das ist kein vollstaendiger Build mit `GGML_OPENMP=OFF`.

Der native Executor bestand den bestehenden MOE-Test mit/ohne Read-ahead sowie `test-barrier 8 1`, einschliesslich wechselnder Workerzahlen und Graphen. Die Modell-Tokenfolgen stimmen mit der Referenz ueberein. Die kleinen TPS-Abstaende reichen angesichts der Streuung und der nur einmal gemessenen Pool-Varianten nicht fuer eine allgemeine Beschleunigungsbehauptung oder ein Deployment. Artefakte: `persistent-pools.cpp`, `native-build-commands.json`, `native-tests.json`, `variant-hashes.json`.

## CPU-Expertenphasen bei Nex

Eine separate instrumentierte Anfrage exportierte die bestehenden CPU-Phasenzaehler direkt vor dem Kontextabbau. Der normale angeforderte Stats-Export hatte beim Serverende keine Datei erzeugt; der zusaetzliche Test-Wrapper liest nur die vorhandenen Zaehler und aendert weder Pools noch Rechenpfad. Profil-TPS werden nicht als Vergleichswert verwendet.

Erfasst wurden 7760 Aufrufe, genau 194 pro Schicht, mit zusammen 1933.05 ms CPU-Kaltpfadzeit. Gate/Up beanspruchte 57.42%, Down 35.63%, Aktivierung 2.55% und die Differenz zur gemessenen Gesamtzeit 4.39%. Die Phasen enthalten ihre instrumentierungsbedingten Barrieren; die Zaehler sind kumulativ und enthalten auch Warmups. Daraus folgt keine exakte Zeit pro Nutzertoken.

Damit entfallen rund 93% dieser CPU-Zeit auf die beiden Matrixproduktphasen. Weitere reine Aktivierungs- oder Verwaltungsoptimierungen haben hier einen begrenzten Hebel. Ein aktueller Bandbreiten- oder Cache-Miss-Nachweis ist damit noch nicht erbracht; fuer Gate/Up/Down sind Gewichtszugriffe, bereits vorhandene Mehrzeilenpfade und sichere Ueberlappung die relevanteren Fragen. Daten: [nex-cpu-phases-summary.json](benchmark-results/perf-review-20261025/nex-cpu-phases-summary.json), `nex-cpu-phases.json`, `cpu-profile-export.cpp`.

## Neue Hotspots aus dem Qwen-Trace

Nsight 2022.4 lieferte CUDA-, Copy-, Runtime- und NVTX-Ereignisse. Der Profiler-Wrapper endete beim Aufraeumen mit Status 139; der separate Import gelang. Die SQLite-Datenbank ist konsistent, aber CUDA-Ereignisse enden rund 663 ms vor dem letzten Decode-NVTX-Bereich. Schritt 31 ist unvollstaendig, die letzten sechs Schritte fehlen. Diese Zeit wurde ausgeschlossen und nicht als GPU-Leerlauf gewertet.

Ausgewertet wurden die ersten 30 vollstaendig erfassten Draft-/Verify-Schritte der letzten Anfrage, einschliesslich der Zwischenraeume. Alle betrachteten Kernel wurden ueber ihre Launch-Korrelation zugeordnet; in diesem Fenster blieb ihre Ausfuehrung innerhalb der zugeordneten Host-Bereiche.

| Messgroesse | Wert |
| --- | ---: |
| Zusammenhaengendes Fenster | 2852.83 ms |
| GPU-Kernelzeit | 927.93 ms |
| GPU-Aktivitaet inklusive Transfers | 1037.41 ms, 36.36% |
| Zeit ohne GPU- oder CUDA-API-Aktivitaet | 1471.69 ms |
| Kernelstarts / Host-API-Zeit | 75225 / 435.34 ms |
| Stream-Synchronisationen / API-Zeit | 8574 / 234.85 ms |
| H2D / D2H | 584.53 MB / 47.48 MB |

API-Zeit und GPU-Arbeit ueberlappen und duerfen nicht addiert werden. Ohne CPU-Sampling ist nicht geklaert, wie viel der Luecken notwendige CPU-Berechnung, Scheduling oder vermeidbares Warten ist.

Die konkretesten noch offenen Kernelversuche sind:

1. **Q8_0-MMVQ mit fuenf Spalten:** 6250 Verify-Kernel mit zusammen 217.50 ms. SM75 nutzt hier zwei Warps und zwei Ausgabezeilen pro Block, 78 Register pro Thread. Die bestehenden Q8-Umgebungsvariablen greifen nur bei einer, zwei oder drei Spalten.
2. **Q6_K-MMVQ mit vier Spalten im Draft:** 30 Kernel mit zusammen 75.08 ms, rund 2.50 ms je Aufruf. Vier Warps, zwei Ausgabezeilen pro Block, 80 Register pro Thread. Die grosse Ausgabedimension passt zur Vokabularprojektion, der Trace beweist den Tensornamen jedoch nicht. Die vorhandenen Q6-Schalter greifen nur bei einer/drei Spalten.

Beide Pfade melden keine lokalen Register-Spills. Ein enger naechster Versuch waere, das bereits vorhandene Template `rows_per_block_override` fuer genau diese Typ-/Spaltenkombinationen auf SM75 mit einer und vier Zeilen zu messen. Warpzahl und Reduktionsreihenfolge sollten dabei unveraendert bleiben. Erst Operatorvergleich und reale Shapes, dann tokenkontrollierte Modellmessungen. Diese Varianten wurden in dieser Runde noch nicht implementiert oder gemessen; ein Speedup ist nicht nachgewiesen.

Die beiden Pfade machen zusammen etwa 31.5% der Kernelzeit des ausgewerteten Fensters aus, nicht 31.5% der Modelllaufzeit. Dispatchstellen: `ggml/src/ggml-cuda/mmvq.cu`, Faelle vier/fuenf Spalten um Zeile 1329/1337; Standardwerte in `calc_nwarps` und `calc_rows_per_block`. Einzelereignisse, Zeitfenster und Quellcode-Zuordnung: [trace-summary.json](benchmark-results/perf-review-20261025/qwen-nsys/trace-summary.json).

Graph-Neubauten bleiben messbar: Im unprofilierten Qwen-Kontrolllauf 76-80 Neubauten pro Anfrage bei 3.91-4.06 ms je Neubau. Das ergibt rund 305-325 ms beziehungsweise 2.72-2.84% der Decode-Zeit. Selbst vollstaendiges Entfernen dieser gemessenen Kosten waere hier nur ein kleiner einstelliger Prozenthebel; ein Graphcache muss zusaetzlich Scheduler-Allokationen, Eingaben und DFlash-Zustand korrekt erhalten. Beleg: [qwen-graph-cost.json](benchmark-results/perf-review-20261025/qwen-graph-cost.json).

Abschlusszustand: alle eigenen Benchmarkserver beendet, agollm weiterhin gestoppt, GPU wieder frei. Die letzte Kontrolle zeigte 1 MiB belegten GPU-Speicher und 0% Auslastung. Nur diese Dokumentation wurde als dauerhafte Aenderung aus der Runde behalten; die Versuche liegen separat im Ergebnisverzeichnis.

---

# Optimierungsrunde vom 2026-10-06

**Modellwechsel am 2026-10-07 optimiert:** agollm haelt den vorhandenen Leistungsmodus jetzt auch waehrend Entladen und Laden. Der kleine Router-Fix ist getestet und aktiv. 24 Wechselmessungen zeigen einen nutzbaren Vorteil, aber bei grossen Modellen weiterhin starken Dateicache-Einfluss. Messwerte und Grenzen stehen im letzten Abschnitt.

**Deployment am 2026-10-07 abgeschlossen:** Die geprueften CPU-/CUDA-Fixes sind jetzt auch im laufenden agollm-Worker aktiv. Fuenf Laufzeitdateien wurden aktualisiert, der Dienst neu gestartet und die geladenen Dateien gegen den getesteten Kandidaten verifiziert. Health-Checks und eine Testantwort ueber die agollm-API waren erfolgreich. Details, Belege und Backup stehen im Abschnitt zum Deployment nach agollm. Lookahead bleibt ausgeschlossen.

**Stand nach der Fortsetzung vom 2026-10-07:** Die offenen Messungen sind abgeschlossen. Weiterhin kein belastbarer Gewinn von 3-5 Modell-TPS. Der warme Kolibri-Vergleich zeigt nur +0.4 bis +0.6 TPS im Median bei ueberlappenden Einzelwerten, Nex praktisch Gleichstand, Qwen mit festen Draft-Caps und identischen Anfragepaaren +0.04 TPS. Top-K hat messbare, aber Shape-abhaengige Mikrogewinne und kleine 512-Experten-Decode-Regressionen. Details und Grenzen stehen im Abschnitt zur Fortsetzung unten. `agollm` laeuft wieder.

Ziel: weitere 3-5 Tokens/s bei den groesseren vorhandenen Modellen durch Codeaenderungen, bessere Inferenzpfade und Kerneloptimierungen. Die bereits optimierten Einstellungen aus `../agollm` waren die Referenz. Nach der Praezisierung des Auftrags wurden keine weiteren Parametersuchen verfolgt.

**Ergebnis: Ein belastbarer Gewinn von 3-5 Tokens/s wurde nicht erreicht.** Kleine Codeverbesserungen und Korrekturen sind im Arbeitsbaum erhalten. Das groessere CPU/GPU-Ueberlappungsexperiment bleibt separat, weil der Nutzen nicht ausreichend belegt ist. Die Runde wurde auf Wunsch des Benutzers beendet; es laufen keine eigenen Benchmarkprozesse mehr. `agollm` wurde wieder gestartet und meldete seine Worker bereit.

## Messgrundlage und Grenzen

- Hardware: Ryzen 7 4800H, 8 Kerne/16 Threads, rund 30 GiB nutzbarer RAM; GTX 1660 Ti Mobile, SM75, 6 GiB VRAM.
- Referenz: bestehender optimierter Release-Build `build-main-sm75/bin`, Git-Ausgangspunkt `9ca564aab`. Dieser Build wurde nicht ueberschrieben.
- Kandidaten: separate Bibliotheken aus den geaenderten Uebersetzungseinheiten, mit den Compile- und Linkoptionen des Referenz-Builds. Andere Objekte wurden wiederverwendet. Kein vollstaendiger Neubau aller Plattformen.
- Modellstart mit den jeweiligen agollm-Rezepten. Zusaetzliche Flags nur fuer ausdruecklich bezeichnete Codeexperimente; waehrend der Messrunden keine dauerhaften Aenderungen an agollm-Konfiguration oder produktiven Binaries. Das anschliessende Deployment ist unten separat dokumentiert.
- Messanfragen: greedy, Temperatur 0, Top-k 1, Seed 42, `ignore_eos=true`. Drei Messlaeufe nach Warmup. Die Tabelle nennt Decode-Tokens/s aus den Server-Timings, nicht den Gesamtdurchsatz einschliesslich Promptverarbeitung.
- Qwen und Nex: derselbe lokale Code-Prompt, 498 Prompt-Tokens, jeweils 512 generierte Tokens. Warmup: Qwen 256, Nex 512 Tokens. Kolibri: Prosa-Prompt mit der Chatvorlage seines Profils, 211 Prompt-Tokens, 256 generierte Tokens und 256 Warmup-Tokens.
- Token-IDs und deren Hashes sowie physische Prozess-I/O wurden gespeichert. Drei aufeinanderfolgende Laeufe ersetzen keinen wiederholten Vergleich in wechselnder Reihenfolge mit kontrollierten Temperaturen, Takten und Dateicache-Zustaenden.
- Bei Qwen waehlt DFlash seine Draft-Laenge zeitabhaengig. Unterschiedliche adaptive Entscheidungen gingen teilweise mit unterschiedlichen Tokenfolgen einher. Solche Vergleiche isolieren den Effekt einer Codeaenderung nicht sauber.

Die Rohdaten liegen unter `benchmark-results/perf-review-20261006/`. Dieser Verzeichnisbaum ist ignoriert und lokal; er wird durch einen Commit von `optimize.md` nicht automatisch mit versioniert.

## Durchsatz der abgeschlossenen Modelllaeufe

| Modell / Rezept | Codevariante / Laufname | Median TPS | Einzelwerte TPS | Bewertung |
| --- | --- | ---: | --- | --- |
| Qwen3.6-35B-A3B hybrid | Referenz, `qwen-code-control` | 48.00 | 45.94 / 48.00 / 49.30 | Ausgangspunkt |
| Qwen | CPU-Pfad + Top-K, `qwen-cpu-topk` | 48.49 | 47.81 / 48.55 / 48.49 | +0.49; korrespondierende Tokenfolgen identisch, kleiner Effekt nicht abgesichert |
| Qwen | Zusaetzliche ADD-Fusion + CPU-Endbarriere, `qwen-kernel-all` | 47.81 | 47.81 / 47.35 / 48.46 | Kein Vorteil; korrespondierende Tokenfolgen identisch |
| Qwen | CPU + Top-K + CUDA-Typcache, `qwen-code-final` | 47.81 | 47.81 / 43.81 / 49.30 | Kein belegter Gewinn; adaptive Draft-Pfade und Tokenfolgen abweichend |
| Qwen | Experimentelle Ueberlappung, `qwen-lookahead-fixed` | 50.24 | 50.13 / 50.24 / 50.83 | Beobachtete +2.24; nur erster Lauf mit identischer Tokenfolge, nicht als gesicherter Gewinn uebernommen |
| Nex N2.5 Mini i1 | Referenz, `nex-i1-code-control` | 35.72 | 35.78 / 35.72 / 35.12 | Ohne spekulatives Decoding |
| Nex | Code inkl. CPU-Quellsynchronisierung, `nex-i1-code-final` | 35.26 | 35.45 / 35.26 / 34.98 | -0.46; alle Tokenfolgen identisch, kein Geschwindigkeitsnachweis |
| Nex | Experimentelle Ueberlappung, `nex-i1-lookahead` | 35.89 | 35.89 / 36.71 / 35.58 | +0.17 gegen Referenz; alle Tokenfolgen identisch, praktisch kein nachgewiesener Nutzen |
| Kolibri-1, grosses Q4_K_M-Modell | Referenz mit Chatvorlage, `kolibri-chat-control` | 26.61 | 19.50 / 26.61 / 27.68 | Dateicache waermt sich noch stark auf |
| Kolibri-1 | Abschliessender Codestand, `kolibri-chat-final` | 27.19 | 26.14 / 27.22 / 27.19 | Identische Tokenfolgen; +0.58 im Median nicht dem Code zurechenbar |

Wichtig zur Versionszuordnung: `qwen-code-final` entstand vor der spaeteren CPU-Quellsynchronisierung im CUDA-Kopierpfad. Der Nex-Kandidat enthaelt diese Korrektur bereits. Beide entstanden vor der abschliessenden atomaren Initialisierung der Read-ahead-Seitengroesse. Nur `kolibri-chat-final` nutzt den kompletten abschliessenden Bibliotheksstand. Die Bezeichnung `final` in aelteren Laufnamen bedeutet deshalb nicht, dass alle spaeteren Aenderungen enthalten sind.

Kolibri ist besonders durch Speicherresidenz beeinflusst: Die Referenz las in den drei Messanfragen zusammen 5.83 GB physisch, der Kandidat nur 0.234 GB. Bereits der letzte Referenzlauf erreichte 27.68 TPS bei fast keinem physischen Nachladen. Der geplante Rueckvergleich `kolibri-chat-control-warm` wurde beim Beenden der Runde vor dem ersten gespeicherten Messlauf gestoppt. Damit fehlt die warme Gegenprobe. Ein frueherer Lauf ohne Chatvorlage, `kolibri-code-control`, brach wegen Wiederholungen schon nach 26 Tokens ab und ist kein brauchbarer Durchsatzvergleich.

Maschinenlesbare Auswertung: [final-code-results.json](benchmark-results/perf-review-20261006/final-code-results.json) und [final-code-results.csv](benchmark-results/perf-review-20261006/final-code-results.csv). Fuer den abgebrochenen Rueckvergleich gilt zusaetzlich dessen [aborted.json](benchmark-results/perf-review-20261006/kolibri-chat-control-warm/aborted.json).

## Erhaltene Codeaenderungen und Erfolge

### CPU-MoE-Pfad

Datei: [ggml/src/ggml-cpu/ggml-cpu.c](ggml/src/ggml-cpu/ggml-cpu.c).

1. Die Aktivierungsquantisierung in `ggml_compute_forward_moe_cold` verteilt Tokenzeilen jetzt mit `t += nth`. Zuvor startete jeder Thread bei `ith`, lief aber mit `t++` weiter. Bei mehreren Tokens wurden dieselben Zeilen mehrfach und nebenlaeufig beschrieben. Die neue Aufteilung vermeidet redundante Arbeit und diese ueberlappenden Schreibzugriffe; die notwendige Produzentenbarriere bleibt erhalten.
2. Die zusaetzliche Barriere nach dem Experten-Read-ahead wird in `MUL_MAT_ID`, `MUL_MAT_ID_COLD` und `MOE_COLD` nur ausgefuehrt, wenn Read-ahead aktiv ist. Bei deaktiviertem Read-ahead entfielen bisher zwar die Seitenzugriffe, nicht aber die zweite Barriere.
3. Die gecachte Read-ahead-Aktivierung und die lazy initialisierte Seitengroesse verwenden atomare Werte. Das beseitigt weitere Datenrennen bei der Initialisierung. Die Konfiguration wird weiterhin als prozessweit konstant behandelt.

Das ist eine kleine, nachvollziehbare Verbesserung des vorhandenen Pfads. Ein modellweiter Gewinn von mehreren TPS ist dadurch nicht belegt; Kolibri aktiviert Read-ahead und spart dort folglich die zusaetzliche Barriere nicht.

### CUDA-Top-K fuer MoE-Routing

Datei: [ggml/src/ggml-cuda/topk-moe.cu](ggml/src/ggml-cuda/topk-moe.cu).

Drei dynamisch indizierte Schreibzugriffe auf lokale Thread-Arrays wurden durch entrollte Schleifen mit konstanten Array-Indizes ersetzt. Das erlaubt dem Compiler, diese Werte in Registern zu halten und lokale Speicherzugriffe zu vermeiden. Auswahlreihenfolge, Gleichstandsbehandlung und Gleitkommaoperationen bleiben erhalten.

Die ersten isolierten Messungen zeigten beispielsweise bei 128 Experten und einer Zeile etwa 4.2 auf 3.0 Mikrosekunden sowie bei 512 Experten und 512 Zeilen etwa 25.5 auf 12.1 Mikrosekunden. Bei 512 Experten im Decode war der Effekt dagegen gering oder leicht negativ. Diese Mikrobenchmarks sind vorlaeufig: Eine zweite Serie mit einer Sekunde GPU-Warmup je Fall und umgekehrter Reihenfolge wurde vorbereitet, wegen des Rundenendes aber nicht mehr ausgefuehrt. Deshalb werden diese Zahlen nicht als abgesicherter Kernel-Speedup oder als Modell-TPS-Gewinn gewertet.

### CUDA-Metadaten und sichere Host-Kopien

Datei: [ggml/src/ggml-cuda/ggml-cuda.cu](ggml/src/ggml-cuda/ggml-cuda.cu).

- Der unveraenderliche Geraetetyp wird bei der Registrierung aus `cudaDeviceProp.integrated` gespeichert. Wiederholtes `ggml_backend_dev_type()` fragt nicht mehr jedes Mal den Treiber ab. Isoliert sanken 1000 Aufrufe von rund 1.77 Sekunden auf rund 10 Mikrosekunden. Die grosse Wirkung trat im experimentellen Scheduler auf, der diesen Getter pro Knoten aufrief; sie darf nicht auf den normalen Tokenpfad hochgerechnet werden. Das intern teilweise absichtlich ueberschriebene `info.integrated` wird nicht als Ersatz verwendet.
- Vor einer asynchronen Kopie aus einem gepinnten CPU-Puffer wird der Quell-Backend synchronisiert. Ein asynchron berechneter CPU-Tensor, einschliesslich Views darauf, muss fertig sein, bevor CUDA ihn liest. Die Kopie und ihre GPU-Verbraucher bleiben im Zielstream geordnet. Diese Aenderung ist eine Korrektheitsreparatur und kein behaupteter Geschwindigkeitsgewinn.

### Pruefung der Korrektheit

- Top-K: 10,976 direkte Vergleichsfaelle je Bibliothek; Gewichtsbits und geordnete Experten-IDs stimmen zwischen Referenz und Kandidat exakt ueberein. Abgedeckt sind verschiedene Expertenzahlen, Tokenzahlen, Auswahlgroessen, Bias/Normalisierung/Gating, Gleichstaende und teilweise NaN-Eingaben.
- Bestehender CUDA-Backendtest fuer `TOPK_MOE`: 320/320 Faelle bestanden.
- [tests/test-moe-multi-row.cpp](tests/test-moe-multi-row.cpp) prueft jetzt auch vollstaendige `MOE_COLD`-Graphen: 1/3/9 Tokens, kalte/gemischte/heisse Expertenmasken, Optimierungspfade an/aus sowie 1/2/4/8 Threads. 72 Graphberechnungen je Read-ahead-Modus, beide Modi bestanden. `GGML_ASSERT` stellt sicher, dass die Pruefungen auch im Release-Build aktiv bleiben.
- CPU-zu-CUDA-Kopierpfad: insgesamt 192 Graphfaelle ueber direkte Tensoren und CPU-Views, mit 192 tatsaechlichen asynchronen CPU-Jobs je Layout; Ausgaben identisch, maximaler Fehler 0. Getestet mit dem urspruenglichen Scheduler, damit dessen experimentelle Schutzmassnahmen die Korrektur nicht verdecken.
- Gecachter CUDA-Typ gegen rohe Treibereigenschaft mit einem und zwei virtuellen Geraeten geprueft.
- Geaenderte Uebersetzungseinheiten und Test gebaut, abschliessender Diff erneut gelesen, `git diff --check` bestanden.

Grenzen: Kein Nachweis auf HIP oder auf Hardware ohne AVX2. Numerisch identische Resultate sind allein kein Race-Detector; insbesondere gleichwertige ueberlappende Schreibzugriffe koennen dabei unentdeckt bleiben. Die Korrektur der Quantisierungsschleife ist durch die eindeutige Aufteilung der Zeilen begruendet.

## Nicht uebernommene Versuche und Misserfolge

| Versuch | Befund | Entscheidung |
| --- | --- | --- |
| CPU/GPU-Lookahead fuer kalte Experten | Zusaetzlicher GPU-Praefix wurde vor der kalten CPU-FFN gestartet. Nach Fehlerbehebung bei Qwen beobachtete +2.24 TPS, aber adaptive DFlash-Pfade abweichend; bei Nex nur +0.17 TPS. | Separater Patch, nicht im Produktcode. |
| Erste Lookahead-Fassung | Wiederholte CUDA-Geraeteeigenschaftsabfragen im Knotendurchlauf vergroesserten den Graphaufbau von Millisekunden auf etwa 2.5 Sekunden. | Diagnosepfad korrigiert, langsame/abgebrochene Laeufe nicht als Leistungsnachweis verwendet. Kein Beleg, dass allein der CPU-Koordinator die Ursache war. |
| Fusion der Addition heisser/kalter Experten mit gewichteter MoE-Reduktion | 588 Vergleichsfaelle bestanden, davon 112 tatsaechlich fusionierte Faelle. Der gemeinsame Qwen-Versuch mit CPU-Endbarriere war nicht schneller. | Als Experiment erhalten, kein belegter isolierter Nutzen. |
| Weglassen einer CPU-Endbarriere bei abschliessenden reinen View-Knoten | Numerische Vergleiche bestanden; im Modellversuch zusammen mit der ADD-Fusion kein Gewinn. | Nicht uebernommen. |
| Geteilte Aktivierungsquantisierung fuer weitere Gate/Up-Pfade | Routing-Gate/Up teilt Quantisierung bereits; Down erhaelt andere Eingaben. Bei dichten Shared-Expert-Pfaden fuer kleine Mehrtoken-Batches blieb etwas doppelte Arbeit. | Nur analysiert: im ausgewaehlten Trace 770 doppelte Aufrufe mit zusammen etwa 1.84 ms GPU-Zeit in 2.1 s, zu klein als alleiniger Hebel fuer 3-5 TPS. |
| Kolibri ohne Chatvorlage | Wiederholungsschutz beendet nach 26 Tokens. | Ausgeschlossen; mit Profil-Chatvorlage wiederholt. |
| Warmer Kolibri-Rueckvergleich und strengere Top-K-Mikromessung | Bei Rundenende noch nicht abgeschlossen bzw. nicht ausgefuehrt. | Offen dokumentiert, keine Ergebnisse erfunden. |

Die experimentelle Ueberlappung erforderte zusaetzliche Abhaengigkeitsregeln fuer CPU-Ergebnisse und Views. Synthetische Gleichheits- und Fallbacktests bestanden, rechtfertigen aber ohne klaren Modellgewinn keine dauerhafte Vergroesserung des Schedulers.

## Verbleibendes Optimierungspotenzial

Die folgenden Punkte sind Hypothesen fuer eine weitere Runde, keine zugesagten Gewinne. Bei etwa 48 TPS erfordern weitere 3-5 TPS ungefaehr 1.2-2.0 ms weniger Zeit pro generiertem Token. Einige eingesparte Mikrosekunden im Routing reichen dafuer alleine nicht.

1. **Kritischen CPU/GPU-Pfad und Graphstartkosten genauer aufloesen.** Der lokale Nsight-Trace zeigte in einem ausgewaehlten Decode-Fenster GPU-Aktivitaet von ungefaehr einem Drittel der Zeit und viele kleine Kernel. Das deutet auf Luecken zwischen GPU-Arbeiten hin, beweist aber keine vermeidbare CPU-Ursache: CPU-Sampling und eine vollstaendige Zuordnung der Wartezeiten fehlen. Naechster sinnvoller Schritt waere, CPU-FFN, Host-Starts, Transfers und Stream-Abhaengigkeiten entlang desselben Tokens zu messen. Danach gezielt vorhandene Graphen wiederverwenden oder zusammenhaengende kleine Operationen fusionieren. Die reine Vorverlagerung eines GPU-Praefixes war bislang nicht ausreichend.
2. **Kalte CPU-FFN bei kleinen Verifikationsbatches.** Pruefen, ob Gate/Up/Down weiterhin dieselben Gewichtsbereiche mehrfach laden oder ob Arbeitspakete und Barrieren zu klein sind. Ansatzpunkte sind gemeinsame Bearbeitung mehrerer Zeilen, Cache-Tiling und Weiterverwendung quantisierter Aktivierungen. Der Fork hat davon bereits mehrere Varianten; sie zuerst profilieren, statt dieselbe Optimierung erneut einzubauen. Quantisierung und numerische Eigenschaften muessen erhalten bleiben.
3. **Bei Kolibri Speicherresidenz als eigene Codefrage behandeln.** Das grosse GGUF liegt ueber dem RAM-Budget. Experten-Read-ahead, Auswahl der vorausgeladenen Slabs und Speicher-Lebenszeiten koennen wichtiger sein als einzelne CUDA-Kernel. Ein neuer Versuch muss kalte und warme Zustaende getrennt auswerten und physische Reads erfassen. Die jetzigen Messungen belegen den Einfluss des Nachladens, noch keine bessere Prefetch-Strategie.
4. **DFlash-Adaptivitaet fuer Codevergleiche nachvollziehbar machen.** Die vorhandenen Entscheidungen samt Tokenfolgen reproduzieren bzw. kontrolliert wiedergeben, um Scheduler-Gewinne von einer geaenderten Draft-Laenge zu trennen. Die beobachteten Tokenabweichungen trotz greedy Anfrage separat auf erwartetes Verhalten und moegliche Korrektheitsprobleme untersuchen. Das ist zuerst eine Mess- und Korrektheitsaufgabe, keine Aufforderung zur erneuten Abstimmung der Produktionsparameter.
5. **Top-K-Mikrobenchmark abschliessen und nach realen Shapes gewichten.** Warmup und umgekehrte Reihenfolge sind vorbereitet. Falls die relevanten 512-Experten-Decode-Shapes dauerhaft langsamer werden, die Registervariante dort begrenzen oder zuruecknehmen. Positive Prefill-Zahlen allein rechtfertigen keine pauschale Beschleunigungsbehauptung.

Niedrige Prioritaet: weitere einzelne Mikrosekunden-Fusionen ohne Anteil am kritischen Pfad, ein grosser neuer Scheduler ohne vorherigen Nachweis und erneute allgemeine Parameter-Sweeps.

## Arbeitsstand nach der ersten Runde und Wiederaufnahme

- Die vier oben genannten Quelldateien/Testdateien enthalten die erhaltenen Aenderungen. Bis zum Abschluss der ersten Runde wurde kein Commit, Push, PR oder Deployment erstellt.
- Bereits vorher vorhandene Aenderungen an `start-lythri-4b.sh` und den Templates `lythri-4b.jinja` / `mimo-v2.6-distill.jinja` wurden erhalten.
- Der komplette letzte Bibliotheksstand der ersten Runde liegt lokal in `benchmark-results/perf-review-20261006/final-complete-bin/`; CPU aus `cpu-final-bin`, CUDA aus `cuda-host-sync/bin`. Das ist ein isoliertes Testruntime-Verzeichnis mit Symlinks auf weitere Referenzbibliotheken, kein eigenstaendiges Distributionspaket. Zum damaligen Zeitpunkt nutzte `agollm` weiterhin seinen bisherigen Build.
- Build-/Testbelege: `cpu-final-build.json`, `cuda-host-sync/`, `cuda-source-sync-validation.json`, `cuda-device-type/validation.json`, `topk-registers/backend-ops.log`, `product-code-review.json`.
- Nicht uebernommene Patches: `cpu-cold-lookahead.patch`, `cpu-tail.patch`, `cpu-async-views.patch` sowie die Artefakte in `combine-add/`.
- Profiling: `qwen-profile-fixed/trace-summary.json` und zugehoerige Trace-Dateien. Profiling-Laeufe sind keine Durchsatzreferenzen.
- Reproduzierbarer Modellrunner: `run_recipe.py`; pro Lauf sind Rezept, Startkommando, Metadaten, Anfragen, Timings und Token-Hashes gespeichert. Neuere Laeufe enthalten ausserdem `loaded-libraries.json` zur Kontrolle der tatsaechlich geladenen Bibliotheken.
- Noch offene Pruefungen: warmer Kolibri-Rueckvergleich, umgekehrter Top-K-Mikrovergleich mit Warmup, streng kontrollierter Vergleich des gesamten letzten Codestands bei Qwen/Nex. Sie wurden nach dem Wunsch, die Runde zu beenden, nicht weiter ausgefuehrt.

## Fortsetzung am 2026-10-07

Auf Wunsch des Benutzers wurden die offenen Tests wieder aufgenommen. Die obigen Angaben beschreiben den Abschluss der ersten Runde. Diese Fortsetzung ist ebenfalls abgeschlossen: vier Top-K-Serien mit je 24 Shapes sowie elf Modellprozesse mit insgesamt 33 vollstaendigen Messanfragen und separaten Warmups. `agollm` wurde nach den Tests wieder gestartet. In dieser Fortsetzung wurden weder Produktquellen noch agollm-Rezepte oder produktive Binaries geaendert.

### Referenz neu abgesichert

`build-main-sm75` war zwischen den Runden neu gebaut worden und enthielt nicht mehr die urspruenglichen Referenzbibliotheken. Deshalb wurden neue A/B-Bibliotheken erstellt: Referenzquellen aus dem unveraenderten HEAD `9ca564aab`, Kandidatenquellen aus dem unveraenderten Arbeitsbaum. Compileoptionen und alle uebrigen Linkobjekte sind je Vergleich identisch. Gemeinsame Server-/Llama-/GGML-Bibliotheken wurden kopiert und fuer die Runde festgehalten. Die Laufzeitaufloesung faellt nicht auf `build-main-sm75` zurueck.

Die drei agollm-Rezepte, ihre expandierten Argumente/Umgebungen und die relevanten Spawn-Quelldateien sind gegenueber der ersten Runde unveraendert. Der Runner erfasst jetzt vor/nach jeder Anfrage CPU-/GPU-Temperatur und GPU-Takte sowie weiterhin physische Lesezugriffe. Das sind Momentaufnahmen, keine kontinuierliche Takt- oder Temperaturkontrolle.

Belege: [ab-runtime-manifest.json](benchmark-results/perf-review-20261007/ab-runtime-manifest.json), [recipe-and-runner-audit.json](benchmark-results/perf-review-20261007/recipe-and-runner-audit.json), [dependency-resolution.json](benchmark-results/perf-review-20261007/dependency-resolution.json).

### Warmer Top-K-Mikrovergleich abgeschlossen

Vier Serien in der Reihenfolge Kandidat, Referenz, Referenz, Kandidat; jeweils eine Sekunde GPU-Warmup pro Fall. Alle 24 getesteten Shapes zeigen in beiden Paarvergleichen dieselbe Richtung. Die maximalen Abweichungen zwischen Wiederholungen lagen bei etwa 0.73% fuer die Referenz und 1.64% fuer den Kandidaten.

| Experten / Batch | Aenderung der Kernelzeit | Einordnung |
| --- | ---: | --- |
| 128 Experten, kleine Decode-/Verify-Batches | -7.06% bis -8.23% | Wiederholbarer Mikrogewinn |
| 256 Experten, kleine Decode-/Verify-Batches | -3.38% bis -5.25% | Wiederholbarer Mikrogewinn |
| 512 Experten, Batch 1 ohne/mit Bias | +3.09% / +3.97% | Kleine Regression |
| 512 Experten, Batch 3/5 mit Bias | +3.04% / +2.57% | Kleine Regression |
| 512 Experten, Batch 3/5 ohne Bias | +0.45% / -1.22% | Geringer Effekt |
| 512 Experten, Batch 512 ohne Bias | -51.48%, 25.7545 auf 12.4973 us | Deutlicher Prefill-Mikrogewinn |
| 512 Experten, Batch 512 mit Bias | -37.21%, 23.8298 auf 14.9628 us | Deutlicher Prefill-Mikrogewinn |

Die fruehen Mikrobenchmarkzahlen der ersten Runde werden damit genauer eingeordnet: Die Verbesserung ist nicht fuer alle Shapes gegeben. Bei 512 Experten steigt die Registerzahl ohne/mit Bias von 48/49 auf 63/90, waehrend der lokale Stack von 128 auf 0 Bytes faellt. Eine getrennte Auswahl der Kernelvariante nach Shape/Bias bleibt eine ungetestete Moeglichkeit. In dieser Fortsetzung wurde der Produktcode nicht geaendert. Vor der letzten Kandidatenserie meldete ein GPU-Snapshot kurz `SW Thermal Slowdown`; die uebrigen Snapshots nicht. Auch deshalb keine Uebertragung der Prozentwerte auf Modell-TPS.

Belege: [Top-K-Ergebnisse](benchmark-results/perf-review-20261007/topk-warm/results.json), [alle Shape-Vergleiche](benchmark-results/perf-review-20261007/topk-warm/shape-diffs.csv).

### Warmer Kolibri-Vergleich abgeschlossen

Drei Prozesse in der Reihenfolge Referenz A, Kandidat, Referenz B. Je Prozess drei vollstaendige Warmup-Anfragen und drei Messanfragen mit jeweils 256 Tokens und identischem formatiertem Prompt.

| Variante | Median TPS | Einzelwerte TPS | Physische Reads der drei Messanfragen |
| --- | ---: | --- | ---: |
| Referenz A | 27.1132 | 26.9610 / 27.4470 / 27.1132 | 667648 Bytes |
| Kandidat | 27.7163 | 27.7163 / 27.7861 / 26.9602 | 0 Bytes |
| Referenz B | 27.3146 | 26.0340 / 27.3146 / 27.4885 | 0 Bytes |

Alle neun Messanfragen haben identische Token-IDs; alle angeforderten Tokens wurden erzeugt. Der Kandidat liegt im Median 0.6031 TPS ueber Referenz A und 0.4017 TPS ueber Referenz B. Die Spannen der Einzelmessungen ueberlappen. Der grosse SSD-Effekt ist aus den Messanfragen weitgehend entfernt, aber ein belastbarer Gewinn von 3-5 TPS liegt weiterhin nicht vor. CPU-Temperatursnapshots lagen je nach Prozess etwa zwischen 91.6 und 95.0 Grad Celsius; eine vollstaendig kontrollierte thermische Messumgebung war das nicht.

Aktuelle Modell-Rohdaten und Auswertung: [round-results.json](benchmark-results/perf-review-20261007/round-results.json), [round-results.csv](benchmark-results/perf-review-20261007/round-results.csv), erzeugt mit `benchmark-results/perf-review-20261007/summarize_round.py`.

### Nex-Vergleich abgeschlossen

Referenz A, Kandidat, Referenz B; je Prozess zwei Warmup-Anfragen und drei Messanfragen zu jeweils 512 Tokens. Keine Aenderung am agollm-Rezept und kein spekulatives Decoding.

| Variante | Median TPS | Einzelwerte TPS | Physische Reads der drei Messanfragen |
| --- | ---: | --- | ---: |
| Referenz A | 35.1677 | 35.4703 / 35.1115 / 35.1677 | 458752 Bytes |
| Kandidat | 35.1473 | 35.1439 / 35.5629 / 35.1473 | 0 Bytes |
| Referenz B | 35.1047 | 34.2803 / 35.1047 / 35.4134 | 0 Bytes |

Alle neun Messanfragen erzeugen dieselben 512 Token-IDs. Der Kandidat liegt zwischen den Referenzmedianen, mit -0.0204 bzw. +0.0426 TPS Unterschied. Das ist praktisch Gleichstand. Die groessere negative Differenz der ersten Runde wiederholt sich damit nicht; ein Leistungsgewinn ergibt sich ebenfalls nicht.

### Qwen-Abweichungen genauer eingegrenzt

Der neue Vergleich mit unveraendertem Profil ergibt 46.4366 TPS fuer die Referenz (46.7183 / 43.1955 / 46.4366) und 47.8159 TPS fuer den Kandidaten (47.8159 / 43.2197 / 49.2245). Die ersten beiden Anfragepaare sind jeweils tokenidentisch und haben identische adaptive Cap-Folgen; ihre Differenzen betragen +1.0976 und +0.0241 TPS. Das dritte Paar unterscheidet sich in der Cap-Folge und ab Tokenindex 507. Der Medianunterschied von +1.3792 TPS ist deshalb kein sauber isolierter Codegewinn. Die Referenz las insgesamt etwa 14.1 MB physisch, der Kandidat 0 Bytes.

Bereits die unveraenderte Referenz der ersten Runde erzeugt zwischen Wiederholungen unterschiedliche Tokenfolgen, teilweise ab Tokenindex 41. Die adaptive Cap-Auswahl wird zwar pro Anfrage zurueckgesetzt, ihre erste zeitabhaengige Bestwahl erfolgt aber fruehestens nach 65 decodierten Tokens. Fruehere Abweichungen koennen daher nicht pauschal dieser Entscheidung in derselben Anfrage zugeschrieben werden.

`LLAMA_EXPERT_WARM_RESET=request` leert die geladenen Warm-Experten nicht, sondern setzt unter anderem Alters-/Admission-Historie zurueck. Zudem bleibt ein asynchroner Admission-Cursor erhalten. Vorherige Anfragen koennen damit beeinflussen, welche Experten ueber CPU oder GPU berechnet werden. Das ist eine weitere plausible Einflussgroesse, beweist weder numerische Harmlosigkeit noch einen Acceptance-/Rollback-Fehler. Der folgende Diagnosevergleich mit `LLAMA_DFLASH_ADAPTIVE=0` und dem unveraenderten Profil-Maximum von vier Draft-Tokens wird getrennt vom normalen Profilvergleich ausgewertet.

Quellstellen, Tokenindizes und Diagnosegrenzen: [dflash-analysis.json](benchmark-results/perf-review-20261007/dflash-analysis.json).

### Qwen mit fester Draft-Obergrenze und erneutes Lookahead-Experiment

Je Variante ein frischer Serverprozess mit zwei Warmup-Anfragen und drei Messanfragen, jeweils 512 Tokens. Die zeitabhaengige Cap-Auswahl wurde nur fuer die Testprozesse abgeschaltet. Das Experiment aktiviert zusaetzlich `LLAMA_EXPERT_CPU_ASYNC=1` und `GGML_SCHED_MOE_COLD_SPLIT=2`. Das ist eine Diagnose unter kontrollierter Obergrenze, kein geaendertes oder neu abgestimmtes Produktionsprofil.

| Variante | Median TPS | Einzelwerte TPS | Ausgaben gegen Referenz |
| --- | ---: | --- | --- |
| Referenz | 48.4550 | 48.4550 / 48.9203 / 48.2350 | Referenzfolge |
| Regulaerer Kandidat | 48.4911 | 47.4166 / 49.4050 / 48.4911 | Alle positionsgleichen Warmup-/Messanfragen exakt identisch |
| Lookahead-Experiment | 49.5745 | 49.5745 / 51.0818 / 48.8467 | Alle drei Messanfragen abweichend |

Alle Messanfragen lesen 0 Bytes physisch. Referenz und regulaerer Kandidat stimmen in allen fuenf positionsgleichen Anfragen inklusive der Warmups bei saemtlichen 512 Token-IDs und den Draft-Zaehlern ueberein. Innerhalb eines Prozesses unterscheiden sich die Ausgaben weiterhin zwischen Anfragepositionen. Die Differenz der Variantenmediane betraegt fuer den regulaeren Kandidaten nur +0.0361 TPS. Die einzelnen paarweisen Differenzen wechseln das Vorzeichen (-1.0384 / +0.4847 / +0.2562 TPS); deren Median ist +0.2562 TPS.

Lookahead liegt rechnerisch +1.1195 TPS ueber der Referenz und +1.0833 TPS ueber dem regulaeren Kandidaten, erzeugt aber andere Tokenfolgen: Erste Abweichungen der Messanfragen an den nullbasierten Tokenindizes 41, 299 und 37. Bereits die Warmups unterscheiden sich. Diese Differenzen bestehen trotz ausgeschalteter Adaptivitaet. Damit ist die Erklaerung allein durch zeitabhaengige Cap-Auswahl fuer dieses Experiment widerlegt. Eine Korrektheitsregression oder andere numerische/Backend-Effekte sind nicht ausgeschlossen; es liegt kein sauberer Speedup-Nachweis bei gleichem Arbeitsvolumen vor. Das Experiment bleibt ausserhalb des Produktcodes.

Die rein lesende Nachanalyse nennt geaenderte Cold-/Hot-Graphreihenfolge, Splitgrenzen, Speicherlebenszeiten und Warm-Expert-Publish-Zeitpunkte als moegliche Einfluesse. CUDA-Fusion haengt unter anderem von Graphgrenzen und konkreten Speicherueberlappungen ab. Es wurde aber weder der Verlust einer bestimmten Fusion noch eine konkrete Race als Ursache nachgewiesen. Details und Quellstellen: [lookahead-fixed4-analysis.json](benchmark-results/perf-review-20261007/lookahead-fixed4-analysis.json).

Beide experimentellen Uebersetzungseinheiten wurden fuer diese Diagnose frisch aus HEAD plus dokumentiertem Patch mit den aktuellen Flags gebaut. Es wurden keine historischen Experimentobjekte fuer den endgueltigen Link uebernommen. Reproduktion und Hashbelege: [lookahead-build.json](benchmark-results/perf-review-20261007/lookahead-build.json), [lookahead-dependencies.json](benchmark-results/perf-review-20261007/lookahead-dependencies.json), `benchmark-results/perf-review-20261007/build_lookahead.py --fresh-objects --build`.

### Prioritaeten nach dieser Fortsetzung

1. Vor weiteren Lookahead-Leistungsversprechen die erste Logit-/Routing-Abweichung mit reproduzierbarem Expertenzustand und gleicher Anfragehistorie isolieren. Synthetische Kopier-/Abhaengigkeitstests genuegen nicht, um die beobachteten Modellabweichungen zu erklaeren.
2. Falls Top-K weiter bearbeitet wird, gezielt die 512-Experten-Decode-Regression adressieren, ohne den nachgewiesenen Prefill-Vorteil zu verlieren. Eine Shape-/Bias-Auswahl wurde noch nicht implementiert oder getestet.
3. Fuer mehrere zusaetzliche Modell-TPS bleibt der kritische CPU-FFN-/Transfer-/Graphstart-Pfad entscheidend. Die jetzt besser kontrollierten Gesamtmessungen rechtfertigen weder eine allgemeine +3-5-TPS-Behauptung noch die Uebernahme eines groesseren Schedulers allein aufgrund der Mikrobenchmarks.

Alle in der ersten Runde konkret offen gebliebenen Vergleiche wurden damit nachgeholt. Weiterfuehrende Logit-/Routing-Diagnostik und neue Kernelvarianten sind moegliche neue Experimente, keine noch laufenden Tests dieser Runde.

## Deployment nach agollm am 2026-10-07

Auf Wunsch des Benutzers wurden die noch fehlende Uebernahme der getesteten Laufzeitdateien und der anschliessende Neustart umgesetzt. Quelle ist der regulaere Kandidat aus `benchmark-results/perf-review-20261007/candidate-bin`, dessen Modell- und Korrektheitstests oben dokumentiert sind. Die experimentelle Lookahead-Variante wurde nicht uebernommen.

Aktualisiert wurden unter `/home/andi/src/agollm/workers/llama/` die fuenf Dateien `llama-server`, `libggml-cpu.so.0`, `libggml-cuda.so.0`, `libggml-base.so.0` und `libllama-common.so.0`. Die vier uebrigen Laufzeitbibliotheken waren bereits byteidentisch zum getesteten Kandidaten. Die Installation enthaelt echte Dateien und haengt nicht von Symlinks in die Benchmarkverzeichnisse ab. `ORIGIN.txt` dokumentiert Herkunft und Backup.

Der vollstaendige vorherige Worker liegt als Backup unter `/home/andi/src/agollm/runtime/llama-build-before-perf-fixes-20261007T091306Z`. Alle 14 Assets und alle 12 Konfigurationsdateien blieben laut Hashvergleich unveraendert. Es wurden keine Parameter neu abgestimmt.

Pruefung nach dem Neustart, abgeschlossen um 09:14:50 UTC:

- Installierte Dateien stimmen mit dem vorbereiteten Paket ueberein; Bibliotheksabhaengigkeiten werden vollstaendig aufgeloest.
- Die geladenen Laufzeitdateien des Workers wurden ueber `/proc/3506682/maps`, Inodes und SHA256 gegen die Installation geprueft. Alle neun Laufzeitdateien entsprechen dem getesteten Kandidaten; keine Bibliothek wird aus einem Benchmarkverzeichnis geladen.
- Router, LLM-, ASR-, Embedding- und Decision-Worker antworten auf ihre Health-Checks jeweils mit HTTP 200 und `status: ok`.
- Eine Chat-Anfrage mit `ling-tiny` ueber `http://127.0.0.1:8080/v1/chat/completions` lieferte die angeforderte Antwort `OK`. Das bestaetigt den Weg durch agollm bis zur Generierung; es ist kein neuer Durchsatzbenchmark der grossen Modelle.

Belege: [Deployment-Manifest](benchmark-results/perf-review-20261007/deployment/manifest.json), [Laufzeitvalidierung](benchmark-results/perf-review-20261007/deployment/validation.json), [API-Testantwort](benchmark-results/perf-review-20261007/deployment/smoke-response.json).

Damit sind die erhaltenen Fixes implementiert, getestet und in agollm aktiv. Die oben dokumentierten Leistungsgrenzen gelten weiterhin: Ein allgemeiner Gewinn von 3-5 TPS ist nicht nachgewiesen; Lookahead-Diagnostik und eine gesonderte Top-K-Auswahl fuer 512-Experten-Decode bleiben moegliche weitere Arbeiten.

## Schnellere Modellwechsel im Router, 2026-10-07

### Ursache und implementierte Aenderung

Der Router forderte den gemeinsamen Busy-Leistungsmodus bisher fuer Sprach-/CPU-Arbeit an, aber nicht fuer einen LLM-Wechsel. Der llama-Worker initialisiert seinen eigenen Power-Controller erst nach dem Modellladen. Damit konnten sowohl die Bereinigung des alten Modells als auch Metadaten, Gewichtsuebertragung und Warmup im Battery-Profil laufen.

In [agollm/internal/app/app.go](../agollm/internal/app/app.go) haelt `AcquireLLM` jetzt innerhalb des vorhandenen Ladezweigs `a.Power.Acquire()` bis zur Rueckkehr aus `Ensure`. Der Guard umfasst das Entladen, Dateivorbereitung, Prozessstart und die Bereitschaftspruefung. Er wird bei Erfolg, Fehler und Abbruch freigegeben; der konfigurierte Idle-Timer gilt unveraendert. Bereits residente Modelle und das Warten auf einen Scheduler-Platz loesen keinen neuen Guard aus. Der gemeinsame Marker verhindert zudem, dass der alte Worker waehrend des Wechsels wieder auf Battery umstellt.

Die Produktcodeaenderung ist eine zusaetzliche Anweisung und ein aktualisierter Kommentar. Keine Aenderung an Modellparametern, Warmup, Quantisierung, KV-Cache oder llama-Laufzeitbibliotheken. Die Testhelper verwenden jetzt einen temporaeren GPU-Lock und deaktivieren echte Power-Kommandos.

### Messungen und Grenzen

Gemessen wurde die Antwortzeit von `POST /router/v1/preload`, also einschliesslich Entladen, Laden, Warmup und Bereitschaftspolling. Jede Serie wechselte von Ling zu LFM, zu Qwen und zurueck zu Ling. Drei Sekunden Abstand lassen den bestehenden Idle-Timer ablaufen. Reihenfolge: urspruenglicher Router, derselbe Router mit manuell gehaltenem Busy-Marker, Referenz, Busy-Marker, Referenz; danach drei Serien mit dem neu gebauten Router ohne manuellen Guard. Insgesamt 24 erfolgreiche Wechsel. Builds und Tests liefen nicht gleichzeitig mit den Messungen.

| Zielmodell / Wechsel | Urspruenglicher Router, aufgewaermte Serien B/C | Diagnose mit gehaltenem Leistungsmodus A/B | Installierter Router A/B/C |
| --- | --- | --- | --- |
| LFM2.5-1.2B, von Ling | 2.105 / 1.973 s | 1.593 / 1.287 s | 2.750 / 1.304 / 1.830 s |
| Qwen3.6-35B-A3B, von LFM | 13.371 / 13.581 s | 8.335 / 5.871 s | 11.326 / 18.192 / 7.149 s |
| Ling-tiny, von Qwen | 8.542 / 9.156 s | 5.722 / 5.325 s | 5.035 / 5.941 / 5.058 s |

Die erste Referenzserie benoetigte 2.933 / 37.013 / 9.731 Sekunden. Sie ist keine warme Referenz. Die erste Serie nach Installation folgte einem Dienstneustart mit erneut gestarteten Hintergrundworkern; Qwens vorhandenes `freeVega` arbeitete dabei teilweise parallel zum Laden.

Der Diagnosevergleich ergibt fuer Qwen 13.476 gegen 7.103 Sekunden Median und fuer Ling 8.849 gegen 5.523 Sekunden. Die beobachtete Beschleunigung ist aber kein pauschales Versprechen fuer jeden Speicherzustand: Physische Reads wurden erst ab Kandidat B erfasst, fuer die frueheren Serien ist gleicher Dateicache-Zustand nicht nachgewiesen. Im installierten Router las Qwen in B rund 15.01 GB physisch und brauchte 18.192 Sekunden; in C waren es 0.89 GB und 7.149 Sekunden. Ling las in B/C 4.92 / 4.79 GB und brauchte trotzdem nur 5.941 / 5.058 Sekunden. Kaltes Nachladen bleibt ein eigenstaendiger Engpass.

Auch das Entladen profitiert teilweise: Bei Qwen lag das beobachtete Verschwinden des alten Prozesses in Referenz B/C bei 1.519 / 1.539 Sekunden, mit Diagnose-Guard bei 0.736 / 0.741 Sekunden und im installierten Router bei 0.427 / 0.859 / 0.804 Sekunden. Das ist eine 20-ms-Prozessbeobachtung ab Anfragebeginn und enthaelt bei aktivem Guard auch dessen Power-Kommando, keine isolierte CUDA-Free-Messung. Bei kleinen alten Modellen ist das Beenden bereits kurz; dort kann der zusaetzliche Power-Aufruf einen Teil des Vorteils aufbrauchen.

Die 20-/30-Sekunden-Werte in `Proc.Stop` sind maximale Wartezeiten, keine festen Pausen. Sie wurden nicht verkuerzt. Session-GC und Server-Queue werden beim Shutdown bereits direkt geweckt. Ein erzwungenes sofortiges Kill ist fuer die gemessenen normalen Wechsel nicht erforderlich.

Rohdaten: [results.json](benchmark-results/model-switch-20261007/results.json), [results.csv](benchmark-results/model-switch-20261007/results.csv). Jede Serie enthaelt ausserdem die Serverlogs, Prozesswechsel und Zeitstempel in ihrem JSON unter `benchmark-results/model-switch-20261007/`.

### Validierung und aktiver Stand

- Neuer [Regressionstest](../agollm/internal/app/power_test.go): Busy-Marker und Kommando vor Ladebeginn, Freigabe nach Erfolg/Fehler/Abbruch, kein erneutes Laden oder Power-Kommando bei residentem Modell. Alle drei Unterfaelle schlagen mit dem alten Produktcode erwartungsgemaess am fehlenden Busy-Marker fehl und bestehen mit dem Fix.
- `go test -race` fuer `internal/app`, `internal/api`, `internal/llama`, `internal/proc`, `internal/scheduler` und `internal/power` bestanden; Build, Konfigurationspruefung und `git diff --check` bestanden.
- Der neue `ai-routerd` wurde installiert und sein Hash gegen `/proc/<pid>/exe` verifiziert. Backup: `/home/andi/src/agollm/runtime/ai-routerd-before-model-switch-20261007T092955Z`.
- Nach den Tests wurden die urspruenglichen Hintergrundworker wieder gestartet. Router, LLM, ASR, Embedding und Decision bestanden alle Health-Checks. Ling-tiny ist wieder geladen, eine Anfrage ueber die agollm-Chat-API lieferte `OK`, und nach dem Idle-Timer blieb kein lebender Busy-Marker zurueck.
- Alle Konfigurationen, Assets und neun llama-Laufzeitdateien sind laut Hashvergleich unveraendert. Die zuvor installierten Inferenzfixes bleiben aktiv.

Belege: [Tests](benchmark-results/model-switch-20261007/tests.json), [Build](benchmark-results/model-switch-20261007/build.json), [Installation und Backup](benchmark-results/model-switch-20261007/deployment.json), [abschliessende Validierung](benchmark-results/model-switch-20261007/final-validation.json).

### Weiteres Potenzial

1. Qwen versucht vor dem Target-Laden eine Draft-Speichermessung, die beim aktuellen DFlash-Loader ohne Target-Kontext immer fehlschlaegt. Der Abschnitt kostet in diesen Logs etwa 0.6 Sekunden mit Busy-Profil bzw. 1.4 Sekunden im Referenzlauf. Eine fruehe Pruefung der tatsaechlichen Draft-Architektur koennte diese Arbeit vermeiden. Nicht implementiert: Ein pauschales Ueberspringen anhand des DFlash-Flags waere bei gemischten MTP-Konfigurationen zu breit und koennte eine heute erfolgreiche VRAM-Reservierung entfernen.
2. `warmFiles` liest synchron bis zu 8 MiB pro Datei nach dem Entladen. Ueberlappung mit der alten Prozessbereinigung ist ein moeglicher kleiner Hebel, aber noch nicht isoliert gemessen.
3. Die Bereitschaft wird alle 200 ms abgefragt. Kuerzeres bzw. ereignisgesteuertes Warten kann hoechstens diesen Polling-Anteil sparen; es erklaert keine mehrsekundigen Ladezeiten.
4. Fuer die grossen Modelle bleibt die Speicherresidenz entscheidend. Weniger Prefaulting oder abgeschaltetes Warmup koennte Ladezeit nur in die erste Anfrage verschieben. Solche Parameter wurden nicht geaendert. Nex und Kolibri wurden in dieser Wechselrunde nicht separat vermessen.

## Offene Punkte der Spielerstellung abgeschlossen, 2026-10-07

### Workspace-Lesen und Reparaturen

Die bisherige Sperre nach zwei Leseaufrufen pro Datei verhinderte auch legitimes Nachlesen nach Aenderungen und das Lesen weiterer Abschnitte. In `agollm/internal/tools/research.go` gilt das Limit jetzt fuer zwei erfolgreiche identische Leseanfragen desselben Bereichs. Fehler zaehlen nicht. Erfolgreiche Schreib-/Editieroperationen setzen die Sperren der betroffenen Datei zurueck. Ein veralteter Zeilenanker erlaubt ebenfalls erneutes Lesen, weil er eine zwischenzeitliche Aenderung bestaetigt. Ausgefuehrte Python-Kommandos invalidieren alle lokalen Lesezaehler, auch bei einem Fehlerexit oder Timeout; solche Programme koennen Dateien bereits veraendert haben.

`workspace_read` kann jetzt auch lange Einzelzeilen vollstaendig in Abschnitten liefern. `next_offset_chars` ist ein fortschreitender Unicode-Zeichencursor; `partial_start` und `partial_end` kennzeichnen unvollstaendige Zeilen. Die Anker beziehen sich weiter auf die gesamte Zeile. Eine abgeschnittene letzte Zeile wird korrekt als `truncated` gemeldet. Die bisherigen Laengen-, Pfad- und Schreibschutzgrenzen bleiben bestehen.

Die neuen Paginierungstests scheiterten am bisherigen Code und bestehen mit der Korrektur. Sie pruefen lange letzte Zeilen, Unicode, CRLF, exakte Abschnittsgrenzen, leere Dateien, EOF und kleine Leselimits. Ein Integrationstest mit echten Workspace-Dateien und Werkzeugen durchlaeuft elf gesteuerte Modellantworten: Erstellen, zweimal Lesen, Editieren, erneut zweimal Lesen, externe Aenderung, Ablehnung eines veralteten Edits, erneutes Lesen, korrigierter Edit und abschliessender Dateivergleich. Dabei ist nur der Modellserver simuliert.

### Beide vorhandenen Spiele repariert

- Aeltere Version, `924cb.../workspace/index.html`: fehlendes `Game.sfx` initialisiert, Audio erst nach Benutzeraktion, 55 korrekt positionierte Gegner, konsistente Geschwindigkeiten in Sekunden, begrenzte Projektile, pixelweise zerstoerbare Schilde, UFO-Bonus, Lebensverlust mit Schutzzeit, Pause, Bodenkontakt, Levelwechsel und Neustart. Der vorherige Startabsturz wurde vor der Korrektur im Browser bestaetigt. Punkte/Leben sind auf kleinen Displays lesbarer; Touch-Knoepfe passen auch im Querformat in den sichtbaren Bereich.
- Neuere Version, `85ed.../space_invaders.html`: fehlende Audioeffekte, vier Schutzschilde, UFO, Pause und Bodenkontakt-Game-over ergaenzt. Dazu kommen vier Gegnertypen mit 55 Gegnern, A/D/Z-Steuerung, ein Spielerschuss gleichzeitig, Schutzzeit nach Treffern, Leveluebergang, Touch-Steuerung und ein einzelner Animationsloop mit festen Simulationsschritten. Der vorherige seitliche Ueberlauf auf dem Handy ist behoben. Die bestehende Retrogestaltung und die jeweils einzelne selbstenthaltene HTML-Datei bleiben erhalten.

### Verifikation und Installation

- 275 Tests und Untertests in `internal/tools`, `internal/tools/workspace`, `internal/routing`, `internal/api`, `internal/app` und `internal/uiapi` mit `go test -race` bestanden. Router-Build und `git diff --check` bestanden.
- Beide Spiele bestehen jeweils vier Chromium-Pruefungen: Desktop, Mobil, `sandbox="allow-scripts"` wie in agollms Vorschau und fehlende Audio-Unterstuetzung. Geprueft wurden Start, echte Tastatur-/Touch-Eingaben, Schusslimit, beide Richtungen der Schilderosion, Alien-/UFO-Treffer, Lebensverlust, Schutzzeit, Pause, naechste Welle, Bodenkontakt und Neustart. Keine unbehandelten JavaScript-Fehler. Eine zusaetzliche Darstellungskontrolle bestaetigt fuer beide Spiele bei 1280x900, 390x844 und 844x390, dass Canvas und Bedienelemente vollstaendig sichtbar sind.
- Der Simulationsvergleich der neueren Version ergibt bei 30, 60 und 120 Hz jeweils 300 Pixel Bewegung pro Sekunde. Das ist eine Zeitintegrationspruefung, kein GPU-Leistungsbenchmark.
- Getesteter Router und Spiele sind installiert. Der Routerhash `96324e6fdadbc8971da3395f61e9f2867046f3ff555fe9691d7b84c7678a69cd` stimmt mit dem laufenden Prozess ueberein. Die ueber die Workspace-API ausgelieferten Spieldateien entsprechen exakt den geprueften Dateien. Router, LLM, ASR, Embedding und Decision bestanden ihre Health-Checks.
- Backup des vorherigen Routers und beider Spieldateien: `/home/andi/src/agollm/runtime/before-game-repair-20261007T115915Z`. Konfigurationen, Modellparameter, Assets und alle neun llama-Laufzeitdateien sind per Hashvergleich unveraendert; die vorherigen Inferenz- und Modellwechselkorrekturen bleiben aktiv.

Belege: [Go-Tests](benchmark-results/game-repair-20261007/tests.json), [Browser-Verhalten](benchmark-results/game-repair-20261007/after/browser.json), [Darstellung](benchmark-results/game-repair-20261007/after/layout.json), [Installationsmanifest](benchmark-results/game-repair-20261007/deployment.json). Screenshots und gesicherte Ausgangsdateien liegen unter `benchmark-results/game-repair-20261007/`.

### Live-Erstellung und Reparatur ueber die aktive API

Mit dem tatsaechlich laufenden `ling-tiny` wurde in einer eigenen temporaeren Testsitzung ein kleines Klickspiel erzeugt. Der absichtlich vorgegebene Testfehler zaehlte pro Treffer zwei Punkte. Das Modell schrieb und las die Datei; der Browser bestaetigte Start, zwei Treffer, Reset und erneuten Start. Anschliessend erhielt das Modell nur den Fehlerbericht und reparierte die Datei per `workspace_edit` auf einen Punkt pro Treffer. Es las die Datei danach erneut. Die zweite Browserpruefung lud die tatsaechlich von agollm ausgelieferte Datei in der Vorschau-Sandbox und bestaetigte die korrigierte Wertung sowie Start/Reset ohne JavaScript-Fehler.

Erstellung dauerte 16.995 Sekunden, Reparatur 11.053 Sekunden. Dabei unterliefen dem Modell zwei reale Werkzeugfehler: zuerst ein unvollstaendiger Zeilenanker, danach ueberlappende Edit-Operationen. Beide wurden abgefangen; das Modell korrigierte die Aufrufe und fuehrte schliesslich genau einen gueltigen Edit aus. Der Lauf belegt auch die Erholung nach Werkzeugfehlern. Er ersetzt keine Qualitaetsgarantie fuer beliebige kuenftige Modellprogramme oder einen erneuten autonomen Gesamtaufbau des grossen Space-Invaders-Spiels; dessen vorhandene Dateien wurden separat repariert und geprueft.

Belege: [Live-Anfragen und Antworten](benchmark-results/game-repair-20261007/live-repair.json), [Browser vor Reparatur](benchmark-results/game-repair-20261007/live-browser-create.json), [Browser nach Reparatur](benchmark-results/game-repair-20261007/live-browser-repair.json), [Werkzeugprotokoll](benchmark-results/game-repair-20261007/live-workspace.log).

Die temporaere Testsitzung und ihre Workspace-Datei wurden anschliessend entfernt; die Nachweise bleiben im Benchmarkverzeichnis. Abschliessend am 2026-10-07 um 12:03 UTC: alle fuenf Health-Checks erfolgreich, Ling-tiny geladen, Warteschlange leer, kein lebender Busy-Marker uebrig. Details: [Abschlusspruefung](benchmark-results/game-repair-20261007/final-validation.json).
