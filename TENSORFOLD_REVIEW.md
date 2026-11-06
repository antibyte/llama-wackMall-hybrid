# TensorFold-Review: uebertragbare Konzepte fuer diesen llama.cpp-Fork

Datum: 2026-11-06. Quelle: `https://github.com/ashhart/TensorFold` (Klone nach `/tmp/tensorfold`, Commit `878ab5b`).

**Ergebnis: Ja, es gibt Konzepte, aber die hohen Hebel sind woanders als erwartet.** TensorFold
ist kein llama.cpp-Derivat, sondern eine Eigenentwicklung (Zig, CUDA/Metal), die nur sm_80+ baut.
Der Vulkan-Pfad ist ein Contract-Stub ohne Modelllauf. Die brauchbaren Muster sind
**algorithmisch und hostseitig** (Cache-Policy, Prefetch-Disziplin, Plan-Kernels, Kostenmodelle),
nicht architektonisch. Davon sind vier im Fork bereits umgesetzt.

**Nachtrag vom selben Tag (Benchmark, Abschnitt 9):** Die vier uebernommenen Ideen wurden
implementiert oder simuliert und gemessen. Ergebnis: **zwei widerlegt** (B1 Vulkan-Debug,
B4 CLOCK), **eine ohne lokales Messfahrzeug** (B2 FA-Kachelbreite, Mechanik geklaert, aber kein
lokales Modell hat GQA-Rate > 8 bei GPU-Last), **eine mit Wirkung** (B3 bailingmoe3 an
Lookahead + Page-Prefetch, Abschnitt 9.3). Der angebliche "Bug im eigenen Vulkan-Backend" aus
der ersten Fassung dieser Analyse **existiert nicht** (B1).

---

## 1. Was TensorFold ist (und was nicht)

| | |
|---|---|
| Eigenentwicklung | Zig-Binary, eigene CUDA- und Metal-Kernels, kein ggml |
| CUDA-Ziel | `README.md:164` - Fatbins nur sm_80/86/120/121; aeltere Karten werden beim Laden abgewiesen (`CHANGELOG.md:752`) |
| Vulkan | **Contract-Stub.** `zig/kernels/vulkan/` hat genau 2 Dateien (`qmm.comp`, `contract_controls.comp`); `zig/src/vulkan/` ist der Runtime-Nachweis. `CHANGELOG.md:736`: "No model runs on it yet". Die "Geraete" im Contract sind Apple M1 Max unter Mesa (`zig/src/core/contract/vulkan_records.zig:14-21`). `docs/` enthaelt keinen einzigen Vulkan-Treffer. |
| Optimierungsort | Metal und HIP. Die Vulkan-Seite liefert fuer den Fork **nichts**, das llama.cpp nicht schon kann. |
| HIP-Messungen | alle von gfx1030/gfx1100/gfx1151/gfx1201, **nie von einer Vega** |

**Konsequenz fuer diesen Fork:** Alles, was sm_80+ braucht, faellt aus: `cp.async`, TMA,
`mbarrier`, Warp-Specialization, NVFP4/FP8-Tensor-Cores, WMMA. Der 1660 Ti ist sm_75 ohne
Tensor-Cores. Uebertragbar sind die **Muster**, nicht die Prozentwerte.

---

## 2. Sofort umsetzbar (Tier 1)

### 2.1 CLOCK-Eviction ueber Use-Counts statt LRU fuer den Kaltpfad

TensorFold: `zig/src/core/fsr/cache.zig:126-148`. CLOCK-Hand mit 2-Bit-Use-Count (`cache.zig:10`).
Ein einmal benutzter Experte geht zuerst wieder raus (`CHANGELOG.md:136`: "the expert pool counts
uses, so an expert used once leaves before one used again"). Demand wird **vor** der Eviction
geschuetzt (`cache.zig:78-94`), ein Forecast, der nie startete, gibt seinen Slot zurueck
(`cache.zig:167-178`), und ein Forecast darf nur Slots mit `uses == 0` nehmen (`cache.zig:120-123`).

Belegter Effekt: `CHANGELOG.md:136` - rund **11 % weniger Plattenlesen** bei 17 und 35 GiB Pool,
**8 % weniger Zeit** bei 35 GiB, gleiche Replies.

**Luecke im Fork:** Der Fork hat keinen Kalt-Pool. Kalt-Experten werden direkt aus dem mmap auf der
CPU gerechnet (`ggml/src/ggml-cpu/ggml-cpu.c:2098`), es gibt keine Slots, keine Uses, keine Leases.
Der Fork **zaehlt** Cold-Auswahlen bereits (`L.cum_cold`, `ggml_moe_count`), und der Residency-Test
existiert (`src/llama-expert-tier.cpp:2809-2821`, `mincore`).

**Ehrliche Einordnung:** TensorFold faehrt einen *residenten* Pool auf Metal, der Fork rechnet
kalt auf der CPU. Die 11 %/8 % sind **keine Prognose** fuer den Fork. Der Fork ist bei Ling
bandbreitenlimitiert (`optimize.md:99`: SSD liest 0,9-1,2 GB/s = Maximum), deshalb waere eine
Reduktion der Kaltbytes pro Token naeherungsweise 1:1 in Tokens/s - **falls** die CPU nicht der
naechste Flaschenhals ist.

**Naechster Schritt (rein analytisch, kein Risiko):** CLOCK-Use-Count-Simulation ueber die
14-Profil-Prompts gegen die heutige LRU-artige Page-Cache-Policy, bei 8/16/32 GB Poolbudget.
Erst wenn die Simulation >= 5 % Kaltbytes spart, implementieren. TensorFold liefert dafuer das
Muster: `tools/fsr_profile.py:134-174` simuliert CLOCK mit identischer Use-Count-Semantik.

### 2.2 FA-Arbeitsaufteilung nach KV-Laenge statt nur nach Query-Breite

TensorFold: `zig/src/hip/launches/plan.zig:66` - `score_keys = 4` fuer Span >= 16384,
`score_keys_short = 2` darunter (`plan.zig:118`). "short spans take fewer keys a warp to fill the card".

Belegter Effekt: `CHANGELOG.md:382` - "the planned decode score takes four keys a warp on long spans
and two on short ones ... so decode runs **7.5-18.9 % faster on Qwen3.6-35B-A3B and Qwen3.8-27B at
32K and 64K** with identical tokens on a gfx1151".

**Luecke im Fork:** `ggml/src/ggml-cuda/fattn-tile.cuh:12-38` tabelliert die Konfiguration nur ueber
`(DKQ, DV, ncols)`. `ncols` ist die **Query**-Breite. Im Decode ist `ncols == 1`, also gilt bei
16k KV und bei 64k KV **dieselbe** Konfiguration. Das Modell aus der Messung (Qwen3.6-35B-A3B) laeuft
im Fork, und die Messung erfolgte bei 32K/64K Kontext - genau der Bereich, in dem der Fork den
Unterschied nicht abbildet.

Das ist der billigste der Tier-1-Punkte: ein zusaetzlicher Parameter in der
`GGML_CUDA_FATTN_TILE_CONFIG_CASE`-Tabelle, der die vorhandene `GGML_CUDA_FA_SCALAR_*`-Override-
Logik auf die KV-Seite spiegelt.

### 2.3 Radix-Select-Top-k statt Vollsortierung

TensorFold: `zig/kernels/cuda/torch_ops/topk.cu:10-15` (monotone float->uint-Abbildung,
NaN -> `UINT32_MAX`), `:63-82` (256-Bucket-Histogramm), `:141-155` (Kompaktieren via
`__ballot_sync`).

Belege: `CHANGELOG.md:684` - MoE-Top-k **2,74 us statt 4,37 us**. `CHANGELOG.md:814` (Metal,
gleicher Algorithmus) - **~29 us statt ~700 us** fuer top_k 1..64.

**Luecke im Fork:** `ggml/src/ggml-cuda/top-k.cu` hat kein Radix-Select; `topk-moe.cu` rechnet den
MoE-Router ueber Argsort. Der Fork hat die CUDA-Variante also nicht.

### 2.4 GPU-seitiger MoE-Plan statt CPU-Plan

TensorFold: `zig/kernels/cuda/experts.cu:149-197` (`place_items` mit Warp-Prefix-Sum fuer balancierte
Kacheln), `:253-274` (`rank_pairs` via `__match_any_sync`), `:276-308` (`plan_rank`/`plan_offsets`).
`__match_any_sync` ist ab sm_70 verfuegbar, also auf sm_75 nutzbar.

**Luecke im Fork:** Der MoE-Plan liegt auf der CPU (`src/llama-graph.cpp` / `src/llama-model.cpp`),
also ein Sync-Overhead pro Layer. Zusatzlich: TensorFold schickt **Padding-Zeilen nie** durch
Router/Top-k/Plan/Combine (`CHANGELOG.md:1252`) - bei Decode mit 1 aktiver Zeile ist das der
Unterschied zwischen "alle Expertengewichte lesen" und "nur die aktiven".

Aufwand hoch, da die ggml-Graph-Abstraktion gebrochen wird. Als P2 einplanen.

### 2.5 Grouped Launches: mehrere Projektionen desselben Inputs in einem Launch

TensorFold: `zig/kernels/cuda/qmm_group.cu:38-58` (`group_kernel` mit `Parts[4]`-Tabelle),
`zig/kernels/cuda/lane_gemv.cu:338-346` (`tf_lane_gemv_many`).

Belege: `CHANGELOG.md:830` - Prefill **+10,8 % @2k, +9,7 % @8k, +7,8 % @32k** (GB10, Qwen3.8-27B).
`CHANGELOG.md:597` - grouped affine-4: Nemotron 8 Streams **+13,3 %**, Qwen3.8-27B DFlash2 Decode
**+3,7-5,0 %**.

Auf sm_75 ist der Launch-Overhead-Anteil groesser als auf GB10, der Hebel also vermutlich groesser -
aber llama.cpp ist ein Datenfluss-Graph, ein Custom-Kernel mit N-Projektions-Deskriptoren bricht
diese Abstraktion. Der Fork hat den merged gate_up-Pfad bereits
(`src/llama-graph.cpp:2333`: "merged gate_up path: one mul_mat_id, then split into gate and up views").

---

## 3. Hostseitige Konzepte (Tier 2)

### 3.1 Cache-Bypass als Messpflicht: erst die echte SSD-Bandbreite kennen

TensorFold behandelt Cache-Bypass als Pflicht, nicht als Option:
`docs/design/fluid-subgraph-residency.md:9-13` - "A cached file read measures memory reuse, not SSD
throughput." und die physische Grenze "Let B be cold expert bytes read per generated token and S be
measured sustained SSD bytes per second ... storage alone bounds single-stream throughput by S / B".
`zig/src/core/fsr/file.zig:52-56` - "Readahead hints are not a cache-bypass substitute", `fcntl(F_NOCACHE)`.
`zig/src/core/direct_io.zig:8` - 4-KiB-Ausrichtung, `:25-35` Fallback auf Page-Cache.

**Luecke im Fork:** `use_direct_io` greift nur im Loader-Lesepfad
(`src/llama-model-loader.cpp:548`, `src/llama-mmap.cpp:190-203`), **nicht** fuer die gemappten
Expert-Slices zur Laufzeit.

Das ist die wichtigste methodische Uebertragung: die gemessenen 0,9-1,2 GB/s aus `optimize.md:99`
sind Page-Cache-Reuse, nicht SSD-Throughput. Solange S unbekannt ist, ist jede Optimierung des
Kaltpfads eine Optimierung ins Blaue. **Erst messen, dann optimieren** - und der Cache-Bypass-Pfad
wird vermutlich *langsamer* sein als der Page-Cache-Pfad, liefert aber die ehrliche Zahl.

**Erledigt am 2026-11-06 (Abschnitt 10).** Die ehrliche Zahl fuer Lings Zugriffsmuster
(verstreute Bloecke von 1,05-1,54 MB ueber 36,9 GiB, Page-Cache vorher per
`posix_fadvise(DONTNEED)` verworfen): 887 MB/s mit einem Thread, **2948 MB/s mit 8 Threads**,
3304 MB/s mit 16, 3390 MB/s mit 32. Die 0,9-1,2 GB/s aus `optimize.md:99` waren also
Page-Cache-Reuse; die Platte kann mit diesem Muster rund **3 GB/s**. Der Hebel liegt damit
beim Mechanismus, nicht bei der Hardware: der Ist-Pfad (`madvise(MADV_WILLNEED)` in
128-kB-Fenstern plus Touch, `ggml/src/ggml-cpu/ggml-cpu.c:1779`) kommt bei demselben Muster
auf 1426-1750 MB/s, `pread` in einen privaten Puffer auf 2984 MB/s (8 Threads). In situ
liefert Ling 0,85 GB/s - also rund ein Drittel der Faehigkeit des eigenen Mechanismus.

### 3.2 Prefetch mit Slot-Reservierung und Lease statt reiner madvise-Hinweise

TensorFold: `zig/src/core/fsr/arena.zig:206-235` (`forecast`), `:178-204` (`demand`, wartet nur auf
den laufenden Read, wenn die Union genau diesen Experten braucht), `zig/src/core/fsr/commands.zig:53-64`
(`overlap`: Command-Buffer-Overlap, damit der CPU-Leser im Hintergrund laeuft, waehrend die GPU
rechnet). Beleg: `CHANGELOG.md:108` - "a request no longer waits for a forecast read it doesn't need,
and a forecast that never started keeps the expert it displaced: about **8 % faster at 35 GiB and 4 %
at 17 GiB**".

Der Fork hat den Praediktor, aber dessen einziger Konsument ist ein `madvise(MADV_WILLNEED)`-Hinweis
(`src/llama-expert-tier.cpp:2847`), und der `madvise`-Hinweis faultet nur - er liest nicht.

### 3.3 Union pro Layer statt Union pro Zeile

TensorFold: `zig/src/families/flashnext/fsr_runtime.zig:92-131`. Die Union ist **pro Layer ueber alle
Zeilen des Ausfuehrungsfensters** (`layer * 512 + expert_id`, `:117`), nicht token-uebergreifend. Der
Prompt-Pfad laedt pro Chunk die komplette Union (`:134-147`).

**Luecke im Fork:** `src/llama-expert-tier.cpp:2833-2848` praediziert top_m **je Zeile** und
madisiert **je Zeile**. Die `mincore`-Pruefung dedupliziert immerhin ueber bereits residente Slabs,
aber nicht ueber das Fenster.

### 3.4 Load-Time Prefetch: waehrend der Loader packt, wird schon gelesen

TensorFold: `docs/recipes/cuda-load-prefetch.md:3` (`TF_LOAD_PREFETCH`, 0..16 Reader-Threads),
`CHANGELOG.md:804` - GLM laedt in **155 s statt 327 s** (TP=2) bzw. **123 s statt 268 s** (TP=3).
`CHANGELOG.md:330` - Experten laden **12 % schneller** mit direktem Laden: "the next chunk is read
while the current one packs, and the store switches shards without reopening them".

**Luecke im Fork:** `start-ling-3-flash.sh:38` setzt `LLAMA_MMAP_PREFETCH=0`. Bei 72,5 GiB und
0,9-1,2 GB/s liegt der reine Leseboden bei 60-80 s. Effekt: **Startzeit**, nicht Decode.

### 3.5 MADV_RANDOM pro Region statt global

TensorFold: `CHANGELOG.md:242` - "advises random access over the n-gram table's mapped tensors, so a
gather's scattered single-row reads fault only their own pages instead of whole readahead windows:
the first request after start decodes at **35.8 tok/s instead of 24.4**, and a cold 7K-token prompt
starts in **28.9 s instead of 48.4 s** on a Strix Halo". `CHANGELOG.md:422` - Page-Cache faellt von
31,5 auf 2,8 GiB, kalter Prompt **28,55 s statt 48,44 s**.

Der Fork setzt `POSIX_MADV_RANDOM` global fuer die ganze Datei (`src/llama-mmap.cpp:514`). Das
uebertragbare Muster ist die **regionale** Trennung: Scatter-Gather-Regionen (N-Gram-Tabelle,
PLE) random, sequentielle Regionen (Expert-Slices) normal.

### 3.6 Checkpoint nach dem Laden aus dem Page-Cache werfen

TensorFold: `CHANGELOG.md:1488` - `madvise` + `fadvise DONTNEED`, 20,1 GiB in 0,26 s.
`CHANGELOG.md:536` - "about 36 GiB of cached shards had left under 1 GiB free, and long prompts slowed
to about **0.73x** from some point in a server's life until a restart".

**Fuer den Fork nur eingeschraenkt uebertragbar:** Bei Ling ist der Page-Cache der eigentliche
Hot-Store der Experten - ein globales DONTNEED waere katastrophal. Uebertragbar ist die *Diagnose*:
der Page-Cache konkurriert mit dem Modell-Page-Cache, und `optimize.md` (Systemebene, 2026-10-31)
notiert bereits, dass `swappiness 180` mit zram "eher fremde Heaps als den Modell-Page-Cache"
verdraengt. Die Formulierung lautet also: **welche Region des GGUF soll im Page-Cache liegen und
welche nicht** - explizit entscheiden statt es der Kernel entscheiden zu lassen.

---

## 4. Konzepte mit Substanz fuer Speculative Decoding

### 4.1 Gelerntes Zeilenkosten-Modell plus Kopie-gegen-Head-Allokator

TensorFold: `zig/src/core/lanes/row_cost.zig:1` - "What a shared round's rows cost past the cost table
on this machine, learned from the rounds it ran"; Least-Squares-Steigung ueber Breiten (`:29-68`),
ein Fit wird verworfen, wenn er nicht das Doppelte der Streuung uebersteigt (`:67`).
`zig/src/core/lanes/allocate.zig:78-82` - Zielgroesse `rate = expected / (cost + overhead_ms)`.
`zig/src/core/lanes/windows.zig:243-263` - Kopie und Head-Proposal landen im **selben** Allokator;
der Allokator kann einem Stream null Zeilen geben.

Beleg: `CHANGELOG.md:206` - "learns what a shared round's row costs on that chip instead of pricing it
from an M5 Ultra's table ... about **14 % more decode throughput at four streams**".

**Bezug zum Fork:** `optimize.md:111` - MTP n=1 ist mit 7,59-7,80 t/s schneller, aber die
Greedy-Ausgabe **weicht ab**; n=2 ist langsamer. Das ist ein Verifikationsproblem, kein
Throughputproblem. Ein Kostenmodell erklaert, **wann** ein Verify-Fenster sich lohnt, und macht den
Pfad erst verifizierbar. Wichtig: TensorFold warnt selbst, dass Akzeptanz allein nicht reicht -
"Spark's copy experiment lost 3.8 to 22.7 % in long-context cells" (`docs/recipes/cuda-lane-copies.md:33`).

---

## 5. Was der Fork bereits hat (Duplikate, nicht neu umsetzen)

| Konzept | TensorFold | Fork |
|---|---|---|
| Hot/Cold-Split der Experten | `fsr/cache.zig` | generisch in `src/llama-graph.cpp:2304-2317, 2464` |
| Cross-Layer-Router-Praediktor | `fsr_runtime.zig:97-100` (Router der naechsten Schicht auf dem aktuellen Hidden-State) | `src/llama-expert-lookahead.cpp`, verdrahtet in `src/models/kolibri1.cpp:126-134` und `src/models/qwen35moe.cpp:219-228` |
| Page-Cache-Prefetch | `arena.zig:206-235` | `src/llama-expert-tier.cpp:2806-2876` (`madvise(MADV_WILLNEED)`, 128-KiB-Chunks) |
| Warm-Slot-Tier mit LRU und async H2D | `cache.zig` | `src/llama-expert-cache.h:52-113`, `src/llama-expert-tier.h:51-63` |
| Graph-Cache je Form | `zig/src/cuda/graph_cache.zig:30-67` | `LLAMA_GRAPH_SMALL_TOKENS` / `_SMALL_SHAPES` (aus dem Avalon-Review, `optimize.md`) |
| Keyed CUDA-Graph-Cache mit Memory-Floor | `graph_cache.zig:78-237` | **teilweise** - der Floor fehlt, der Allokator-Hysterese ersetzt ihn (`optimize.md`) |
| EXL3-Kernels | `CHANGELOG.md:1467` (Prefill +38-42 %) | vorhanden: `ggml/src/ggml-cuda/exl3*.cuh` |
| Direct I/O | `core/direct_io.zig` | `src/llama-mmap.cpp:190-203`, aber nur im Loader-Pfad |
| RoPE-Fusion | `qwen36_mtp.cu:79-131` | teilweise (Mage-VL-Graph), nicht im LLM-Pfad |
| Norm+RoPE in einem Kernel | `CHANGELOG.md:796` (ein Sync statt zwei) | `ggml/src/ggml-cuda/norm.cu:180` - Cross-Warp-Reduktion ueber Shared Memory |

---

## 6. Fund im eigenen Fork: das Vulkan-Backend schreibt Debug-Zeilen pro Knoten

**Korrektur vom 2026-11-06: Der Fund war eine Fehlinterpretation. Es gibt keinen Zeitverlust.**
Die Belege stehen in Abschnitt 9 (B1); die urspruengliche Schaetzung ist widerlegt.

`ggml/src/ggml-vulkan/ggml-vulkan.cpp` enthaelt 164 `std::cerr`-Ausgaben (das CUDA-Backend
`ggml/src/ggml-cuda/ggml-cuda.cu`: 0). Zwei davon sitzen in den Hauptpfaden:

- `ggml-vulkan.cpp:8938-8941` in `ggml_vk_mul_mat_q_f16`
- `ggml-vulkan.cpp:9843-9846` in `ggml_vk_mul_mat_id_q_f16`

Alle 164 stehen aber **als Argument des `VK_LOG_DEBUG`-Makros**, nicht als eigener Aufruf. Die
Klammerbilanz des Makroarguments laeuft ueber die vier Zeilen hinweg von +1 auf -1 - deshalb
sehen die Zeilen in einer Zeilensuche wie unbedingte Aufrufe aus. Ohne
`-DGGML_VULKAN_DEBUG` ist der ganze Block `((void) 0)`.

**Konsequenz:** Die urspruengliche Hochrechnung (24-66 ms pro Token, 18-50 % der
Vega-Tokenzeit) ist hinfaellig, und an den Vulkan-Zeilen wird nichts geaendert. Wer
`GGML_VULKAN_DEBUG` einschaltet, will diese Zeilen sehen.

---

## 7. Priorisierte Naechste Schritte

**Stand nach dem Benchmark vom 2026-11-06 (Abschnitt 9).** Die Spalte "Ergebnis" ersetzt die
urspruengliche Erwartung, wo der Benchmark die Sache entschieden hat.

| Prio | Schritt | Aufwand | Ergebnis |
|---|---|---|---|
| P0 | Vulkan-Debug-Ausgaben raus (Abschnitt 6) | gering | **entfaellt** - die Zeilen sind `VK_LOG_DEBUG`-Argumente und ohne `-DGGML_VULKAN_DEBUG` nicht kompiliert (B1) |
| P0 | FA-Arbeitsaufteilung nach KV-Laenge (2.2) | gering | **ohne lokalen Effekt** - nur bei `gqa_ratio > 8` sinnvoll, kein lokales Modell erfuellt das bei GPU-Last; Patch liegt hinter Env (B2) |
| P1 | CLOCK-Use-Count-**Simulation** (2.1) | gering, analytisch | **-1,6 bis -5,3 %** statt 11 % - zu schwach, nicht implementiert (B4) |
| P1 | Echte SSD-Bandbreite S messen (3.1) | gering | Zahlenbasis; derzeit misst der Fork Page-Cache-Reuse |
| P1 | `bailingmoe3` (Ling) an Lookahead + Page-Prefetch | gering | umgesetzt: gruppenrouting-konformer Praediktor in `src/models/bailingmoe3.cpp`, Ergebnis in 9.3 |
| P2 | Radix-Select-Top-k fuer den MoE-Router (2.3) | mittel | TensorFold: 4,37 -> 2,74 us. Offen: der Fork routet bereits ueber `ggml_top_k` (Index-Op, kein Argsort) |
| P2 | Zeilenkosten-Modell fuer das MTP-Verify-Fenster (4.1) | mittel | TensorFold: +14 % Decode bei 4 Streams; im Fork erst zur Verifikation von MTP n=1 noetig |
| P2 | Load-Time Prefetch (3.4) | mittel | Startzeit: TensorFold 155 s statt 327 s |
| P3 | GPU-seitiger MoE-Plan (2.4) | hoch | Hebel unklar, kein Zahlenbeleg |
| P3 | Grouped Launches (2.5) | hoch | TensorFold Prefill +7,8-10,8 %, bricht die ggml-Graph-Abstraktion |

---

## 8. Grenzen dieser Analyse

1. **Keine Messung von mir auf Zielhardware.** Alle TensorFold-Zahlen stammen aus
   `CHANGELOG.md` und `docs/`; alle Fork-Zahlen aus `optimize.md` und den Startskripten.
   Gebaut oder gemessen wurde nichts.
2. **Die Uebertragbarkeit der Prozentwerte ist nicht belegt.** TensorFold faehrt Metal auf einem
   M3 Ultra mit residentem Pool, der Fork faehrt CUDA auf einer 1660 Ti mit 6 GB VRAM, 32 GB RAM
   und rechnet Kalt-Experten auf der CPU aus dem mmap. Die Prozentwerte sind
   Obergrenzenindikatoren fuer die **Mechanismen**, keine Prognosen.
3. **Die Vulkan-Seite von TensorFold ist ohne Wert.** Der Stub laeuft kein einziges Modell. Wer
   dort Ideen sucht, sucht ins Leere.
4. **Keine der Zahlen stammt von einer Vega.** Alle AMD-Messungen sind gfx1030/gfx1100/gfx1151/gfx1201.
   Wellenbreite (32), LDS-Groesse (64 KiB/WGP) und Registerbudget (256 VGPR/SIMD) der Vega
   (gfx906, RDNA1) unterscheiden sich. `zig/kernels/hip/common/arch.hpp:27-28` verweigert RDNA1
   explizit, und gfx906 fehlt in allen Tuning-Tabellen.
5. **`CHANGELOG.md:676`** (der groesste Prefill-Gewinn, -32 % durch Spill-Vermeidung im K-Loop)
   ist keiner konkreten Datei zugeordnet - vor einer Portierung zuerst lokalisieren.
6. **Der Fork-Zustand ist ein Arbeitsbaum mit offenen Aenderungen** (`git status`: `src/llama-model.cpp`,
   `src/models/mellum.cpp`, `optimize.md` u.a.). Zeilenangaben zu `llama-expert-tier.cpp` und
   `llama-graph.cpp` beziehen sich auf diesen Stand.

**Stand 2026-11-06, nach dem Benchmark (Abschnitt 9):** Punkt 1 gilt nur noch fuer die
TensorFold-Zahlen. Der Fork ist gebaut und gemessen; Punkt 6 gilt weiterhin.

---

## 9. Benchmark vom 2026-11-06: die vier Ideen implementiert und gemessen

Messdisziplin wie in `optimize.md`: A/B/A/B ohne Cache-Leerung zwischen den Laeufen, gleiche
Prompts und Seeds, Greedy-Ausgabe per SHA verglichen, Rohdaten unter
`benchmark-results/tensorfold-review-20261106/`. `agollm` war die ganze Zeit gestoppt
(`systemctl stop agollm.service`, VRAM 1 MiB zu Beginn, `is-active: inactive`).

### 9.1 B1: Vulkan-Debug-Ausgaben - widerlegt, nichts geaendert

Die These aus Abschnitt 6 war: 164 `std::cerr` im Vulkan-Backend kosten pro Decode-Token
24-66 ms. Vier unabhaengige Pruefungen:

| Pruefung | Ergebnis |
|---|---|
| Klammerbilanz der Fundstellen | Argument des `VK_LOG_DEBUG`-Makros, Bilanz +1 -> -1 ueber vier Zeilen |
| Praeprozessor-Output ohne `-DGGML_VULKAN_DEBUG` | nur `((void) 0)`, keine Format-Strings |
| Format-Strings im gebauten Binary (`strings`) | 0 Treffer |
| Debug-Zeilen im Baseline-Bench ueber Vulkan | 0 |

**Ergebnis: Die These ist falsch.** Ohne `GGML_VULKAN_DEBUG` wird der Block nie kompiliert.
An den Vulkan-Zeilen wurde nichts geaendert; sie sind in diesem Zustand korrekt.

### 9.2 B2: FA-Arbeitsaufteilung nach KV-Laenge - Mechanik geklaert, kein lokales Messfahrzeug

**Was `ncols2` wirklich ist.** In der FA-MMA-Variante ist `ncols2` die Zahl der **Query-Koepfe,
die sich einen KV-Kopf teilen** (`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1224`:
`zt_gqa*ncols2 + c < gqa_ratio`). Das K/V wird damit `gqa_ratio/ncols2` Mal gelesen. Die
TensorFold-Idee "breitere Kachel bei langem KV" ist genau: `ncols2` erhoehen, damit das K/V
weniger oft gelesen wird.

**Wo der Hebel liegt.** Der generische Upstream-Pfad waehlt `ncols2 = min(gqa_ratio, 8)`
(`ggml/src/ggml-cuda/fattn.cu:163-179`), der MLA-Pfad (DKQ=576) bei 16 (`:304-307`). Ein Hebel
existiert also nur bei `gqa_ratio > 8` (bzw. > 16 im MLA-Fall). Genau dort hat Upstream eine
Sonderbehandlung fuer GLM 4.7 Flash (`gqa_ratio == 20`, `fattn.cu:290-301`).

**Alle Modelle im lokalen Zoo** (`tools/gguf_attn_scan.py`, liest nur den GGUF-Header):

| Modell | Arch | n_head | n_head_kv | GQA-Rate | head_dim | Hebel |
|---|---|---:|---:|---:|---:|---|
| Ling-3.0-flash | bailingmoe3 | 32 | 1 | **32** | 576 | ja (MLA, 7 Vollattention-Layer) |
| Ling-3.0-tiny | bailingmoe3 | 16 | 1 | 16 | 576 | nein (16 = Deckel) |
| Qwen3.6-35B-A3B | qwen35moe | 16 | 2 | 8 | 256 | nein (8 = Deckel) |
| Hy-MT2-30B | hy_v3 | 32 | 4 | 8 | 128 | nein |
| MiniCPM5-2B | llama | 16 | 2 | 8 | 128 | nein |
| Qwen3-1.7B | qwen3 | 16 | 8 | 2 | 128 | nein |
| Qwen3.8B / Mage-VL | qwen3 | 32 | 8 | 4 | 128 | nein |

Nur Ling-3.0-flash hat GQA-Rate 32 - und das Modell ist SSD-gebunden: waehrend des Decodes
laeuft die GPU mit **4-7 % Auslastung** bei 5514 MiB Hot-Set. Der FA-Kernel ist dort kein
Engpass, der Datentransport ist es.

**Implementiert (hinter Env-Schaltern, Standardverhalten unveraendert):**

- `ggml/src/ggml-cuda/fattn.cu`: `ggml_cuda_fa_ncols2()` (Env `GGML_CUDA_FA_NCOLS2`, 0 = auto,
  8 = aus, 16/32 = erzwingen) und `ggml_cuda_fa_kv_wide_threshold()` (Env
  `GGML_CUDA_FA_KV_WIDE_THRESHOLD`, Default 16384).
- Der generische Pfad (DKQ=128) benutzt die breite Kachel nur, wenn `gqa_ratio > 8`,
  `ncols2 <= min(gqa_ratio, 32)`, `use_gqa_opt` und das KV ueber der Schwelle liegt.
- Der MLA-Pfad (DKQ=576) erweitert die Upstream-Sonderbehandlung von `gqa_ratio == 20` auf alle
  Raten > 16.
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`: DKQ=128 von der
  `ncols2 in (16, 32)`-Sperre befreit -> 5 neue Instanzen
  (`fattn-mma-f16-instance-ncols1_{1,2}-ncols2_32.cu`, `..._ncols1_{1,2,4}-ncols2_16.cu`).

**Wichtig: die erste Fassung des Patches war falsch.** Sie verbreiterte die Kachel auch bei
`gqa_ratio = 2` (Qwen3-1.7B) - das sind dann 16 maskierte GQA-Slots je Block, also reine
Scheinarbeit. Der Guard `ncols2 <= min(gqa_ratio, 32)` kam erst nach der Analyse der
`ncols2`-Semantik dazu.

**Messung (Prefill, `llama-bench`, KV 20480, Qwen3-1.7B, `fa-tile-sweep/`):**

| `GGML_CUDA_FA_NCOLS2` | Prefill t/s |
|---|---|
| auto | 903,92 |
| 16 | 907,75 |
| 8 (alt) | 905,00 |

Alle drei innerhalb 0,4 % - erwartbar, denn der Prefill laeuft ueber den Tile-Pfad mit
`Q->ne[1] > 4`, die Verbreiterung greift dort nicht.

**Messfehler, der beim ersten Versuch passiert ist:** `llama-bench` fahrt Prefill und Decode
als getrennte Tests; die `tg`-Zeile startet bei `n_prompt = 0`, also bei KV = 32. Die
Decode-Werte 168,05 / 168,12 / 168,82 t/s (+-0,4 %) sagen nichts ueber den langen KV aus.
Fuer den Decode mit langem KV braucht es den Server (`tools/bench_fa_longkv.py`); der erste
Versuch scheiterte an HTTP 400, weil der Prompt 69.456 Tokens hatte statt 20.000 (BPE splittet
jedes Wort in ~3,5 Tokens) - jetzt kalibriert das Harness ueber `/tokenize`.

**Ergebnis: kein belegter Effekt, weil kein lokales Modell die Bedingung erfuellt.** Der Patch
bleibt hinter Env-Schaltern und aendert das Default-Verhalten nicht. Er ist die Generalisierung
der Upstream-Sonderbehandlung fuer GLM 4.7 auf alle GQA-Raten > 16; messbar waere er erst auf
einem Modell mit GQA-Rate > 8, das GPU-gebunden laeuft.

**Nebenbefund:** `llama-cli` haengt in diesem Fork (23 Threads, 100 % CPU, 0 % GPU) auch bei
`llama-cli -p "hi"`. `llama-bench` und `llama-server` laufen. Ursache nicht geklaert (kein
gdb/perf installiert); der Umweg ueber den Server war der Grund fuer das
`bench_fa_longkv.py`-Harness.

### 9.3 B3: bailingmoe3 (Ling) an Lookahead + Page-Prefetch - umgesetzt, die Wanduhr raehrt sich nicht

Der Fork hat den Cross-Layer-Praediktor und den Page-Prefetch bereits fuer `kolibri1`
(`src/models/kolibri1.cpp:118-152`, `:212-220`) und `qwen35moe`. In
`src/models/bailingmoe3.cpp` (Ling) war **keine** Verdrahtung.

**Umgesetzt in `src/models/bailingmoe3.cpp`:**

- Lookahead-Praediktor und Page-Prefetch nach dem Muster von `kolibri1.cpp`. Der Praediktor
  legt den Router der Zielschicht (`ffn_gate_inp` plus `ffn_exp_probs_b`) auf den
  Residual-Stream der aktuellen Schicht.
- **Lings Router ist gruppiert**: 512 Experten in 8 Gruppen, Top-2 je Gruppe, die besten
  4 Gruppen, daraus Top-8 Experten (`src/llama-graph.cpp:2193-2229`). Ein globales Top-k
  waere die falsche Vorhersage, es naeme Experten aus Gruppen, die der Router verwirft.
  Der Praediktor bildet die Gruppenauswahl deshalb exakt nach (`ggml_top_k` je Gruppe,
  Gruppensumme, `ggml_top_k` der Gruppen, Maske, `ggml_top_k` der Experten).
- Die ersten zwei Schichten von Ling sind dense (`leading_dense_block_count = 2`) und haben
  keinen Router - fuer sie wird nichts vorhergesagt (sonst Null-Tensor im Router).
- Trace-Ausgaben (vorhergesagte und tatsaechliche Experten) wie in `kolibri1.cpp`, damit
  die Praediktor-Guete messbar ist.

**Praediktor-Guete (Trace `ling-ab/trace/ling-2.lookahead.json`, 2520 Records = 60 Tokens
x 42 Schichten):**

| Praediktionsmenge | Top-1-Treffer | Recall | Kalt-Recall | nuetzliche Kalt-Treffer | Fehlalarme |
|---|---:|---:|---:|---:|---:|
| 8 | 75,2 % | 77,9 % | 78,1 % | 13.282 | 4.022 |
| 12 | 75,2 % | 87,1 % | 87,4 % | 14.853 | 11.536 |
| 16 | 75,2 % | 89,6 % | 89,8 % | 15.276 | 20.399 |

Der Praediktor traegt also auch bei Lings gruppiertem Router (Referenz Qwen3.6-35B-A3B:
73,3 % Top-1, 86,1 % Recall bei 16).

**Messung (Ling-3.0-flash, agollm-Rezept, A/B/A/B ohne Cache-Leerung, ctx 32768, 3,9k-Token-Prompt
plus go/code/essay mit je 256 Greedy-Tokens; `ling-ab/`). Drei Runden, Runde 3 in umgekehrter
Arm-Reihenfolge, damit die Page-Cache-Erwarmung nicht mit dem Arm zusammenfaellt:**

| Arm | R1 t/s | R2 t/s | R3 t/s | gepaart gegen base | Slab-Faults (fehlende Seiten) |
|---|---:|---:|---:|---:|---|
| base | 5,75 | 6,02 | 6,42 | - | 32,90 % / 38,08 % / 32,87 % |
| prefetch (`LLAMA_EXPERT_PAGE_PREFETCH=8`) | 5,52 | 6,19 | 6,47 | **-4,0 % / +2,8 % / +0,8 %** (Mittel -0,2 %) | **21,88 % / 21,13 % / 21,56 %** |
| fa32 (`GGML_CUDA_FA_NCOLS2=32`) | 5,83 | - | 6,58 | +1,3 % / +2,4 % (Mittel +1,9 %) | 34,43 % / - / 33,29 % |

Die Rundenvarianz (5,75 bis 6,42 t/s fuer dieselbe Konfiguration) ist groesser als jeder
Armunterschied, deshalb zaehlt nur der gepaarte Vergleich je Runde.

**Token-Gleichheit:** base, prefetch und fa32 erzeugen je Prompt byteweise dieselbe
Greedy-Ausgabe (SHA identisch, `tools/summarize.py`). Der Prefetch aendert die Rechnung
nicht, er aendert nur, wann die Bytes kommen.

**Ergebnis: Der Prefetch funktioniert, bringt aber keine Zeit.**

- Die Zahl der fehlenden Page-Faults sinkt in **jeder** Runde um ein Drittel bis zur Haelfte
  (32,9 % -> 21,9 %, 38,1 % -> 21,1 %, 32,9 % -> 21,6 %), die Fault-Wait im warmen Lauf von
  751 s auf 382 s (-49 %). Die Vorhersage trifft also, und die Daten kommen frueher an.
- Die **Wanduhr bewegt sich nicht**: gepaart -4,0 %, +2,8 %, +0,8 %, im Mittel -0,2 %. Auf dem
  langen Prompt ist der Prefetch in zwei von drei Runden schlechter (4,45 / 5,21 / 5,64 gegen
  5,59 / 5,44 / 5,78).
- Der Grund ist die Art des Engpasses: Ling liest mit 0,9-1,2 GB/s von der SSD, das ist die
  Grenze der Platte. Ein Prefetch holt dieselben Bytes frueher, er erzeugt keine zusaetzliche
  Bandbreite. Aus "warte auf den Page-Fault" wird "warte auf die Bandbreite" - der
  Gesamtfortschritt ist derselbe.
- Der FA-Arm (breitere GQA-Kachel fuer die MLA-Schichten, `GGML_CUDA_FA_NCOLS2=32`) liegt mit
  +1,9 % im Rauschen und aendert die Fault-Rate nicht (33-34 % wie base): bei Ling laeuft die
  GPU mit 4-7 % Auslastung, der FA-Kernel ist nicht der Engpass.

**Konsequenz:** Die Verdrahtung ist implementiert und kann bleiben (der Schalter ist
aus, `LLAMA_EXPERT_PAGE_PREFETCH` steht auf keinem Rezept). Sie ist aber **kein Gewinn** und
wird nicht als solcher behauptet. Der Prefetch lohnt sich erst, wenn die SSD nicht
bandbreiten-, sondern latenzgebunden ist - das ist bei Ling nicht der Fall.

**Methodischer Hinweis:** Das Harness benutzt einen festen Port (18093), es kann immer nur
ein Server laufen. Ein erster Trace-Lauf ist deshalb an einem belegten Port gescheitert
und hat stattdessen einen fremden Server gemessen; die Zahlen wurden verworfen und der Lauf
wiederholt. Ebenfalls verworfen: ein Lauf von `fa32`, der parallel zu einem anderen Server
startete.

### 9.4 B4: CLOCK-Eviction - simuliert, zu schwach, nicht implementiert

CLOCK (nach `zig/src/core/fsr/cache.zig:126-148`) zaehlt Nutzungen bis 3, senkt den Zaehler bei
"Hand"-Zugriffen und verdraengt bei `uses == 0`. Die Simulation lief ueber den vorhandenen
Lookahead-Trace (`benchmark-results/transient-oracle-trace-sm75-20260803T2230Z/LT-run1.lookahead-1.json`:
Qwen3.6-35B-A3B, 40 Layer, 256 Experten, top-8, 4953 Records) und verglich gegen LRU
(`tools/simulate_expert_clock_cache.py`).

| Variante | Misses gegen LRU |
|---|---|
| pro Layer (8 Slots) | -0,4 % bis -2,2 % |
| global (256 Slots) | -1,6 % bis -5,3 % (bester Fall -5,29 %: 9632 -> 9122 Misses) |

TensorFold nennt 11 % weniger Disk-Zugriffe und 8 % weniger Zeit. Die Simulation kommt auf
**weniger als die Haelfte** davon, und das ohne den Aufwand einer CLOCK-Implementierung in
`llama-expert-tier.cpp`. **Nicht implementiert.**

### 9.5 Was daraus folgt

| Idee | Status | Beleg |
|---|---|---|
| B1 Vulkan-Debug-Zeilen | widerlegt, keine Aenderung | 4 Pruefungen, Abschnitt 9.1 |
| B2 FA-Kachelbreite | implementiert hinter Env, kein Effekt messbar | Scan aller Modelle + Prefill-Sweep, Abschnitt 9.2 |
| B3 bailingmoe3 Lookahead + Prefetch | umgesetzt, kein Zeitgewinn (Faults -33 bis -50 %, Wanduhr 0 %) | Ling-A/B + Trace, Abschnitt 9.3 |
| B4 CLOCK-Eviction | simuliert, zu schwach, nicht implementiert | `tools/simulate_expert_clock_cache.py` |

**Fazit aus dem Benchmark:** Die TensorFold-Ideen sind nicht uebertragbar, weil sie
falsch sind, sondern weil sie die **falsche Flaeche** adressieren. Der Fork haengt bei
Ling an der SSD-Bandbreite (0,9-1,2 GB/s) und bei der CPU, nicht an Cache-Politik,
Prefetch-Disziplin oder FA-Kachelbreite. Der eine Mechanismus, der nachweislich
funktioniert (der Page-Prefetch halbiert die Page-Faults), hilft genau deshalb nicht,
weil er Bytes umsortiert, aber keine Bytes erzeugt. Was die Wanduhr bewegen wuerde,
ist mehr SSD-Bandbreite oder weniger Bytes pro Token - beides ist keine
TensorFold-Idee.

## 10. Der Kaltpfad von Ling gemessen: die SSD ist dreimal so schnell wie der Pfad, der sie benutzt

Die Zahlen aus Abschnitt 9 sagen "mehr SSD-Bandbreite", aber nicht, wie viel davon
ueberhaupt erreichbar ist. Das wurde am 2026-11-06 nachgemessen - mit dem
Cache-Bypass, den TensorFold in 3.1 zur Pflicht macht.

### 10.1 Das Zugriffsmuster

Ling-3.0-flash: 512 Experten je Layer, 41 MoE-Layer, je Expert drei Matrizen
(`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) mit 512 Experten in der dritten
Dimension (`tools/gguf_tensor_info.py`). Je Expert und Layer sind das

| Tensor | Typ | Dims | Bytes je Expert |
|---|---|---|---|
| `ffn_gate_exps` | Q4_K | [2560, 768, 512] | 1,05 MiB |
| `ffn_up_exps` | Q4_K | [2560, 768, 512] | 1,05 MiB |
| `ffn_down_exps` | Q6_K | [768, 2560, 512] | 1,54 MiB |

Der Stride je Expert liegt bei 1,05-1,54 MiB, also **unter** der 8-MiB-Grenze des
Fault-Pfads (`ggml/src/ggml-cpu/ggml-cpu.c:1782`) - der Readahead greift also ueberhaupt.
Die Bloecke liegen verstreut ueber 36,9 GiB je Split: das ist kein sequenzieller Strom,
sondern viele kleine Bloecke an zufaelligen Offsets.

### 10.2 Was die SSD kann

`tools/measure_nvme_random.py` liest verstreute 1-MiB-Bloecke, der Page-Cache des
Testbereichs wird vorher per `posix_fadvise(DONTNEED)` verworfen:

| Threads | Bandbreite |
|---|---|
| 1 | 887 MB/s |
| 8 | **2948 MB/s** |
| 16 | 3304 MB/s |
| 32 | 3390 MB/s |

Der Zufalls-Controller der K3 ist also **kein** Engpass, sobald genug Streams offen sind.
Die alte Notiz in `optimize.md:228` ("zufaellig gelesen nur 0,9-1,0 GB/s, schon ab QD 4")
gilt fuer 4-KiB-Zufallsreads, nicht fuer 1-MiB-Bloecke.

### 10.3 Was der Fault-Pfad daraus macht

`tools/bench_fault_random.py` baut den Ist-Pfad nach (madvise plus Touch) und stellt ihn
gegen `pread` in einen privaten Puffer:

| Variante (8 Threads, verstreute 1-MiB-Bloecke) | Bandbreite |
|---|---|
| madvise 128-kB-Fenster plus Touch (Ist-Zustand) | 1750 MB/s |
| madvise 1-MiB-Fenster plus Touch | 732 MB/s |
| `pread` in privaten Puffer | **2984 MB/s** |

Die 1-MiB-Fenster sind schlechter, weil der Kernel je `madvise`-Aufruf nur **ein**
Readahead-Fenster liest. Deshalb ist die Fenstergroesse selbst ein Hebel -
`tools/bench_fault_chunks.py` (8 Threads, verstreute Bloecke):

| Fenster | 32 kB | 64 kB | 128 kB | 256 kB | 512 kB | 1 MiB |
|---|---|---|---|---|---|---|
| MB/s | 981 | 1330 | **1426** | 580 | 543 | 653 |

**Die Konstante 128 kB im Code ist richtig** - sie ist das Maximum der Kurve. Eine
Aenderung daran waere eine Verschlechterung.

### 10.4 Was in situ herauskommt

Im echten Lauf liest die SSD 0,73-0,85 GB/s bei ~8300 IOPS und ~88 kB mittlerer
Request-Groesse (`runs/base.disk`). Das ist **ein Drittel** der Faehigkeit des eigenen
Mechanismus (1426-1750 MB/s) und ein Viertel dessen, was `pread` in einem privaten
Puffer erreichen wuerde.

Die Phasen-Uhr aus dem Fork (`LLAMA_EXPERT_STATS=1`, `LLAMA_EXPERT_TIMING=1`) schliesst
die Luecke (`runs/diag/server.log`, 40 Layer, 836 Laeufe je Layer):

| Je Token ueber 40 Layer | ms |
|---|---|
| kalte Experten gesamt | 138,4 |
| davon Rechnen (Gate/Up + Aktivierung + Down) | 33,6 |
| davon Fault-Wait und Barrieren | 104,8 |

Die gemessene Tokenzeit liegt bei 136 ms - die kalten Experten **sind** die Tokenzeit.
Und sie sind zu drei Vierteln Warten, nicht Rechnen.

### 10.5 Was daraus folgt

Die Reihenfolge der Hebel ist damit klar und alle drei sind messbar:

1. **Mehr Parallelitaet im Fault-Pfad.** Der Pfad benutzt den Rechen-Threadpool (8
   Threads) und arbeitet streng seriell: erst alle Faultholen, dann rechnen. Der
   Readahead-Vorgang und die Rechnung eines Layers ueberlappen nie, die SSD steht in
   der Rechenphase leer. Ein eigener Lese-Pool (oder das Uebergeben der Faultholen an
   einen Pool mit mehr Threads als der Rechner) hebt die Parallelitaet, ohne die
   Mechanik zu aendern.
2. **`pread` statt `madvise`, mit privatem Puffer.** Der Compute-Pfad muesste dann aus
   dem Puffer lesen statt aus dem Mapping - das ist der Eingriff in
   `ggml-cpu.c`, der die 1426 MB/s auf ~3000 MB/s heben wuerde.
3. **Weniger Bytes pro Token.** Der Hot-Set deckt 23 % der Wahlen (S=12, 136 MiB je
   Slot, `optimize.md:112`); der VRAM gibt nur 2,13 GB frei, davon 500 MiB Reserve.
   S auf 16 zu heben braucht ~540 MiB mehr VRAM, als der Kontext
   (`LLAMA_ARG_CTX_SIZE=32768`, MLA-KV rund 530 MiB) freigibt, wenn man ihn auf 16384
   halbiert.

Die Reihenfolge ist wichtig: Punkt 1 ist billig, Punkt 2 ist teuer, Punkt 3 ist
VRAM-gebunden. Und Punkt 1 ist begrenzt durch Punkt 2s Faehigkeitsgrenze - die
beiden gehoeren zusammen.

### 10.6 Die naheliegenden Verdaechtigen bringen nichts

Weil der Decode an der SSD-Wartezeit haengt, liegen die naheliegenden Verdaechtigen
nicht auf der SSD-Seite, sondern beim Rechenpfad: die vier nie dokumentierten
CPU-Knobs des Forks (`LLAMA_EXPERT_CPU_ACT_PARALLEL`, `..._DOWN_PREFETCH`,
`..._REUSE_ROWS`, `..._MULTI_ROW`) und `LLAMA_ARG_THREADS=16`. Gepaart gemessen,
zwei Runden, Arm-Reihenfolge in Runde 2 umgekehrt, Greedy-Ausgabe per SHA in allen
Armen identisch (`benchmark-results/tensorfold-review-20261106/ling-ab/`,
`tools/confirm2.sh`, `tools/summarize_paired.py`):

| Arm | Runde 1 | Runde 2 | Mittel | Siege |
|---|---|---|---|---|
| `LLAMA_EXPERT_CPU_ACT_PARALLEL=1` | -3,03 % | -1,05 % | -2,04 % | 0/2 |
| `LLAMA_EXPERT_CPU_DOWN_PREFETCH=4` | +0,09 % | +0,04 % | +0,06 % | 2/2 |
| `LLAMA_ARG_THREADS=16` | -53,71 % | -48,26 % | -50,98 % | 0/2 |

`ACT_PARALLEL` senkt die Fault-Wartezeit messbar (99,3 s und 94,1 s gegen 123,7 s
und 92,8 s), die Wanduhr aber nicht - die Wartezeit wandert nur in die Rechenphase.
`REUSE_ROWS` und `MULTI_ROW` sind im Decode No-ops, weil jeder Expert-Slot dort
genau eine Spalte hat. `THREADS=16` kostet die Haelfte: 8 physische Kerne, SMT
verdoppelt die Faehthoehe und halbiert den Durchsatz je Thread, der Prefill faellt
von rund 8 auf 4,3 t/s.

**Damit ist die Rechnung vollstaendig:** die Rechenpfad-Knobs sind eskaliert und
kommen nicht ueber die Rundenvarianz (7,04-7,26 t/s) hinaus. Was fehlt, ist
Bandbreite - und die ist eine Eigenschaft des Fault-Mechanismus, nicht der Platte
und nicht der Konfiguration.

### 10.7 Die zweite Runde: der Decode verliert an den Swap, nicht an die SSD

Die Formulierung in 10.5 ("es fehlt Bandbreite") war die halbe Wahrheit. Drei
Messungen vom 2026-11-06 (zweiter Teil) drehen die Reihenfolge der Hebel um.

**Erstens: die Reihenfolge von Hinweis und Zugriff ist der Hebel, nicht die
Request-Groesse.** `tools/bench_fault_interleave.py` baut den Ist-Pfad nach und
stellt vier Varianten gegenueber (8 Threads, 384 verstreute 1-MiB-Bloecke, Span
8 GiB, Page-Cache vorher verworfen):

| Variante | Bandbreite |
|---|---|
| interleave (Hinweis und Zugriff je Block, so wie der Code heute arbeitet) | 916 MB/s |
| batched (alle Hinweise, dann alle Zugriffe) | 2048 MB/s |
| fadvise (Hinweis und Zugriff je Block, aber ueber die Datei) | 2974 MB/s |
| `pread` in privaten Puffer | 2993 MB/s |

Und die Request-Groesse ist *nicht* die Grenze: `tools/measure_nvme_random.py` mit
8 Threads und verstreuten Bloecken liefert 2152 MB/s bei 64 KiB, 2825 bei 128 KiB,
2300 bei 256 KiB und 2888 bei 1 MiB. Der Auftrag muss vor dem Zugriff stehen,
sonst ist die Warteschlange leer, wenn das Geraet arbeiten soll.

**Zweitens: in situ laeuft der Kernel in einen Reclaim- und Swap-Sturm.**
`tools/vmstat_delta.py` misst die Kernel-Zaehler ueber einen Lauf (128 Token, ein
Prompt, `LLAMA_EXPERT_FAULT_STATS=1`):

| Zaehler | Ling-Lauf | Mikro-Benchmark |
|---|---|---|
| `pgscan_direct` | +2.940.843 (11,5 GB) | 0 |
| `pgsteal_direct` | +910.436 (3,5 GB) | 0 |
| `allocstall_normal` | +8.540 | 0 |
| `pswpout` | +167.628 (654 MB) | 0 |

`vm.swappiness` stand auf 180 (Maximum 200). Der Kernel tauscht die anonymen Seiten
des Servers aus, statt die sauberen Dateiseiten des 72-GiB-Modells zu reclaimed - ein
72-GiB-Modell in 32 GB RAM, 25 GiB davon im Page-Cache. Gepaart gemessen, zwei
Runden, Arm-Reihenfolge in Runde 2 umgekehrt, SHA in allen vier Armen identisch:

| Arm | Runde 1 | Runde 2 | Mittel | Siege |
|---|---|---|---|---|
| `vm.swappiness=0` | +8,47 % | +6,35 % | **+7,41 %** | 2/2 |

Bemerkenswert: der Arm mit `swappiness=0` liest **mehr** von der SSD (22,2 % und
21,8 % fehlende Slabs gegen 18,5 % und 20,3 %) und wartet laenger (93,9 s und 88,3 s
Fault-Wait gegen 69,1 s und 86,5 s) - und ist trotzdem 7 % schneller. Der
Swap-Rauschen kostet mehr als die zusaetzlichen Bytes.

**Drittens: `posix_fadvise` statt `madvise` im Fault-Pfad bringt 3,7 %.** Der Pfad
liegt hinter `LLAMA_EXPERT_FADVISE=1` (`ggml/src/ggml-cpu/ggml-cpu.c:1847` mit dem
Resolver aus `src/llama-expert-tier.cpp`), gepaart gemessen, zwei Runden,
umgekehrte Reihenfolge, SHA in allen vier Armen identisch:

| Arm | Runde 1 | Runde 2 | Mittel | Siege |
|---|---|---|---|---|
| `LLAMA_EXPERT_FADVISE=1` | +4,22 % | +3,23 % | **+3,73 %** | 2/2 |

Die SSD-Rate bleibt dabei gleich (752-784 MB/s in allen vier Laeufen, 84-89 KiB
mittlere Request-Groesse). Der Gewinn ist eine kuerzere Wartezeit, keine hoehere
Bandbreite.

**Viertens: der Hinweis vor dem Zugriff wirkt auf den Mechanismus, nicht auf die
Wanduhr.** Der Layer-voraus-Prefetch (`src/models/bailingmoe3.cpp:420`, jetzt mit
`posix_fadvise` statt `madvise` ueber `prefetch_slab` in `src/llama-expert-tier.cpp`)
fuellt die Warteschlange des Geraets:

| Groesse | `LLAMA_EXPERT_FADVISE=1` | `+ LLAMA_EXPERT_PAGE_PREFETCH=8` |
|---|---|---|
| MEAN tg (Runde 1 / Runde 2) | 6,52 / 7,15 t/s | 7,49 / 7,17 t/s |
| Leserate der SSD | 788 / 806 MB/s | **1100 / 1120 MB/s** |
| Warteschlange (qd) | 6,8 / 6,2 | **18,5 / 20,2** |
| fehlende Slabs | 21,3 / 19,5 % | **7,2 / 7,7 %** |
| Fault-Wait | 95,3 / 75,0 s | **40,1 / 43,5 s** |
| Bytes von der SSD | 130,8 / 124,2 GB | 173,7 / 180,4 GB |

Gepaart: Runde 1 +14,9 %, Runde 2 +0,3 %. Die Mechanik ist eindeutig (dreifache
Warteschlange, 40 % mehr Leserate, dreifache weniger fehlende Slabs), die Zeit nicht:
der Prefetch liest 40 % mehr Bytes, weil die Vorhersage nur 78 % Recall hat, und
tauscht damit Wartezeit gegen Geraetezeit. Die naechste Stellschraube ist die
Praezision der Vorhersage (`top_m`), nicht der Mechanismus.

**Was das fuer die Puffer-Idee heisst.** `tools/measure_ram_bandwidth.c`
(gcc -O2 -pthread) misst die Kopierbandbreite des RAM: 10,3 GB/s mit einem Thread,
13,1 mit vier, 15,2 mit acht. Lings Decode beruehrt je Token 40 Layer x 8 Experten
x 3,64 MiB = 1,19 GB Gewichte. Die Kopie in einen privaten Puffer kostet damit
78 ms je Token, 57 % der heutigen Tokenzeit (138 ms), bevor der schnellere
SSD-Zugriff ueberhaupt wirkt. Der private Puffer ist damit ein Nettoverlust; die
Kopie *ohne* Puffer - `posix_fadvise` auf die Modelldatei - bringt denselben
Page-Cache-Effekt ohne Kopierkosten.

**Die Hebel in der richtigen Reihenfolge:**

1. **Speicherverwaltung des Servers** (`MemorySwapMax=0` in der Service-Unit,
   `vm.swappiness=0` messweise): 7,4 %, eine Zeile, keine Code-Aenderung.
2. **`posix_fadvise` im Fault-Pfad** (`LLAMA_EXPERT_FADVISE=1`): 3,7 %, wenige
   Zeilen im CPU-Pfad plus ein Resolver im Tier-Modul.
3. **Hinweis vor Zugriff** (Layer-voraus-Prefetch ueber `posix_fadvise`, der
   Prefetch liegt in `src/models/bailingmoe3.cpp:420` und benutzt bisher `madvise`):
   der Auftrag steht dann vor dem Zugriff und die Warteschlange ist tief - die
   Mechanik wirkt (qd 6,5 auf 19,5), die Zeit noch nicht, weil der Prefetch 40 %
   mehr Bytes liest. Naechster Schritt ist die Praezision der Vorhersage.
4. **Weniger Bytes** (Hot-Set, VRAM) und **der private Puffer** (`pread`): der
   Puffer ist nach Punkt 1-3 widerlegt, die Hot-Set-Frage bleibt VRAM-gebunden.

Punkt 1 gehoert nicht in diesen Fork, sondern in die Unit von `agollm`; Punkt 2 und
3 sind Commits in diesem Baum.

### 10.8 Die MMQ-Kachel auf Turing war nicht optimal

Der andere schwere Kern - der quantisierte Matmul, 71,9 % der Prefill-Kernelzeit -
wurde im Rahmen dieser Runde ebenfalls angefasst. Der Build laeuft mit
`GGML_CUDA_TURING_MMQ_DP4A=ON` (DP4A-MMQ statt MMA-Datenlayout auf sm_75), die
Kachelkonfiguration kam aus der Pascal-Tabelle: I=64, J bis 64, ohne Stream-K.
Neu ist `ggml/src/ggml-cuda/mmq-config-turing.cuh` mit den Compile-Time-Knobs
`GGML_CUDA_TURING_MMQ_{I,J_MAX,STREAM_K}` (Default = die gemessene beste Kachel,
`ggml/src/ggml-cuda/mmq.cuh:234` bindet die Tabelle ein, `:258` und `:279` nutzen sie
in Host-Auswahl und Kernel). `tools/mmq_tile_sweep.sh` schreibt die Werte um, baut
neu und misst mit llama-bench (Prefill 4096, `-r 3`):

| Konfiguration | MiniCPM5-2B (Q4_K_M) | Intern-Decision-2B (Q8_0) |
|---|---|---|
| I=64, J=64 (bisher) | 1295,9 / 1292,7 t/s | 1986,2 / 1981,5 t/s |
| I=128, J=64 | 1010,3 t/s (-22,0 %) | 2057,3 t/s (+3,6 %) |
| **I=64, J=128** | **1385,2 / 1387,5 t/s (+7,1 %)** | **2184,0 / 2184,7 t/s (+10,1 %)** |
| I=64, J=128, Stream-K | 1378,5 t/s (+6,5 %) | 2261,6 t/s (+14,0 %) |
| I=128, J=64, Stream-K | 1067,2 t/s (-17,7 %) | 2112,4 t/s (+6,4 %) |

Die erste und die letzte Zeile jeder Spalte sind derselbe Aufbau am Anfang und am
Ende der Runde; der Drift liegt bei 0,25 %. Decode bleibt unveraendert. Eine Kachel
mit 128 Spalten laedt das Gewichts-Band halb so oft - der Prefill liest die Gewichte
bandbreiten- und nicht rechengebunden. I=128 verdoppelt dagegen das x-Band der
Kachel: der Block braucht doppelt so viel Shared Memory, es passt nur einer je SM
statt zwei, und die Latenzverdeckung geht verloren (-22 % bei Q4_K). Stream-K ist
typabhaengig und bleibt aus.

Die Ausgabe ist unveraendert: derselbe Prompt mit demselben Seed ergibt bei J=64
und J=128 den SHA `71f5cc0f7ad629fb` (`tools/gpu_output_check.sh`).
