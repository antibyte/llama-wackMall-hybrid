# tandem.cpp (ewhacc) gegen unseren Fork geprueft vom 2026-11-07

**Ergebnis: tandem.cpp ist ein Patch-Set gegen llama.cpp, das MoE-Modelle beschleunigt, die nicht in den VRAM passen. Der Kern-Mechanismus (VRAM-Experten-Cache plus CPU/GPU-Split der Fehltreffer) loest genau das Problem nicht, das wir haben: unser Ling-3.0-flash ist 2,4-mal so gross wie der RAM, und tandem.cpp weist fuer genau diese Groessenordnung selbst eine Verlangsamung aus (DeepSeek-Zeile: 0,87x). Uebertragbar sind vier kleinere Dinge, davon zwei mit messbarem Potenzial fuer unseren CPU-Pfad: der Mehrspalten-Dot-Product mit beliebigen Spalten (tandems `llama_moe_iq`), das Packen der Host-Eingaben in einen einzigen H2D-Kopiervorgang (`GGML_SCHED_BATCH_INPUTS=3`) und der Wegfall des Vorhersage-Pruefens beim Layer-voraus-Prefetch durch eine echte Uebergabe der Gewaehlten Experten (`GGML_OP_MOE_DOORBELL`). Drei Fehler im Patch sind beim Uebernehmen zu beachten: ein 16-MiB-Leak, toter Schnellpfad und ein fehlender Null-Check.**

Quelle: <https://github.com/ewhacc/tandem.cpp>, Commit `e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683` (2026-11-03), Patch `tandem.patch` (8731 Zeilen) gegen llama.cpp `f5222f25c`. Lokal geklont nach `/tmp/tandem.cpp` (nur Lesezugriff, nichts uebernommen).

## 1. Was tandem.cpp macht

- **Experten-Cache im VRAM.** Die Experten liegen im Host-RAM, der Rest des VRAM nimmt einen LRU-Cache der zuletzt benutzten Experten auf (`--moe-cache-mib auto`: der Rest nach Modell, KV und Compute-Puffern, minus `--moe-cache-margin-mib`).
- **Split der Fehltreffer.** Die Experten, die ein Token braucht und die nicht im Cache liegen, werden zwischen CPU und GPU aufgeteilt: die CPU rechnet ihren Anteil, waehrend die GPU die gecachten rechnet; der Rest wird in den Cache hochgeladen. Die Aufteilung folgt der Regel L/(L + k*R) aus Strata: L = gemessene PCIe-Bandbreite, R = gemessene CPU-Rate auf Fehltreffern, k = 1,85.
- **Automatische A/B-Schleuse.** Die CPU-Schicht hat Fixkosten je Layer; wenn wenige Experten fehlen, ist Hochladen schneller. Der Patch schaltet die CPU-Schicht periodisch aus (Probe-Fenster) und behaelt, was schneller war.
- **Host-Vorlesen.** `LLAMA_MOE_PREREAD=1`: der CPU-Pool liest nach seinem Job die vorhergesagten Experten der naechsten Layer, die nicht im Cache liegen, in seinen eigenen Cache - nutzt RAM-Bandbreite, die zwischen den Jobs ungenutzt ist.
- **Eigene CPU-Kernels.** `llama-moe-iq.h`: ein Gewichtsblock wird einmal dekodiert und auf bis zu acht Token-Aktivierungen angewendet (bitidentisch zu ggml `vec_dot`). Dazu ein Q2_0-Schnellpfad mit vier Zeilen gegen eine Aktivierung.
- **Scheduler- und Server-Knobs.** `GGML_SCHED_BATCH_INPUTS` (Host-Eingaben in einen H2D-Vorgang), `GGML_CUDA_Q8_CACHE` (q8_1-Aktivierungen wiederverwenden), `GGML_CUDA_GRAPH_KEY_SHAPE` (ein CUDA-Graph je Graph-Form), `LLAMA_GRAPH_POOL` (Pool aus Scheduler plus Graph fuer Ubatches bis 8 Tokens), `LLAMA_SRV_USER_BREAK` (kein Checkpoint an User-Nachrichten), `LLAMA_MTP_DEFER` (Catch-up-Decode des Drafts auf den naechsten Draft-Schritt legen).

## 2. Warum der Kern-Transfer nicht greift

| | tandem.cpp | unser Setup |
|---|---|---|
| Testmaschine | i5-12400F, 64 GB, RTX 4090, PCIe 4.0 x16 | Ryzen Renoir, 6 Kerne, 30 GB RAM, GTX 1660 Ti, PCIe 3.0 x8 (6,7 GB/s) |
| CPU-Rate auf Experten | 43 bis 78 GB/s RAM, 10 Kerne | 15,2 GB/s RAM, 6 Kerne, unter Last thermisch auf 3,75 GHz begrenzt |
| VRAM | 24 bis 96 GB frei fuer den Cache | 6 GB gesamt, davon 4,7 GB vom GPU-Preload belegt |
| Modell | passt in den RAM (64 GB bei 78 GiB) | Ling 72,5 GiB bei 30 GB RAM: passt nicht |

tandem.cpp nennt die Konstellung "models much larger than RAM" selbst als bekanntes Problem: "the CPU path does not take the misses, so most of the gain is lost there", in der DeepSeek-Zeile 0,87x gegenueber llama.cpp. Genau das ist unsere Lage. Dazu kommt: das Cache-Subsystem braucht freien VRAM fuer Experten - bei uns sind das 1,4 GB, waehrend allein ein Layer von Ling 1,87 GB Experten traegt. Der VRAM-Cache faellt fuer Ling und Kolibri-1 (44 GiB, ebenfalls groesser als der RAM) ersatzlos weg. Fuer die Modelle, die in den VRAM passen (Lythri, Spark, MiniCPM, LFM2.5), ist ein Experten-Cache sinnlos, weil das ganze Modell dort liegt.

Der CPU/GPU-Split hilft nur dort, wo ein Modell in den RAM passt, aber nicht in den VRAM: bei uns Qwen3.6-35B-A3B (22,4 GB). Das ist ein moeglicher Zielfall, aber die Rechnung faellt duenn aus: mit 6,7 GB/s Link und einer CPU, die unter Last auf 3,75 GHz faellt, ist der CPU-Anteil klein, und der Aufwand (ein Cache-Subsystem mit LRU, Split-Logik, Schleuse und Vorlesen) ist ein mehrtaegiges Projekt in Code, den wir nicht haben.

## 3. Uebertragbar, priorisiert

### P1: Mehrspalten-Dot-Product auf beliebige Spalten, auch fuer Q6_K (unser eigener CPU-Pfad)

tandems `llama_moe_iq` dekodiert einen Gewichtsblock einmal und wendet ihn auf bis zu acht Token-Aktivierungen an. Unser Fork hat dasselbe Prinzip schon: `vec_dot_g` in `ggml/src/ggml-cpu/ggml-cpu.c` rechnet eine Gewichtszeile gegen bis zu vier Spalten. Aber zwei Bedingungen schraenken es praktisch ein:

- **Nur aufeinanderfolgende Tokens.** Der Lauf laeuft nur, wenn `next.i2 == rm.i2 + run` gilt, also wenn die Spalten der Expertenmatrix auf fortlaufende Token-Indizes zeigen (Zeile 2594 und 2817 derselben Datei; `matrix_rows` traegt je Spalte `{ id, t }` mit `t` = Token-Index, Zeile 2505). Bei Ling mit 512 Experten und 8 aktiven trifft das fuer 8/512 = 1,6 % der benachbarten Token-Paare zu. Bei 2048 Token je Batch kommen auf einen Experten im Mittel 32 Token, davon entfallen etwa 0,5 Laeufe mit zwei Spalten - der Amortisationsfaktor bleibt bei rund 1. Der Pfad ist praktisch tot. In der Messrunde vom 2026-11-06 galt `LLAMA_EXPERT_CPU_MULTI_ROW` deshalb als No-op, gemessen wurde aber nur im Decode (dort hat jeder Experten-Slot genau eine Spalte), nie im Prefill.
- **Nur Q4_K fuer gate/up, gar nicht fuer down.** `multi_gate_up = multi_row && type_g == GGML_TYPE_Q4_K`, `multi_down = multi_row && type_d == GGML_TYPE_Q5_K`. Ling hat gate/up in Q4_K, down in **Q6_K** (`blk.N.ffn_down_exps.weight`, 787,5 MB je Layer bei 512 Experten, gate und up je 540 MB Q4_K). Die Down-Projektion, also der groesere Teil der Expertengewichte, bekommt den Pfad also nie.

Beide Einschraenkungen sind im eigenen Code aufzuheben: statt der Spaltenfortsetzung eine Liste der Spaltenzeiger uebergeben (so wie tandem es mit `const void * const * y` macht) und den Q6_K-Pfad fuer down ergaenzen. Der Ertrag ist offen, bis der Prefill-Anteil gemessen ist: aus 37 t/s Prefill und 7,5 t/s Decode laesst sich der Prefill-Anteil von rund 20 ms CPU-Rechenzeit je Token hochrechnen (hochgerechnet, nie phasenprofiliert), waehrend der Decode nach eigener Messung bandbreiten-, nicht rechengebunden ist (1,19 GB Gewicht je Token bei 33,6 ms). Der Prefill dagegen liest die Gewichte einmal je Batch und ist damit rechen- und dequantisierungsgebunden - dort duerfte die amortisierte Dekodierung der Zeile den groessten Teil des Effekts liefern.

Vorsicht: Der Gewinn ist nur dann echt, wenn der Prefill nicht ohnehin am SSD haengt. Das ist zuerst zu messen (Phasen-Profiling `LLAMA_EXPERT_TIMING=1` plus `tools/vmstat_delta.py`), sonst baut man eine Aenderung ohne Wirkung.

**Gemessen am 2026-11-07: der Prefill haengt am SSD, der Rechenanteil betraegt 6,5 %.** Von 166 s Wanduhr fuer 5192 Token (31,3 t/s) entfallen 10,7 s auf die CPU-Rechenzeit der kalten MoE-Knoten, davon 2,4 s in den drei gemessenen Phasen. Die SSD lieferte 172 GB bei 890 MB/s durchgehend, 66,8 % der Experten-Slabs fehlten, die Warteschlange stand bei qd 283. **Damit ist P1 fuer den Prefill tot:** der Decode hat je Experten-Slot genau eine Spalte (kein Amortisationsfaktor ueberhaupt), und im Prefill ist der Anteil, den die Mehrspalten-Dekodierung theoretisch einspart, 6,5 % der Zeit. Selbst eine Halbierung der Dekodierkosten braeuchte 3 % Prefill.

**Praezisierung vom selben Tag (Runde 5):** Der Decode laeuft ueber MTP-Spec-Decoding, und im Verify-Schritt verarbeitet der Graph **zwei** Tokens (Draft und Target). Damit hat der Cold-Kern im Decode zwei Spalten je Experten-Slot statt einer - die Amortisationsbedingung von P1 ist im Decode also *nicht* per se erfuellt, sondern nur fuer das Q6_K-Down fehlend. Genau da liegt der Rest: `multi_down = multi_row && type_d == GGML_TYPE_Q5_K` (Zeile 2560 ff.), Ling hat down in Q6_K. **P1 ist damit fuer den Decode wieder offen**, sofern (a) der Q6_K-Pfad fuer `vec_dot_g` ergaenzt wird und (b) die Spaltenfortsetzung (`next.i2 == rm.i2 + run`) im Verify-Schritt zutrifft. Beides ist zu messen, bevor Code entsteht: mit `LLAMA_EXPERT_TIMING=1` am Decode-Arm laesst sich der Rechenanteil der Cold-Knoten (gate_up/activation/down) gegen die Tokenzeit stellen, und die Spaltenzahl laesst sich an den Fault-Statistiken oder einem Debug-Zaehler ablesen. Der Decode-Arm vom 2026-11-07 kam ohne Phasen-Profiling daher, deshalb steht die Zahl noch aus.

### P2: Host-Eingaben in einen einzigen H2D-Vorgang packen (`GGML_SCHED_BATCH_INPUTS=3`)

Modus 3 des Patch sammelt alle Host-Eingaben eines Graphen in einem gepinnten Staging-Puffer (4 MiB, doppelt gepuffert), macht **eine** `cudaMemcpyAsync` und danach **einen** Streu-Kernel statt einer Kopie pro Eingangstensor. Modus 2 und 3 schliessen ausdruecklich "the results of a CPU split" ein - also genau die Ergebnisse, die unser CPU-MoE-Pfad auf dem Host ablegt und die die GPU danach liest.

Unser Fork hat `GGML_SCHED_DEFER_HOST_INPUTS` (Host-Eingaben werden zurueckgestellt), aber keine Buendelung in einen Upload. Der Nutzen ist umso groesser, je mehr Host-Eingaben ein Graph hat; bei unserem CPU-MoE-Pfad ist das pro Layer eine Ergebnis-Tensor plus die Slot-Map. Der Patch ist ein gutes Stueck Arbeit: Scheduler-Aenderung, CUDA-Backend-Hook (`ggml_backend_set_tensors_packed`), Staging-Puffer, Entpack-Kernel. Er ist backend-neutral formuliert, der CUDA-Teil aber CUDA-only. Erwartbar eher Prefill als Decode.

### P3: Die gewaehlten Experten statt einer Vorhersage melden (`GGML_OP_MOE_DOORBELL`)

Unser Layer-voraus-Prefetch (`LLAMA_EXPERT_PAGE_PREFETCH`) nutzt eine **Vorhersage** der naechsten Layer und liegt bei 78 % Recall: 40 % mehr Bytes von der SSD, keine gesicherte Wanduhr. tandem loest das mit einer Uebergabe: eine neue Operation `GGML_OP_MOE_DOORBELL` schreibt die vom Router gewaehlten Experten waehrend des Graphenlaufs in einen Kontrollblock im gepinnten Speicher und laesst den Host-Thread darauf reagieren (Request/Ack-Handshake). Der Host laeuft also mit den **tatsaechlichen** Experten und nicht mit einer Vorhersage.

Das ist fuer unsere SSD-Lage die naheliegendste Verbesserung: keine Fehltreffer, keine zusaetzlichen Bytes, und der Hinweis kommt genau dann, wenn die Layer fertig gerechnet ist. Unser Fork hat bereits eine Host/Device-Kopplung fuer Experten (CUDA expert bridge in `ggml/src/ggml-cuda/ggml-cuda.cu`, Scratch pro Layer), aber keine Uebergabe der Gewaehlten. Die Reihenfolge ist umgekehrt zu tandem: deren Host-Seite liest die IDs aus dem Kontrollblock und laedt die Experten hoch; bei uns muesste der Host die IDs bekommen und den **SSD-Zugriff** vorbereiten (unser Fall ist nicht der Upload, sondern das Lesen). Der Kern ist klein (eine Op, ein Kontrollblock, ein Handshake), die Einpassung in unseren Tier-Code ist der Aufwand.

### P4: Die Aufteilregel L/(L + k*R) und die automatische A/B-Schleuse

Unser Fork hat rund 20 `LLAMA_EXPERT_*`-Schalter, viele davon in der Messrunde als No-op im Decode eingestuft. tandem loest die gleiche Frage mit einer Regel aus zwei gemessenen Groessen (Link-Bandbreite, CPU-Rate) und einer periodischen Probe, die die schnellere Variante behaelt. Das ist kein Code, den man uebernimmt, sondern eine Methode: die Schalterwahl als Messung statt als Vermutung. Bei uns waere der Kandidat "wie gross ist der RAM-Hot-Set S" - S = 12 stammt aus `ling-3-flash-chat.csv`, nicht aus einer Messung.

### Nachrangig, aber billig zu pruefen

- **`GGML_CUDA_GRAPH_KEY_SHAPE`**: ein CUDA-Graph je Graph-Form statt je erstem Knoten-Zeiger, damit Verify-Fenster unterschiedlicher Breite sich nicht gegenseitig verdraengen. Unser Fork hat auf llama-Ebene einen Graph-Cache (`LLAMA_GRAPH_SMALL_TOKENS/SHAPES`), die CUDA-Ebene kennt aber weiterhin nur `nodes[0]` als Schluessel.
- **`LLAMA_GRAPH_POOL`**: Pool aus Scheduler plus Graph fuer Ubatches bis 8 Tokens, gegen den Our-Fork-Graph-Cache vermutlich redundant.
- **`GGML_CUDA_Q8_CACHE`**: q8_1-Aktivierung einmal quantisieren und fuer mehrere quantisierte Matvecs wiederverwenden (16 MiB global je CUDA-Kontext, Invalidierung je Graph-Auswertung). Wirkt nur auf den GPU-Decode mit quantisierten Aktivierungen; unser GPU-Decode ist schnell, der Ling-Decode laeuft ohnehin auf der CPU.
- **`GGML_CUDA_MMVF_MAX=8`**: Vektor-Kernel statt Tile-Kernel fuer Float-Gewichte bis 8 Spalten. Unsere Modelle sind quantisiert, der Hebel ist damit klein.
- **`LLAMA_MTP_DEFER`**: den Catch-up-Decode des MTP-Drafts auf den naechsten Draft-Schritt legen, sodass er sich einen Decode teilt. Relevant fuer unsere MTP-Modelle (SenseNova, Mellum), nicht fuer Ling.

## 4. Was wir schon haben (Ueberschneidung)

| tandem.cpp | unser Fork | Bewertung |
|---|---|---|
| `LLAMA_MOE_TAIL_SKIP` (Experten ab Rang r ueberspringen, approximativ) | `LLAMA_EXPERT_TAIL_SKIP` | vorhanden |
| Vorhersage der naechsten Layer (`LLAMA_MOE_LOOKA`) | `src/models/bailingmoe3.cpp`, `llama-expert-lookahead` | vorhanden, unsere Version berechnet die Vorhersage im Graph |
| Vorlesen der Experten in den Host-Cache (`LLAMA_MOE_PREREAD`) | `LLAMA_EXPERT_PAGE_PREFETCH` (fadvise auf die Datei) | vorhanden, aber am Datei- statt am RAM-Pfad; bei uns liest der Prefetch 40 % mehr Bytes |
| Gepinnter Host-Puffer fuer die Split-Ergebnisse | `GGML_SCHED_DEFER_HOST_INPUTS`, gepinnte H2D-Kopien aus dem 2026-10-07-Stand | teilweise |
| CUDA-Graph je Form | `LLAMA_GRAPH_SMALL_TOKENS/SHAPES` (llama-Ebene) | teilweise, CUDA-Ebene fehlt |
| Host-Schicht fuer Experten (MoE-Cache) | `llama-expert-tier`, `CUDA expert bridge` | eigene Loesung, anderer Zweck |

## 5. Nicht uebertragbar, mit Begruendung

- **VRAM-Experten-Cache**: kein Platz (1,4 GB frei), und die Modelle, fuer die er gedacht ist, sind groesser als unser RAM.
- **CPU/GPU-Split der Fehltreffer**: der Nutzen entsteht aus RAM-Bandbreite und PCIe, beides ist bei uns klein (15,2 GB/s, 6,7 GB/s), und die CPU ist der langsamste Teil der Kette, nicht ein ungenutzter Reserve-Pool.
- **i-Quant- und Q2_0-Kernels**: unsere Experten sind Q4_K (gate/up) und Q6_K (down), die Kernel decken IQ2/IQ3/Q2_0 ab. Die Idee traegt, der Code nicht.
- **`LLAMA_SRV_USER_BREAK`**: entfaellt, weil unser Server den Checkpoint an User-Nachrichten gar nicht anlegt (`is_user_start` kommt in unserem Stand nicht vor). tandem misst dort 112 MiB Zustandskopie pro Checkpoint - ein Problem, das wir nicht haben.
- **Qwen3.8-spezifische Schalter** (`LLAMA_HC_FUSE`, `LLAMA_GDN_DIRECT`, `LLAMA_GDN_GATES`, `LLAMA_CONV_CONT/TAIL`, `LLAMA_PLE_GPU`): alle an `LLM_ARCH_QWEN4EXP` bzw. `per_layer_tok_embd` gebunden, wir fahren kein Qwen3.8.
- **`LLAMA_MOE_PIN_EXPS`**: kopiert gemappte Experten in gepinnten Speicher, wenn der RAM sie fasst. Bei 72,5 GiB Modell und 30 GB RAM greift die Bedingung nicht.

## 6. Fehler im Patch, falls Code uebernommen wird

- **`q8_cache_mem` wird nie freigegeben.** 16 MiB je CUDA-Kontext bleiben beim Destruktor liegen; der Patch raeumt nur die `pack_*`-Ressourcen ab.
- **Der neue Kompatibilitaets-Schnellpfad ist toter Code.** Er verlangt `cgraph->uid != 0`, aber `ggml_cgraph::uid` wird in llama.cpp nirgends gesetzt (Init 0, `ggml_graph_next_uid()` ohne Aufrufer). Die Kompatibilitaetspruefung laeuft weiterhin ueber alle Knoten (gemessen 47 us bei 7000 Knoten).
- **`atoi(getenv("LLAMA_GDN_GATES"))` ohne Null-Check** in `src/models/qwen4exp.cpp` - Absturz, wenn die Variable nicht gesetzt ist. Abgesichert nur dadurch, dass `common_init()` sie vorher setzt.
- **Die Defaults kommen aus `setenv` in `common_init()`**, nicht aus dem Code: der Code-Default ist meist "aus", der Release-Default "an". Wer den Patch prueft, muss `LLAMA_MOE_NO_DEFAULTS=1` benutzen, um den echten llama.cpp-Zustand zu sehen.

## 7. Vorschlag fuer die naechste Runde, Stand nach der Messrunde vom 2026-11-07

Die Vorbedingung ist gemessen (Abschnitt P1): der Prefill liest 172 GB bei 890 MB/s, die CPU rechnet 6,5 % der Zeit. Daraus folgt die Reihenfolge:

1. **Erledigt: den Swap-Sturm im Prefill abschalten.** `vm.swappiness=0` bringt gepaart +9,7 % Prefill (A/B/A vom 2026-11-07, Ausgabe-SHA identisch). In agollm ist das als `MemorySwapMax=0` bereits aktiv - die 31 t/s aus der Messung sind also der Zustand ohne den Hebel.
2. **Erledigt und belegt: die Byte-Menge senken.** Der RAM-Lock (`LLAMA_EXPERT_RAM_LOCK_GIB=12`) bringt gepaart +14,7 % Prefill (36,67 gegen 32,04 t/s, SHA identisch), `LLAMA_ARG_CACHE_RAM=0` bringt +7,8 % (34,49 t/s). Runde 6 hat den Lock skaliert: das Optimum liegt bei 12 GiB (+15,3 % gegen den frischen Basisarm); 6 GiB bringt nur +3,9 %, 18 GiB +13,0 % bei bereits 744,8 MB/s Geraeterate, weil dem Page-Cache dann zu wenig RAM bleibt. Die Kombination beider Hebel bringt nichts zusaetzlich - sie greifen an derselben Stelle. Offen bleibt nur die Betriebsfrage: 12 GiB RAM sind dauerhaft blockiert.
3. **Erledigt (Negativbefund): die Rechenzeit hinter das Warten legen.** `LLAMA_EXPERT_CPU_ASYNC` ist im Decode ein No-op (-0,4 %, Runde 5 mit Cache-Leerung) und im Prefill von der Rechenzeit her gedeckelt bei 6,5 %.
4. **Erledigt: P1 (Mehrspalten-Dot) streichen.** Der Decode hat eine Spalte je Slot, der Prefill hat 6,5 % Rechenanteil. Der Aufwand steht in keinem Verhaeltnis zum Ertrag. (Ausnahme siehe die Praezisierung unten: im MTP-Verify-Schritt gibt es zwei Spalten, das ist vor einem Codeeingriff zu messen.)
5. **Erledigt (Negativbefund): die Fenstergroesse des fadvise-Hinweises.** Der Ist-Pfad (128 KiB) liegt auf dem Optimum; 64 KiB ist gleichwertig, 256 KiB bis 1 MiB sind schlechter (0,75 bis 0,86 s statt 0,57 s Gesamtzeit im Micro-Benchmark). Kein Codeeingriff noetig.
6. **Neu: der Decode haengt weder an der SSD noch am VRAM.** Nur 10,9 % der Tokenzeit sind Fault-Warten; die fehlenden Slabs sind Host-RAM-Fehler, keine VRAM-Luecken, und die PCIe-Last je Token liegt bei rund 6,7 MB. Der Hebel ist die CPU-Rechenzeit der kalten Experten und die Host-RAM-Bandbreite (1,07 GiB Gewicht je Token gegen 15,2 GB/s). Erster Schritt: den Decode-Arm mit `LLAMA_EXPERT_TIMING=1` messen (`tools/round7.sh`), um den Rechenanteil zu quantifizieren.
7. **Neu: der VRAM begrenzt den Hot-Set.** `LLAMA_EXPERT_S=16` wird vom Server auf 12 geklemmt ("manual S = 16 exceeds free VRAM"). Im agollm-Betrieb kostet der 4,7-GiB-GPU-Preload eines anderen Dienstes Ling Hot-Slots - ein Verkleinern dieses Preloads ist ein bisher ungesehener Decode-Hebel.
8. **P2 (`GGML_SCHED_BATCH_INPUTS=3`) und `GGML_CUDA_GRAPH_KEY_SHAPE` beobachten:** kleine, umrissene Aenderungen mit messbarem Effekt auf Prefill bzw. Graph-Umschaltung.
9. **P3 (`GGML_OP_MOE_DOORBELL`) ist im Prefill nicht erreichbar.** Der Layer-voraus-Prefetch haengt am selben `LLAMA_EXPERT_TMAX` wie die Warm-Admission; mit TMAX 4096 sprengt die Warm-Admission den VRAM (Prefill-Compute-OOM, -40 % Prefill). Ein eigener Grenzwert nur fuer den Prefetch waere der erste Schritt.

Nicht anfassen: der VRAM-Cache und der CPU/GPU-Split. Beides loest das Problem nicht, das wir haben, und kostet Tage fuer eine Maschine, die es nicht gibt.
