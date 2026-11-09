# Schrumpfender VRAM bei W0: cuBLAS legt pro Graph-Capture einen Workspace an, vom 2026-11-09

**Ergebnis: Nicht der VMM-Pool war die Ursache, sondern cuBLAS. Ohne eigenen Workspace legt cuBLAS bei jeder Stream-Capture einen neuen Workspace an (4 MiB unterhalb von Hopper) und gibt ihn erst bei `cublasDestroy` frei. ggml ruft vor jedem GEMM `cublasSetStream` auf, und das setzt den Workspace auf den Default-Pool zurueck. Bei Qwen mit W0 laufen die 17-Token-Verify-Batches der ngram-Entwuerfe als CUDA-Graphen. Jeder Layer-Split, der dabei einen Float-Matmul ueber cuBLAS rechnet, haelt danach 4 MiB, auch wenn der Graph beim Phasenwechsel verworfen wird: rund 320 MiB je Anfrage mit 16er-Entwurf. Neu ist ein fester Workspace pro Stream (`GGML_CUDA_CUBLAS_WORKSPACE_MIB`, Default = Groesse des cuBLAS-Pools, 0 = altes Verhalten). Damit laeuft der W0-Robustheitstest fehlerfrei durch.**

| Qwen W0, 3 Prefetch-Slots, 12 Anfragen, Sampling abwechselnd | Treiberanteil ausserhalb von ggml ("other") | Ende |
|---|---|---|
| ohne Fix (auch: Fix-Binary mit `GGML_CUDA_CUBLAS_WORKSPACE_MIB=0`) | 114 -> 261 -> 583 -> 902 -> 1163 MiB | 609-MiB-Prefill-Puffer passt nicht mehr, Step-down auf 1536, danach HTTP 500 im Decode |
| mit Fix | 114 -> 261 MiB, danach konstant | fehlerfrei, ab der 6,1k-Anfrage zwei statt drei Slots |

- **Messung:** `GGML_CUDA_MEM_TRACE=1` (neu, Default aus, im Log ab `-lv 4`) protokolliert jeden ggml-Puffer, jedes Pool-Wachstum, jedes Verwerfen der Graphen und jede Graph-Instanziierung, die den Treiberanteil um 1 MiB oder mehr veraendert, jeweils mit dem freien VRAM. "other" ist belegt minus ggml-Puffer minus Pools. Der VMM-Pool waechst nur bis zur Hochwassermarke des Prefills (104 MiB).
- **Eingrenzung:** Teuer waren genau die 17-Token-Graphen (82 je Decode-Phase, je 3,9-4,0 MiB), die 1- und 5-Token-Graphen kosteten nichts. Ein Backtrace von `cublasSetStream` waehrend einer Capture zeigt `ggml_cuda_mul_mat_cublas` aus `ggml_cuda_compute_forward`. Isoliert (`cublasSgemm` in 20 Graph-Captures) bleiben 80 MiB auch nach dem Zerstoeren der Graphen belegt, mit `cublasSetWorkspace` sind es 0 MiB. Die Gegenprobe mit derselben Binary und `..._MIB=0` bringt das Leck zurueck.
- cuBLAS startet seine Kernels ueber die Driver-API. `cudaGraphKernelNodeGetParams` liefert fuer sie keinen Funktionszeiger, und eine Kernelliste ueber die Runtime-Zeiger uebersieht sie. Die erste Zuordnung (MMQ in allen teuren Graphen) war deshalb nur eine Korrelation.
- **Betroffen** sind Konfigurationen, deren Verify-Splits als CUDA-Graph laufen und einen Float-Matmul ueber cuBLAS enthalten. Die agollm-Rezepte von Qwen und Clover (W4) sind es nicht: Dort laufen diese Splits eager (MUL_MAT_ID mit Warm-Slots). In 18 Anfragen blieb der Treiberanteil mit und ohne Fix bei 105 MiB, der Prefill war identisch, Decode 36,8 / 37,0 t/s. SenseNova (ngram-simple + MTP) blieb ebenfalls flach.
- **W0 bleibt keine Alternative zu W4.** Der Treiber haelt danach ~260 MiB fuer bis zu ~2200 gleichzeitig lebende Graph-Instanzen. Das ist eine Hochwassermarke, die wiederverwendet wird. Fuer den dritten Slot fehlen dann ~40 MiB. Mit drei Slots ist der Prefill um 11-14 % schneller als bei W4 (1,2k / 2,6k / 3,4k: 364 / 637 / 451 gegen 329 / 561 / 403 t/s), mit zwei Slots nicht mehr (349 / 538 / 405 und 6,1k 572 gegen 335 / 569 / 404 / 596 t/s). Decode ist gleich (36,6 gegen 37,0 t/s).

**Codeaenderungen** (Fork, nicht committet; nicht in `workers/llama`, weil die aktuellen Rezepte nicht betroffen sind und die dortige `libggml-cuda` vom 2026-11-06 stammt):
- `ggml-cuda/common.cuh`, `ggml-cuda.cu`: `ggml_backend_cuda_context::cublas_set_stream(device, stream)` ersetzt die vier `cublasSetStream`-Aufrufe (`mul_mat_cublas`, `out_prod`, `exl3`, `solve_tri`). Der Workspace pro Stream wird beim ersten GEMM per `cudaMalloc` angelegt und im Kontext-Destruktor freigegeben. Scheitert `cudaMalloc`, behaelt dieser Stream den Default-Pool (Warnung). HIP/MUSA unveraendert.
- `GGML_CUDA_MEM_TRACE`: die Speicherspur oben.
- `test-backend-ops -b CUDA0`: MUL_MAT 1177/1177, OUT_PROD 71/71, SOLVE_TRI 24/24.

Laeufe: `benchmark-results/bigmodel-opt-20261109/runs/vt-*` (`vt-w0-n3b`: Spur, `vt-w0-n3-fix`/`-final`: Fix, `vt-w0-n3-ws0`: Gegenprobe, `vt-w4-old`/`-fix`: Produktionsrezept), Einzeltests in `tools/vram/` (`cb.cu`: cuBLAS in Graph-Captures, `g.cu`/`s.cu`/`l.cu`: Graph-Speicher ohne cuBLAS, `gg.cpp`: ggml-Graphen).

# Grosse Modelle: zweiter und dritter Experten-Prefetch-Slot im Prefill, Kolibri-Fault-Pfad vom 2026-11-09

**Ergebnis: Der Prefill der 35B-MoE-Modelle mit DFlash-Draft (Qwen3.6, Clover) lief mit nur einem Prefetch-Slot fuer die Experten-Uploads ueber PCIe. Upload und Matmul wechselten sich deshalb strikt ab: Im nsys-Profil eines 3,4k-Prompts belegte die Kopie 5,96 s und die Kernels 4,8 s, ueberlappt haben sie sich nur 1,2 s. Mit 4 statt 8 Warm-Slots passt ein zweiter Slot: Prefill +15 bis +28 %, Decode -4 bis -5 % (Entscheidung des Nutzers: uebernommen). Nex und Sage hatten schon zwei Slots und bekommen ohne Decode-Preis einen dritten: Prefill bis +22 % bzw. +14 %. Kolibri uebernimmt die Ling-Sieger `LLAMA_EXPERT_FADVISE=1` und `GGML_CPU_EXPERT_FAULT_TOUCH=0`: Decode +3,7 bis +8,0 %, Prefill +10 bis +17 %. In allen Armen ist die Ausgabe-SHA innerhalb einer Konfiguration identisch.**

| Prefill (t/s, Worker-Libs, Rezept vorher -> nachher) | ~2,6k Tokens | ~3,4k Tokens |
|---|---|---|
| Qwen3.6-35B-A3B (W8 -> W4, 1 -> 2 Slots) | 438 -> **562** | 339 -> **401** |
| Clover-35B (W8 -> W4, 1 -> 2 Slots) | 439 -> **564** | ~345 -> **401** |
| Nex-N2.5-mini (2 -> 3 Slots) | 553 -> **670** | 417 -> **464** |
| Sage-A1.5 (2 -> 3 Slots) | 602 -> **664** | 421 -> **456** |

**Warum nur ein Slot.** `ggml_backend_sched_prefetch_init` legt Slot 0 bei der Reservierung an, weitere Slots erst im ersten Prefill aus dem dann freien VRAM, und nur wenn `Slotgroesse + 256 MiB` frei sind. Der groesste Experten-Tensor von Qwen (Q6_K-`down` in einigen Layern) braucht 210 MiB. Mit S=20, W=8 und dem DFlash-Draft waren beim Anlegen nur 124 MiB frei (`expert prefetch reduced to 1 slot(s)`). Die Meldung nennt jetzt Slotgroesse, freien VRAM und Headroom. Die Zeitachse eines Layers (~149 ms) zeigt die Folge: Der Gate-Upload laeuft parallel zu Attention/GDN, der Up-Upload startet aber erst nach dem Gate-Matmul und der Down-Upload erst nach dem Up-Matmul. Holo4, Nex und Sage haben ohne Draft genug VRAM und liefen schon mit zwei Slots, Holo4 gleicher Bauart wie Qwen mit 614 statt 438 t/s.

| Qwen gemischte Last (A/B/A/B, Prompts 1,1k/2,6k/3,4k/6,2k plus Decode 256 Tokens, Spekulation an) | A: W8, 1 Slot | B: W4, 2 Slots, Headroom 64 |
|---|---|---|
| Prefill 1,1k | 287 / 285 | 329 / 329 (**+15 %**) |
| Prefill 2,6k | 438 / 437 | 563 / 560 (**+28 %**) |
| Prefill 3,4k | 339 / 338 | 405 / 405 (**+20 %**) |
| Prefill 6,2k (2 Ubatches) | 473 / 474 | 597 / 595 (**+26 %**) |
| Decode-Mittel | 45,3 / 44,4 | 42,4 / 43,1 (**-4,7 %**) |
| Decode ohne Spekulation (eigene A/B/A/B) | 37,1 / 36,7 | 36,1 / 34,7 (**-4,0 %**) |

  Je Slot braucht es etwa vier Hot- oder Warm-Slots pro Layer (72,9 MiB je Slot ueber alle Layer). S=16 statt W4 kostete im Decode -3,4 % bei gleichem Prefill, W0 mit drei Slots brachte +45 % Prefill, scheiterte aber nach einigen Anfragen (siehe unten). Beispielrechnung fuer den Nutzer: 2,6k-Prompt plus 500 Tokens Antwort wird 0,8 s schneller, 1k-Prompt plus 900 Tokens Thinking 0,5 s langsamer.

| Dritter Slot (nur im Prefill belegt, kein Decode-Preis) | 1,1k | 2,6k | 3,4k | 6,2k |
|---|---|---|---|---|
| Nex 2 -> 3 | 346 -> 361 | 553 -> 674 (**+22 %**) | 417 -> 463 | 604 -> 723 (**+20 %**) |
| Sage 2 -> 3 | 334 -> 334 | 602 -> 668 (**+11 %**) | 421 -> 453 | 640 -> 730 (**+14 %**) |
| Holo4 2 -> 3 (Headroom 128) | 329 | 679, spaeter 612 | 450, spaeter 426 | 632 |

  Holo4 fiel im Robustheitstest nach einigen Anfragen auf zwei Slots zurueck und bleibt beim Rezept. Robustheitstest: je 12 Prefills und Decodes mit abwechselnd greedy und Rezept-Sampling (mit Backend-Sampling loest ein Sampler-Wechsel eine Neureservierung aus): Qwen W4, Nex, Sage, Holo4 ohne Fehler.

| Kolibri-1 (A/B, Seiten-Cache vor jedem Arm geleert) | Decode A -> B | Prefill 1k-Prompt A -> B |
|---|---|---|
| Paar 1 (A1/B1, mlock 3,84 GiB) | 15,66 -> 16,92 (**+8,0 %**) | 59-64 -> 71-73 (**+17 %**) |
| Paar 2 (B2/A2, mlock 12 GiB wie im Dienst) | 17,08 -> 17,71 (**+3,7 %**) | 63-64 -> 70 (**+10 %**) |

  B = `LLAMA_EXPERT_FADVISE=1` + `GGML_CPU_EXPERT_FAULT_TOUCH=0`. Die Fault-Wartezeit faellt von 184 s auf 0,16 s (Thread-Summe). Paar 2 lief B vor A, die SSD-Drift beguenstigt also A2.

**Codeaenderungen** (Fork, nicht committet; Libs in `workers/llama`, ORIGIN-Eintrag `bigmodel-prefetch=20261109T155837Z`):
- `GGML_SCHED_PREFETCH_HEADROOM_MIB` (`ggml/src/ggml-backend.cpp`, Default 256 = bisher): freier VRAM, der neben einem zusaetzlichen Prefetch-Slot bleiben muss. Qwen/Clover setzen 64; mit W4 sind 416 MiB frei, nach Slot 1 bleiben 206 MiB fuer den MMQ-Pool.
- Die Warnung `expert prefetch reduced to N slot(s)` nennt Slotgroesse, freien VRAM und Headroom.
- `llama_context::sched_reserve` gibt die Prefetch-Slots vor der Worst-Case-Reservierung frei, wie es `set_runtime_ubatch` schon tat. Ein Sampler-Wechsel kann die Reservierung mitten im Decode ausloesen. Die Slots entstehen beim naechsten Prefill neu.

**Verworfen, mit Zahlen** (Patches in `benchmark-results/bigmodel-opt-20261109/rejected-*.patch`):
- `LLAMA_EXPERT_PREFILL_COUNT_SKIP`: Mit Warm-Slots haengt im Prefill an jedem MoE-Layer ein CPU-`MOE_COUNT`, und bei `ADAPT=0` werden diese Zaehlungen gar nicht verwendet (`admit_graph` ist bei breiten Graphen false). Ohne den Op und den GPU-Sync pro Layer wurde der Prefill aber **langsamer**: -10 % (2,6k), -8 % (3,4k), mit einem und mit zwei Slots. Der Sync haelt den Host-Loop zurueck, das ordnet die Uploads guenstiger.
- `GGML_CPU_EXPERT_FAULT_ORDERED` (ein Thread gibt die Hinweise aller fehlenden Slabs in Dateireihenfolge): Kolibri-Prefill -14 % (3,4k) und -30 % (1k). Der eine Thread wird selbst zum Engpass (Seitenallokation, Reclaim), die anderen warten an der Barriere.
- Privater O_DIRECT-Puffer fuer den SSD-Prefill: Die SSD ist ueber die volle Modellspanne selbst langsam. `tools/bench_odirect_span.c`, Kolibri-GGUF, Zufallsreads 1 MiB: 1,39 GB/s ueber 8 GiB, 0,83-0,87 GB/s ueber 44 GiB (8, 16 oder 32 Threads, 128 KiB oder 1 MiB). Das ist die In-situ-Rate von Ling und Kolibri; der Page-Cache-Pfad ist nicht die Grenze. Ein einzelner sequenzieller Strom schaffte 1,1-1,8 GB/s, zwei bis vier parallele nur 0,62-0,66 GB/s (Laufwerk `K3`, `hmpre=0`).

**Fallstricke der Messung:**
- Die Login-Shell hat `ulimit -l` 3,84 GiB, der agollm-Dienst `LimitMEMLOCK=infinity`. Ohne Anheben scheitert der Kolibri-RAM-Lock (`mlock failed after 3.84 GiB`) und das `cudaHostRegister` des Modells. `tools/arm.sh` setzt das Limit per `sudo -n prlimit`.
- nsys 2022.4 findet seinen Importer nicht; `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i prof.qdstrm` direkt aufrufen, dann `nsys export --type sqlite`. Auswertung: `tools/nsys_window.py`.
- `LLAMA_EXPERT_STATS=1` schreibt beim Beenden per SIGTERM nichts.
- Mit W0 schrumpft der freie VRAM von Prefill zu Prefill (240 -> 230 -> 120 MiB), bis der Prefill-Puffer (609 MiB) beim Phasenwechsel nicht mehr passt. Geklaert und behoben: cuBLAS-Workspace pro Graph-Capture (Abschnitt "Schrumpfender VRAM bei W0").

Messprogramme: `benchmark-results/bigmodel-opt-20261109/tools/` (`arm.sh`, `mbench.py` mit `--sampling alt`, `kolibri_fault.sh`, `qwen_prof.sh`, `qwen_pp*.sh`, `qwen_tg*.sh`, `qwen_final.sh`, `robust.sh`, `span_bench.sh`, `nsys_window.py`), Laeufe in `runs/`. Ein Haengenbleiber vom Vortag (`llama-cli`-Smoke-Test im Interaktivmodus auf `/dev/null`, 17 h, 95 % eines Kerns, 68-GB-Log) wurde vor den Messungen beendet.

# Der Eingriff in die Fault-Phase: die Hinweise stehen lassen, den Touch-Pass streichen vom 2026-11-08

**Ergebnis: der freigegebene Eingriff - die fehlenden Cold-Slabs ueber einen atomaren Index verteilen statt statisch round-robin - ist implementiert und wirkt, bringt aber nur +3,5 % (7,36 gegen 7,11 t/s). Der Hebel ist ein anderer: der Touch-Pass. Die Fault-Phase besteht aus zwei Teilen, der Residenzpruefung mit Device-Hinweis (`posix_fadvise(WILLNEED)`) und dem Touch-Pass, der die Slabs einholt und danach an der Barriere wartet. **Loescht man nur den Touch-Pass (`GGML_CPU_EXPERT_FAULT_TOUCH=0`), holen die Hinweise die Daten weiterhin asynchron und tief in die Device-Queue, aber die Rechnung wartet nicht mehr an der Barriere auf den langsamsten Thread: Decode +14,6 % im frischen Zustand (8,00 gegen 6,97 und 6,99 t/s) und +13,3 % im belasteten Zustand (8,53 und 8,14 gegen 7,36 t/s), der Prefill bleibt bei 36,3 t/s, die Ausgabe-SHA sind in allen Armen identisch.** Der Eingriff in die Verteilung ist damit nicht der Produktionshebel, die reine Weglassung des Touch-Passes ist es - und sie ist eine Konfigurationszeile, kein Umbau.**

| Decode Runde 10 (frischer Cache, MEAN tg, go/code/essay, 256 Token je Prompt) | tg t/s | go | code | essay | fehlende Slabs | Fault-Wait | gegen die eingeklammerte Basis |
|---|---|---|---|---|---|---|---|
| a statisch (Basis) | 7,25 | 7,80 | 6,85 | 7,11 | 23,54 % | 115.564 ms | - |
| b dynamisch | 7,36 | 8,49 | 6,83 | 6,77 | 21,17 % | 119.780 ms | **+3,5 %** |
| c statisch (Basis) | 6,97 | 8,16 | 6,35 | 6,39 | 21,14 % | 82.645 ms | - |
| d nur Hinweise | 8,00 | 8,87 | 7,78 | 7,34 | 20,05 % | 201 ms | **+14,6 %** |
| e Schwelle 16 | 8,22 | 9,81 | 7,44 | 7,41 | 0 % | 0 ms | +17,8 % |
| f statisch (Basis) | 6,99 | 7,62 | 6,59 | 6,75 | 20,81 % | 80.243 ms | - |

| Decode Runde 15 (belasteter Cache, gleicher Aufbau) | tg t/s | go | code | essay |
|---|---|---|---|---|
| nur Hinweise | 8,53 | 8,87 | 8,29 | 8,42 |
| statisch (Basis) | 7,36 | 8,56 | 6,37 | 7,14 |
| nur Hinweise | 8,14 | 8,59 | 7,95 | 7,90 |

  Die drei Basisarme in Runde 10 (7,25 / 6,97 / 6,99) sind die Kontrollgruppe: jede Variante sitzt zwischen zwei Basisarmen, der Vergleich gilt gegen deren Mittel (b gegen a und c: 7,11; d und e gegen c und f: 6,98). Der Eingriff ist in `ggml_fault_expert_slabs` (`ggml/src/ggml-cpu/ggml-cpu.c`): die Threads schreiben jeden fehlenden Slab in eine geteilte Liste und nehmen sie nach einer Barriere einzeln von dort, statt jeden nur die eigenen Experten zu faulten. **Er wirkt, ist aber der kleinere Hebel.** Der Grund steht in den Phasenzeiten aus Runde 7b: von den 91 ms "Rest" je Token sind nur rund 13 ms die nachweisbare Touch-Wartezeit - der Rest ist die Barriere selbst. Die Verteilung des Touchs auf acht Threads zu optimieren kann deshalb hoechstens einen Teil dieser 13 ms holen; die Barriere wegzunehmen holt die ganzen 91 ms.

**Die Schwellenvariante ist ein Gegenbeispiel zur Nachfrage-Variante, obwohl beide im frischen Zustand gleich gut dastehen.** `GGML_CPU_EXPERT_FAULT_MIN_TOKENS=16` laesst die ganze Fault-Phase fuer Knoten mit einem Token entfallen (Runde 10: 8,22 t/s, +17,8 %), der Kernel laedt die Slabs beim Zugriff nach. Im belasteten Zustand ist dieselbe Konfiguration aber **28 % langsamer** (Runde 14: 5,14 und 4,89 gegen 7,10 t/s statisch): der Page-Fault landet jetzt im Rechenpfad, und steht der Cache unter Druck, kostet jeder Fault zusaetzlich den Reclaim. Die Variante "nur Hinweise" leidet nicht darunter, weil die Reads asynchron im Geraet stehen und die Rechnung nicht auf sie wartet.

| Zustand | statisch | dynamisch | nur Hinweise | Schwelle 16 |
|---|---|---|---|---|
| frischer Cache (Runde 10) | 6,97 - 7,25 | 7,36 | **8,00** | 8,22 |
| belasteter Cache (Runde 14/15) | 7,10 / 7,36 | - | **8,53 / 8,14** | 5,14 / 4,89 |

**Was der belastete Zustand ist.** Ab dem Neustart am 2026-11-08 um 15:26 kippte das Verhalten der Maschine: derselbe Arm las dieselbe Byte-Menge (129 GB) mit derselben aktiven Geraeterate (1017 MB/s), aber die SSD war in 242 von 286 Proben leer statt in 25 von 154. **Der Prozess wartet, waehrend das Geraet nichts liefert** - der Engpass war nicht das Geraet. Ausgeschlossen als Ursache: die CPU-Temperatur (Mittel 89-93 Grad, Maximum 103-109 Grad in allen Armen), die SSD-Temperatur (41 Grad, keine Drosselung, keine Warnzeit), der Swap (20,7 von 20,9 GB frei) und konkurrierende Prozesse (agollm gestoppt, keine weiteren Worker). Zwei Fehler in der Messkette wurden dabei gefunden und behoben: ein liegengebliebener Messserver blockierte den VRAM ("unable to allocate CUDA0 buffer"), seitdem prueft jedes Messskript den VRAM vor dem Arm; und nach einem Prefill-Arm (50 GB gelesen) reicht der 48-GB-Cache-Buster nicht mehr, weil die Bloecke den Device-Cache fuellen - solche Arme sind nicht mit den uebrigen vergleichbar.

| Prefill (5192 Token, Cache vor dem Arm geleert) | pp t/s | Ausgabe-SHA |
|---|---|---|
| statisch (Runde 10) | 36,3 | a8933235e37195a3 |
| nur Hinweise (Runde 10) | 36,3 | a8933235e37195a3 |
| Schwelle 16 (Runde 11) | 36,4 | a8933235e37195a3 |

  Der Touch-Pass ist also auch im Prefill entbehrlich: die Hinweise allein ergeben dieselbe Rate bei 56 % fehlenden Slabs. Die Schwelle fasst den Decode-Knoten nicht an (ein Token je Knoten), den Prefill-Knoten schon (2048 Tokens je Knoten, 73 % fehlende Slabs in Runde 11) - so bleiben beide Phasen bei ihrer jeweiligen Bestform.

**Produktionsempfehlung: `GGML_CPU_EXPERT_FAULT_TOUCH=0`.** Der Schalter liegt in `libggml-cpu.so.0` und muss mit der Bibliothek in den agollm-Worker (`workers/llama/`); das Rezept setzt ihn neben `GGML_CPU_EXPERT_READAHEAD`. Die drei Schalter sind in `ARCHITECTURE.md` Abschnitt 3 beschrieben, alle mit dem bisherigen Verhalten als Vorgabe.

**Der Rollout ist vor dem Produktionseinsatz geprueft (Runde 16).** Der Worker in agollm laeuft nicht mit dem Build des Forks, sondern mit vendorierten Bibliotheken (libllama und libggml-cuda vom 06.11., libggml-base vom 02.11.). Deshalb wurde die Kombination aus dem vendorierten Stand und der neuen libggml-cpu vorab gemessen: das vendoriert gebaute llama-server mit der neuen Bibliothek gegen den alten Worker-Stand, echtes Rezept, Cache vor jedem Arm geleert, Greedy-Ausgabe per SHA verglichen.

| Runde 16 (Produktionskombination) | tg t/s | fehlende Slabs | Fault-Wartezeit | Ausgabe-SHA |
|---|---|---|---|---|
| vendoriert (06.11.) | 6,86 | 21,65 % | 89.305 ms | 3x identisch |
| neue Lib, ohne Schalter | 7,10 | 20,54 % | 86.199 ms | 3x identisch |
| **neue Lib, `GGML_CPU_EXPERT_FAULT_TOUCH=0`** | **7,83** | 20,10 % | **204,7 ms** | 3x identisch |

  Vor dem Kopieren geprueft: die neue Lib braucht kein Symbol, das das vendoriert gebaute libggml-base (02.11.) nicht auch hat (45 ggml_-Importe, alle vorhanden); die Importe von libllama und libggml-cuda sind mit der neuen Lib genauso aufgeloest wie mit dem alten Stand, es kommt kein neues Symbol dazu; der Resolver moe_file_resolver liegt im vendorierten libllama an derselben Adresse (0x149bd0) wie im Fork-Build, also dieselbe ABI. Der erste Kopierversuch hat den Symlink des Build-Verzeichnisses statt der Datei kopiert und damit einen toten Symlink im Worker hinterlassen - das Skript kopiert jetzt mit -L und prueft die Ziel-Datei.

**Im laufenden Dienst gemessen (Produktions-A/B/A).** Der Worker im Dienst laeuft mit dem echten Rezept (RAM-Lock 12 GiB, Hot-Set S=12, FADVISE an); die Messung lief direkt am Worker-Port 18080, weil der Router bei den Chat-Werkzeugen in eine Schleife laeuft und den Vergleich gestoert haette. Je Arm Dienstneustart, Worker ueber den Preload-Endpunkt geladen, dann drei Anfragen mit dem Benchmark-Prompt "go" (256 Tokens, greedy, Cache-Waerme wie im Dienst).

| Produktions-A/B/A | 1. Anfrage | 2. Anfrage | 3. Anfrage | Mittel 2+3 | Ausgabe-SHA |
|---|---|---|---|---|---|
| nur Hinweise (Schalter 0) | 8,39 | 11,99 | 12,65 | **12,32** | 8510841615eb2a81 |
| Touch-Pass (Schalter 1) | 7,64 | 11,16 | 10,52 | 10,84 | 8510841615eb2a81 |
| nur Hinweise (Schalter 0) | 9,06 | 12,28 | 12,32 | **12,30** | 8510841615eb2a81 |

  Die beiden Hinten-Arme liegen bei 12,32 und 12,30 t/s, der Touch-Pass-Arm bei 10,84 t/s: **+13,6 %** gegen den eingeklammerten Mittelarm, mit identischem Text in allen neun Anfragen (SHA 8510841615eb2a81, derselbe Wert wie in den Messarmen und im Sandbox-Test). Die erste Anfrage nach dem Neustart ist immer langsamer, weil der Cache erst gefuellt wird - die Folgeanfrage ist der Zustand, der im Dienst zaehlt.

Messprogramme: `tools/round10.sh` (Varianten des Fault-Pfads, Decode und Prefill), `tools/round11.sh` (Verifikation der Schwelle), `tools/round12.sh` bis `tools/round15.sh` (Ausreisser, Zustandsvergleich und der finale A/B/A-Vergleich), `tools/summarize_round89.py` (t/s, SSD-Rate, Queue-Depth, Temperatur je Arm), `tools/vendor_fault_switch.sh` (gezieltes Kopieren der Bibliothek in den Worker).

# Die Fault-Phase trennt I/O und Rechnen: ohne sie +18 % Decode, aber -56 % Prefill vom 2026-11-07

**Ergebnis: der Decode wartet nicht an der SSD, sondern an der Barriere hinter der Fault-Phase. Abschalten der expliziten Fault-Phase (`GGML_CPU_EXPERT_READAHEAD=0`) bringt gepaart +18,2 % Decode (8,33 gegen 7,01 und 7,08 t/s, Ausgabe-SHA in allen Armen identisch), kostet aber im Prefill 56 % (14,4 gegen 32,8 t/s). Der Unterschied ist die Queue-Depth: die `posix_fadvise(WILLNEED)`-Hinweise der Fault-Phase fuellen die Device-Queue (qd 205, 818 MB/s), der bedarfsgesteuerte Kernel-Readahead laesst sie leer (qd 3,8, 419 MB/s). Im Decode ist die tiefe Queue kein Vorteil, weil die Barriere hinter der Fault-Phase das Rechnen erst nach dem Nachladen beginnen laesst; im Prefill ist sie der ganze Unterschied. Die beiden anderen Kandidaten sind damit tot: der VRAM-Hot-Set kann die Cold-Slabs strukturell nicht abdecken (Top-12 trifft 23,1 %, Top-16 26,4 %, Top-32 36,7 % der Selektionen), und der RAM-Lock haelt die falschen Experten fest (die fehlenden Slabs sinken nur von 20,59 % auf 19,62 %). Der Arbeitssatz des Decodes ist 55 bis 64 GiB bei 30 GiB RAM - der Decode ist ein Streaming-Workload, und die Frage ist nur, wie die Schreibspitze behandelt wird.**

| Decode Runde 8 (MEAN tg, go/code/essay, 256 Token je Prompt) | tg t/s | go | code | essay | SSD MB/s | qd | fehlende Slabs |
|---|---|---|---|---|---|---|---|
| Basis A | 7,01 | 8,21 | 6,29 | 6,54 | 862 | 7,13 | 21,08 % |
| **ohne Fault-Phase** | **8,33** | **10,17** | **7,46** | **7,37** | 810 | 3,28 | - |
| Basis B | 7,08 | 7,88 | 6,65 | 6,73 | 870 | 6,27 | 20,69 % |

  Die Fault-Phase ist der Block in `ggml_compute_forward_moe_cold`, der vor der Rechnung die Cold-Slabs prueft (`mincore`), dem Geraet die Reads vorlaeuft (`posix_fadvise(WILLNEED)` je 128-KiB-Chunk) und sie dann mit einem Touch-Pass einholt; danach kommt die Barriere. Der Arm ohne sie laesst den Kernel die Slabs beim Zugriff nachladen. Beide Basisarme stimmen auf 7,01 und 7,08 t/s, der Variantenarm liegt zwischen ihnen: **+18,2 % gegen Basis A, +17,7 % gegen Basis B** - der Unterschied ist also nicht der thermische Drift, den der Arm zwischen zwei Basen eingeklammert hat. Die Ausgabe-SHAs sind in allen drei Armen identisch (`8510841615eb2a81`, `5f8f0f8d01004f3b`, `2ba13b0b87b14003`): der Eingriff aendert nur den I/O-Pfad, nicht die Rechnung.

| Prefill Runde 9 (5192 Token, Cache vor dem Arm geleert) | pp t/s | SSD MB/s | IOPS | qd | Ausgabe-SHA |
|---|---|---|---|---|---|
| Fault-Phase an | **32,8** | 817,9 | 7.142 | **205,5** | a8933235e37195a3 |
| Fault-Phase aus | 14,4 | 418,8 | 48.487 | 3,84 | a8933235e37195a3 |

  Der Prefill braucht die Hinweisphase, weil sie die Geraete-Queue tief fuellt: ein Prefill-Batch beruehrt je Layer fast alle 512 Experten, die Hinweise legen die Reads des ganzen Layers vor, und das Geraet laeuft mit 818 MB/s. Ohne die Hinweise bleibt die Queue bei qd 3,8, das Geraet liefert nur 419 MB/s, und der Prefill braucht 1,8-mal so lange. Im Decode ist es umgekehrt: dort beruehrt ein Knoten nur acht Experten, die Queue ist mit den Hinweisen nicht tiefer als mit dem Kernel, und die Barriere kostet die Ueberlappung von I/O und Rechnen. **Der Hebel ist deshalb nicht "Fault-Phase an" oder "aus", sondern "welche Phase fuer welche Knotengroesse"** - genau das setzt der Eingriff in Runde 10 um.

**Der VRAM-Hot-Set ist als Hebel tot - offline geprueft, ohne Messlauf.** Das Chat-Profil des Rezepts (14 agollm-Prompts, 3,3k Tokens, 479 verschiedene Experten je Layer) deckt mit den Top-12 je Layer 23,1 % der Selektionen. Das Profil aus dem Benchmark selbst (`LLAMA_EXPERT_USAGE`, `LLAMA_EXPERT_USAGE_MODE=session`) deckt 22,7 % - **die Heat-Datei ist also fast optimal, es gibt nichts zu verbessern**. Selbst das Profil trifft den Nachbarprompt nur zu 12,8 %, weil Lings Routing sehr flach ist: 512 Experten, Top-8, und die Verteilung hat kaum eine Zipf-Form.

| S (Slots je Layer) | Abdeckung der Selektionen | VRAM-Bedarf | erreichbar? |
|---|---|---|---|
| 8 | 19,1 % | 1,06 GiB | ja |
| 12 (Auto-Fit) | 23,1 % | 1,60 GiB | ja |
| 16 | 26,4 % | 2,13 GiB | nein: 2,13 GB frei, 2,27 GiB noetig |
| 32 | 36,7 % | 4,26 GiB | nein |
| 64 | 51,3 % | 8,51 GiB | nein |

  Der Auto-Fit sitzt bei S=12 (`init: AUTO-FIT ENGINE -> set S = 12 (Free VRAM: 2.13 / 5.61 GB, per-slot size: 136.23 MB)`), und der geklemmte S=16-Arm aus Runde 6 (6,96 t/s) ist deshalb sinnlos: selbst S=16 bruechte nur 26,4 % Abdeckung, und S=16 passt nicht in den VRAM. Der Versuch mit einem zum Benchmark passenden Heat-Profil (`LLAMA_EXPERT_HOT=profiles/profile_bench.csv`, Abdeckung 22,7 %) brachte 7,16 t/s gegen 7,01 und 7,08 der Basis - keine Wirkung, dazu ein **anderer Ausgabe-SHA** (`e53738a670b758a9` statt `8510841615eb2a81`): die Ausgabe haengt am VRAM-Hot-Set, weil die heissen Experten auf der GPU und die kalten auf der CPU gerechnet werden, und das Rundungsunterschied ist. Ein Profilwechsel ist damit keine Messung mit identischer Ausgabe, sondern ein anderes Modell im Wortlaut.

**Der RAM-Lock haelt die falschen Experten fest.** Der Lock nimmt die 94 Experten mit dem hoechsten Seed-Score (die besten nach dem Chat-Profil) und haelt 12,04 GiB mit `mlock` im RAM. Der Decode verlangt aber gerade die *anderen* Experten: die fehlenden Slabs sinken nur von 20,59 % auf 19,62 %, die Wanduhr von 7,02 auf 7,09 t/s (+1,0 %, innerhalb der Drift). Zum Vergleich: der Prefill braucht den Lock (+15,3 %), weil dort dieselben Experten wiederholt werden.

**Der Arbeitssatz des Decodes ist grosser als der RAM.** Das Benchmark-Profil zaehlt 16.537 Experten-Layer mit je 3,41 MB (136,23 MB je Slot ueber alle 40 Layer) = **55,0 GiB**; das Chat-Profil 19.156 = 63,7 GiB. Der RAM hat 30 GiB, davon rund 26 GiB Page-Cache. Der Decode kann den Arbeitssatz also nicht einloesen, er streamt - die Frage ist nur, ob das Geraet die Spitze vertraegt.

| Thread-Sweep Runde 9 (Decode, Fault-Phase aus) | tg t/s |
|---|---|
| 8 Threads | 6,41 und 7,66 (zwei Arme, gleiche Konfiguration) |
| 12 Threads | 5,39 |
| 16 Threads | 2,8 (abgebrochen) |

  Die Maschine hat 8 physische Kerne (16 logische CPUs, `nproc` = 16). Mehr Threads als physische Kerne machen den Decode langsamer, nicht schneller - die I/O-Wartezeit laesst sich mit den Rechenkernen nicht beliebig ueberdecken, und SMT kostet Rechenleistung. **Der Rezeptwert 8 bleibt richtig.** Der Sweep zeigt zugleich die Messunsicherheit dieser Maschine: derselbe Arm liefert 6,41 und 7,66 t/s (Tctl zwischen 94 und 105 Grad). Nur gepaarte Vergleiche zaehlen, die Absolutwerte sind ohne Belastung des Rezeptes zu pessimistisch.

Messprogramme: `tools/round8.sh` (Decode-Varianten), `tools/round9.sh` (Prefill-A/B und Thread-Sweep), `tools/heat_coverage.py` und `tools/heat_cross.py` (Abdeckung der Heat-Dateien offline, ohne Messlauf), `tools/summarize_round89.py` (t/s, SSD-Rate, Queue-Depth je Arm). Die Rohdaten liegen in `runs/r8d-*/` und `runs/r9d-*/`, `runs/r9p-*/`; das Benchmark-Profil in `profiles/profile_bench.csv`.

# Decode-Phasen: der Decode wartet auf den langsamsten Thread, nicht auf die SSD vom 2026-11-07

**Ergebnis: Lings Decode verbringt 75,9 % der kalten Knotenzeit mit Warten und nur 24,1 % mit Rechnen - aber die nachweisbare SSD-Wartezeit ist davon nur rund ein Zehntel. Die restlichen gut 80 ms je Token entstehen an der Barriere nach der Fault-Phase, weil die kalten Experten-Slabs statisch round-robin auf die acht Threads verteilt werden: der Thread mit den kaltesten Slots wartet, die anderen warten auf ihn. Das erklaert im Nachhinein alle bisherigen Negativbefunde im Decode (RAM-Lock, Async, Cache-Ram, groesserer Hot-Set): sie machen die wartenden Threads schneller, aber sie aendern nicht das Maximum. Der Hebel ist die Verteilung der Fault-Arbeit, nicht die SSD und nicht die Rechenzeit - ein umrissener Codeeingriff in `ggml_fault_expert_slabs`.**

| Decode-Anteil (Differenz Decode-Arm minus Prefill-Arm) | Wert |
|---|---|
| Laeufe (Summe ueber 40 Layer) | 30.600 (je Layer 765) |
| Knotenzeit des kalten MoE-Kerns | 91,72 s, **119,9 ms je Token** |
| davon Rechnen (gate_up + activation + down) | 22,07 s, 28,9 ms je Token, **24,1 %** |
| davon Rest (Fault-Phase, Kopie, Barrieren) | 69,65 s, 91,0 ms je Token, **75,9 %** |
| gemessene Tokenzeit des Decodes | 133,80 ms (der kalte Knoten belegt 89,6 %) |
| nachweisbare SSD-Wartezeit (Touch-Pass, Thread-Summe durch 8) | 81,8 ms je Token, also **10,2 ms je Thread** |

  Die Trennung von Prefill und Decode ueber die Phasensumme: `LLAMA_EXPERT_TIMING=1` akkumuliert ueber den ganzen Serverprozess, und der Prefill-Anteil ist nicht klein (ein Batch mit 61 Tokens waehlt 488 Expert-Slots und trifft damit praktisch alle 512 Experten je Layer). Der Prefill-only-Arm (1 Token Decode) und der Decode-Arm (256 Token je Prompt, drei Prompts) unterscheiden sich deshalb nur um den Decode-Anteil. Beide Arme: gleiche Prompts, Seeds und Ausgabe-SHA, vor jedem Arm beide Caches geleert, `LLAMA_EXPERT_RAM_LOCK_GIB=0` (das agollm-Rezept enthaelt den Lock seit dem 2026-11-07 mit 12 GiB, sonst waere die Basis keine Basis).

- **Die Proportion ist alt, nur die Genauigkeit ist neu.** Der Lauf vom 2026-11-06 (`runs/diag/server.log`, ohne Cache-Leerung) kam auf 138,4 ms kalte Experten je Token, davon 33,6 ms Rechnen und 104,8 ms "Fault-Wait und Barrieren" - 24 % gegen 76 %. Die neue Messung mit Cache-Leerung kommt auf 119,9 / 28,9 / 91,0 ms, also 24,1 % gegen 75,9 %. Zwei unabhaengige Laeufe, dieselbe Proportion: **das Rechnen ist ein Viertel, der Rest drei Viertel.**
- **Der Rest ist nicht die SSD.** Der `fault wait` aus der Fehlerstatistik zaehlt nur den Touch-Pass und summiert ueber alle acht Threads: 81,8 ms je Token, das sind 10,2 ms je Thread im Mittel. Die Knotenzeit braucht aber 91 ms. Die Differenz von rund 80 ms je Token ist **Wartezeit an der Barriere**, nicht Lesezeit. Zum Vergleich: der Prefill-Anteil braucht 9,21 ms je Lauf bei 1,80 ms Rechnen - dort ist der Rest die echte SSD-Zeit (der Prefill liest 74 GiB je Batch).
- **Der Mechanismus ist die statische Verteilung der Fault-Arbeit.** `ggml_fault_expert_slabs` verteilt die Cold-Slabs mit `for (int e = ith; e < n_expert; e += nth)` (`ggml/src/ggml-cpu/ggml-cpu.c:1830`) - runder robin, statisch, ohne Rucksicht auf Last. Je Layer-Lauf gibt es 24,5 Cold-Slots auf acht Threads, also etwa drei je Thread; 21 % davon fehlen im Page-Cache (5,2 kalte Slabs je Layer-Lauf). Landen die drei kalten Slabs eines Threads auf diesem Thread, wartet er dreimal die SSD-Latenz, und alle anderen warten an der nachfolgenden Barriere (`ggml-cpu.c:2550`) auf ihn. Das erklaert, warum die gemessene Wanduhr fast unabhaengig von der Gesamtzahl der fehlenden Slabs ist.
- **Das erklaert die Negativbefunde.** Der RAM-Lock senkte die fehlenden Slabs von 20,95 auf 19,62 % und die Fault-Wait um 6,6 %, brachte aber keine Wanduhr (+1,0 %, innerhalb der Drift). `LLAMA_EXPERT_CPU_ASYNC` brachte -0,4 %, `CACHE_RAM=0` -0,1 %, ein groesserer VRAM-Hot-Set -0,9 %. Alle vier senken oder verschieben Arbeit, aber keine aendert **welcher** Thread am laengsten wartet. Der Prefill dagegen hat keine Barriere ueber die Slots hinweg - dort hilft der RAM-Lock (+15,3 %) sehr wohl, weil er die Bytesmenge senkt, die der einzelne Thread nachladen muss.
- **Der Hebel ist die Verteilung, nicht die Bandbreite.** Zwei Moeglichkeiten, beide in `ggml_fault_expert_slabs`: (a) alle Threads pruefen die Residentz aller Slabs und teilen sich nur die *fehlenden* ueber einen atomaren Index (die Pruefung ist billig, das Nachladen teuer); (b) die fehlenden Slabs nach erwarteter SSD-Latenz sortieren und die schweren zuerst verteilen. Variante (a) ist kleiner und trifft den Punkt direkt. Erwartbar ist der Gewinn dort, wo die kalten Slabs ungleich verteilt sind, also im Decode; im Prefill (5,2 kalte Slabs je Layer ueber 512 Slots, aber 24.000 Slots je Batch) ist die Verteilung bereits glatter und der Hebel entsprechend kleiner.

Messprogramme: `benchmark-results/tensorfold-review-20261106/ling-ab/runs/r7p-prefill-{a,b}/` und `runs/r7d-decode-{a,b}/` (Serverlog mit Phasen je Layer, `summary.json`), `tools/round7b.sh` (Prefill- und Decode-Arm mit Phasen-Profiling), `tools/summarize_phases.py` (Summen und Differenzrechnung). Zwei Feinheiten, die den Lauf gepraegt haben: `LLAMA_EXPERT_TIMING=1` allein schreibt nichts - die Phasenzeilen haengen am STATS-Block (`src/llama-expert-tier.cpp:1240`), und das Profiling selbst kostet rund 7 % (6,66 gegen 6,83 t/s im Mittel), weshalb nur die Verhaeltnisse und nicht die Absolutzeiten zwischen den Armen verglichen werden.

# RAM-Lock und Cache-Ram: die Prefill-Byte-Menge sinkt, der Decode haengt nicht an der SSD vom 2026-11-07

**Ergebnis: der RAM-Lock (`LLAMA_EXPERT_RAM_LOCK_GIB=12`) bringt gepaart +14,7 % Prefill (36,67 gegen 32,04 t/s, Ausgabe-SHA in allen Armen identisch), `LLAMA_ARG_CACHE_RAM=0` bringt weitere +7,8 % (34,49 t/s). Runde 6 hat den Lock skaliert: das Optimum liegt bei 12 GiB (+15,3 % gegen den frischen Basisarm, 6 GiB nur +3,9 %, 18 GiB +13,0 % mit bereits 744,8 MB/s Gereterate), und die Kombination mit `CACHE_RAM=0` bringt nichts zusaetzlich. Beide Hebel greifen an derselben Stelle: sie nehmen dem Page-Cache das RAM, das das 72-GiB-Modell zum Nachladen braucht. Der Decode dagegen haengt weder an der SSD noch am VRAM - dort sind nur 10,9 % der Tokenzeit Fault-Warten, und weder Lock (+1,0 %) noch Async (-0,4 %) noch Cache-Ram (-0,1 %) noch ein groesserer Hot-Set bringen etwas. `LLAMA_EXPERT_TMAX=4096` ist im Prefill schaedlich (-40 %), weil der warme VRAM-Pfad dann fuer 2048 Tokens admitted wird und der Prefill-Compute in ein OOM laeuft. Und die Methode selbst hat einen Fehler, der jede Decode-Messung dieser Maschine bisher verdorben hat: die SSD liefert spaetere Arme schneller, weil sie die zuletzt gelesenen Bloecke aus ihrem eigenen Cache nimmt - ohne Cache-Leerung lag der Basisarm am Ende 33 % schneller als am Anfang.**

| Prefill (5192 Token) | pp t/s | gegen Basis A | gegen Basis B | SSD MB/s | fehlende Slabs | Fault-Wait |
|---|---|---|---|---|---|---|
| Basis A | 32,04 | - | -1,9 % | 830,2 | 67,16 % | 211.272 ms |
| RAM-Lock 12 GiB | **36,67** | **+14,7 %** | **+16,9 %** | 813,7 | **58,35 %** | 192.998 ms |
| TMAX 4096 | 19,22 | -40,0 % | -38,8 % | 916,3 | 23,15 % | 404.362 ms |
| Cache-Ram 0 | 34,49 | +7,8 % | +9,9 % | 883,0 | 66,93 % | 193.369 ms |
| Basis B | 31,42 | - | - | 863,3 | 66,89 % | 202.144 ms |

| Decode (MEAN tg, go/code/essay, 256 Token je Prompt) | tg t/s | gegen Basis A | gegen Basis B | fehlende Slabs | Fault-Wait |
|---|---|---|---|---|---|
| Basis A | 6,92 | - | -1,0 % | 20,95 % | 96.477 ms |
| RAM-Lock 12 GiB | 6,92 | 0,0 % | -1,0 % | 19,91 % | 90.135 ms |
| Async | 6,89 | -0,4 % | -1,4 % | 21,19 % | 95.406 ms |
| Basis B | 6,99 | - | - | 20,92 % | 99.346 ms |

  Vor jedem Arm beide Caches geleert: Page-Cache des Modells per `--evict`, SSD-Cache per 48-GB-O_DIRECT-Schreibvorgang (`tools/flush_drive.sh`). Der Cache-Buster kostet dem Prefill etwa 12 % gegenueber einem Lauf ohne (Runde 4: 36,3 t/s, Runde 5: 32,0 t/s) - die paarweisen Vergleiche bleiben gueltig, die Absolutwerte sind pessimistischer als im Betrieb. Die Drift zwischen den beiden Basisarmen liegt bei -1,9 % (Prefill) und +1,0 % (Decode), beide Effekte liegen deutlich darueber.

**Runde 6 hat den Lock skaliert und die Hebel kombiniert.** Die vier "frischen" Arme (Basis A, Lock 6, Lock 12, Lock 18) liefen alle direkt nach einem Cache-Buster und sind untereinander vergleichbar; der Basisarm B kam nach den Lock-Armen und hatte einen warmen Page-Cache (die gelockten Seiten bleiben nach dem Prozessende im Cache), er ist deshalb als Kontrollarm zu hoch.

| Prefill Runde 6 (5192 Token) | pp t/s | gegen Basis A | gegen Basis B | SSD MB/s | fehlende Slabs | Lock-Zeit |
|---|---|---|---|---|---|---|
| Basis A | 30,70 | - | -5,5 % | 833,8 | 67,25 % | - |
| RAM-Lock 6 GiB | 31,90 | +3,9 % | -1,8 % | 807,8 | 61,51 % | 8,1 s |
| RAM-Lock 12 GiB | **35,40** | **+15,3 %** | **+8,9 %** | 810,9 | 58,53 % | 15,4 s |
| RAM-Lock 18 GiB | 34,70 | +13,0 % | +6,8 % | 744,8 | **53,12 %** | 22,7 s |
| Basis B | 32,50 | +5,9 % | - | 859,4 | 67,65 % | - |
| Lock 12 GiB + Cache-Ram 0 | 35,40 | +15,3 % | +8,9 % | 777,6 | 58,46 % | 15,4 s |

| Decode Runde 6 (MEAN tg, go/code/essay) | tg t/s | gegen Basis A | gegen Basis B | fehlende Slabs |
|---|---|---|---|---|
| Basis A | 7,02 | - | -0,6 % | 20,59 % |
| Cache-Ram 0 | 7,01 | -0,1 % | -0,7 % | 20,89 % |
| RAM-Lock 12 GiB | 7,09 | +1,0 % | +0,4 % | 19,62 % |
| Hot-Set S 16 (geklemmt auf 12) | 6,96 | -0,9 % | -1,4 % | 20,69 % |
| Basis B | 7,06 | +0,6 % | - | 20,86 % |

  Das Optimum liegt bei **12 GiB**: 6 GiB bringt fast nichts (+3,9 %, das liegt in der Drift), 18 GiB liefert zwar die wenigsten fehlenden Slabs (53,12 %), aber die SSD-Rate faellt auf 744,8 MB/s, weil dem Page-Cache nur noch rund 12 GB RAM bleiben. Der Lock ist damit kein "mehr ist besser"-Hebel, sondern ein Gleichgewicht zwischen gesicherten Experten im RAM und Page-Cache fuer den Rest. Die Kombination mit `CACHE_RAM=0` bringt **nichts zusaetzlich** (35,40 wie Lock 12 allein): beide Hebel greifen an derselben Stelle, dem RAM fuer den Page-Cache, und der Lock hat den Vorrang. Im Decode sind beide neutral (Lock +1,0 %, Cache-Ram -0,1 %, Drift +0,6 %) - die im Rezept notierte "-4 %" des RAM-Prompt-Caches ist im Mittel ueber drei Prompts nicht zu sehen, die Einzelwerte streuen (go -0,8 %, code -4,0 %, essay +4,4 %).

- **Der RAM-Lock senkt die Byte-Menge, die vom SSD kommen muss.** Die 12,04 GiB (10833 Slabs, in 16,0 s gelockt) sind vom Page-Cache nicht mehr verdraengbar. Die Folge: 58,35 % statt 67,16 % fehlende Slabs, 192.998 statt 211.272 ms Fault-Wait, 20,5 s weniger Wanduhr. Die SSD-Rate sinkt dabei leicht (813,7 gegen 830,2 MB/s) - es werden weniger Bytes geliefert, aber die verbleibenden kommen schneller durch, weil der Kernel nicht mehr um den Platz der Modellseiten kaempft. **Der Preis: 12 GiB RAM sind dauerhaft nicht verfuegbar**, und die Lock-Zeit faellt einmalig beim Serverstart an (16,0 s). Im Betrieb amortisiert sie sich ueber jeden weiteren Prefill; nebenbei leiden die anderen Modelle desselben Hosts, weil der Swap-Druck steigt. Der Decode zahlt nichts dafuer - dort sind 12 GiB nur 10 Token Material.
- **Der RAM-Prompt-Cache kostet Prefill.** `LLAMA_ARG_CACHE_RAM=8192` (der Wert im agollm-Rezept) reserviert 8 GiB RAM fuer gespeicherte Chats. Bei einem 72-GiB-Modell in 30 GB RAM ist das der Page-Cache, der dem Nachladen fehlt: mit `CACHE_RAM=0` steigt die SSD-Rate von 830,2 auf 883,0 MB/s und die Fault-Wait sinkt von 211.272 auf 193.369 ms, bei **gleicher** Slab-Fehlerrate (66,93 % gegen 67,16 %). Der Mechanismus ist hier also nicht die Byte-Menge, sondern der Speicherdruck: weniger Reclaim, mehr Bandbreite. Das Rezept nennt fuer den Cache-Ram bisher nur die Decode-Seite ("decode -4 %"), der Prefill-Gegenwert ist neu.
- **Der Decode haengt nicht an der SSD, und er haengt auch nicht am VRAM.** 768 Tokens in 111 s sind 144,5 ms je Token, davon 15,7 ms Fault-Warten (10,9 %, Thread-Summe durch 8 Threads). Der Rest ist CPU-Rechnen und Host-RAM-Bandbreite. Die "fehlenden Slabs" sind **Host-RAM-Fehler, keine VRAM-Luecken**: der `mincore`-Test in `ggml-cpu.c:1836` prueft die Residentz im Page-Cache, und der Cold-Kern rechnet direkt aus dem Host-mmap (`ggml-cpu.c:2565`). Der VRAM-Hot-Store (`w_hot`, 1,73 GiB fuer 12 Experten je Layer) wird ausschliesslich von der GPU-Seite gelesen (`src/llama-expert-tier.cpp:2787`), die CPU-Seite bekommt nie VRAM-Gewichte; der PCIe-Verkehr je Token liegt bei rund 6,7 MB (D2H der Router-Ids und Layer-Inputs, H2D des Cold-Ergebnisses, Logits) und damit bei etwa 1 ms. **Ein groesserer VRAM-Hot-Set ist deshalb kein Decode-Hebel**, sondern eine Verlagerung von CPU- auf GPU-Rechenzeit. Daraus folgt: (1) der RAM-Lock senkt die Fault-Wait um 6,6 %, bringt aber keine Wanduhr - die gesparte Wartezeit fliesst in Rechenzeit, (2) `LLAMA_EXPERT_CPU_ASYNC` ist ein No-op, (3) der Decode-Hebel ist die CPU-Rechenzeit der kalten Experten und die Host-RAM-Bandbreite (15,2 GB/s bei 8 Threads gegen rund 1,07 GiB Gewicht je Token), nicht die SSD.
- **Der Decode laeuft ueber MTP-Spec-Decoding.** Je 256 dekodierte Tokens zaehlt der Serverlog 255 Verify-Schritte (151,75 ms je Schritt), der groesste Teil der Tokenzeit steckt also im Verify-Schritt. Dessen Zeit waechst mit dem Kontext (127,46 ms bei 69 Token Prompt, 154,48 ms bei 306 Token Prompt). Das ist bei kuenftigen Decode-Vergleichen zu beachten: die Tokenzeit ist keine konstante, und der Prompt-Anteil liegt bei 136 ms je Token.
- **`LLAMA_EXPERT_TMAX=4096` sprengt den VRAM im Prefill.** `LLAMA_EXPERT_TMAX` (Default 16) sperrt im Prefill drei Dinge: den fusionierten `MOE_COLD`-Pfad (`begin_moe_cold`), den Layer-voraus-Prefetch (`page_prefetch_top_m`) und die Warm-Admission (`warm_admit_from_counts`: peak_count <= tmax). Mit 4096 sind alle drei frei - und der warme Pfad fuer 2048 Tokens passt nicht mehr in den VRAM: `prefill compute OOM (ret = -2); falling back to decode ubatch`. Der Prefill zerfaellt in 64-Token-Stuecke, laeuft mit 19,2 t/s statt 35 (-40 %) und liest dabei nur 23,15 % fehlende Slabs mit einer praktisch leeren Warteschlange (qd 17,6 statt 350). **Der Layer-voraus-Prefetch ist im Prefill ueber TMAX also nicht erreichbar**, ohne den VRAM zu sprengen; dafuer braucht es einen eigenen Grenzwert, der nur den Prefetch freigibt.
- **Huge Pages sind auf dieser Maschine nicht verfuegbar.** THP steht auf `madvise`, aber der Kernel hat kein `CONFIG_READ_ONLY_THP_FOR_FS`: `madvise(MADV_HUGEPAGE)` auf dem Modell-Mapping wird akzeptiert (`VmFlags: ... hg`), erzeugt aber keine Huge Pages (smaps: kein `FileThp`, `AnonHugePages: 0 kB`). Ein Codeeingriff in `src/llama-mmap.cpp` (ein `madvise` mehr) bringt hier also nichts.
- **Der fadvise-Hinweis bringt 27 %, und seine Fenstergroesse ist bereits richtig.** Der Ist-Pfad legt je fehlendem Experten-Slab 128-KiB-Fenster mit `POSIX_FADV_WILLNEED` auf die Datei und faultet danach jede 4-KiB-Seite einzeln (`ggml-cpu.c:1847`). `tools/sweep_fadvise_chunk.sh` misst genau diesen Durchgang (8 Threads, 2 GiB aus der Modelldatei, vor jedem Durchgang die ganze Datei per `fadvise(DONTNEED)` aus dem Page-Cache und der SSD-Cache geleert):

| Fenster des Hinweises | Gesamtzeit | langsamster Thread | Rate |
|---|---|---|---|
| keiner (nur 4-KiB-Faults) | 0,78 s | 0,78 s | 2.615,5 MiB/s |
| 64 KiB | 0,57 s | 0,09 s | 3.574,7 MiB/s |
| **128 KiB (Ist-Pfad)** | **0,57 s** | **0,08 s** | **3.562,0 MiB/s** |
| 256 KiB | 0,86 s | 0,62 s | 2.370,4 MiB/s |
| 512 KiB | 0,75 s | 0,63 s | 2.739,5 MiB/s |
| 1 MiB | 0,75 s | 0,69 s | 2.724,7 MiB/s |

  Der Hinweis ist asynchron, deshalb wartet der langsamste Thread bei 64/128 KiB nur noch 0,08 bis 0,09 s statt 0,78 s: die SSD liefert im Hintergrund, der Rechen-Thread wartet nicht. **Groessere Fenster sind schlechter**, nicht besser (0,75 bis 0,86 s): je groesser der Einzel-Hinweis, desto laenger wartet der eigene Thread auf seinen Anteil, und die Gesamtzeit steigt ueber die 128-KiB-Marke. Der Ist-Pfad liegt mit 128 KiB genau auf dem Optimum - **an der Fenstergroesse ist nichts zu holen.** Die Einzelzahl 82,5 MiB/s aus dem THP-Versuch gilt nur fuer einen Thread und 8 MB (4-KiB-Faults ohne Vorlauf); mit 8 Threads und 2 GiB sind es ohne jeden Hinweis 2.615 MiB/s.
- **Der VRAM begrenzt den Hot-Set, nicht die Messung.** `LLAMA_EXPERT_S=16` wird vom Server abgelehnt: "init: manual S = 16 exceeds free VRAM, clamping to 12" (`src/llama-expert-tier.cpp`, Clamp beim Init). Bei 6 GiB VRAM und 1,73 GiB fuer 12 Experten je Layer sind 16 Slots nicht darstellbar. **Im agollm-Betrieb kostet der 4,7-GiB-GPU-Preload eines anderen Dienstes Ling zusatzliche Hot-Slots** - ein Verkleinern dieses Preloads ist damit ein Decode-Hebel, den man bisher nicht gesehen hat. Der Arm selbst ist ein No-op (gleiche Konfiguration wie die Basis) und dient nur als Driftmessung.
- **Die SSD-Cache-Drift hat jede Decode-Messung dieser Maschine verdorben.** Ohne Cache-Leerung wird ein spaeterer Arm schneller, weil die Platte die zuletzt gelesenen Bloecke aus ihrem eigenen Cache liefert. In Runde 4 lag der Basisarm am Ende bei 7,26 gegen 5,45 t/s am Anfang (+33 %) bei identischer Konfiguration. Deshalb sind die Decode-Zahlen dieser Maschine ohne Cache-Leerung nicht vergleichbar - die Runde-5-Zahlen (6,89 bis 6,99 t/s, Drift +1,0 %) sind die belastbaren.

Messprogramme: `benchmark-results/tensorfold-review-20261106/ling-ab/runs/r5{p,d}-*/` und `runs/r6{p,d}-*/` (Serverlog mit Phasen, `summary.json`, `.disk` mit der SSD-Rate je Sekunde), `tools/round5.sh` und `tools/round6.sh` (Prefill und Decode mit Cache-Leerung vor jedem Arm), `tools/summarize_rounds.py` (Tabelle ueber alle Runden), `tools/flush_drive.sh` (SSD-Cache leeren) - alle Messskripte liegen unter `benchmark-results/tensorfold-review-20261106/ling-ab/tools/` und damit im gitignorierten Messbereich. Allgemein nuetzlich sind die drei Werkzeuge im Repo: `tools/bench_thp_file.c` (THP-Nachweis), `tools/bench_fadvise_chunk.c` (Fenster-Groesse des fadvise-Hinweises), `tools/evict_file.py` (Page-Cache leeren).

# Ling-Prefill-Phasen: der Prefill liest das ganze Modell vom SSD vom 2026-11-07

**Ergebnis: der Prefill haengt noch staerker am SSD als der Decode. Von 166 s Wanduhr fuer 5192 Token (31,3 t/s) entfallen 10,7 s auf die CPU-Rechenzeit der kalten MoE-Knoten (6,5 %, davon nur 2,4 s in den drei gemessenen Phasen) - der Rest ist Warten auf die Platte: 172 GB bei 890 MB/s, mittlere Warteschlange qd 283, 66,8 % der geprueften Experten-Slabs fehlten. `vm.swappiness=0` bringt gepaart +9,7 % Prefill (A/B/A, Ausgabe-SHA in allen Armen identisch). Die Idee aus tandem.cpp - den Mehrspalten-Dot-Product auf beliebige Spalten zu verallgemeinern - zielt am Prefill auf die falsche Stelle: sie greift auf die 6,5 % Rechenanteil, nicht auf die 93 % Wartezeit.**

| Arm | swappiness | Prefill 5192 Token | Wanduhr | CPU-MoE-Knoten | davon Rechnen | SSD | fehlende Slabs |
|---|---|---|---|---|---|---|---|
| Referenz (vorheriger Lauf) | 180 | 31,2 t/s | 166 s | 10,72 s | 2,38 s | 890,9 MB/s | 66,79 % |
| A | 180 | 31,1 t/s | 167 s | 10,54 s | 2,47 s | 865,7 MB/s | 66,77 % |
| B | **0** | **33,8 t/s** | 154 s | 10,90 s | 2,49 s | **928,8 MB/s** | 67,04 % |
| C | 180 | 30,5 t/s | 170 s | 12,78 s | 2,71 s | 840,4 MB/s | 67,28 % |

  Gepaart, Arm-Reihenfolge A/B/A, vor jedem Lauf der Page-Cache des Modells geleert (`--evict`), gleiche Prompts und Seeds, Greedy-Ausgabe per SHA `a8933235e37195a3` in allen vier Laeufen identisch. Die beiden 180-Arme liegen 2 % auseinander (Drift), der 0-Arm liegt **+8,7 %** ueber A und **+10,8 %** ueber C, im Mittel **+9,7 %**. Der Gewinn kommt von der hoeheren Geraeterate (865,7 und 840,4 gegen 928,8 MB/s), nicht von weniger Bytes: die Zahl der fehlenden Slabs ist in allen Armen gleich.

- **Der Prefill liest das Modell von vorn bis hinten.** Der Prompt (5192 Token aus `data/long.txt`, 16000 Zeichen) laeuft in drei CMOE-Batches (2048/2048/1096). Jeder Batch beruehrt praktisch alle 512 Experten je Layer: 32 Experten-Layer (10 bis 41) x 512 x 3,64 MiB = **78,3 GiB** je Batch, theoretisch 235 GiB fuer die drei Batches. Die SSD lieferte 172 GB - der Page-Cache steuerte rund 63 GiB bei, also etwa 27 % Wiederverwendung, der Rest kam von der Platte. Die Rate liegt durchgehend bei 890 bis 1300 MB/s (Maximum 2643 MB/s), die Warteschlange bei qd 200 bis 283 (Maximum 1920), 7707 IOPS. **Die Platte laeuft die ganze Zeit durch.**
- **Die CPU rechnet nur 6,5 % der Prefill-Zeit.** Das Phasen-Profiling (`LLAMA_EXPERT_TIMING=1` plus `LLAMA_EXPERT_STATS=1`) misst je Layer die Wanduhr des kalten MoE-Knotens (`cpu_cold_ms`, enthaelt die Fault-Phase) und die drei Phasen darin: `cpu_gate_up_ms` 1,45 s, `cpu_activation_ms` 0,04 s, `cpu_down_ms` 0,89 s ueber alle Layer und Laeufe. Die restlichen 8,34 s des Knotens sind Kopie, Barrieren und die Teile des Pfads ausserhalb der drei Phasen. Der Decode hat zum Vergleich 33,6 ms Rechenzeit bei 138 ms Tokenzeit (24 %) - **der Prefill ist also noch staerker SSD-gebunden als der Decode**, und die Rechenphase des Decodes liegt zu 76 % im Warten.
- **Der Swap-Sturm ist im Prefill groesser als im Decode.** `tools/vmstat_delta.py` ueber den Lauf: `pgscan_direct` **+11.927.048 Seiten** (46,5 GB direct reclaim, zum Vergleich +2.940.843 im Decode), `pgsteal_direct` +5.498.252 (21 GB), `allocstall_normal` **+103.078** (im Decode +8.540), `pswpout` +194.785 (761 MB), `pswpin` +29.600 (116 MB). Der Kernel verdraengt die Modell-Seiten aus dem Page-Cache, waehrend die SSD sie nachliefern soll - genau der Effekt, den `vm.swappiness=0` im Decode mit +7,4 % ausgeraeumt hat. Im Prefill ist er mit **+9,7 %** groesser, weil dort die Bytesmenge um eine Groessenordnung hoeher ist.
- **Die fehlenden Slabs nehmen zum Ende der Layers hin ab:** Layer 12 bei 96,4 %, Layer 25 bei 79,6 %, Layer 40 bei 51,4 % (heisser gewordener Hot-Set). Ueber alle Layer 78,6 % von 1.371.904 Pruefungen. Der Hot-Set liegt mit S = 12 von 512 Experten je Layer im VRAM (2,13 GB frei von 5,61 GB), der Rest muss je Batch ueber RAM oder SSD kommen.
- **Was das fuer die Hebel bedeutet.** (1) `vm.swappiness=0` ist bereits als `MemorySwapMax=0` in agollm aktiv - die Prefill-Zahl hier (31 t/s bei 180) ist also der Zustand ohne den Hebel. (2) Der Rechenanteil (6,5 %) laesst sich mit dem asynchronen Ueberlapp (`LLAMA_EXPERT_CPU_ASYNC`, im Rezept aus) hinter das Plattenwarten legen - gedeckelt bei etwa 6 %. (3) Der groesste Hebel ist die Byte-Menge: der Page-Cache thrasht ueber 78,3 GiB, weil pro Batch alles gebraucht wird und die RAM 30 GB nur ein Drittel fasst. Der Gegenzug liegt im RAM-Lock (`LLAMA_EXPERT_RAM_LOCK_GIB`, im Rezept 0), der die kalten Experten mit den besten Seed-Scores per `mlock` im RAM haelt, statt sie zwischen den Batches zu verdraengigen - der naechste Messkandidat. (4) **Die Mehrspalten-Optimierung aus tandem.cpp ist fuer den Prefill tot:** der Decode hat je Experten-Slot genau eine Spalte (kein Amortisationsfaktor), und im Prefill ist der Rechenanteil nur 6,5 % - selbst eine Halbierung der Dekodierkosten braeuchte 3 %.

Messprogramme: `benchmark-results/tensorfold-review-20261106/ling-ab/runs/prefill-{timing-1,sw180-a,sw0-b,sw180-c}/` (Serverlog mit Phasen, `summary.json`, `.disk` mit der SSD-Rate je Sekunde), `benchmark-results/tensorfold-review-20261106/ling-ab/tools/prefill_phases.sh` und `tools/prefill_ab.sh` (Prefill-Lauf mit Phasen-Profiling, vmstat-Differenz und SSD-Statistik je Lauf, A/B/A ueber `vm.swappiness`).

# Integration der Sieger in agollm vom 2026-11-06

**Ergebnis: die drei belegten Sieger der Runde vom 2026-11-06 sind in agollm aktiv - Swap-Limit in der Service-Unit, `posix_fadvise` im Ling-Rezept, MMQ-Kachel J_MAX 128 in den vendierten CUDA-Libs. Der private Experten-Puffer mit `pread` (die erste Wahl des Auftrags) und der Layer-voraus-Prefetch bleiben draussen: der Puffer ist ein Nettoverlust, der Prefetch liest 40 % mehr Bytes, ohne die Wanduhr zu gewinnen.** Verifiziert nach dem Neustart: Dienst `active`, `MemorySwapMax=0`, `MemorySwapCurrent=0`, `LimitMEMLOCK=infinity`, Router `/health` HTTP 200, VRAM 4706 MiB vom GPU-Preload, Ling mit dem neuen Rezept 7,65 t/s auf "go" bei unveraenderter Ausgabe (SHA `ed287f50e17fa16b`).

| Hebel | Wo in agollm | Wirkung | Siege |
|---|---|---|---|
| `MemorySwapMax=0` | `scripts/agollm.service` (+ installierte Unit) | Decode +7,41 % | 2/2 |
| `LLAMA_EXPERT_FADVISE=1` | `config/llama/ling-3-flash.yaml` | Decode +3,73 % | 2/2 |
| MMQ `GGML_CUDA_TURING_MMQ_J_MAX=128` | `workers/llama/libggml-cuda.so.0` | Prefill +7,1 % (Q4_K), +10,1 % (Q8_0) | 2/2 |

- **Das Swap-Limit ist das systemd-Aequivalent von `vm.swappiness=0`.** Es wirkt auf den ganzen agollm-Dienst (auch whisper, TTS, kiwix). Rueckweg: eine Zeile in der Unit entfernen. Risiko: ein Modell, das den RAM wirklich uebersteigt, trifft dann einen OOM statt auf Swap.
- **`LimitMEMLOCK=infinity` kam mit.** Die Zeile stand bisher nur in der installierten Unit; ohne sie bricht ein frisches Install den mlock des LFM2.5-24B (ein 14-GB-Puffer am Standardlimit von 8 MB). Sie ist jetzt im Unit-Template `scripts/agollm.service` mitgeschrieben.
- **Der Rezept-Schalter bleibt ein Env-Schalter.** `LLAMA_EXPERT_FADVISE=1` steht nur im Ling-Rezept, weil nur Ling seine Experten von der SSD streamt. `LLAMA_EXPERT_PAGE_PREFETCH` bleibt Default aus.
- **Die Libs wurden gezielt kopiert, nicht ueber `vendor-binaries.sh`.** Geaendert sind `libggml-cpu.so.0` (neuer Export `ggml_set_moe_file_resolver`, fadvise-Fault-Pfad), `libggml-cuda.so.0` (MMQ J_MAX 128) und `libllama.so.0` (Resolver im Tier-Modul); `libggml.so.0`, `libllama-common.so.0`, `libllama-server-impl.so` und `llama-server` waren SHA-identisch und blieben unberuehrt. `vendor-binaries.sh` ueberschreibt `workers/llama` vollstaendig und haette den aufgebauten Stand samt ORIGIN-Historie plattiert. Backup mit den drei vorherigen Libs: `~/src/agollm/runtime/llama-build-before-fadvise-mmq-tile-20261106T225816Z/`, ORIGIN.txt-Eintrag `fadvise-mmq-tile=20261106T225816Z` (Fork-HEAD `6ad6cf87b45e4cdb25ab48b0613843068084cfd2` plus uncommittete Aenderungen).
- **Was nicht integriert wurde und warum.** Der private Experten-Puffer mit `pread` kostet 78 ms Kopie je Token (1,19 GB Gewicht je Token bei 15,2 GB/s RAM-Bandbreite) und waere damit ein Nettoverlust; `posix_fadvise` auf die Modelldatei erreicht denselben Page-Cache-Effekt ohne Kopie. Der Layer-voraus-Prefetch fuellt die SSD-Warteschlange dreifach, liest aber 40 % mehr Bytes; die Wanduhr folgt nicht (+14,9 % / +0,3 %, Runde 1 gegen Runde 2 widersprechen sich).

Messprogramme: `benchmark-results/tensorfold-review-20261106/ling-ab/`, `benchmark-results/mmq-tile-sweep/`. Integration: `~/src/agollm/README.md` (Abschnitt "Decode-Hebel vom 2026-11-06"), `~/src/agollm/workers/llama/ORIGIN.txt`.

# Ling-3.0-flash, zweite Runde vom 2026-11-06: der Decode verliert an der Speicherverwaltung, nicht an der SSD

**Ergebnis: Lings Decode wartet nicht an der Grenze der SSD, sondern im Swap des Kernels. `vm.swappiness=0` bringt 7,4 % (gepaart, 2/2 Runden), `posix_fadvise` statt `madvise` im Fault-Pfad bringt 3,7 % (2/2 Runden). Der Layer-voraus-Prefetch fuellt die SSD-Warteschlange dreifach (Leserate 788 auf 1100 MB/s, fehlende Slabs 21 auf 7 %), liest aber 40 % mehr Bytes und bringt deshalb keine gesicherte Wanduhrzeit. Der private Experten-Puffer mit `pread` - die erste Wahl - ist ein Nettoverlust: die Kopie der Gewichte kostet 78 ms je Token, mehr als die halbe Tokenzeit.** Die SSD-Rate bleibt in den A/B-Laeufen bei rund 790 MB/s; sie ist nicht die Grenze. Die Grenze ist die Reihenfolge aus Hinweis und Zugriff (Warteschlange leer, wenn das Geraet arbeiten soll) und der Speicherdruck des Page-Cache (72-GiB-Modell in 32 GB RAM).

- **Der Swap ist der teuerste Posten im Decode.** `tools/vmstat_delta.py` misst die Kernel-Zaehler ueber einen Lauf (128 Token, ein Prompt): `pgscan_direct` **+2.940.843 Seiten** (11,5 GB direct reclaim), `pgsteal_direct` +910.436 (3,5 GB), `allocstall_normal` **+8.540**, `allocstall_movable` +8.605, `pswpout` +167.628 (654 MB), `pswpin` +11.102. Der gleiche Zugriff im Mikro-Benchmark (`tools/bench_fault_interleave.py`): null Reclaim, keine Stalls, kein Swap. `vm.swappiness` stand auf **180** (Maximum 200): der Kernel tauscht die anonymen Seiten des Servers aus, statt die sauberen Dateiseiten des Modells zu reclaimed. Gepaart, zwei Runden, Arm-Reihenfolge in Runde 2 umgekehrt, gleiche Prompts und Seeds, Greedy-Ausgabe per SHA in allen vier Armen identisch:

| Arm | Runde 1 | Runde 2 | Mittel | Siege |
|---|---|---|---|---|
| `vm.swappiness=0` | +8,47 % | +6,35 % | **+7,41 %** | 2/2 |

  6,85 und 6,96 t/s bei 180 gegen 7,43 und 7,40 t/s bei 0. Der Arm mit `swappiness=0` liest **mehr** von der SSD (22,2 % und 21,8 % fehlende Slabs gegen 18,5 % und 20,3 %) und wartet laenger (93,9 s und 88,3 s Fault-Wait gegen 69,1 s und 86,5 s) - und ist trotzdem schneller: der Swap-Rauschen kostet mehr als die zusaetzlichen Bytes. Fuer den Betrieb ist das eine Zeile in der Service-Unit (`MemorySwapMax=0`), nicht im Code.
- **`posix_fadvise` statt `madvise` im Fault-Pfad: +3,7 %.** Neuer Pfad hinter `LLAMA_EXPERT_FADVISE=1`: `ggml/src/ggml-cpu/ggml-cpu.c:1847` loest die Adresse ueber einen vom Tier-Modul installierten Resolver (`src/llama-expert-tier.cpp`, `ggml_set_moe_file_resolver`) auf und legt die 128-kB-Fenster mit `posix_fadvise(WILLNEED)` auf die Modelldatei, statt sie nur im Mapping zu markieren. Der Resolver trifft die Datei in 802.560 von 758.240 Aufrufen (die beiden Zaehler werden an verschiedenen Stellen im Mutex erhoeht, die Summe ist deshalb nicht exakt). Gepaart, zwei Runden, umgekehrte Reihenfolge, SHA in allen vier Armen identisch:

| Arm | Runde 1 | Runde 2 | Mittel | Siege |
|---|---|---|---|---|
| `LLAMA_EXPERT_FADVISE=1` | +4,22 % | +3,23 % | **+3,73 %** | 2/2 |

  6,46 und 6,81 t/s gegen 6,74 und 7,03 t/s. Die **SSD-Rate bleibt dabei gleich** (752-784 MB/s in allen vier Laeufen, 84-89 KiB mittlere Request-Groesse, qd 5,4-7,9): der Gewinn ist eine kuerzere Wartezeit, keine hoehere Bandbreite.
- **Der Hinweis vor dem Zugriff fuellt die Warteschlange - die Wanduhr folgt nicht.** `LLAMA_EXPERT_PAGE_PREFETCH=8` legt die vorhergesagten Experten der naechsten Layer mit `posix_fadvise` auf die Modelldatei, waehrend die aktuelle Layer rechnet (`src/models/bailingmoe3.cpp:420`; der Prefetch benutzte bisher `madvise`, jetzt `prefetch_slab` in `src/llama-expert-tier.cpp`). Der Mechanismus ist eindeutig messbar, die Zeit nicht:

| Groesse | `LLAMA_EXPERT_FADVISE=1` | `+ LLAMA_EXPERT_PAGE_PREFETCH=8` |
|---|---|---|
| MEAN tg (Runde 1 / Runde 2) | 6,52 / 7,15 t/s | **7,49 / 7,17 t/s** |
| Leserate der SSD | 788 / 806 MB/s | **1100 / 1120 MB/s** |
| Warteschlange (qd) | 6,8 / 6,2 | **18,5 / 20,2** |
| fehlende Slabs | 21,3 / 19,5 % | **7,2 / 7,7 %** |
| Fault-Wait (Thread-Summe) | 95,3 / 75,0 s | **40,1 / 43,5 s** |
| Bytes von der SSD | 130,8 / 124,2 GB | 173,7 / 180,4 GB |

  Gepaart, zwei Runden, umgekehrte Reihenfolge, SHA in allen vier Armen identisch: Runde 1 **+14,9 %**, Runde 2 **+0,3 %** (Mittel +7,6 %, 2/2 Siege, aber die Runden widersprechen sich). Der Grund steht in derselben Tabelle: der Prefetch liest **40 % mehr Bytes** (die Vorhersage hat nur 78 % Recall bei 8 Vorhersagen), und die gesparte Wartezeit wird als Geraetezeit ausgegeben. Der Prefetch bleibt aus, bis die Vorhersage treffgenauer ist oder der Overhead sinkt - die Stellschraube ist `top_m` (4, 8, 16), nicht der Mechanismus.
- **Die Grenze ist die Reihenfolge, nicht die Request-Groesse.** `tools/bench_fault_interleave.py` baut den Ist-Pfad nach (8 Threads, 384 verstreute 1-MiB-Bloecke, Span 8 GiB): **interleave** (Hinweis und Zugriff je Block, so wie der Code heute arbeitet) **916 MB/s**, **batched** (alle Hinweise, dann alle Zugriffe) **2048 MB/s**, **fadvise** (Hinweis und Zugriff je Block, aber ueber die Datei) **2974 MB/s**, **pread** 2993 MB/s. Die Request-Groesse ist nicht die Grenze: `tools/measure_nvme_random.py` mit 8 Threads und verstreuten Bloecken liefert 2152 MB/s bei 64 KiB, 2825 bei 128 KiB, 2300 bei 256 KiB, 2888 bei 1 MiB. Der Auftrag muss **vor** dem Zugriff stehen, sonst ist die Warteschlange leer, wenn das Geraet arbeiten soll.
- **Der private Experten-Puffer mit `pread` ist ein Nettoverlust - die erste Wahl des Auftrags ist damit widerlegt.** `tools/measure_ram_bandwidth.c` (gcc -O2 -pthread) misst die Kopierbandbreite des RAM: **10,3 GB/s** mit einem Thread, 11,9 mit zwei, 13,1 mit vier, **15,2 GB/s** mit acht. Lings Decode beruehrt je Token 40 Layer x 8 Experten x 3,64 MiB = **1,19 GB** Gewichte. Die Kopie in einen privaten Puffer kostet damit 1190/15,2 = **78 ms je Token**, also 57 % der heutigen Tokenzeit (138 ms), bevor der schnellere SSD-Zugriff ueberhaupt wirken kann. Der theoretische Gewinn der Platte (Device-Zeit 140 ms auf 36 ms) wird von der Kopie mehr als aufgezehrt; ein echter Gewinn ist nur mit einem privaten Puffer *und* perfekt ueberlappter Kopie zu erwarten, also mit einem sehr grossen Eingriff in den Matmul-Pfad. Die Kopie ohne Puffer - `posix_fadvise` auf die Modelldatei - bringt denselben Page-Cache-Effekt ohne Kopie und kostet nichts; deshalb ist `posix_fadvise` der Weg, nicht `pread`.

Messprogramme: `benchmark-results/tensorfold-review-20261106/ling-ab/` (`tools/ab_disk.sh` mit SSD-Statistik je Lauf, `tools/ab_sysctl.sh` mit System-Schalter je Arm, `tools/summarize_disk.py`, `tools/run.sh`), `benchmark-results/tensorfold-review-20261106/ling-ab/runs/{base-1,fadv-1,fadv-2,base-2,sw180-1,sw0-1,sw0-2,sw180-2}/`. Werkzeuge: `tools/bench_fault_interleave.py`, `tools/vmstat_delta.py`, `tools/measure_ram_bandwidth.c`, `tools/summarize_disk.py`.

# MMQ-Kachel auf Turing: 128 Spalten statt 64 bringen 7-10 % Prefill vom 2026-11-06

**Ergebnis: der quantisierte Matmul - 71,9 % der Prefill-Kernelzeit - war auf der 1660 Ti nicht optimal konfiguriert. Eine Kachel mit 128 statt 64 Spalten bringt 7,1 % (Q4_K) und 10,1 % (Q8_0) Prefill, ohne die Ausgabe zu aendern (SHA `71f5cc0f7ad629fb` bei J=64 und J=128). Mehr Zeilen je Kachel (I=128) kostet dagegen 22 %.** Der Build nutzt `GGML_CUDA_TURING_MMQ_DP4A=ON` (DP4A-MMQ statt MMA-Datenlayout auf sm_75); die Kachelkonfiguration kam aus der Pascal-Tabelle (`ggml/src/ggml-cuda/mmq-config-pascal.cuh`: I=64, J bis 64, ohne Stream-K). Neu ist `ggml/src/ggml-cuda/mmq-config-turing.cuh` mit den drei Compile-Time-Knobs `GGML_CUDA_TURING_MMQ_{I,J_MAX,STREAM_K}`, eingebunden in `ggml/src/ggml-cuda/mmq.cuh:234` und verwendet in der Host-Auswahl (`:258`) wie im Kernel (`:279`). `tools/mmq_tile_sweep.sh` schreibt die Werte um, baut neu und misst mit llama-bench (Prefill 4096 Tokens, `-r 3`, beide Modelle voll auf der GPU):

| Konfiguration | MiniCPM5-2B (D128, Q4_K_M) | Intern-Decision-2B (D256, Q8_0) |
|---|---|---|
| I=64, J=64 (Pascal-Tabelle, bisher) | 1295,9 / 1292,7 t/s | 1986,2 / 1981,5 t/s |
| I=128, J=64 | 1010,3 t/s (-22,0 %) | 2057,3 t/s (+3,6 %) |
| **I=64, J=128** | **1385,2 / 1387,5 t/s (+7,1 %)** | **2184,0 / 2184,7 t/s (+10,1 %)** |
| I=64, J=128, Stream-K | 1378,5 t/s (+6,5 %) | 2261,6 t/s (+14,0 %) |
| I=128, J=64, Stream-K | 1067,2 t/s (-17,7 %) | 2112,4 t/s (+6,4 %) |

  Die erste und die letzte Zeile jeder Spalte sind derselbe Aufbau (Baseline am Anfang und Ende der Runde): der Drift liegt bei 0,25 %, die Messung ist stabil. Decode bleibt unveraendert (134,9-135,1 t/s und 108,5-108,9 t/s), der Knopf beruehrt nur den Prefill-Pfad. **Warum J=128 hilft:** die Kachel laedt das Gewichts-Band einmal fuer J Spalten, eine groessere Kachel liest die Gewichte also halb so oft - der Prefill ist bandbreiten- und nicht rechengebunden. **Warum I=128 schadet:** das x-Band der Kachel verdoppelt sich, der Block belegt doppelt so viel Shared Memory und es passt nur ein Block je SM statt zwei - die Latenzverdeckung geht verloren. **Stream-K** ist typabhaengig (+3,6 % bei Q8_0, -0,5 % bei Q4_K) und bleibt deshalb aus. Der Default ist jetzt `J_MAX=128`; `GGML_CUDA_TURING_MMQ_J_MAX=64` stellt die alte Kachel wieder her.

Messprogramme: `benchmark-results/mmq-tile-sweep/{mmq,mmq2}/` (Log je Build und Modell). Werkzeuge: `tools/mmq_tile_sweep.sh`, `tools/gpu_output_check.sh` (Ausgabe-SHA eines Builds gegen einen anderen).

# Ling-3.0-flash: SSD-Pfad, Phasen-Timing und CPU-Knops; FA-Tile-Sweep D128/D256 vom 2026-11-06

**Ergebnis: Lings Decode ist zu drei Vierteln SSD-Wartezeit, nicht Rechenzeit - und die Platte kann mit Lings Zugriffsmuster rund 3 GB/s, waehrend der Fault-Pfad daraus 1,4-1,75 GB/s holt und in situ 0,85 GB/s ankommen.** Die vier nie dokumentierten CPU-Knobs bringen keine Zeit (16 Threads kosten 51 %). Auf der GPU-Seite ist die FA-Kachel fuer Kopfgroesse 128 und 256 bereits optimal, der FA-Kern ist im Prefill nur 12,6 % der Kernelzeit. Die alte Notiz "die SSD kann 0,9-1,2 GB/s, das ist ihre Grenze" ist damit widerlegt: gemessen wurde die **Wiederverwendung** des Page-Cache, nicht die Platte. Details in `TENSORFOLD_REVIEW.md` Abschnitt 10.

- **Phasen-Timing: die kalten Experten sind die Tokenzeit, und sie warten.** `LLAMA_EXPERT_STATS=1` + `LLAMA_EXPERT_TIMING=1` (`runs/diag/server.log`, 40 Layer, 836 Laeufe je Layer) liefert je Layer `cpu_cold_ms`, `cpu_gate_up_ms`, `cpu_activation_ms`, `cpu_down_ms`. Je Token ueber 40 Layer: **138,4 ms** kalte Experten, davon **33,6 ms Rechnen** (Gate/Up + Aktivierung + Down) und **104,8 ms Fault-Wait und Barrieren**. Die gemessene Tokenzeit liegt bei 136 ms. Der Anteil der fehlenden Slabs betraegt 19,8 % (162.372 von 818.958 Pruefungen), die Fault-Wartezeit 84,3 s (Thread-Summe).
- **Die SSD kann das Muster dreimal so schnell, wie der Pfad es abholt.** `tools/measure_nvme_random.py` liest verstreute 1-MiB-Bloecke (so liegen Lings Experten: 1,05 MiB gate/up, 1,54 MiB down je Expert und Layer) mit leerem Page-Cache (`posix_fadvise(DONTNEED)`): 887 MB/s mit einem Thread, **2948 MB/s mit 8**, 3304 mit 16, 3390 mit 32. Die fruehere Notiz "zufaellig gelesen nur 0,9-1,0 GB/s, schon ab QD 4" galt fuer 4-KiB-Reads, nicht fuer 1-MiB-Bloecke.
- **Die Grenze ist der Mechanismus, nicht die Platte.** `tools/bench_fault_random.py` baut den Ist-Pfad nach: madvise mit 128-kB-Fenstern plus Touch (der Code, `ggml/src/ggml-cpu/ggml-cpu.c:1779`) kommt auf 1750 MB/s, madvise mit 1-MiB-Fenstern auf 732 MB/s (der Kernel liest je Aufruf nur ein Readahead-Fenster), **`pread` in einen privaten Puffer auf 2984 MB/s** (8 Threads). In situ liefert Ling 0,73-0,85 GB/s bei ~8300 IOPS und ~88 kB mittlerer Request-Groesse - also ein Drittel der Faehigkeit des eigenen Mechanismus. Ursache: der Fault-Pfad arbeitet streng seriell, erst alle Faultholen dann rechnen; die SSD steht waehrend der Rechenphase leer.
- **Die Fenstergroesse 128 kB ist richtig eingestellt.** `tools/bench_fault_chunks.py` (8 Threads, verstreute Bloecke): 32 kB 981 MB/s, 64 kB 1330, **128 kB 1426**, 256 kB 580, 512 kB 543, 1 MiB 653. Jede Aenderung an `const uintptr_t chunk = 128 * 1024;` waere eine Verschlechterung.
- **Die vier CPU-Knobs bringen keine Zeit.** Gepaart, zwei Runden, Arm-Reihenfolge in Runde 2 umgekehrt, gleiche Prompts und Seeds, Greedy-Ausgabe per SHA in allen Armen identisch (Korrektheit neutral). `LLAMA_EXPERT_CPU_REUSE_ROWS` und `LLAMA_EXPERT_CPU_MULTI_ROW` sind im Decode No-ops und wurden nicht weiter verfolgt: im Decode hat jeder Expert-Slot genau eine Spalte, die Zeilen-Wiederverwendung greift erst ab zwei Spalten, also erst im Prefill.

| Arm | Runde 1 | Runde 2 | Mittel | Siege |
|---|---|---|---|---|
| `LLAMA_EXPERT_CPU_ACT_PARALLEL=1` | -3,03 % | -1,05 % | **-2,04 %** | 0/2 |
| `LLAMA_EXPERT_CPU_DOWN_PREFETCH=4` | +0,09 % | +0,04 % | **+0,06 %** | 2/2 |
| `LLAMA_ARG_THREADS=16` + `LLAMA_ARG_THREADS_BATCH=16` | -53,71 % | -48,26 % | **-50,98 %** | 0/2 |

  `ACT_PARALLEL` senkt die Fault-Wartezeit zwar (99,3 s und 94,1 s gegen 123,7 s und 92,8 s der Basis), die Wanduhr bewegt sich aber nicht - die Wartezeit wandert nur in die Rechenphase. `DOWN_PREFETCH=4` ist mit +0,06 % Rauschen (2/2 Siege, aber 0,04-0,09 % sind keine Zeit). **`LLAMA_ARG_THREADS=16` kostet die Haelfte**: 8 physische Kerne mit SMT, der Prefill faellt von ~8 auf 4,3 t/s, und die Zahl der fehlenden Slabs steigt (21,1 % und 20,6 % gegen 24,4 % und 22,0 %). Das Rezept setzt `LLAMA_ARG_THREADS=8` bereits auf die Zahl der physischen Kerne; SMT verdoppelt die Faehthoehe und halbiert den Durchsatz je Thread.
- **GPU: die FA-Kachel fuer D128 und D256 ist auf der 1660 Ti bereits optimal.** Der FA-Tile-Kern nimmt 32 Query-Spalten je Block, sobald mehr als `16/ncols2` Spalten anliegen - das ist der Prefill-Pfad (`ggml/src/ggml-cuda/fattn-tile.cuh:27-86`, Zeile `ncols=32`; der D64-Knopf wirkt aus demselben Grund nur auf den Vision-Encoder). Neu hinter Compile-Time-Makros `GGML_CUDA_FATTN_TILE_D{128,256}_{NTHREADS,NBATCH_FA,NBATCH_K}` (Default = Upstream). `tools/fa_tile_sweep.sh`, Prefill 4096 Tokens, `-r 3`:

| Konfiguration | MiniCPM5-2B (D128, gqa 8) | Intern-Decision-2B (D256, gqa 4) |
|---|---|---|
| 256 Threads (Upstream) | **1300,3 t/s** | **1994,6 t/s** |
| 128 Threads | 1173,5 t/s (-9,8 %) | 1936,0 t/s (-2,9 %) |
| 512 Threads | 1247,6 t/s (-4,1 %) | 1975,3 t/s (-1,0 %) |

Decode bleibt in allen drei Konfigurationen unveraendert (135,4 / 135,6 / 135,2 t/s), wie erwartet - der Knopf beruehrt nur die `ncols=32`-Zeile. `nbatch_K=128` ist fuer D256 nicht baubar (59.392 Bytes Shared Memory ueber dem sm_75-Limit von 49.152).
- **FA ist im Prefill eine Nebenrolle.** `ncu --metrics gpu__time_duration.sum` (als root, `nsys` hat keinen Importer) auf MiniCPM5-2B mit 1024 Prefill-Tokens: `mul_mat_q` 71,9 %, `flash_attn_tile` 12,6 %, `mul_mat_vec_q` 6,7 %, Rest 8,8 %. Der Prefill wird vom quantisierten Matmul dominiert; die FA-Kachel kann daran nur die 12,6 % verschieben.

Messprogramme: `benchmark-results/tensorfold-review-20261106/ling-ab/` (`tools/confirm2.sh` mit umgekehrter Arm-Reihenfolge, `tools/summarize_paired.py`, `tools/run.sh`), `benchmark-results/fa-tile-sweep/{baseline,d128t128-d256t128,d128t512-d256t512}/`, `benchmark-results/gpu-bench/ncu-d128.csv`. Werkzeuge: `tools/measure_nvme.py`, `tools/measure_nvme_random.py`, `tools/bench_fault_io.py`, `tools/bench_fault_random.py`, `tools/bench_fault_chunks.py`, `tools/gguf_tensor_info.py`, `tools/bench_gpu.sh`, `tools/fa_tile_sweep.sh`.

# TensorFold-Konzepte: getestet und gemessen vom 2026-11-06

**Ergebnis: Von den vier uebernommenen TensorFold-Ideen ist keine ein Gewinn. Zwei sind widerlegt, eine hat auf dieser Maschine keinen Effekt, eine funktioniert mechanisch, bringt aber keine Zeit.** Die Analyse (`TENSORFOLD_REVIEW.md`) wurde gegen den Fork geprueft statt geglaubt: der angebliche "Bug im Vulkan-Backend" existiert nicht (B1), die CLOCK-Eviction ist zu schwach (B4), die FA-Kachelbreite hat nur bei GQA-Rate > 8 Luft - die kein lokales Modell GPU-gebunden trifft (B2). Der Page-Prefetch fuer Ling senkt die Zahl der Page-Faults um ein Drittel bis zur Haelfte, die Wanduhr bewegt sich nicht, weil die SSD bei 0,9-1,2 GB/s bandbreiten- und nicht latenzgebunden ist (B3). Gemessen wurde A/B/A/B ohne Cache-Leerung zwischen den Laeufen, gleiche Prompts und Seeds, Greedy-Ausgabe per SHA verglichen. `agollm` war gestoppt (`systemctl stop agollm.service`, VRAM 1 MiB).

- **B1 - Vulkan-Debug-Ausgaben: widerlegt, nichts geaendert.** Die 164 `std::cerr` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` sind **Argumente des `VK_LOG_DEBUG`-Makros**, keine eigenen Aufrufe (Klammerbilanz laeuft ueber vier Zeilen von +1 auf -1). Ohne `-DGGML_VULKAN_DEBUG` ist der Block `((void) 0)`: Praeprozessor-Output ohne Format-Strings, Binary ohne Format-Strings (0 Treffer), Baseline-Bench ueber Vulkan mit 0 Debug-Zeilen. Die urspruengliche Schaetzung (24-66 ms/Token, 18-50 % der Vega-Tokenzeit) ist hinfaellig.
- **B2 - FA-Arbeitsaufteilung nach KV-Laenge: Mechanik geklaert, kein lokaler Effekt.** `ncols2` im FA-MMA-Kernel ist die Zahl der Query-Koepfe, die sich einen KV-Kopf teilen (`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1224`); das K/V wird `gqa_ratio/ncols2` Mal gelesen. Upstream decelt die Kachel bei `min(gqa_ratio, 8)` (`fattn.cu:163-179`), im MLA-Pfad (DKQ=576) bei 16 (`:304-307`) - ein Hebel gibt es nur bei `gqa_ratio > 8` bzw. > 16. Ein Scan aller GGUF-Header im lokalen Zoo (`tools/gguf_attn_scan.py`) zeigt: nur Ling-3.0-flash hat GQA-Rate 32 (MLA, 7 Vollattention-Layer), alle anderen liegen auf dem Deckel (Qwen3.6-35B-A3B 8, Hy-MT2-30B 8, MiniCPM5-2B 8, Qwen3-1.7B 2). Ling laeuft aber SSD-gebunden (GPU 4-7 % Auslastung, 5514 MiB Hot-Set), dort ist der FA-Kernel kein Engpass. Implementiert hinter `GGML_CUDA_FA_NCOLS2` (0 = auto, 8 = aus, 16/32 = erzwingen) und `GGML_CUDA_FA_KV_WIDE_THRESHOLD` (Default 16384), Default-Verhalten unveraendert; Prefill-Sweep Qwen3-1.7B: 903,9 / 907,8 / 905,0 t/s fuer auto/16/8 (alle innerhalb 0,4 %). Fuenf neue FA-Instanzen fuer DKQ=128 (Generator `generate_cu_files.py`).
- **B3 - bailingmoe3 (Ling) an Lookahead + Page-Prefetch: umgesetzt, kein Zeitgewinn.** `src/models/bailingmoe3.cpp` hatte keine Verdrahtung; jetzt sind Lookahead-Praediktor und Page-Prefetch drin, nach dem Muster von `kolibri1.cpp`. Lings Router ist **gruppiert** (512 Experten in 8 Gruppen, Top-2 je Gruppe, Top-4 Gruppen, Top-8 Experten) - der Praediktor bildet die Gruppenauswahl exakt nach, ein globales Top-k naeme Experten aus verworfenen Gruppen. Der Trace zeigt, dass die Vorhersage traegt: 75,2 % Top-1-Treffer, 77,9 % Recall bei 8 Vorhersagen (Referenz Qwen3.6-35B-A3B: 73,3 % / 86,1 %). Der Prefetch senkt die fehlenden Page-Faults **in jeder Runde** von 32,9/38,1/32,9 % auf 21,9/21,1/21,6 % und die Fault-Wait im warmen Lauf von 751 s auf 382 s. Die **Wanduhr bleibt aber stehen**: gepaart je Runde -4,0 %, +2,8 %, +0,8 % gegen die Basislinie (im Mittel -0,2 %), auf dem langen Prompt in zwei von drei Runden schlechter (4,45 / 5,21 / 5,64 gegen 5,59 / 5,44 / 5,78 t/s). Grund: Ling liest mit 0,9-1,2 GB/s von der SSD. Ein Prefetch holt dieselben Bytes frueher, er erzeugt keine zusaetzliche Bandbreite. (Die Deutung "das ist die Grenze der Platte" ist durch die Messung vom 2026-11-06 widerlegt: die Platte schafft mit Lings Zugriffsmuster rund 3 GB/s, siehe der Eintrag oben und `TENSORFOLD_REVIEW.md` Abschnitt 10. Der Prefetch scheitert nicht an der Bandbreite, sondern daran, dass der Fault-Pfad mit Rechnung und Lesen strikt abwechselt.) Der Schalter `LLAMA_EXPERT_PAGE_PREFETCH` bleibt aus. Drei Runden, Runde 3 in umgekehrter Arm-Reihenfolge (die Rundenvarianz von 5,75 bis 6,42 t/s ist groesser als jeder Armunterschied); Greedy-Ausgabe per SHA in allen Armen identisch.
- **B4 - CLOCK-Eviction: simuliert, zu schwach, nicht implementiert.** `tools/simulate_expert_clock_cache.py` rechnet CLOCK (nach `zig/src/core/fsr/cache.zig:126-148`) ueber den Lookahead-Trace (Qwen3.6-35B-A3B, 40 Layer, 256 Experten, top-8, 4953 Records) gegen LRU: pro Layer -0,4 bis -2,2 %, global -1,6 bis -5,3 % (bester Fall 9632 -> 9122 Misses). TensorFold nennt 11 % weniger Disk und 8 % weniger Zeit - weniger als die Haelfte, und ohne Implementierungsaufwand in `llama-expert-tier.cpp` nicht zu holen.

Messprogramme: `benchmark-results/tensorfold-review-20261106/` (`ling-ab/` mit `tools/abn.sh`, `round.sh`, `trace.sh`, `summarize.py`, `mbench.py`, `run.sh`; `fa-tile-sweep/`; `fa-longkv-server/`), Werkzeuge `tools/gguf_attn_scan.py`, `tools/gguf_attn_config.py`, `tools/simulate_expert_clock_cache.py`, `tools/bench_fa_longkv.py`.

# Mage-VL-Cowork: Bildzeit vom 2026-11-02

Mage-VL-Cowork 4B (Q4_K_M, Q8_0-mmproj, Projektor `magevl` aus PR 26796 portiert). Bildzeit = Encode + Prefill der Bild-Token bei gecachtem System-Prompt, 1920x1080:

| | GTX 1660 Ti | Vega 7 |
|---|---:|---:|
| vorher | 5,97 s | 37,7 s |
| volle Aufloesung | 5,16 s | 32,0 s |
| `--image-max-tokens 1024` | 2,24 s | 12,6 s |
| `--image-max-tokens 768` | 1,55 s | 8,5 s |

- CUDA Tile-FA Kopfgroesse 64 / 32 Spalten: 128 statt 256 Threads (Encode 2,78 -> 2,42 s, LFM2.5 unveraendert). Zurueck mit `-DGGML_CUDA_FATTN_TILE_D64_NTHREADS=256`.
- `half_kq`: K*Q-Summen mit hfma2 fuer Kopfgroesse 40/64/72/80, nur wenn die Op keine F32-Praezision verlangt. Die LLM-Graphen verlangen sie immer; mtmd nur ohne `LLAMA_MTMD_FA_PREC=f16`. Aus mit `GGML_CUDA_FA_TILE_HALF_KQ=0`. Encode -10 %.
- RoPE im magevl-Graph ohne Permute/Cont: identische Ausgabe, -4 %.
- `GGML_VK_FA_SCALAR_{BR,BC,WG,ROW_SPLIT,OCC}`: Overrides der Vulkan-Scalar-FA fuer grosse Batches; die Defaults sind auf Vega fast optimal.
- `--image-max-tokens` wirkt jetzt bei magevl. Auf Vega sind FA (600 GFLOPS) und Prefill (135 t/s) rechengebunden; dort hilft nur die Tokenzahl.
- Die GPU-FA-Kernel bilden K*Q in FP16; beim Mage-ViT weicht das deutlich von der CPU ab (Cosinus 0,984 CUDA, 0,991 Vulkan, Q8_0-mmproj 0,982). Klick- und Lesetests aendern sich dadurch nicht.

Artefakte (gitignored): `benchmark-results/mage-vl-cowork-20261102/` (`vlbench.py`, `embdump.cpp`, `README.md`). Start: `start-mage-vl-cowork.sh`.

# CMOE-Prefill bei knappem VRAM vom 2026-11-01

**Ergebnis: Passte der Prefill-Ubatch nach einer Decode-Phase nicht mehr in den VRAM, fiel der Server direkt auf den Decode-Ubatch 64 zurueck und versuchte vor jedem 64er-Batch erneut die volle Reservierung. In agollm rechnete Clover-35B so einen 36,7k-Tokens-Prompt mit 30 t/s (20 Minuten). Neu: `LLAMA_CMOE_PREFILL_STEPDOWN` (Default 1) halbiert den Prefill-Ubatch bis knapp ueber den Decode-Ubatch und behaelt die passende Groesse, bis die Slots leer laufen. Unter gleicher Klemme (500 MiB belegt) 222,5 statt 40,0 t/s.**

- **Ursache:** Der Decode-Wechsel schrumpft den Compute-Puffer (gallocr-Shrink). Waehrenddessen belegen CUDA-Pool, Expertenpuffer und der DFlash-Draft den freien Platz; der naechste Prefill braucht 606 MiB fuer Ubatch 3072 und bekommt sie nicht. Jeder Fehlversuch raeumt zusaetzlich die CUDA-Graphen ab.
- **Test** (Clover-Rezept auf eigenem Port, `hog` haelt per cudaMalloc 500 MiB nach dem ersten Decode, frei bleiben 358 MiB):
  - Neu: 3072 und 1536 scheitern, 768 passt. 9852-Token-Prompt 222,5 t/s, 3276 Tokens 205,9 t/s.
  - Alt (`LLAMA_CMOE_PREFILL_STEPDOWN=0`): 3276 Tokens 40,0 t/s, 48 gescheiterte Reservierungen.
  - Ohne Klemme danach: wieder 3072, 9852 Tokens 405,7 t/s (Grenze wird beim Leerlauf zurueckgesetzt).
- **Absturz behoben:** Bei so wenig VRAM scheiterte danach die Reservierung des DFlash-Draft-Graphen (41 MiB). `ggml_backend_sched_alloc_splits` ignorierte den Rueckgabewert von `ggml_gallocr_reserve_n`, `ggml_gallocr_alloc_graph` legte die Tensoren in einen NULL-Puffer (Segfault). Jetzt meldet beides einen Fehler, der Draft schreibt "llama_decode returned -2" und das Modell decodiert ohne Entwurf weiter.
- Alle Startskripte mit CMOE-Prefill setzen den Knopf, ebenso die 18 agollm-Rezepte. Ein leerer Wert gilt als Default.

Messprogramme: scratch `stepdown/hog.cu`, `run.sh`, `long.json`/`mid.json`.

# SenseNova-Loop-9B-Preview Q3_K_M vom 2026-11-01

**Ergebnis: Die Architektur `sensenova_loop` (Qwen3.5-9B-Hybrid, Schichten 4..27 laufen zweimal, zwei mHC-Residual-Streams mit Sinkhorn-Router um jede Teilschicht) ist in den Fork portiert und laeuft voll auf der 1660 Ti: Decode 23,2 t/s (erste Portierung 20,4), Prefill ~255 t/s, Code-Edits mit ngram-simple 63,8 t/s. CPU und GPU liefern denselben Greedy-Text. Nachgeruestet: der trainierte MTP-Kopf aus dem HF-Repo. ngram-simple plus MTP bringt mit Server-Sampling Code 29,5, Thinking 27,7-28,3, Edits 70,3 und deutsche Prosa 23,9 t/s (nur ngram: 22,2 / 20,8-21,6 / 64,6 / 22,6). Dafuer sinkt der Kontext auf 16384 mit f16-KV (VRAM-Spitze 5470-5516 MiB).**

Messungen, Werkzeuge und Profile: `benchmark-results/sensenova-loop-9b-20261101/` (README). agollm war gestoppt.

- **Portierung (aus GuideboardLabs/llama.cpp-sensenova-loop, dort nur CPU/Vulkan):**
  - Neue Ops `ggml_mhc_route`/`ggml_mhc_merge` (hinter `GGML_OP_GLU` angehaengt, Sinkhorn-Iterationen und Epsilon als Op-Parameter), CPU und neue CUDA-Kernel (`ggml-cuda/mhc.cu`), `test-backend-ops` test_mhc.
  - `llama_model_sensenova_loop` erbt von qwen35 und liest nur `mhc.streams`/`mhc.layer_kinds`. Der qwen35-Lader ueberspringt die Bloecke 28..51 und kopiert die Layer 4..27; der qwen35-Graph hat den Zwei-Stream-Pfad.
  - `--decision-seqs 8` (Default) kostet hier 703 MiB Recurrent-State; Rezept und Skript setzen 0.
- **Q3_K-Matvec ist der Engpass:** 57 % der Kernelzeit, 134 GB/s gegen 283 GB/s bei Q4_K; begrenzt durch Lade- und Integerbefehle, nicht durch DRAM.
  - `ggml_cuda_vsub4_i8`: `((a | 0x80808080) - b) ^ 0x80808080` statt `__vsubss4`, das NVIDIA seit Maxwell emuliert. An allen 14 Stellen (Q3_K/Q6_K-Matvec, MMQ-Loader Q4_0/Q5_0/Q3_K/Q6_K, FA-Dequant Q4_0/Q5_0) sind die Bytes kleiner 0x80 und keine Spur saettigt, das Ergebnis ist gleich. `-DGGML_CUDA_SWAR_VSUB=0` schaltet zurueck.
  - Q3_K-MMVQ mit VDR 2 (`GGML_CUDA_Q3_K_MMVQ_VDR`): zwei benachbarte ints teilen Skalen und q8_1-Bloecke. VDR 1: 22,57, VDR 2: 23,21, VDR 4: 21,30 t/s.
  - `GGML_CUDA_MMVQ_Q3_K_NCOLS1_ROWS=2` im Rezept und Skript (0: 20,7, 1: 20,0, 2: 23,2, 4: 20,9 t/s). Der Default bleibt 0, weil derselbe Knopf geroutetes Q3_K (MUL_MAT_ID) trifft.
- **A/B gegen die alten Kernel** (gleiche CMake-Optionen inkl. `GGML_CUDA_TURING_MMQ_DP4A`, A/B/A/B): SenseNova tg128 21,13 -> 22,79 (+7,9 %), Spark-X2.6 Q6_K 62,5 -> 64,0 (+2,5 %), FrogNano 55,5 -> 56,0 (+0,9 %), Prefill unveraendert. Ein Build mit Standardoptionen hat kein DP4A-MMQ (Prefill 290 statt 800 t/s) und taugt nicht als Vergleich.
- **Nicht uebernommen:** Router-Reduktion auf 8/16 Bloecke verteilt plus Abschlusskernel und Sinkhorn parallel zum Schreiben (beides 23,1-23,4 t/s, Router nur 3,7 % der Zeit). CUDA-Graphen fuer dichte Modelle wieder einschalten (`llama-expert-tier` setzt `GGML_CUDA_DISABLE_GRAPHS=1` fuer jedes Modell): MiMo 40,7 / 40,7, FrogNano 55,0 / 54,9 t/s, kein Gewinn.
- **Kontext:** f16-KV schlaegt q8_0 (tg128 23,2 / 22,0; bei 16k 21,2 / 18,8; q8_0 bei 32k 16,4). ctx 20480 f16: 17,5k-Prompt mit 221 t/s, Decode 20,9 t/s, 5468 MiB; ctx 24576 erreichte 5696 von 5748 MiB. KVFlash aus (LRU nimmt verdraengte Bloecke aus der Attention).
- **Spekulation:** ngram-simple Code-Edit 63,8 t/s (855/1200 angenommen) gegen 22,7 ohne, Prosa -1 bis -3 %; ngram-mod (n-match 16) 47,9 t/s. Die Greedy-Ausgabe weicht erst an einer Beinahe-Gleichheit ab (Batch-Verify-Rundung).
- **MTP-Kopf:** Die guideboardlabs-GGUF laesst ihn weg; `mtp.safetensors` (243M Parameter, BF16) liegt im HF-Repo, die Modellkarte nennt MTP trainiert.
  - `scripts/sensenova_loop_add_mtp.py` haengt ihn als Block 56 an (Norm +1 wie der Rumpf, per Range-Request an Schicht 3 geprueft); `download-sensenova-loop-9b.sh` laedt und baut ihn (Q4_0, +131 MiB; gleiche Trefferquote wie Q8_0).
  - Fork: `sensenova_loop` bekommt fuer den MTP-Kontext den reinen Attention-KV-Cache wie qwen35.
  - `LLAMA_MTP_PROCESS_CHUNK` (Default 8 wie bisher, 0 = Draft-Ubatch): Der MTP-Nachlauf rechnete in 8er-Haeppchen mit Logits je Haeppchen und las so bei einem 13k-Prompt 1668-mal den LM-Kopf. Mit 0: Prefill 199,8 -> 221,8 t/s (nur ngram 225,4), Edit 63,2 -> 70,1 t/s, Trefferquote gleich. Rezept und Skript setzen 0.
  - FrogNano (nur MTP, ohne mmproj wie in agollm), A/B/A/B: 6k-Prompt 569/552 -> 675/662 t/s (+19 %); Decode, Trefferquote und VRAM gleich, weil die Verify-Batches (3 Tokens) unter 8 liegen. `start-frognano-4b.sh` und das agollm-Rezept setzen 0.
  - Ling-3.0-tiny mit NextN (`start-ling-tiny-mtp.sh`), A/B/A/B: 6k-Prompt 1008 -> 1174 t/s, 1k-Prompt ~1085 -> ~1280 t/s; greedy Edit 45,3 / 46,8 t/s bei gleicher Trefferquote. Beide Ling-tiny-Skripte setzen 0.
  - Qwen3.6-35B-A3B im MTP-Modus (`start1660.sh`, sonst DFlash): 0 und 256 lassen den 6k-Prefill gleich (405 / 411 t/s, die CPU-Experten dominieren), Decode +2-4 % an der Rauschgrenze, aber die VRAM-Spitze steigt 5390 -> 5570 MiB. Bleibt 8, ebenso `start.sh`/`start1080.sh` (gleicher Kopf, andere Maschine). Ornith hat keinen NextN-Kopf. Alle MTP-faehigen Startskripte haben den Knopf jetzt ausdruecklich.
- **FrogNano mit mmproj (Bilder) repariert:** `start-frognano-4b.sh` startete mit Projektor nicht mehr bzw. stuerzte ab. Drei Fehler, alle im Fork behoben bzw. im Skript beruecksichtigt:
  - `server_tokens::get_tokens()` verlangte "kein mmproj geladen". Der Spekulationspfad nimmt bei Text-Prompts die Tokens per Referenz, also `GGML_ASSERT(!has_mtmd)` bei jeder Textanfrage mit Projektor und MTP. Jetzt reicht "kein Bild im Prompt".
  - Ein gescheitertes Bild machte `slot.release(); continue;`. Aus der frueheren Slot-Schleife ist im Fork ein Lambda mit `while (true)` geworden, der Code las danach den freigegebenen Task (Segfault). Jetzt `return`. Betraf jedes Modell mit Projektor, in agollm holo4.
  - KVFlash ordnet jeder Position einen Slot zu, M-RoPE-Bildtokens teilen sich eine Position: Bilder gehen nur ohne KVFlash. Die Meldung sagt das jetzt ("tokens that share a position"), das Skript bricht bei Projektor plus KVFlash mit Erklaerung ab.
  - Speicher: Der MTP-Draft-Kontext bekommt kein KVFlash und haelt den ganzen Kontext (f16 bei 262144: 1024 MiB). Projektor auf der GPU passt nicht (OOM schon bei 32768), auf der CPU kodiert er ein 300-Token-Bild in ~3,3 s. Neuer Default im Skript: Projektor auf der CPU, KVFlash aus, ctx 49152 (35k-Prompt 431 t/s, Decode 45 t/s, Spitze 5432 MiB; 65536 passt nicht). Draft-KV q8_0 braucht einen FA-Umwandlungspuffer von 205 MiB (Spitze 5494), bleibt also f16 (`DRAFT_TYPE_K/V`). Nur Text wie agollm: `MMPROJ="" LLAMA_KVFLASH="8192" CONTEXT="262144"`.
  - Nicht behoben: Beim Beenden per SIGTERM segfaultet der Server gelegentlich in `ggml_backend_cuda_synchronize` (dmesg, 3 Faelle).
  - Greedy, ctx 8192: MTP n_max 1 Edit 31,0, Deutsch 24,7 (57 % angenommen), Code 29,5; n_max 2 Deutsch 21,0 (34 %); p_min 0,5/0,75 nicht besser. Abweichungen zum Lauf ohne Spekulation nur an Beinahe-Gleichstaenden (0,445 gegen 0,439; ngram kippt dasselbe Token).
  - VRAM: Kopf 131, zweiter Recurrent-State-Slot fuer den Rollback 88 MiB, dazu KV und Compute. ctx 20480 5712 MiB (zu knapp), 18432 5590, 16384 5470-5516. Prefill-Ubatch 256 statt 512: gleich schnell, 70 MiB weniger.
- **Qualitaetsprobe (Thinking an):** 5/6 (Schlaeger/Ball, Werktage Maerz 2024, drei deutsche Saetze mit "Im", Tool-Call, merge_intervals; roemische Zahlen nach 4928 Tokens falsch).
- **agollm:** Rezept `sensenova-loop-9b` (Aliase `sensenova`, `sensenova-loop`), nur manuell waehlbar (alle Skills 1), mit MTP-GGUF, `ngram-simple,draft-mtp`, ctx 16384. `workers/llama` aus `build-main-sm75` aktualisiert, Backups `runtime/llama-build-before-sensenova-20261101T012802Z` und `...-before-sensenova-mtp-20261101T121302Z`. Test ueber agollm: 38,8 t/s mit Thinking, 200 von 247 Drafts angenommen.

Dateien: `download-sensenova-loop-9b.sh`, `start-sensenova-loop-9b.sh`, `scripts/sensenova_loop_add_mtp.py`.

# Systemebene: PCIe, Treiber, Kernel, Kuehlung vom 2026-10-31

**Ergebnis: Kein wirksamer Software-Hebel. Die PCIe-Paketgroesse (Max Payload 256 statt 128 Byte per `pci=pcie_bus_perf`) aenderte nach einem Neustart weder Bandbreite noch Qwen-Prefill und wurde wieder entfernt. Alles andere ist Hardwaregrenze oder schon optimal; die CPU laeuft unter Qwen-Last thermisch begrenzt.**

- **PCIe:** Die GPU kann x16, der Renoir-Root-Port nur x8 Gen3 (Hardware). Gepinnt H2D 6,64 / D2H 6,68 GB/s, ~85 % der Rohrate (7,88 GB/s); das passt rechnerisch zu 128-Byte-TLPs, war aber nicht die Ursache (siehe unten). Root-Port kann MPS 512, die GPU 256, beide stehen auf 128. `pci=pcie_bus_perf` setzte Root-Port 512 und GPU 256 (MRRS 256 statt 512), keine AER-Fehler. Gemessen (gleichzeitig kamen Kernel 7.2.6 und Treiber 595.104 dazu): H2D/D2H 6,69/6,76 -> 6,70/6,76 GB/s, Qwen-Prefill eines 3k-Prompts (`tools/qwenpp.sh`, 6 Laeufe, tokengleich) 346,4-347,6 -> 347,2-348,3 t/s. Die Bandbreite begrenzt also nicht die TLP-Groesse (der Root-Complex liefert Lese-Completions offenbar weiter in kleinen Stuecken); Parameter wieder entfernt.
- **IOMMU:** translated, lazy. Als Bandbreitengrenze nicht ausgeschlossen, `iommu=pt` nicht getestet (Neustart); erwarteter Gewinn hoechstens klein, da 85 % der Rohrate fuer Gen3 x8 ueblich sind.
- **CPU:** acpi-cpufreq (kein amd_pstate auf diesem Renoir-BIOS), Governor performance, Boost aktiv: 4,28 / 4,12 / 3,92 GHz bei 1 / 8 / 16 Threads. Unter Qwen (CPU-Experten plus GPU) Tctl im Mittel 96 C, 77 von 83 s ueber 95 C, 3,75-3,87 statt 4,1 GHz. GPU auf 1500 MHz begrenzt (`nvidia-smi -lgc`, 18 W weniger): CPU-Temperatur und -Takt unveraendert, Qwen 49,7 statt 50,8 t/s; verworfen. `fan_mode` 0 vs 4 (ideapad): identisch. Kein Software-Weg zu mehr Kuehlung.
- **GPU:** 80-W-Limit fest (mobil), unter Qwen-Prefill zeitweise SW Power Cap. Runtime-PM aus, ASPM auf dem GPU-Link aus. Persistenzmodus: CUDA-Start 373 ms mit und ohne.
- **RAM/NVMe:** 2x16 GB DDR4-3200 Dual-Rank, Dual-Channel (optimal). NVMe K3 laeuft Gen3 x4 (Slot), der Controller begrenzt Random-Reads vorher (0,9-1,0 GB/s); APST greift erst nach 100 ms Leerlauf, Interrupt-Coalescing aus, kyber = none (Strata-Runde).
- **Speicherverwaltung:** swappiness 180 mit 16 GiB zram verdraengt eher fremde Heaps als den Modell-Page-Cache: guenstig fuer Ling/Kolibri. THP madvise; `READ_ONLY_THP_FOR_FS` ist nicht gesetzt, Modell-mmaps bleiben 4-KiB-Seiten.
- **Mitigations:** Retpoline, STIBP always-on, Safe RET aktiv. `mitigations=off` waere moeglich, schwaecht aber die microVM- und Browser-Isolation; nicht empfohlen.

Messprogramme: scratch `sysbench/bw.cu` (PCIe), `init.cu` (CUDA-Start), `sample.sh` (Telemetrie); Laeufe in `benchmark-results/avalon-review-20261031/runs/qtherm`, `ktherm-1`, `qclk-*`.

# Avalon-Review: Prompt-Cache, asynchrone Split-Kopien und Graph-Cache vom 2026-10-31

**Ergebnis: Aus der Analyse von freriadoc/Avalon (llama.cpp-Fork fuer Qwen3.8-Flash-Next auf RX 7900 XT) wurden drei Punkte gemessen. Avalons Regel, den RAM-Prompt-Cache bei Modellen groesser als der RAM abzuschalten, gilt hier nicht: Ling behaelt ihn. Asynchrone Eingangskopien der CPU-Splits bringen Kolibri +2,5 % und Ling +1 %, tokengleich; beide Rezepte und Startskripte setzen sie jetzt. Ein Graph-Cache pro Form fuer spekulatives Decoding laeuft tokengleich, bringt Qwen aber nur +0,5 % (Rauschen).**

Messungen und Skripte: `benchmark-results/avalon-review-20261031/`. agollm war gestoppt.

- **Prompt-Cache (Avalon 57c58e00):** `tools/cachebench.py` fuehrt vier verschiedene Chats (1,7k-2,4k Tokens, `cache_prompt: true`), einen Decode-Test und eine Folgefrage zu Chat 1 auf einem Server aus. Mit `--cache-ram 8192` belegen vier gespeicherte Chats ~2 GiB, die dem Page-Cache fehlen: Decode 7,33 statt 7,63 t/s (-4 %), 32 statt 29-31 GB SSD-Reads. Die Folgefrage stellt dafuer 1803 Tokens aus dem Cache wieder her und rechnet 29 neu (3,5-3,8 s statt 42-50 s Prefill). Avalon prefillt mit ~1600 t/s, dort kostet ein Neuaufbau wenig. Das Ling-Rezept setzt `LLAMA_ARG_CACHE_RAM` jetzt ausdruecklich auf 8192.
- **Asynchrone Split-Kopien (Avalon 98a8dc2b, dort +5,3 %):** entspricht unserem `GGML_SCHED_ASYNC_D2H_COPY=1` plus `GGML_SCHED_DEDUP_DST_SYNC=1`, das bei Kolibri und Ling fehlte. Kolibri, vier verschachtelte Runden nach einem Aufwaermlauf: 17,45 -> 17,89 t/s, vorn in 3 von 4 Paaren, kleinste Streuung. Ling: 7,50 -> 7,58 t/s (im Rauschen; die SSD begrenzt). Cold-First plus Defer bei Kolibri: Mittel 18,50, Median 18,0 t/s, aber von einem schnellen Lauf getragen (Slab-Faults schwanken je Lauf zwischen 3,8 und 9 %); nicht uebernommen.
- **Graph-Cache pro Form (Avalon a9a8168e):** Spekulatives Decoding wechselt die Graphformen (Verify 5 Tokens, Decode 1 Token); `llama_context` behielt nur den letzten Graphen. Umgesetzt: Scheduler-Plaene (`ggml_backend_sched_plan_*`), ein Graph je kleiner Form (`LLAMA_GRAPH_SMALL_TOKENS` 16, `LLAMA_GRAPH_SMALL_SHAPES` 8, 0 schaltet ab) und gesicherte Eingabe-Bindungen der Backend-Sampler (ohne sie las ein wiederverwendeter Graph veraltete Indizes: illegal memory access in `set_rows`). Nebenbei gefunden: Der Allokator legte die Compute-Puffer bei jedem Formwechsel neu an (CUDA0 12,63 <-> 13,30 MiB, Host 0,50 <-> 0,82 MiB, ~25-mal in 128 Tokens); er verkleinert jetzt nur noch auf hoechstens die Haelfte und um mindestens 64 MiB (`GGML_ALLOC_SHRINK_RATIO` 0.5, `GGML_ALLOC_SHRINK_MIN_MIB` 64; 1 und 0 stellen das alte Verhalten her). Qwen3.6 DFlash, drei verschachtelte Runden, alle 54 Antworten identisch: alte Binary 50,02, nur Allokator 50,27, Allokator plus Plaene 50,31 t/s. Die Neubauten sinken von 72/38 auf 22/19 je Request, der Gewinn bleibt im Rauschen: Die restlichen Neubauten entstehen, wenn der KV-Bereich ueber eine 256er-Stufe waechst. Beides bleibt standardmaessig an; `start1660.sh`, `router1660.sh`, `start-kolibri-1.sh`, `start-ling-3-flash.sh` und die agollm-Rezepte qwen3.6-35b-a3b-hybrid, kolibri-1 und ling-3-flash setzen die vier Parameter ausdruecklich.

# Ling-3.0-flash Q4_K_M von der SSD vom 2026-10-30

**Ergebnis: inclusionAI/Ling-3.0-flash (127B MoE, 512 Experten top-8 plus ein Shared Expert, KDA/MLA-Hybrid, eine MTP-Schicht) laeuft als bartowski-Q4_K_M (72,5 GiB, 2,4x RAM) auf der 1660 Ti mit ~7,5 t/s Decode. Ein Prompt mit 3,9k Tokens prefillt mit 33-41 t/s, kurze Prompts mit 7-11 t/s. Die Experten streamen von der SSD; beim Decode liest sie 0,9-1,2 GB/s, ihr Maximum. Eingebaut als `start-ling-3-flash.sh` und agollm-Rezept `ling-3-flash` (nur von Hand waehlbar).**

Der Fork kann `bailingmoe3` seit b0ce1363b (auch in `build-main-sm75` und agollm `workers/llama`). Auf der GPU liegen Attention, KDA, Router, Shared Experts, die zwei dichten Schichten und ein statisches Hot-Set; der Rest der 20 480 Experten-Slices (je 3,4 MB) kommt aus dem Page-Cache oder der SSD. Wichtig: Fuer `bailingmoe3` startet das Tiering nur mit `LLAMA_EXPERT_HOT` oder `LLAMA_EXPERT_S`; ohne beides rechnet die CPU alle Experten und `LLAMA_EXPERT_USAGE` schreibt nichts. Das Profil `profiles/ling-3-flash-chat.csv` stammt aus 14 agollm-artigen Prompts (3,3k Tokens, S=8 mit ADAPT=1 aufgezeichnet): Top-16 je Schicht decken 26 %, Top-64 51 %, Top-128 70 % der Expertenwahlen. Das Auto-Fit setzt S=12 (2,13 GB frei, 136 MB je Slot), 23 % Abdeckung.

Messung: agollm gestoppt, A/B/A/B ohne Cache-Raeumung, je ein 3,9k-Token-Prompt (sein Prefill zieht alle Experten durch den Cache), dann drei Kurzprompts mit 256 Greedy-Tokens.

| Variante | Decode t/s (Mittel je Runde) | Ergebnis |
| --- | ---: | --- |
| Basis: Hot-Set S=12, RAM-Lock 12 GiB | 7,23 / 7,11 / 7,25 / 7,32 | |
| Cold-First + Defer-Host-Inputs | 7,21 / 7,27 | tokengleich, kein Gewinn |
| RAM-Lock 0 GiB | 7,39 / 7,46 | **uebernommen** (Lock senkt Fehlseiten 35 -> 30 %, spart aber keine Zeit) |
| RAM-Lock 18 GiB | 7,21 / 7,16 | |
| ADAPT=1 (adaptives Hot-Set) | 7,45 / 7,37 | gegen Basis 7,61 / 7,43 |
| MTP-Draft n=1 | 7,59 / 7,80 | jeder Draft angenommen (127/127 je Prompt) und Greedy-Ausgabe weicht ab: Verify-Pfad verdaechtig, nicht uebernommen |
| MTP-Draft n=2 | 6,98 / 6,71 | langsamer |
| Prefill-ubatch 4096 | 7,71 / 7,54 | Prefill 15 statt 37 t/s, S faellt auf 8 |

Ohne Hot-Set aus kaltem Cache (erster Lauf): 6,7 t/s ueber die 14 Profil-Prompts. Ueber agollm: 41-Token-Frage nach 58 s beantwortet (10 s Laden, 4,8 s Prefill, 298 Tokens mit 7,2 t/s), Denken (englisch) und deutsche Antwort sauber getrennt. Nicht gemessen: Decode bei langem Kontext (MLA-KV ist mit 7 Schichten zu 576 Werten je Token klein) und Threads ueber 8 (bei Kolibri ohne Gewinn).

Artefakte (gitignored): `benchmark-results/ling-3-flash-20261030/` (`tools/mbench.py`, `tools/abn.sh`, `runs/`, `ab*.txt`, `profile-corpus.jsonl`). Modell: `~/models/ling-3-flash/` (sha256 geprueft, `download-ling-3-flash.sh`).

# Phonon-2.5 und Parakeet (ASR) gegen granite-speech vom 2026-10-26

**Ergebnis: FermionResearch/Phonon-2.5 (Parakeet-TDT-0.6B-v3 in Low-Bit-Form, 156 MB, CPU) und das volle Parakeet-TDT-0.6B-v3 ueber audio.cpp auf der Vega transkribieren 6-15x schneller als die aktive agollm-ASR granite-speech-5.0-2b, machen aber mehr Wortfehler. Beide sind in agollm als Auswahl eingebaut; granite bleibt aktiv.**

Testsatz: 25 FLEURS-de-Testaufnahmen (echte Sprache, Mittel 14 s) und 24 Assistenten-Saetze (phonon-TTS Annika/Anton, 2-5 s). WER ohne Zahl-Token, weil Phonon und Parakeet Ziffern schreiben, granite Zahlwoerter.

| Engine | WER FLEURS | WER Assistent | Latenz Median FLEURS / Assistent |
| --- | ---: | ---: | ---: |
| granite-speech 5.0 2B Q8_0 (agollm, Vega) | 3.82 % | 1.63 % | 3.02 / 1.33 s |
| Parakeet-TDT-0.6B-v3 Q8_0 (audio.cpp 820db90f, Vega) | 4.96 % | 2.72 % | 0.56 / 0.24 s |
| Parakeet-TDT-0.6B-v3 F16 (audio.cpp, Vega) | 4.96 % | 2.72 % | 0.56 / 0.23 s |
| Phonon-2.5 (fermion-research 0.2.15, CPU 8 Threads) | 6.11 % | 3.80 % | 0.32 / 0.11 s |
| Parakeet ueber agollm | 4.96 % | 2.72 % | 0.49 / 0.20 s |
| Phonon-2.5 ueber agollm | 6.11 % | 3.80 % | 0.36 / 0.16 s |

Roh-WER (Zahlen mitgezaehlt) FLEURS: granite 5.24, Parakeet Q8_0 5.06, F16 4.89, Phonon 5.93 %. Q8_0 und F16 sind gleich genau; agollm nutzt Q8_0. Fehler von Parakeet und Phonon: getrennte Komposita ("ueber morgen"), Flexion ("Stelle", "mache"), Eigennamen; Phonon zusaetzlich einzelne Fehlwoerter ("Huerdenschadens" statt "Hirnschadens"). Beide ignorieren `language` und erkennen die Sprache selbst.

Einbau in agollm (`internal/asr`): `config.ASR.Kind` = `fermion` (`fermion serve <Cache-Verzeichnis> --served-model-name phonon-2.5`, `HF_HUB_OFFLINE=1`, Start 2 s) oder `audiocpp` (`audiocpp_server --config runtime/asr-audiocpp.json`, Vulkan0, Start 3-4 s). Dabei aufgefallen: Push-to-Talk in der UI schickt WebM/Opus, das llama-server (und audio.cpp) nicht dekodiert -- auch granite lieferte "Failed to load image or audio file". agollm wandelt jetzt jeden Nicht-WAV-Upload per ffmpeg in 16-kHz-Mono-WAV.

Artefakte (gitignored): `benchmark-results/asr-phonon25-20261026/` (`tools/asr_eval.py`, `tools/wer_summary.py`, `asr-eval*.json`, FLEURS-Ausschnitt, synthetische Saetze). Modelle: `~/models/parakeet-tdt-0.6b-v3/` (Q8_0 und F16, sha256 geprueft), Phonon im fermion-Cache.

# Ming-3.0-TTS 0.5B und audio.cpp auf der Vega vom 2026-10-26

**Ergebnis: Ming 3.0 TTS (audio-cpp Q8_0) laeuft mit audio.cpp (Commit 820db90f, Vulkan-Build) auf der Vega, aber langsamer als Echtzeit: RTF 1.33 (CPU mit 8 Threads 1.0). Die aktive agollm-Engine phonon 40M liegt auf der CPU bei RTF 0.09. Verstaendlich sind beide (WER 0 % auf drei deutschen Texten). audio.cpp lohnt sich fuer dieses Modell auf dieser Hardware nicht; agollm bleibt bei phonon.**

| Warm, deutsche Texte 3-18 s Audio | RTF | Hinweis |
| --- | ---: | --- |
| phonon 40M (agollm, CPU, aktiv) | 0.086-0.095 | volle WAV nach 0.3-1.5 s |
| Ming Q8_0, audio.cpp, Vega Vulkan | 1.32-1.38 | 107 ms Rechenzeit je 80-ms-Patch |
| Ming Q8_0, audio.cpp, CPU 8 Threads | ~1.0 | |
| Modellkarte: audio.cpp auf RTX 5090 | 0.17 | Python BF16 dort 0.65 |

Warum audio.cpp auf CUDA 3.8x schneller als Python ist: Je 80-ms-Patch laufen ein Backbone-Schritt (n=1), zehn Flow-Schritte mit CFG (3 Token x 2 Zweige = 6 Spalten) und ein Aggregator. In PyTorch eager sind das Hunderte kleiner Kernel mit Python-Dispatch je Patch, auf einer RTX 5090 also CPU-gebunden. audio.cpp baut die Graphen einmal und verwendet sie wieder, packt QKV und Gate/Up, schreibt einen statischen KV-Cache per set_rows, nutzt Flash-Attention, rechnet beide CFG-Zweige in einem Batch und laedt Q8-Gewichte.

Auf der Vega zaehlt das kaum, die GPU ist busy: 4.31 s GPU-Zeit bei 4.35 s Wandzeit (Vulkan-Perf-Logger). Matrix-Vektor mit 6 Spalten (Flow-Kopf, 10 x 107 MB je Patch) erreicht nur 18-26 GB/s gegen 41-44 GB/s bei einer Spalte (Backbone); Flow 46 %, ROUND_BF16 fuer die BF16-Paritaet 9 %, kleine elementweise Ops 18 %.

Eigene Hebel auf audio.cpps Vulkan-Pfad (Patch `benchmark-results/ming-tts-20261026/audiocpp-vega-experiments.patch`, Checkout wieder auf Upstream):
- ROUND_BF16 weglassen: -9 % je Audiosekunde, WER weiter 0 %.
- Subgroup-Reduktion auf GCN: -5 % je Patch.
- Schlechter: Matmul statt Matrix-Vektor ab 2-4 Spalten (RTF 2.4-2.5), 1, 4 oder 8 statt 2 Zeilen je Workgroup (+17 bis +50 %).

Zusammen waere RTF ~1.15 erreichbar, also weiter nicht echtzeitfaehig. Die Techniken, mit denen audio.cpp gegenueber Python gewinnt, hatte unsere vkrt-Runtime fuer Humaneness schon (gebatchtes CFG, geteilter KV mit Views, FA ohne Umpacken; ~1.05x Echtzeit auf der Vega); die verbleibende Grenze ist die Speicherbandbreite bei zehn Flow-Durchlaeufen je 80 ms. Schneller ginge es nur mit Modellaenderungen (weniger Flow-Schritte, Q4-Flow-Kopf), mit Qualitaetsrisiko, nicht getestet.

Artefakte (gitignored): `benchmark-results/ming-tts-20261026/` (`tools/`, `out/` mit allen WAVs, `exp.txt`, `exp2.txt`, Perf-Log). Modell: `~/models/ming-3.0-tts/`, audio.cpp: `~/src/audio.cpp` (Build `build/linux-vulkan-release`).

# Sage-A1.5 als Planer vom 2026-10-26

**Ergebnis: Sage-A1.5 (InternScience, Feintune von Qwen3.6-35B-A3B; mradermacher i1-Q4_K_M) laeuft mit dem Hybrid-Setup von Qwen3.6 und ist als agollm-Rezept `sage-a1.5` installiert. Am schnellsten ist es ohne Draft (37.3 Token/s; Qwen3.6 mit seinem DFlash-Draft 44.9). Als Planer schreibt es auf fuenf Code-Aufgaben in allen Faellen PLAN.md (Qwen3.6: 4/5) und braucht 62 statt 77 s je Plan; inhaltlich ist es gleichwertig, bei zwei Aufgaben etwas weniger konkret. Die Router-Zuordnung ist unveraendert.**

| Sage-A1.5, greedy, 3 Prompts x 384 Token, A/B/A/B | Token/s |
| --- | ---: |
| Qwen3.6-Rezept (UD-Q4_K_M, DFlash-Draft) zum Vergleich | 44.9 |
| ohne Draft | 37.3 |
| DFlash-Draft von Qwen3.6 (59-84 % angenommen) | 36.6 |
| eigener MTP-Kopf, n_max 3 (85-88 % angenommen, Warm-Slots aus) | 33.1 |
| eigener MTP-Kopf, n_max 3, S=26 | 34.4 |
| ohne Draft, eigenes Heatprofil statt Qwen3.6-Profil | +0.9 % (Rauschen) |

| Plan-Schritt ueber agollm (Planer festgepinnt, Abbruch beim Start der Umsetzung) | Sage-A1.5 | Qwen3.6 |
| --- | ---: | ---: |
| PLAN.md geschrieben | 5/5 | 4/5 (Snake: kein workspace_write) |
| Zeit je Plan (inkl. Laden) | 57-71 s, Mittel 62 | 60-93 s, Mittel 77 |
| Bibliotheken (Three.js, Tailwind) ueber workspace_libraries | beide | beide |

Inhalt: Snake gut (Umkehrschutz, Pfeiltasten-Scrollen). Sonnensystem bei beiden gut, beide empfehlen fuer OrbitControls ein CDN, obwohl der Umsetzer nichts aus dem Netz laden darf. Duplikatfinder: Sage ohne Namenskonflikte im Papierkorb, Qwen mit. Kanban: Sage ohne `darkMode: 'class'` fuer Tailwind, Qwen konkreter (Datenmodell, Events). Rate-Limiter gleichwertig.

- Rezept `sage-a1.5`: Hybrid-Werte von Qwen3.6 (S=20, W=8, q4_0-KV, 64k), kein Draft, Heatprofil aus 12 eigenen Prompts (`workers/llama/assets/sage-a1.5-chat.csv`), Sampling laut Karte (temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5), Chat-Template aus dem GGUF (Tool-Calls, `verification`). Skills vorlaeufig unter Qwen3.6, damit `installed_test.go` dieselbe Zuordnung haelt.
- Die Qwen3.6-DFlash-Vorlage passt nicht gut zu Sage; der MTP-Kopf trifft oft, entwirft aber Token fuer Token und schaltet die Warm-Slots ab.

Artefakte (gitignored): `benchmark-results/sage-a15-20261026/` (`tools/`, `runs/`, `plans/` mit allen PLAN.md, `profile/`, `t1-spec.txt`, `t2-profile.txt`, `t3-planner.txt`).

# Strata-Ideen vom 2026-10-26

**Ergebnis: Kolibri-1 decodiert im agollm-Rezept mit +13 % (16.3 -> 18.4 Token/s, A/B/A/B mit 3.5k-Token-Prompt) und streut kaum noch. Zwei Strata-Ideen tragen: weniger Kontext-VRAM fuer mehr heisse Experten (16k statt 64k) und ein mlock-Budget fuer die heissesten CPU-Experten. Tail-Skip ist schnell, aber zu verlustbehaftet, und bleibt opt-in. Rezept und `workers/llama` (libggml-cpu, libllama) sind in agollm installiert. Kein Commit.**

Grundlage: Strata 0.2.6 (github.com/niko1221/strata), Doku und `src/core/expert_source.cpp`. GTX 1660 Ti, Ryzen 7 4800H, Performance-Profil, agollm gestoppt, greedy, 3 Thinking-Prompts x 384 Token, Werte in Token/s.

| Hebel | Modell | Basis | Variante | Ergebnis |
| --- | --- | ---: | ---: | --- |
| Kontext 16k statt 64k (S 19 -> 27) | Kolibri | 14.81 | 18.53 | +25 %, 2/2 Paare; 32k (S=24) +14 % |
| mlock 12 GiB heisse CPU-Experten, nach 3.5k-Prefill | Kolibri 16k | 16.39 | 18.03 | +10 %, 2/2 Paare; fehlende Slabs 16 -> 11 % |
| Altes Rezept gegen 16k + mlock 12 GiB | Kolibri | 16.34 | 18.42 | +13 %, 2/2 Paare; Prefill gleich oder schneller |
| Tail-Skip Rang 6 / Raenge 5-6 | Kolibri 16k | 17.27 | 19.46 / 23.53 | +13 / +36 %, aber KL 0.24 / 0.32, Top-1 81 / 78 % |
| Tail-Skip Rang 8 / Raenge 7-8 (mit DFlash) | Qwen3.6 | 48.36 | 51.26 / 54.34 | +6 / +12 %, KL 0.027 / 0.065, Top-1 91 / 89 % |
| Next-Layer-Prefetch (8) ab kaltem Page-Cache | Kolibri 16k | 13.21 | 13.72 | +4 %, nur 1/2 Paare; bleibt aus |
| CPU-Pinning, ein Thread je Kern (`-C 0x5555 --cpu-strict 1`) | Kolibri / Qwen3.6 | 19.08 / 48.16 | 18.08 / 47.31 | -5 / -2 %; verworfen |

## Was eingebaut ist

- `LLAMA_EXPERT_RAM_LOCK_GIB` (llama-expert-tier.cpp): sperrt nach dem Laden die CPU-Experten mit den hoechsten Seed-Scores per `mlock` im Page-Cache, in einem Hintergrund-Thread (12 GiB in 9-26 s). Nur fuer statische Tiers (ohne Adaption und Warm-Slots). Braucht ein memlock-Limit ueber dem Budget; `agollm.service` hat `LimitMEMLOCK=infinity`, `start-kolibri-1.sh` hebt das Limit soweit erlaubt an.
- `LLAMA_EXPERT_TAIL_SKIP=R` (ggml-cpu.c, fusionierter kalter MoE-Op): ein kalter Experte, den alle Token des Batches nur auf Top-k-Rang >= R (0-basiert) waehlen, wird weder gelesen noch gerechnet; Beitrag null, keine Renormierung (wie Stratas `STRATA_ROUTE_TAIL_SKIP`). Opt-in, nirgends gesetzt.
- `LLAMA_EXPERT_FAULT_STATS=1`: geprueft/fehlende Expert-Slabs und Wartezeit im Fault-Pfad beim Beenden.
- agollm `kolibri-1.yaml`: Kontext 16384, `LLAMA_EXPERT_RAM_LOCK_GIB=12`, `tokens_per_second` 18. Router-Prompts sind auf 4000 Zeichen begrenzt, die Kompaktierung passt direkte Chats an den Kontext an, und der Szenario-Router laesst Kolibri bei zu langen Stufen aus.

## Was nicht traegt

- **Tail-Skip:** Strata misst KL 0.003-0.006 bei top-10 aus 512 (Raenge 8-10). Kolibri (top-6) und Qwen3.6 (top-8) verlieren mit den letzten Raengen deutlich mehr; Qwen Rang 8 entspricht etwa einer Quant-Stufe. Rauschboden: Kolibri 0, Qwen 0.0005 (ubatch 1, Warm-Slots aus).
- **SSD-Lese-Pool mit hoher Queue-Tiefe** (Strata: 32 Threads, 128 tief): nicht gebaut. Die SSD (K3, PCIe 3.0 x4) schafft zufaellig gelesen nur 0.9-1.0 GB/s, schon ab QD 4 (fio, direct, 128k-1M); kyber gegen none ohne Unterschied. Im Betrieb liegen wir bei 0.44-0.54 GB/s in aktiven Sekunden; die Fault-Wartezeit ist ~10 % der Laufzeit, der Gewinn waere hoechstens ~5 %.
- **Gate fuer den Next-Layer-Prefetch:** der Prefetch bringt auch ab kaltem Cache nur ~4 %; ein Gate lohnt den Umbau (Graph-Invalidierung) nicht.
- **iommu=pt:** 12 IOMMU-Gruppen laufen mit DMA-FQ. Nicht getestet (Kernel-Parameter und Neustart); mit `--no-op-offload` geht bei Kolibri wenig ueber PCIe.
- **Draft nur wenn es sich lohnt** (DSpark mit Option "kein Draft"): nicht gebaut; bringt nur bei repetitivem Text etwas, Kolibri liefert in agollm Prosa.

Messmethode: `tools/kl.py` laesst `LLAMA_CMOE_*` und Backend-Sampling weg, sonst ersetzen die Phase-Batches `-ub 1` und alles laeuft ueber den Prefill-Pfad. Kolibri-KL mit S=27 (Wert bei 16k), 4 x 512 Token Prosa aus optimize.md und den agollm-Dokumenten.

Artefakte (gitignored): `benchmark-results/strata-levers-20261026/` (`tools/`, `runs/`, `kl/`, `t1-ctx.txt` bis `t6-final.txt`, `kl-*.txt`, `deployment.json`, Backups).

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
