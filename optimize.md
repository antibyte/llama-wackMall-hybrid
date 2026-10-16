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
