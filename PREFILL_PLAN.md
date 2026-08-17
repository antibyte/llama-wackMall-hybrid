# Prefill / Prompt-Processing TPS Plan (≥10%)

**Ziel:** Prompt-Processing-Durchsatz (`prompt_tps` / TTFT) um **mindestens 10%**
steigern, ohne Decode um mehr als **3%** zu regressieren (HYBRID_DESIGN gate).

**Hardware / Stack:** GTX 1660 Ti (sm_75, 6 GiB), Qwen3.6-35B-A3B UD-Q4_K_M,
DFlash + Turbo4 KV + KVFlash, `start1660.sh`.

**Status:** P0 + **1856 e2e-Winner** (2026-08-17).  
S=28 / prefill **1856** / decode 64 → Long PP **+33.8%**, Decode **+4.7%** vs S30/p1024.  
`start1660.sh` promoted.

---

## 1. Baseline (gemessen)

Quelle: `benchmark-results/start1660-ctx-20260812T073923Z`  
Konfig: CONTEXT=24576, prefill **1024/1024**, decode **64/64** (Bench),
Turbo4 K/V, S≈26, DFlash n_max=5 n_min=2.

| Regime | prompt_n | prompt_tps | ms/tok | TTFT | Decode tps |
|--------|---------:|-----------:|-------:|-----:|-----------:|
| Short SC | 37 | **25.7** | 38.9 | 1.46 s | ~32.6 |
| Mid (×16) | 607 | **98.5** | 10.2 | 6.19 s | ~31.6 |
| Long (×48) | 1823 | **132.6** | 7.54 | 13.8 s | ~28.9 |

**Aktuelles `start1660.sh` (2026-08-16):**

| Knob | Wert | Hinweis |
|------|------|---------|
| `CMOE_PREFILL_{BATCH,UBATCH}` | 1024 | Phase-Prefill aktiv |
| `CMOE_BATCH` / `CMOE_UBATCH` | 128 | Base + Default-Decode-Phase |
| `CMOE_DECODE_*` | leer | Decode-Phase = Base **128** (nicht 64) |
| `LLAMA_EXPERT_S` | 30 | VRAM-Tradeoff vs. Prefill-Graph |
| `LLAMA_KVFLASH` | 12288 | Pool groß → Clamp greift **nicht** bei ubatch 1024 |
| `GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH` | 2 | FP16-FA-Pfad für Multi-Token |

Header-Kommentar („prefill 768, batch 64“) ist veraltet.

### Gate / Zielzahlen

| Regime | Baseline prompt_tps | Ziel (≥+10%) |
|--------|--------------------:|-------------:|
| Mid (~600 tok) | 98.5 | **≥108.4** |
| Long (~1.8k tok) | 132.6 | **≥145.9** |
| Short (~37 tok) | 25.7 | **≥28.3** (optional; Overhead-dominiert) |

Primärmetrik: **Median prompt_tps** über 3 Runs bei **Mid + Long** (produktionsrelevant).  
Sekundär: TTFT, effective S, VRAM peak, decode_tps (≤3% Regression), Output-Hash-Stabilität optional.

---

## 2. Analyse: Was limitiert Prompt TPS?

### 2.1 Historische Evidenz (stärkster Hebel)

Aus `HYBRID_EXPERIMENTS.md` — physisches **ubatch**, nicht logisches batch:

| Batch/ubatch | Prompt tok/s | TTFT (2k-ish) |
|-------------:|-------------:|--------------:|
| 32/32 | 19.8 | 103 s |
| 256/256 | 57.8 | 35 s |
| 512/512 | **94.4** | 22 s |

Nur `n_batch` erhöhen bei ubatch=32: **kein** Gewinn (19.83).  
Prefill-Sweep 2026-08-10 (q4, n_predict=1): 256→768 ubatch ≈ **56 → 109** tok/s; S 32 vs 34 **neutral**.

**Fazit:** Prompt TPS skaliert mit **physikalischem Prefill-ubatch** bis GPU/VRAM-Sättigung. Wir sind schon bei 1024 — nächster Schritt ist 1280–2048 **oder** die restliche Overhead/Inject-Last.

### 2.2 Regime-Split

```
prompt_time ≈ T_fixed + N_prompt / R_compute(ubatch, S, kernels)
```

| Regime | Dominant | Hebel |
|--------|----------|-------|
| **Short** (≪ ubatch) | Fixed: Graph-Reserve, Phase-Switch, DFlash begin, 1 Graph | Fixed-cost (CUDA graph, weniger Setup) |
| **Mid/Long** | Compute: FA + MoE-Hot + CPU-Cold + DFlash inject pro Chunk | Größeres ubatch, weniger Chunks, Kernel-Pfad |
| **KVFlash klein** (Pool≪ubatch) | Clamp `n_ubatch → (blocks−reserved)×chunk` | Nicht Prod (Pool 12k); nur kleine Pool-Smokes |

### 2.3 Code-Pfade (relevant)

1. **Phase batching** (`common/arg.cpp`, `server-context.cpp`):
   - Reserve = `max(prefill, decode)` → Graph-VRAM skaliert mit Prefill-Cap.
   - Runtime: `llama_set_runtime_ubatch` vor Prefill/Decode.
   - Decode-Cap spart **kein** Reserve-VRAM (nur Runtime-Graph-Größe).

2. **KVFlash clamp** (`llama-context.cpp` ~764–773):
   ```
   max_ubatch = (pool/chunk − sink − 1) * chunk
   ```
   Bei Pool=12288, chunk=64: max ≈ 12k → **1024 unberührt**.  
   Bei Pool=512: Clamp 512→384 (in Smokes beobachtet) — Prefill-Bremse nur dort.

3. **DFlash Prefill** (`speculative.cpp` process + combined inject):
   - Target-Prefill und Draft-Inject laufen pro ubatch-Chunk.
   - Combined inject misst ~2.3 ms/call im Decode; im Prefill steckt Feature-Extract + Draft-KV im Prompt-Pfad.
   - Draft-ubatch = `min(prefill_ubatch, llama_n_ubatch(ctx_dft))` — Draft-Context muss ≥ Prefill sein.

4. **Turbo4 FP16 Prefill** (bereits ON): Multi-Token nutzt FP16-FA-Scratch statt reinem Vector-Pfad. Weitere Gewinne eher Tile/Graph als neuer Quant-Pfad.

5. **S / Experts:** Prefill-Sweep S32 vs S34 ≈ gleich → Prefill ist **nicht** primär cold-CPU-MoE-bound bei großem ubatch. S senken freigibt VRAM für größeres Prefill-Graph.

### 2.4 Erwartete Hebel-Größenordnung (geschätzt)

| Hebel | Erw. Mid/Long | Risiko | Aufwand |
|-------|---------------|--------|---------|
| Prefill ubatch 1024→1536/2048 | **+8–20%** (wenn VRAM/S hält) | S-Drop, OOM | Config-Sweep |
| Explizit DECODE=64 (vs 128) | ~0% Prefill; leicht weniger Decode-Reserve-Druck nur wenn max sinkt | gering | Config |
| S 30→28 für Prefill-Headroom | indirekt (ermöglicht größeres ubatch) | Decode −1–3% | Config |
| THREADS_BATCH 8→12/16 | 0–5% falls CPU-Cold sichtbar | Contention | Config |
| KVFlash Prefill-Fastpath (bulk page-out, kein Scorer) | 0–5% Long | Korrektheit Pager | Code |
| DFlash Prefill: Draft-ubatch/Overlap | 2–8% TTFT wenn Draft/Inject seriell eng | Spec-State | Code |
| CUDA-Graph stabile 1024-Chunks | 3–10% Mid/Long nach Warm | Rebuild bei Shape-Change | Code |
| Short-only Fixed-Cost | Short +10% möglich; Mid/Long ≈0 | gering | Profil |

**Wahrscheinlichster Weg zu ≥10% ohne großen Code:** Prefill-ubatch-Sweep oberhalb 1024 bei fester Decode-Phase und S-Headroom.

---

## 3. Plan (Phasen)

### Phase 0 — Messprotokoll (Pflicht vor Opt)

**Skript:** erweitere `scripts/ab_prefill_sweep.sh` (oder neues `scripts/ab_prefill_prod.sh`):

| Fix | Wert |
|-----|------|
| Model / DFlash / Turbo4 / KVFlash | wie `start1660.sh` |
| CONTEXT | 24576 |
| n_predict | 1 (Prefill-only) **und** 256 (End-to-End mit Decode-Gate) |
| prompt_n | 37 / 607 / 1823 (bestehende SC-Prompts / prompt_repeat) |
| REPEATS | 3, Median |

**Sweep-Matrix (Phase 1):**

| PREFILL | DECODE | S | KVFlash |
|--------:|-------:|--:|--------:|
| 1024 (baseline) | 64, 128 | 28, 30 | 12288 |
| 1280 | 64 | 28, 30 | 12288 |
| 1536 | 64 | 28, 30 | 12288 |
| 2048 | 64 | 26, 28 | 12288 |

Stop-Kriterien pro Zelle: OOM, Xid, effective S unerwartet, Decode-Regression >3% (e2e-Läufe), Hash-Korruption.

**Artefakte:** `benchmark-results/prefill-prod-YYYYMMDD…/{runs.csv,summary.md}`.

---

### Phase 1 — Config-only (höchster ROI, ~0.5–1 Tag)

**P1.1** Prefill-ubatch über 1024  
- Erwartung: Long/Mid ≥+10% wenn 1536/2048 fit.  
- Bei Auto-Fit S-Drop: Winner = größtes ubatch mit S≥28 und freiem VRAM ≥150 MiB.

**P1.2** `CMOE_DECODE_{BATCH,UBATCH}=64` in `start1660.sh`  
- Align mit gemessener DFlash-Decode-Geometrie und start1660-ctx-Benches.  
- Prefill-Reserve bleibt max(prefill,64)=prefill; Decode-Runtime schlanker.

**P1.3** Optional `THREADS_BATCH` / `DRAFT_THREADS_BATCH` 8 vs 12  
- Nur promoten wenn Median ≥+3% Prefill ohne Decode-Regression.

**P1.4** Promotion  
- Winner-Werte in `start1660.sh` + Kommentar-Header aktualisieren.  
- Re-Bench Mid+Long+Short + 256-tok Decode.

**Exit Phase 1:** ≥10% Mid **oder** Long → Plan erfüllt, Phase 2 optional.  
Sonst Phase 2.

---

### Phase 2 — KVFlash Prefill-Pfad (Code, ~1–2 Tage)

Nur wenn Phase 1 <10%.

| PR | Inhalt |
|----|--------|
| 2a | Prefill: Scorer/QK-Invalidate batchen oder skip bis Decode |
| 2b | Sequential write: bulk `mark_written` / weniger Host-Roundtrips |
| 2c | **Nicht** blindly Clamp entfernen bei kleinem Pool — stattdessen: multi-chunk ubatch nur wenn Pager `page_in` für fehlende Chunks vor Graph sicherstellt; Unit-Tests `test-kvflash` |

Gate: Long prompt_tps + KVFlash Stats (page-ins während reinem Prefill ≈0).

---

### Phase 3 — DFlash Prefill (Code, ~1–2 Tage)

| PR | Inhalt |
|----|--------|
| 3a | Log/assert: `llama_n_ubatch(ctx_dft) ≥ prefill_ubatch` nach Phase-Switch |
| 3b | Profiling: Anteil Target-decode vs DFlash process im Prefill-TTFT |
| 3c | Wenn Draft-Pfad eng: größere Draft-Chunks (bereits an Target gekoppelt) oder Pipeline Overlap (nächster Target-Chunk || Draft inject) — nur mit State-Invarianten |

Gate: TTFT Mid/Long; Acceptance/Decode unverändert.

---

### Phase 4 — Kernel / Graph (nur bei Restlücke)

| Item | Hinweis |
|------|---------|
| CUDA graph capture für volle Prefill-ubatch-Chunks | Shape-stabil (1024/1536); Invalidate bei Phase-Switch |
| FA/Turbo4 Tile-Audit sm75 bei n_tokens≥512 | NVTX/ncu Stichprobe |
| Kein neuer Quant-Pfad ohne Qualitätsgate | Turbo4 PPL/KLD bereits dokumentiert |

---

### Phase 5 — Abschluss

- [ ] Winner in `start1660.sh`
- [ ] `HYBRID_EXPERIMENTS.md` Abschnitt „Prefill ≥10%“
- [ ] Header-Kommentar (prefill/decode/S) synchron
- [ ] Optional: TTFT-Max Preset vs Balanced (S vs ubatch) wie historisch 376/S33 vs 512/S32

---

## 4. Nicht-Ziele / Fallen

- **Decode-TPS** als Proxy für Prefill — CMOE_BATCH-Sweep (16–128) änderte Decode ~±1%, nicht Prefill.
- **Nur logisches n_batch** erhöhen — historisch wirkungslos.
- **KVFlash Pool verkleinern** für VRAM — clamp’t Prefill-ubatch und schadet PP.
- **Tree-verify / DDTree** — aktuell Decode-AL, nicht Prefill-Hebel.
- **Short-Prompt +10%** erzwingen mit riesigem ubatch — T_fixed dominiert; separater Fixed-Cost-Track.

---

## 5. Empfohlene Reihenfolge (konkret)

```
Tag 0   Phase 0 Mess-Harness + Baseline-Replika (1024/64, S=30, 3× Mid+Long)
Tag 0–1 Phase 1 Sweep 1280/1536/2048 × S; DECODE=64
        → wenn ≥10%: promote, fertig
Tag 2+  Phase 2 KVFlash Prefill-Fastpath
Tag 3+  Phase 3 DFlash Prefill-Profil + gezielte Fix
Tag 4+  Phase 4 nur bei Rest <10%
```

**Erster Ausführungs-Schritt (nach Freigabe):** Phase 0/1 Sweep, kein Code außer ggf. Sweep-Skript-Erweiterung.

---

## 6. Entscheidungen (Operator 2026-08-16)

| # | Frage | Antwort |
|---|-------|---------|
| 1 | Primärmetrik | **Mid + Long** reicht; Short optional / nicht gate |
| 2 | S bis 28 | **Ja**, wenn Prefill-Gewinn stark und Decode ≤ ~2–3% |
| 3 | Spec-Modi | **Beides**: primär DFlash (Prod) **und** isoliert `SPEC_MODE=none` |

---

## 7. Phase-1 Messergebnisse (2026-08-16)

Artefakt: `benchmark-results/prefill-prod-20260816T145511Z`  
Skript: `scripts/ab_prefill_prod.sh`  
Stack: Turbo4, KVFlash env 12288, DECODE 64, CONTEXT 24576, n_predict=1.

### Baseline (DFlash, S=30, prefill 1024)

| Regime | prompt_n | prompt_tps | Ziel +10% |
|--------|---------:|-----------:|----------:|
| Mid | 607 | **98.78** | 108.7 |
| Long | 1823 | **133.56** | 146.9 |

### DFlash (Prod)

| S | Prefill | Mid | Long | vs base Long | Status |
|--:|--------:|----:|-----:|-------------:|--------|
| 30 | 1024 | 98.78 | 133.56 | +0.0% | ok |
| 30 | ≥1280 | — | — | — | **CUDA OOM** |
| 28 | 1024 | 98.51 | 134.03 | +0.3% | ok |
| 28 | 1280 | 98.34 | 136.99 | +2.6% | ok |
| 28 | 1536 | 98.06 | **140.46** | **+5.2%** | best DFlash |
| 28 | 2048 | — | — | — | **CUDA OOM** |

### Isolation SPEC=none (Target-only)

| S | Prefill | Mid | Long | vs DFlash base Long |
|--:|--------:|----:|-----:|--------------------:|
| 30 | 1024 | 100.94 | 136.05 | +1.9% |
| 30 | 1536 | 100.47 | 143.86 | +7.7% |
| 30 | **2048** | 100.06 | **187.16** | **+40.1%** |
| 28 | 2048 | 100.12 | 186.97 | +40.0% |

### Interpretation

1. **Mid ist bereits ein Chunk** (607 ≤ 1024). Größeres Prefill-ubatch ändert Mid kaum (±1%); +10% Mid braucht Kernel/Fixed-Cost, nicht nur ubatch.
2. **Long skaliert**, sobald Chunks sinken. Bei 1823 Tokens: ubatch 1024→2 Chunks, **2048→1 Chunk** (+40% ohne DFlash).
3. **DFlash + S≥28 blockiert 2048** (OOM während Prefill-Graph). Bestes DFlash-Config: **S=28 / 1536 → +5.2% Long** — **unter 10%**.
4. DFlash-Overhead vs none am gleichen Punkt: ~1–2.5% (klein). Der große Hebel ist **VRAM-Headroom für Target-Prefill-Graph**, nicht DFlash-Inject-CPU.
5. Draft-Context wird bei Phase-Batching bereits auf Decode-Geometrie begrenzt (`common_base_params_to_speculative`) — Draft-Reserve ist nicht der Haupt-OOM-Treiber; Target-Aktivierung bei großem ubatch + DFlash-Layers/S ist es.

### Folge Phase 1b — S≤26 (gemessen, `prefill-prod-followup-20260816T151345Z`)

DFlash fit ab **S≤26** für prefill 1792/2048:

| S | Prefill | Long tps | vs base Long | Mid Δ |
|--:|--------:|---------:|-------------:|------:|
| 26 | 1536 | 141.0 | +5.6% | −0.6% |
| 26 | **1792** | **176.0** | **+31.8%** | −1.7% |
| 26 | **2048** | **180.9** | **+35.4%** | −1.9% |
| 24 | 2048 | 181.0 | +35.5% | −1.6% |
| 22 | 2048 | 181.1 | +35.6% | −1.5% |

**Long ≥10% klar erreicht** ab prefill **1792** (fast single-chunk für 1823 tok).  
**Mid bleibt flach** — Config allein erreicht Mid +10% nicht.

⚠️ Operator erlaubte S≥**28**. S=26 liegt darunter → Decode-Tradeoff und Freigabe nötig, **oder** 1792 bei S=28 fitten.

### Folge Phase 1c — S≥28 Edge (gemessen, `prefill-prod-s28-1792-20260816T152259Z`)

| S | Prefill | Long tps | vs base | Mid Δ | Status |
|--:|--------:|---------:|--------:|------:|--------|
| 28 | 1664 | 143.4 | +7.4% | −0.9% | ok |
| 28 | 1728 | 142.8 | +6.9% | −1.2% | ok |
| **28** | **1792** | **176.0** | **+31.8%** | −1.5% | **Winner-Kandidat** |
| **28** | **1856** | **181.7** | **+36.0%** | −1.5% | **Winner-Kandidat** |
| 29–30 | ≥1664 | — | — | — | OOM |

**Innerhalb freigegebenem S≥28: Long +10% erreicht** mit `CMOE_PREFILL=1792` (oder 1856) und `LLAMA_EXPERT_S=28`.

### Decode-Gate e2e (`N_PREDICT=256`, `REPEATS=3`)

Base e2e: **S=30 / p=1024** → Long prompt 134.2 / decode 31.51; Mid 98.7 / 34.00.

| Config | Long prompt | Long decode | Mid prompt | Mid decode | e2e |
|--------|------------:|------------:|-----------:|-----------:|-----|
| S28 p1024 | −0.4% | −1.6% | −0.2% | **+3.1%** | ok |
| S28 p1792 | PP-only +32% | — | −1.5% | **+2.5%** | **Long OOM** |
| S28 p1856 | — | — | −1.6% | +1.2% | Long OOM |
| S26 p1536 | **+4.7%** | −3.6% | −1.1% | −1.8% | ok |
| **S26 p1792** | **+31.1%** | **−7.6%** | −1.5% | −2.4% | ok, decode-gate fail |

**Fazit Config-only:**

1. **Long ≥+10%** erreichbar ab prefill **1792**, aber:
   - bei S≥28: pure PP ok, **Long e2e OOM**
   - bei S=26: e2e ok, Decode **−7.6%** (über 3%-Gate)
2. **Mid +10%** mit Config **nicht** erreichbar (schon 1 Chunk bei 1024).
3. Decode-Strafe bei S=26 p=1792 ≈ S-Drop (~−3.5%) + großes Prefill-**Reserve**-Graph (~−4%).

### Empfohlene nächste Schritte (Operator-Entscheidung)

| Option | Prefill Long | Decode | Mid | Aktion |
|--------|-------------:|-------:|----:|--------|
| **A** TTFT-Max | +31% (S26/p1792) | −7.6% | ~0% | start1660: S=26, PREFILL=1792, DECODE=64 — nur wenn Decode-Trade ok |
| **B** Balanced S28 | +0% | 0 bis +3% | ~0% | bleiben bei p=1024 S=30/28 |
| **C** Code Phase-2/3 | Ziel +30% @ S≥28 | ≤−3% | +10% separat | Prefill-Graph nach Phase-Switch freigeben; Mid Kernel/Fixed-Cost |
| **D** Hybrid-Preset | zwei Profiles | — | — | `start1660-ttft.sh` (A) + Prod (B) |

**Code-Hebel für C (höchste Hebelwirkung):** Phase-Batching reserviert `max(prefill,decode)` dauerhaft → 1792er Graph-Peak bleibt im Decode. Nach Prefill Graph/Backend-Buffer auf Decode-ubatch schrumpfen würde Long e2e bei S=28 und Decode-Strafe bei S=26 reduzieren.

### Revidierte Hebel-Priorität nach Messung

| Prio | Hebel | Erwartung |
|-----:|-------|-----------|
| 1 | **DFlash + prefill 2048 fit** (S senken oder VRAM freigeben) | Long +25–40% wenn single-chunk |
| 2 | Mid: FA/MoE Kernel + Fixed-Cost (Graph, Phase-Switch) | Mid +10% ohne ubatch-Gewinn |
| 3 | Zwischengröße 1792 (2→1 Chunk-Grenze bei ~1823) | Long ~+10–30% |
| 4 | Phase 2 KVFlash Prefill-Fastpath | marginal wenn Pool schon groß |

---

## 8. Code-Audit: Optimierungspotenzial (2026-08-16)

Priorität relativ zu gemessenen Engpässen (Long e2e OOM bei S28/p1792, Decode-Strafe, Mid flach).

### P0 — Compute-Buffer schrumpfen bei Phase-Switch (höchster Hebel)

**Befund:** `llama_set_runtime_ubatch` ändert nur den Skalar und wirft CUDA-Graphs weg — **kein** erneutes `graph_reserve` mit Decode-Größe.

```1448:1468:src/llama-context.cpp
bool llama_context::set_runtime_ubatch(uint32_t n_ubatch) {
    ...
    // only: synchronize + cuda_evict_graphs + runtime_n_ubatch = n_ubatch
}
```

**Gallocr wächst nur, schrumpft nie:**

```920:922:ggml/src/ggml-alloc.c
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
```

Selbst nach `dyn_tallocr_reset` bleibt der **physische** CUDA-Compute-Buffer auf Prefill-Peak (1792). Decode behält den VRAM-Ballast → erklärt Long-e2e-OOM bei S28/p1792 und ~4% Decode-Strafe S26 p1792 vs p1536.

**Fix-Skizze:**
1. `set_runtime_ubatch`: nach Setzen `graph_reserve(runtime_n_ubatch, …)` aufrufen (analog `memory_update`).
2. Gallocr: bei `new_chunk_size < cur_chunk_size` (optional mit Hysterese) Buffer freigeben und neu allokieren; oder explizites `ggml_gallocr_shrink` / free-all vor Decode-Reserve.
3. Server: nach Prefill→Decode-Switch erzwingen (bereits `set_cmoe_batch_phase(false)`).

**Erwartung:** S28+p1792 Long e2e möglich; Decode-Regression bei großem Prefill stark reduziert. **Das ist Option C.**

### P1 — DFlash Prefill-Inject-Pfad

**Combined-Inject nur für kleine Batches:**

```107:113:src/llama-context.cpp
    uint32_t max_combined_tokens() const {
        return std::min({
            target.cparams.n_ubatch,   // reserve max, not runtime
            draft.cparams.n_ubatch,   // draft = decode-sized (~64)
            draft.cparams.n_sampling_outputs_per_seq_max + 1, // ~n_max+1
        });
    }
```

Große Prefills gehen **immer** standalone (korrekt per Kommentar). Standalone:

```1178:1194:common/speculative.cpp
        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft); // max, not runtime
        // host memcpy: 3 layers × n_chunk × n_embd floats per chunk
```

- Host-Gather der Layer-Features + `llama_encode`/`llama_decode` draft in 64er Chunks.
- Draft ist absichtlich decode-groß → VRAM-schonend, aber serielle Inject-Last nach jedem Target-Ubatch.

**Optimierungen:**
| ID | Idee | Nutzen | Risiko |
|----|------|--------|--------|
| P1a | GPU-seitige Feature-Pack (kein Host-memcpy) | TTFT Mid/Long | Graph-Umbau |
| P1b | Draft-Prefill-ubatch separat (128–256) nur für Inject | weniger Inject-Chunks | Draft-VRAM |
| P1c | Overlap: Draft-Inject Chunk *i* \|\| Target Chunk *i+1* | Pipeline | State/Sync |
| P1d | `llama_get_runtime_ubatch` in process_rows | Korrektheit | gering |

Hinweis: `need_embd()==false` bei DFlash; Layer-Inp wird separat aktiviert — ok. Warnung `ctx_dft pos_max < N-1` in Benches → Draft-KV unvollständig nach Prefill (Qualität/Decode-TPS), Prefill-Zähler selbst target-dominiert.

### P2 — Mid Fixed-Cost (607 tok, 1 Chunk)

Config-ubatch hilft Mid nicht. Code-Hebel:

| ID | Ort | Idee |
|----|-----|------|
| P2a | CUDA graphs | Stabile Prefill-Shape (z. B. 607 / 1024) capturen; Phase-Switch invalidiert (bereits `evict_graphs`) |
| P2b | `server-context` Checkpoints | Prod `CTX_CHECKPOINTS=4` splitet Prompt nahe Ende (`4+n_ubatch`, `4`) → Extra-Graphs; Prefill-Bench mit 0 war unberührt |
| P2c | Phase-Switch Timing | `set_cmoe_batch_phase` + Sync vor erstem Token; Prefill-only-Slots ohne unnötigen Decode-Switch vor Warmup |
| P2d | Expert CPU cold | `THREADS_BATCH`, fused paths — historisch oft neutral |
| P2e | Turbo4 FA Tiles | `fattn.cu` Turing-Pfad für große `Q->ne[1]`; Profiling ob 607 auf optimaler Specialization landet |

### P3 — Init-Reserve / Doppelte PP-Reserve

```1278:1337:src/llama-context.cpp
    // reserve pp graph first
    // reserve tg
    // reserve dflash combined + verify
    // reserve pp again
```

Notwendig für worst-case, aber `n_tokens = cparams.n_ubatch` (max Phase). Mit P0 entkoppelt: Init könnte mit **max** allokieren ODER lazy beim ersten Prefill wachsen (komplexer).

`n_outputs_pp` bei DFlash-Target bereits auf `n_seqs` begrenzt (gut, spart Prefill-Output-VRAM).

### P4 — KVFlash

- Clamp `n_ubatch` nur bei kleinem Pool relevant; Prod pool=12288 unkritisch.
- Prefill-Fastpath (bulk mark_written, Scorer skip): marginal wenn Pool groß und sequential write.
- Sicherstellen dass Target-KVFlash wirklich an ist (Log-Zeile `kvflash_pool`); Draft-Warnung „KVFlash requires…“ ist **Draft-Context**, normal.

### P5 — Server Batching / Checkpoints

```4064:4077:tools/server/server-context.cpp
// checkpoint_offsets[] = {4 + n_ubatch, 4}
```

Bei großem `n_ubatch` und `n_ctx_checkpoints>0` erzwungene Brüche kurz vor Prompt-Ende → zusätzliche kleine Prefill-Graphs (Overhead, nicht Durchsatz-Killer bei Long, spürbar bei Mid).

### P6 — Was **nicht** lohnt (gemessen / strukturell)

- Nur logisches `n_batch` erhöhen ohne physisches ubatch.
- S weiter senken ohne P0 (Decode leidet, Mid nicht).
- Tree-verify / DDTree für Prefill.
- KVFlash-Pool verkleinern (clamp’t Prefill).

### Empfohlene Implementierungs-Reihenfolge

```
1. P0  set_runtime_ubatch → graph_reserve + gallocr shrink   [DONE]
2. Mess-Gate: Long prompt ≥+10%, decode ≤−3% vs S30/p1024  [prompt PASS; decode −3.6%]
3. P1a/P1b DFlash inject nur falls Long noch unter Ziel oder Mid-TTFT
4. P2 Mid fixed-cost / checkpoints / FA profile
```

### P0 Implementierung (Code)

| Datei | Änderung |
|-------|----------|
| `ggml/src/ggml-alloc.c` | Gallocr reallocated auch bei **Shrink** (`new_size < cur_size` / chunk ≠) |
| `src/llama-context.cpp` | `reserve_for_runtime_ubatch()` + `set_runtime_ubatch` re-reserve nach Phase-Switch; `backend_buf_exp_size` sync |
| `src/llama-context.h` | private `reserve_for_runtime_ubatch()` |
| `start1660.sh` | S=28, PREFILL=1792, DECODE=64 |

### P0 e2e-Gate (`prefill-p0-gate-20260816T170253Z`, REPEATS=3, n_predict=256)

| Config | Long prompt | Long decode | Mid prompt | Mid decode |
|--------|------------:|------------:|-----------:|-----------:|
| Base S30 p1024 | 132.3 | 30.74 | 97.0 | 33.21 |
| S28 p1024 | 132.3 (−0%) | 30.22 (−1.7%) | 96.9 | 34.22 (+3%) |
| **S28 p1792** | **171.1 (+29.3%)** | **29.62 (−3.6%)** | 95.3 (−1.8%) | 33.92 (+2.1%) |

- Vorher: S28 p1792 Long e2e **CUDA OOM**
- Nachher: 3/3 Runs ok; Compute-Buffer schrumpft (Destructor: ~2.9 MiB vs expected 103 MiB peak-history)
- Prompt-Ziel ≥+10%: **PASS**
- Decode-Ziel ≤−3%: **knapp verfehlt (−3.6%)**; vs S28/p1024 nur **−2.0%** (Prefill-Effekt); Rest ≈ S-Drop
- Mid +10%: weiterhin nicht (1 Chunk); Decode Mid ok

---

## 10. Post-P0 Re-Audit Prefill + Generation (2026-08-16)

Stand nach P0-Shrink, Graph-OOM-Fix, STARTED-Draft-Fix. Config: S=28, prefill 1792, decode 64, DFlash n_max=6.

### Gemessen (aktuell)

| Regime | Prefill | Decode | Bemerkung |
|--------|--------:|-------:|-----------|
| Long 1823 e2e | 171 tok/s (**+29%** vs 1024) | 29.6 (−3.6%) | P0-Gate |
| Mid 607 | ~95–98 (flach) | ~34 | schon 1 Chunk |
| Short ≤64 | 15–27 (Overhead) | **37–40** | bleibt auf decode-64 |
| Short Req2 (kein Cache) | — | **39.9** | nach STARTED-Fix kein Drop |
| Isolation none p=2048 Long | **187** | — | +40% vs DFlash-Base; Target-only Headroom |

### Was schon ausgereizt ist

- Physisches Prefill-ubatch bis **1792** (nächster Sprung nur 1856/2048 = 1 Chunk für ≥1823).
- Phase-Switch schrumpft Compute-Buffer (P0).
- Short Follow-up bleibt auf Decode-Graphs (STARTED-Fix).
- Combined DFlash-Inject nur für Verify (max ≈ n_max+1), Prefill immer standalone — strukturell korrekt.

### Offene Hebel (priorisiert)

| Prio | Hebel | Erw. Wirkung | Aufwand | Risiko |
|-----:|-------|--------------|---------|--------|
| **1** | **e2e-Retest prefill 1856/2048 @ S=28** nach P0 | Long PP **+3–7%** extra (181 vs 171); Decode evtl. wie 1792 | Config-Sweep | OOM Prefill-Peak (Shrink hilft Decode, nicht Prefill) |
| **2** | **`CTX_CHECKPOINTS=4` split** (`4+n_ubatch` und `4`) | Mid/Long TTFT: Extra-Graph der letzten 4 Tokens vermeiden | klein | Checkpoint-Qualität |
| **3** | **Draft-Inject-ubatch 128–256** nur Prefill (`process_rows`) | Long Prefill: 1792/64 = 28 Draft-encode/decode → 7–14 Calls | mittel | Draft-VRAM während Prefill |
| **4** | **DFlash n_max 6 → 3** (Kommentar sagt Winner n_max=2) | Decode **+2–8%** wenn Verify ungenutzter Draft teuer; AL prüfen | Config | AL/Qualität |
| **5** | Host-Gather Layer-Features (3× n_embd × N) auf GPU packen | Prefill TTFT gering (DFlash vs none war nur ~2%) | hoch | Graph |
| **6** | CUDA-Graph für volle Prefill-1792-Chunks | 2. Request gleiche Shape | mittel | Evict bei Phase-Switch |
| **7** | S=30 während Prefill | Prefill-Peak, nicht Decode — historisch S≈neutral für PP | Config | OOM |
| **8** | `THREADS_BATCH` 8→12 | CPU-Cold Prefill | Config | Contention |

### Generation-spezifisch

- Verify-Schritt ~43–69 ms (DFlash n_max=6). Historisch n_max=3 ≈ 36 decode tok/s, n_max=5 n_min=2 ≈ 32.
- Header `start1660.sh` sagt noch „DFlash winner n_max=2“, Config ist **n_max=6 n_min=1** — lohnt A/B.
- Nach jedem Long-Prefill: Graphs evict + Recapture (unvermeidbar, Buffer-Realloc). Warmup ≈ 2 Verify-Steps, dann ~40 auf kurzem Follow-up.

### Nicht-Ziele (weiterhin)

- Nur logisches `n_batch` ohne ubatch.
- Combined-Inject für 1792 Tokens (würde Draft-Graph auf Prefill-Größe aufblasen).
- KVFlash-Pool verkleinern (clamp’t Prefill).
- Mid +10% allein durch größeres ubatch (607 ≤ 1024).

### Experiment 1 — 1856/2048 e2e @ S=28 (DONE, `prefill-e2e-1856-2048-20260817T063155Z`)

REPEATS=3, n_predict=256, DFlash, decode 64. Alle Zellen **ok** (kein OOM).

| Prefill | Long PP | Long decode | Mid PP | Mid decode |
|--------:|--------:|------------:|-------:|-----------:|
| 1792 | 171.4 | 29.71 | 95.45 | 33.96 |
| **1856** | **177.0 (+3.2%)** | **32.17 (+8.3%)** | 95.22 | 33.43 (−1.6%) |
| 2048 | 176.3 (+2.8%) | 29.36 (−1.2%) | 94.78 | 33.76 |

vs historisch S30/p1024: p1856 Long **+33.8% PP**, Decode **+4.7%** (P0-Gate 1792 war Decode −3.6%).

**Winner: 1856.** Ein Chunk für 1823 Tokens; 2048 bringt kein PP extra, Decode fällt zurück. 3/3 Reps stabil (decode 32.15–32.36).

Promoted in `start1660.sh`.

### Experiment 2+3 — Checkpoints + n_max (DONE, `ckpt-nmax-20260817T081222Z`)

**Checkpoints 0 vs 4** (prefill 1856, n_max=6, 3×):

| ckpt | Mid PP | Mid TTFT | Mid decode | Long PP | Long TTFT | Long decode |
|-----:|-------:|---------:|-----------:|--------:|----------:|------------:|
| 0 | 95.64 | 6370 | 33.65 | 176.95 | 10332 | 32.22 |
| 4 | 95.30 (−0.4%) | 6393 (+0.4%) | 33.62 | 176.72 | 10342 | 32.24 |

→ **ckpt=4 behalten** (kein messbarer Prefill-Schaden, Resume-Nutzen bleibt).

**n_max 2 / 3 / 6** (short 37 tok, ckpt=0, 3×):

| n_max | decode_tps | acc | AL | vs n_max=6 |
|------:|-----------:|----:|---:|-----------:|
| **2** | **35.33** | 0.913 | 2.32 | **+8.5%** |
| 3 | 29.00 | 0.794 | 2.23 | −11.0% |
| 6 | 32.57 | 0.968 | 2.60 | — |

→ **n_max=2** in `start1660.sh` (passt zum Header-Winner; n_max=3 ist klar schlechter).

---

## 9. Referenzen

- `HYBRID_EXPERIMENTS.md` — Physical prefill ubatch sweep  
- `HYBRID_DESIGN.md` — Physical prefill ubatch sizing / phase batching  
- `scripts/ab_prefill_sweep.sh`, `scripts/ab_prefill_prod.sh`  
- `benchmark-results/prefill-sweep-20260810T143805Z`  
- `benchmark-results/prefill-prod-20260816T145511Z`  
- `benchmark-results/start1660-ctx-20260812T073923Z`  
- `src/llama-context.cpp` KVFlash n_ubatch clamp  
- `tools/server/server-context.cpp` `set_cmoe_batch_phase`  
- `common/speculative.cpp` `common_base_params_to_speculative` draft decode sizing  
- `start1660.sh` CMOE_PREFILL_* / KVFlash / Turbo4 F16  
