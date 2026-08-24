# FreeToken GTX 1080 Handoff

Date: 2026-08-24

Status: measurement handoff. The 1660 Ti work is committed. This document tells
the GTX 1080 host what landed, what that GPU already disproved, and which
FreeToken pieces are still worth an sm_61 screen. Do not treat 1660 W=0 as a
1080 production conclusion.

Paper: FreeToken (arXiv:2608.16157). Mapping onto wackMall:

| FreeToken idea | wackMall status after 1660 | 1080 interest |
| --- | --- | --- |
| q* miss split (fetch fraction of cold experts over PCIe, rest on CPU) | measured q*~0.10, cpu-heavy, production W=0 | **primary new measurement** |
| overflow LRU W on top of fixed S | W=2 async was -3.2% vs W=0 | re-open if q* is hybrid |
| LRU-vs-S replay | simulator landed; 1660 replay favored static S | re-run with 1080 q* and traces |
| semantic GDN/KV checkpoints | landed; 1660 production `cache-reuse=0`, checkpoints=8 | **correctness + multi-turn** |
| leftover / CUDA-graph batching | landed; 1856<->64 eviction was a TPS knick | 1024<->128 on 1080 |
| overlapped H2D prefetch | 1660 copy hidden in isolation, still lost e2e | 1080 PCIe is faster, CPU slower |
| Device-LUT for CUDA graphs | **not implemented**, gated | only if W>0 wins |
| elastic VRAM W | existing S/W auto-fit only; no new allocator | do not add a subsystem |
| DFlash algorithm | left alone except live KV bugs | already on 1080 production |

## Expected target

```text
repository: /root/llama-wackMall-hybrid
branch:     codex/transient-expert-feasibility
commit:     4bf36d8db  (or later on the same branch)
            hybrid: keep DFlash KV, GDN reuse, and on-demand router
GPU:        NVIDIA GTX 1080, 8 GiB, SM 61, active PCIe Gen3 x16
CPU:        Intel Core i7-4770, 4 cores / 8 threads, AVX2
CUDA:       12.4, host GCC/G++ 13
binary:     build-turbo-opt-sm61/bin/llama-server
model:      /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
draft:      /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf
profile:    find a validated heat CSV; do not copy 1660 S=28 layouts blindly
results:    /root/gtx1080-hybrid-results/freetoken-YYYYMMDD/
```

Do not use `profiles/gtx1660-expert-bw.json` on this machine.

## Why 1080 is not 1660

1660 Ti (Ryzen 4800H, PCIe Gen3 x8, 6 GiB):

```text
pinned H2D:     ~6.15 GB/s
CPU expert:     ~59.8 GB/s
q*:             0.1029
recommend:      cpu-heavy
production:     S=28, W=0, DFlash n_max=2, decode 64, prefill 1856
cold path:      CPU is ~10x faster than a full-expert H2D
```

1080 (i7-4770, PCIe Gen3 x16, 8 GiB), already measured in
`TRANSIENT_EXPERT_EXPERIMENTS.md`:

```text
pinned H2D:     ~9.84 to 9.89 GiB/s  (full expert ~0.167 ms)
CPU Gate+Up L0: ~0.112 ms  (break-even with the Gate+Up copy)
CPU cold share: ~45% of decode time
resident GPU:   ~0.025 to 0.028 ms per expert
production:     S=58, W=0, DFlash n_max=8, decode 128, prefill 1024
copy overlap:   92 to 98% of pinned copy hidden under a synthetic window
```

q* is `pcie_gbs / cpu_gbs` (equivalently `cpu_ms / pcie_ms` for the same
expert). On 1660 that ratio is ~0.10. On 1080 the older DDR3 CPU and the
faster x16 link push it toward 0.5..1.0. A 1660 "never copy" policy can be
wrong here even if every other knob stays the same.

Do not guess the 1080 q*. Measure it with `tools/bench_expert_bw.py`.

## Protected rules

- Do not implement a Device-LUT, graph cache, or new elastic-VRAM allocator
  until the ordered gates below pass. The 1660 graph-cache attempt aborted
  (`dist_cumsum` in a CUDA0 buffer that cannot run SUB) and was reverted.
- Do not change the DFlash draft algorithm. Combined inject, DDTree, and
  n_max=8 stay as in `start1080.sh`. The 1660 DFlash work was KV/lifetime
  only (stale `pos_max=0`, leftover 3-token GDN split, keep-draft fill).
- Do not enable `GGML_CUDA_MOE_MULTI_FUSION` on Pascal.
- Do not copy `LLAMA_EXPERT_WARM_SLOTS` from 1660 into a "win" without a
  1080 3x512 (or 3x2000) hash-identical A/B.
- `LLAMA_EXPERT_STATIC_NO_SYNC=1` is only legal with W=0, adapt off, and
  stats off. Any W>0 run must accept the tier barrier again.
- DFlash does **not** trip the MTP warm guard (`configure_mtp` is MTP-only).
  Production `SPEC_MODE=dflash` can test W without
  `LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=1`. MTP-1/3 still need that flag; MTP-2
  currently unguards W but remains a correctness experiment.
- `scripts/bench_hybrid.sh` cases C/D/E/F also set `ADAPT=1`. FreeToken q*
  tests must keep `ADAPT=0` like production. Prefer a `start1080.sh` A/B, or
  case `SC` as the W=0 control.

## What already landed (commit `4bf36d8db`)

Code, not 1080 measurements:

- `src/llama-expert-bw-profile.h`, `LLAMA_EXPERT_BW_PROFILE`
  (`cpu-heavy` / `hybrid` / `pcie-heavy`, q* admit cap).
  `cpu-heavy` keeps W only when `0 < q* < 1`; otherwise it forces W=0.
  Prefill-sized graphs (`counts_wide` or peak count > `LLAMA_EXPERT_TMAX`)
  do not admit H2D fills.
- `tools/bench_expert_bw.py`, `tools/simulate_expert_decode_cache.py`
- GDN-safe reuse: `common_prompt_kv_shift_safe`, LCP vs checkpoint,
  semantic chat anchors, checkpoint density/evict/radix helpers
- CMOE leftover policy: stay on decode graphs for follow-up leftovers;
  ngram-only leftover batch of 8; DFlash leftovers must not split to 3 tokens
- Combined DFlash: wipe on `Y != X+1` using the **min** pos of the seq;
  host `spec_ckpt.update_pos`; `common_spec_dflash_keep_draft_token` so
  verify `n_tokens` stays `n_max+1`
- Router: `child-env-file`, `router1080.sh`, `router1660.sh`
- Tests: `tests/test-hybrid-lcp.cpp` (build includes `../src`)

`start1080.sh` / `router1080.sh` still have the **old** 1080 cache knobs
(`cache-reuse=16`, `ctx-checkpoints=0`, no bw profile, W=0). That is
intentional until this host measures. 1660 production is `cache-reuse=0`
and `ctx-checkpoints=8`.

## Order of work

Stop at the first failed gate. Do not skip q* to "just try W=2".

### 0. Snapshot and build

```bash
cd /root/llama-wackMall-hybrid
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline --decorate

RESULT_ROOT=/root/gtx1080-hybrid-results/freetoken-$(date -u +%Y%m%d)
mkdir -p "$RESULT_ROOT"
nvidia-smi > "$RESULT_ROOT/nvidia-smi.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv \
    > "$RESULT_ROOT/gpu-link-idle.csv"
lscpu > "$RESULT_ROOT/lscpu.txt"
```

If the worktree is dirty, document it. Do not reset.

Rebuild the existing turbo-opt tree (do not reuse an sm_75 binary, do not
force MMQ):

```bash
cmake --build build-turbo-opt-sm61 -j 8 --target \
    llama-server llama-cli \
    llama-expert-transport-bench \
    test-hybrid-lcp test-dspark-type \
    test-expert-warm-cache test-expert-adaptation test-expert-placement
```

If that build dir cannot compile the new tests, configure a sibling
`build-freetoken-sm61` with `-DCMAKE_CUDA_ARCHITECTURES=61` and
`-DGGML_CUDA_FORCE_MMQ=OFF` as in `GTX1080_HANDOFF.md`.

```bash
python3 -m unittest tests/test-expert-streaming-tools.py
./build-turbo-opt-sm61/bin/test-hybrid-lcp
./build-turbo-opt-sm61/bin/test-expert-warm-cache
```

`test-hybrid-lcp` must pass before any GDN cache-reuse experiment.

Record the active PCIe gen/width **while** the transport bench runs, not only
at idle (1080 idles at Gen1 x16 P8).

### 1. Gate: native q* profile (do this first)

Reuse or recapture a transport summary plus CPU timing. Prefer a fresh
`pinned_h2d_cpu_overlap` run; the 2026-08-04 numbers are a prior, not this
commit.

```bash
MODEL=/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
TRANSPORT_RUN="$RESULT_ROOT/transport-sm61"

python3 tools/bench_expert_transport.py \
    --model "$MODEL" \
    --binary build-turbo-opt-sm61/bin/llama-expert-transport-bench \
    --output-dir "$TRANSPORT_RUN" \
    --runs 3 --repeats 200 --warmups 20 \
    --working-set-mib 32 --overlap-us 500
```

If the wrapper still needs a stats JSON for CPU GB/s, take the median
`EXPERT_TIMING=1` DFlash or no-MTP control (four threads) as in
`GTX1080_HANDOFF.md` and pass `--stats-json`.

```bash
python3 tools/bench_expert_bw.py \
    --transport-json "$TRANSPORT_RUN/summary.json" \
    --output "$RESULT_ROOT/gtx1080-expert-bw.json"
```

Do not overwrite an existing output path; pick a new name.

Interpretation:

| recommended | q* | 1080 action |
| --- | ---: | --- |
| cpu-heavy | ~0.10 like 1660 | keep production W=0; skip live W A/B; still do GDN/cache and DFlash KV |
| cpu-heavy | 0.3..0.5 | live W A/B **with** the profile admit cap; copies are rationed |
| hybrid | ~0.5..0.9 | **main** live experiment: W=1/2/4 + q* |
| pcie-heavy | ~1.0 | fill up to W; still cap by VRAM vs S |

Copy the JSON into `profiles/` only after the number is stable across two
transport processes. `start1080.sh` should keep W=0 until step 3 promotes.

`scripts/bench_hybrid.sh` does not yet export `LLAMA_EXPERT_BW_PROFILE`.
For live W runs, export it yourself:

```bash
export LLAMA_EXPERT_BW_PROFILE="$RESULT_ROOT/gtx1080-expert-bw.json"
```

### 2. Offline: LRU vs static S (cheap, 1660 had no live win)

Need a routing trace. Reuse a held-out 1080 lookahead JSON if it still
matches this model, otherwise collect `CASES=LT` once (observational; hashes
must match a phase-matched L0). Then:

```bash
python3 tools/simulate_expert_decode_cache.py \
    --trace "$REPLAY_TRACE" \
    --slots-per-layer 54,58,62 \
    --warm-slots 1,2,4 \
    --bw-profile "$RESULT_ROOT/gtx1080-expert-bw.json" \
    --output "$RESULT_ROOT/lru-vs-s.json"
```

1660 result: static S beat a global LRU that stole the same slot budget, and
overflow W only helped in the simulator when copies were cheaper than CPU.
On 1080, report:

- static S miss rate vs per-layer LRU with the same slot count
- static S plus overflow W=1/2/4 miss rate
- `cpu_only_ms` vs `qstar_ms` vs `fill_all_ms`

If `qstar_ms` is not clearly below `cpu_only_ms`, do not spend VRAM on W.

### 3. Live: W on DFlash (only if step 1 is hybrid or q* >= ~0.3)

Production control is `start1080.sh`: S=58, W=0, `STATIC_NO_SYNC=1`,
DFlash n_max=8, turbo4 KV, skip-sentinel, shared-hot-ids, CPU multi-row,
4 threads, prefill 1024, decode 128.

VRAM: one Qwen expert is ~1.8-2.0 MiB, ~40 routed layers.

```text
W=2 ~ 150 MiB
W=4 ~ 300 MiB
```

S=58 vs S=62 was already a 1080 decode trade. Do **not** add W on top of
S=58 until a load log shows free VRAM after DFlash draft KV. Safer first
pair:

```text
A: S=58 W=0   STATIC_NO_SYNC=1   (current production)
B: S=54 W=2   STATIC_NO_SYNC=0   BW_PROFILE=gtx1080-expert-bw.json
             WARM_PREFETCH=0 then 1
```

Same prompt, fresh server, 3x512 minimum, hashes must match. Watch:

- load OOM or stream abort (S=65 + n_max>=8 already failed on 8 GiB)
- `expert_tier: bandwidth profile ... recommended=... q_star=...`
- warm promotions / H2D bytes vs decode tok/s
- CUDA graph recapture / `set_runtime_ubatch` 1024<->128
- think-loop or `mean_draft_out` collapse (1660 DFlash KV bug class)

Promotion bar: median decode >= production and hash-identical, with no extra
OOM. A simulator-only win is not enough.

Prefetch (`LLAMA_EXPERT_WARM_PREFETCH=1`, one stream, max inflight 2) is the
second W variant, not the first. 1080 already proved copy-engine overlap in
isolation; e2e still has to beat W=0.

### 4. GDN / prompt cache (1660 needed this; 1080 still on old knobs)

Qwen3.6 is hybrid GDN. Recurrent state is prefix-dependent. `seq_add` chunk
reuse and context shift are unsafe when `n_swa==0`. 1660 production therefore
uses `--cache-reuse 0`. `start1080.sh` still has `--cache-reuse 16` and
`--ctx-checkpoints 0`.

This is a **correctness** screen, not a speed hunt:

1. Multi-turn think / tool chat with `cache-reuse=16` vs `0`. Compare
   tokens, not just tok/s. IMA, truncated think, or diverging follow-ups
   fail the 16-path.
2. `ctx-checkpoints=0` vs `8` with reuse=0. 1660 needed 8 because 4 evicted
   last-user / turn-end too quickly. GDN snapshots are tens of MiB; the new
   density/split helpers skip think-tag splits and keep fill-wide prefills.
3. Follow-up leftover must stay on decode 128. A 64->1856->64 style phase
   switch was a 1660 TPS knick; 1080 equivalent is 128->1024->128.

Ling-tiny on 1080 (`start-ling-tiny-1080.sh`, `router1080.sh`) still has
`cache-reuse=16` and checkpoints=4. Ling is KDA+MLA; treat reuse=0 as the
safe default until a dedicated Ling A/B says otherwise. Do not enable
`--cpu-moe` for Ling (`no-cpu-moe=true` / `LLAMA_ARG_NO_CPU_MOE=1`).

### 5. Combined DFlash KV (already in binary; confirm on n_max=8)

1660 bugs that showed up as "10 tok/s gone" and think-loops:

- combined inject with draft `pos_max=0` vs target Y; skip-inject never
  wiped; `seq_rm(dft,1,-1)` then dropped prompt KV
- ngram leftover cap applied to 3-token DFlash/GDN leftovers
- empty draft (1 token) vs 3-token verify rebuilt the graph every step

Fixes are in `common_dflash_pos_jump`, host `update_pos`,
`common_spec_clear_draft_kv`, `common_spec_dflash_keep_draft_token`.
1080 runs n_max=8, so verify should stay at 9 tokens. Confirm on a long
think prompt:

- no `Y=X+1` spam in the log after the first token
- `mean_draft_out` not stuck near 0
- acceptance stays in the historical ~0.94 band for the 3x512 recipe
- follow-up turns do not dump draft KV and recapture 1024-wide graphs

Do not retune n_max, p_min, or DDTree in this handoff.

### 6. Router smoke

```bash
./router1080.sh
```

One model in VRAM (`--models-max 1`). OpenWebUI / API names:

```text
qwen3.6-35b-a3b-hybrid-gtx1080
ling-tiny
```

Check: first Qwen load, generate, switch to Ling (Qwen child dies, Ling
gets `no-cpu-moe`), switch back. Child env files live under
`.router-runtime/1080/` (gitignored). `cpu-moe=false` in an INI is a no-op;
Ling must keep `no-cpu-moe=true` plus `LLAMA_ARG_NO_CPU_MOE=1`.

### 7. Device-LUT (gated: do not implement)

CUDA graphs capture the expert LUT. W>0 uploads `lut_host` after fills and
recaptures. FreeToken's device LUT would mutate the mapping without a
host-side recapture.

Implement only if **all** of these hold:

1. Step 1 recommended hybrid or pcie-heavy.
2. Step 3 W>0 is faster than W=0 on a 3x512 hash-identical A/B.
3. Server logs show graph recapture / rebuild as the remaining cost, not
   H2D bytes or CPU cold.

Until then leave LUT upload as-is. Do not revive the 1..8-token graph-slot
cache.

## What 1660 already killed (do not re-litigate without new 1080 data)

These are 1660 conclusions. They stay rejected there. 1080 may still test
the starred items; the others are architecture or algorithm, not GPU.

| Item | 1660 result | 1080? |
| --- | --- | --- |
| routing-exact sync H2D of a cold expert | H2D 0.29 ms vs CPU 0.05 ms, always lose | re-check via q*; 1080 Gate+Up is break-even |
| mapped-host zero-copy experts | mapped read already slower than CPU | only if mapped read < CPU phase |
| production W>0 (sync or async) | async W=2 = -3.2% | **re-open after q*** |
| Device-LUT | not built; W=0 made it moot | gated on W win |
| ggml graph cache (1..8 token slots) | CUDA abort `dist_cumsum` | do not port |
| `cache-reuse>0` on GDN | unsafe; prefix-dependent RS | **set 0**, do not "tune" 16 |
| ngram leftover graph on DFlash | IMA / think loops | keep gated `!ctx_dft` |
| `GGML_CUDA_MOE_MULTI_FUSION` | sm_75-only | keep 0 |
| CPU_ASYNC / DOWN_PREFETCH | 1080 already -4.4% / neutral | leave off |
| lookahead / expert bridge as production | both hosts rejected | out of scope here |
| DFlash n_max retune | 1660 winner n_max=2 | 1080 winner n_max=8; do not mix |

## What is already a 1080 winner (do not A/B again)

Keep these from `start1080.sh` while testing FreeToken:

- `LLAMA_EXPERT_SKIP_SENTINEL=1` (+3.31% 3x512)
- `LLAMA_EXPERT_SHARED_HOT_IDS=1`
- `LLAMA_EXPERT_CPU_MULTI_ROW=1`, `CPU_REUSE_ROWS=1`
- `GGML_CUDA_ASYNC_HOST_COPY=1`
- DFlash n_max=8, p_min=0.75, S=58, prefill 1024, decode 128, 4 threads
- Turbo4 target+draft KV on the turbo-opt binary

## Promotion back into `start1080.sh`

Only after hash-identical 3x512 (then 3x2000 if W changed):

1. `profiles/gtx1080-expert-bw.json` + `LLAMA_EXPERT_BW_PROFILE=...`
2. W>0 and/or prefetch, with the S that still loads DFlash
3. `--cache-reuse 0` and a measured checkpoint count (likely 8)
4. Router env files matching those knobs

Leave Device-LUT, graph cache, and elastic W unimplemented unless step 7
unblocks them.

## Record

Write medians, hashes, VRAM, PCIe gen/width, q*, recommend, W, S, and the
exact binary path under `$RESULT_ROOT`. Append a short section to
`TRANSIENT_EXPERT_EXPERIMENTS.md` or a sibling `FREETOKEN_EXPERIMENTS.md`
only with numbers this host actually ran. Do not copy 1660 tok/s into a
1080 claim.
