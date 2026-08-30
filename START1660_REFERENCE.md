# start1660.sh live reference (GTX 1660 Ti)

Date: 2026-08-27

Status: snapshot of the current `start1660.sh` editable block plus the
observed first-prompt run. Not a promotion claim for untested hardware.
Do not overwrite this file from autotune.

## Observed on this machine

```text
GPU:     NVIDIA GeForce GTX 1660 Ti, SM 75, 6 GiB
binary:  build-main-sm75/bin/llama-server
model:   Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
draft:   Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf
listen:  0.0.0.0:8080
```

First prompt (live 2026-08-27, 13-token leftover prefill):

```text
prompt eval:  13 tok,  19.41 tok/s
eval:       3781 tok,  45.04 tok/s  (22.20 ms/tok)
peak tg_3s:            57.48 tok/s
graphs reused:         784
draft acceptance:      0.80552 (2655 / 3296), mean len = 4.22
dflash: draft 8.42 ms/call + verify 65.79 + inject 0.01 = 74.22
        mean_block_tok=5.00 mean_draft_out=2.92 combined_inject=1127
```

Fitted at load (cpu-heavy q*=0.103):

```text
HYBRID AUTO-FIT -> fixed S = 20, warm W = 8 (slot budget 36, per-slot 72.90 MiB)
20 fixed + 8 warm slots/layer, 120 tensors, 2.06 GiB pinned, seed coverage 59.7%
warm admission frequency window=200 graphs replace_ratio=2.50
async prefetch: streams=1 max_inflight=4 staging=1.95 MiB
STATIC_NO_SYNC rejected: warm slots are enabled
```

## Live knobs (as in start1660.sh)

These values are what the launcher actually exports today.

| Area | Value |
| --- | --- |
| SPEC_MODE | dflash (`draft-dflash`) |
| SPEC_DRAFT_N_MAX / N_MIN / P_MIN | 4 / 1 / 0.75 |
| DFLASH_COMBINED | 1 |
| DFLASH_TARGET_TENSOR_OVERRIDE | `^blk[.]40[.]=CPU` |
| MTP fallback (inactive under dflash) | 2 |
| CONTEXT | 65536 |
| N_PREDICT | 32768 |
| N_PARALLEL | 1 |
| CPU_MOE | 1 |
| TARGET/DRAFT KV | turbo4_k / turbo4_k |
| FLASH_ATTN | on |
| LOAD_MODE | mmap |
| KVFlash resident / pool | 4096 / 8192, policy lru, tau=64, stats=0 |
| CMOE prefill batch/ubatch | 1856 / 1856 |
| CMOE decode batch/ubatch | 64 / 64 |
| LLAMA_EXPERT_S | 20 |
| LLAMA_EXPERT_WARM_SLOTS | 8 |
| PROFILE_KIND | specialist (`profiles/specialist-benchprompt.csv`) |
| WARM_ADMISSION / WINDOW | frequency / 200 |
| WARM_REPLACE_RATIO | 2.5 |
| WARM_PREFETCH / STREAMS / INFLIGHT | 1 / 1 / 4 |
| STATIC_NO_SYNC | 1 (rejected while W>0) |
| VRAM_RESERVE_MIB | 250 |
| LLAMA_EXPERT_BW_PROFILE | `profiles/gtx1660-expert-bw.json` (cpu-heavy q*=0.103) |
| CTX_CHECKPOINTS | 8 |
| CACHE_RAM | 4096 MiB |
| CACHE_PROMPT / CACHE_REUSE / IDLE | 1 / 0 / 1 (GDN must not seq_add-shift) |
| REASONING / BUDGET / PRESERVE | on / 6000 / 1 |
| JINJA | 1 |
| CHAT_TEMPLATE_FILE | `models/templates/Qwen-Fixed-v22.3.jinja` (froggeric v22.3) |
| REASONING_FORMAT | deepseek |
| THREADS (target/draft) | 8 / 8 |
| SKIP_SENTINEL / SHARED_HOT_IDS | 1 / 1 |
| MOE_MULTI_FUSION / COMBINE_FUSION | 1 / 1 |
| CPU fused Gate/Up / multi-row / reuse-y | 1 / 1 / 1 |
| ASYNC_HOST_COPY / DEDUP_DST_SYNC / ASYNC_D2H | 1 / 1 / 0 |
| GGML_CUDA_REGISTER_HOST | 1 (host pin; 21 GiB cudaHostRegister failed with "operation not supported"; does not free VRAM) |
| GGML_SCHED_PREFETCH_EXPERTS | 2 (persistent 210 MiB staging, reused across prompts; gallocr does not allocate a second copy) |
| n-cpu-moe register-host / prefetch-experts | 1 / 2 |
| LOOKAHEAD / BRIDGE | 0 / empty layer lists (consume=0) |
| ADAPT / CPU_ASYNC | 0 / 0 |
| CUDA MMVQ rows | Q8n1=0 Q8n2=0 Q8n3=4 Q6n1=2 Q6n3=4 |
| CUDA concat | flat-dim0=1 block=128 |
| Turbo4 guards | V=1 MTP=1 DFlash=1 draft=1; FP16-threshold=2 convert=0 wht-shuffle=0 Q8-layers=none |
| MTP head | none (trace=0) |

Kernel rows (sm_75 winners kept explicit): Q8 ncols3=4, Q6_K ncols1=2, Q6_K ncols3=4, CONCAT block=128, CONCAT_FLAT_DIM0=1.

## Expected startup warnings (not regressions)

- `API_KEY` empty and CORS `*`: LAN-open, no auth. Restrict before any untrusted network.
- `dflash requires ctx_other to be set` during memory fitting: normal; draft still loads afterwards.
- `failed to measure draft model memory`: same fitting pass; ignore if `draft-dflash` is added later.
- `tensor overrides to CPU are used with mmap enabled`: expected with `DFLASH_TARGET_TENSOR_OVERRIDE`.
- `KVFlash requires Flash Attention and a causal, single-sequence, ...; disabling`: **draft context only**. Target KVFlash 4096/8192 stays on.
- `static no-sync rejected`: expected while W>0; synchronized tier updates are used.

## Second prompt / cache notes

Prefetch slots (210 MiB Down-expert, `256 * 860160`) are allocated once and
bound as the graph copy, so the second prompt must not `cudaMalloc` that
tensor again. S and context do not size that allocation.

Follow-up with a blown think-budget retokenizes history and used to force
prefill 1856 before `continue_from_cached`. The server now estimates the
continue leftover (`n_have = n_cached`, `n_left = last user turn`) before
the phase switch. Restart the server after that change.

The remaining second-prompt drop (~5 tok/s vs first prompt) is DFlash
acceptance and slower draft (`accept ~0.69`, draft `~11 ms/call`), not
target `process_ubatch` (~60 ms, unchanged). That path is still open.

## Recent first-prompt ladder (same machine, DFlash n_max=4)

| S / W | extra | tok | tok/s | accept |
| --- | --- | --- | --- | --- |
| 28 / 1 | window 200 | 4389 | 42.14 | 0.75 |
| 27 / 2 | window 200 | - | 42.49 | 0.77 |
| 28 / 2 | reserve 400 | 4822 | 43.24 | 0.79 |
| 28 / 2 | replace_ratio 2.0 | 3060 | 44.06 | 0.77 |
| **20 / 8** | **ratio 2.5, inflight 4, reserve 250** | **3781** | **45.04** | **0.806** |

CUDA graphs stay off while W>0. Frequency admission with replace_ratio 2.5
cut copy churn vs ratio 1.0.

## What this is not

- Not the 1080 stack (`start1080.sh`: S=58, n_max=8, prefill 1024, decode 128).
- Not n-cpu-moe. Prefetch-experts was measured +58% pp512 on `-ncmoe 99` only.
- Not a claim that REGISTER_HOST pins 21 GiB on this laptop.
- Not a claim that second-prompt TPS matches the first prompt.

## Reproduce

```bash
./start1660.sh
```

No extra CLI args. Edit the block at the top of `start1660.sh`, then rerun.
Empty `CHAT_TEMPLATE_FILE=""` falls back to GGUF metadata.
