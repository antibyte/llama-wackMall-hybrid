# start1660.sh live reference (GTX 1660 Ti)

Date: 2026-08-24

Status: snapshot of the current `start1660.sh` editable block. Not a promotion
claim for untested hardware. Do not overwrite this file from autotune.

## Observed on this machine

```text
GPU:     NVIDIA GeForce GTX 1660 Ti, SM 75, 6 GiB
binary:  build-main-sm75/bin/llama-server
model:   Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
draft:   Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf
decode:  ~40-42 tok/s sustained, ~52 tok/s peak
```

Follow-up (second prompt) previously logged:

```text
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 210.00 MiB on device 0: cudaMalloc failed: out of memory
```

210.00 MiB is one Qwen Down expert tensor (`256 * 860160`). Prefetch and
the wide-prefill graph copy both wanted a buffer of that size. Phase switch
`1856 -> 64` shrank gallocr and freed the copy, so the next `64 -> 1856`
tried `cudaMalloc` again with KV and decode graphs already resident. S and
context do not size that tensor. Prefetch slots are now allocated once and
bound as the graph copy, so the second prompt reuses the first allocation.

## Live knobs (as in start1660.sh)

These values are what the launcher actually exports today. Older comments in
the script (S=28, n_max=2, KVFlash 12288) are stale.

| Area | Value |
| --- | --- |
| SPEC_MODE | dflash |
| SPEC_DRAFT_N_MAX / N_MIN / P_MIN | 4 / 1 / 0.75 |
| DFLASH_COMBINED | 1 |
| DFLASH_TARGET_TENSOR_OVERRIDE | `^blk[.]40[.]=CPU` |
| CONTEXT | 32768 |
| N_PREDICT | 8192 |
| N_PARALLEL | 1 |
| CPU_MOE | 1 |
| TARGET/DRAFT KV | turbo4_k / turbo4_k |
| FLASH_ATTN | on |
| LOAD_MODE | mmap |
| KVFlash resident / pool | 4096 / 8192, policy lru |
| CMOE prefill batch/ubatch | 1856 / 1856 |
| CMOE decode batch/ubatch | 64 / 64 |
| LLAMA_EXPERT_S | 30 |
| PROFILE_KIND | specialist (`profiles/specialist-benchprompt.csv`) |
| WARM_SLOTS | 0 |
| STATIC_NO_SYNC | 1 |
| VRAM_RESERVE_MIB | 500 |
| LLAMA_EXPERT_BW_PROFILE | `profiles/gtx1660-expert-bw.json` (cpu-heavy q*~0.10) |
| CTX_CHECKPOINTS | 8 |
| CACHE_RAM | 2048 MiB |
| CACHE_PROMPT / CACHE_REUSE | 1 / 0 (GDN must not seq_add-shift) |
| REASONING / BUDGET / PRESERVE | on / 2000 / 1 |
| JINJA | 1 |
| CHAT_TEMPLATE_FILE | `models/templates/Qwen-Fixed-v22.3.jinja` (froggeric v22.3) |
| REASONING_FORMAT | deepseek |
| THREADS (target/draft) | 8 / 8 |
| SKIP_SENTINEL / SHARED_HOT_IDS | 1 / 1 |
| MOE_MULTI_FUSION / COMBINE_FUSION | 1 / 1 |
| ASYNC_HOST_COPY / DEDUP_DST_SYNC | 1 / 1 |
| GGML_CUDA_REGISTER_HOST | 1 (host pin; 21 GiB cudaHostRegister failed with "operation not supported"; does not free VRAM) |
| GGML_SCHED_PREFETCH_EXPERTS | 2 (persistent 210 MiB staging, reused across prompts; gallocr does not allocate a second copy) |
| LOOKAHEAD / BRIDGE | 0 / empty layer lists |
| ADAPT / WARM / CPU_ASYNC | 0 / 0 / 0 |

Kernel rows (sm_75 winners kept explicit): Q8 ncols3=4, Q6_K ncols1=2, Q6_K ncols3=4, CONCAT block=128, CONCAT_FLAT_DIM0=1.

## What this is not

- Not the 1080 stack (`start1080.sh`: S=58, n_max=8, prefill 1024, decode 128).
- Not n-cpu-moe. Prefetch-experts was measured +58% pp512 on `-ncmoe 99` only.
- Not a claim that REGISTER_HOST pins 21 GiB on this laptop.

## Reproduce

```bash
./start1660.sh
```

No extra CLI args. Edit the block at the top of `start1660.sh`, then rerun.
Empty `CHAT_TEMPLATE_FILE=""` falls back to GGUF metadata.
