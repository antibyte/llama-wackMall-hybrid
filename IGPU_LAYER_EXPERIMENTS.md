# AMD Renoir iGPU layer-offload experiment

Date: 2026-08-06

This experiment asks whether an integrated AMD GPU can supplement the primary
NVIDIA GPU. It is intentionally backend- and capacity-parameterized; the
negative Renoir result does not disable testing on newer or faster APUs.

## Test host and scope

- CPU/APU: Ryzen 7 4800H with Radeon Graphics (RADV RENOIR)
- primary GPU: GeForce GTX 1660 Ti, 6 GiB, CUDA sm_75
- iGPU backend: Vulkan, reported as an integrated UMA device
- model: Qwen3.6-35B-A3B Q4_K_M with MTP tensors
- benchmark mode: no MTP, 2,048 context, q8_0/q4_0 KV, 128/128 batch,
  eight CPU workers, S=33 unless stated otherwise
- each number below is a single short characterization run, not a median or a
  production performance claim

The dual-backend build was kept separate from the production CUDA build:

```bash
cmake -S . -B build-hybrid-cuda-vulkan-sm75 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_VULKAN=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DLLAMA_BUILD_TESTS=ON
cmake --build build-hybrid-cuda-vulkan-sm75 -j 8 \
    --target llama-server llama-cli test-backend-ops
```

On this host the Vulkan headers, shader compiler, and tools were unpacked in a
temporary local directory because installing system packages required an
interactive administrator password. A reproducible machine should install its
distribution packages for Vulkan development and `glslc` instead. Access to
`/dev/dri/renderD128` is also required; the temporary ACL used here does not
survive a reboot. Membership of the distribution's `render`/`video` groups is
the preferable permanent setup.

## Backend capability check

`llama-cli --list-devices` exposed CUDA0, Vulkan0 (Renoir), and Vulkan1 (the
NVIDIA Vulkan device). RADV reported fp16 support but no bf16, integer dot, or
matrix-core path. The generated Vulkan backend support matrix included:

| Operation | Supported cases |
|---|---:|
| `MUL_MAT_ID` | 863 / 863 |
| `FLASH_ATTN_EXT` | 4,757 / 5,097 |
| `CONCAT` | 112 / 192 |
| `RMS_NORM` | 21 / 21 |
| `ROPE` | 448 / 448 |
| `SSM_CONV` | 45 / 45 |
| `SSM_SCAN` | 3 / 4 |

The wackMall-specific `MUL_MAT_ID_COLD` and `MOE_COLD` operations remain CPU
operations. This test moves normal model layers and their fixed hot experts; it
does not provide a Vulkan implementation of the custom cold-expert kernels.

## Result

The device order matters. With layer split mode, the first listed device owns
the first split. `Vulkan0,CUDA0` with tensor splits `1,40`, `2,39`, and `4,37`
therefore places the requested leading layers on Renoir and leaves the output
and remaining layers on CUDA.

| Placement | Decode tok/s | Change vs CUDA | Prompt tok/s | GTX peak MiB |
|---|---:|---:|---:|---:|
| CUDA baseline, S=33 | 39.497 | baseline | 27.879 | 5,024 |
| first 1 layer on Renoir | 32.725 | -17.1% | 14.714 | 4,592 |
| first 2 layers on Renoir | 30.985 | -21.5% | 14.690 | 4,550 |
| first 4 layers on Renoir | 27.744 | -29.8% | 14.687 | 4,472 |
| all layers on Renoir, S=33 | 10.015 | -74.6% | 13.999 | effectively zero |

Partial offload saves roughly 0.4--0.55 GiB of NVIDIA VRAM, but the serial iGPU
work and cross-backend boundary cost substantially more than the saved CUDA
work. Full Renoir execution is functional: its monitor reached about 491 MiB
dedicated VRAM, 4,829 MiB GTT, 99% busy, and about 79% mean busy while active.

An initial full-iGPU run exposed a real multi-backend bug: fixed `.hot` expert
tensors were allocated on the first registered discrete GPU (CUDA0), although
all model layers were assigned to Vulkan0. `llama-expert-tier.cpp` now chooses
the actual uniform layer device, including an integrated GPU. Mixed-device
models retain the old discrete-GPU fallback until hot storage becomes
per-layer/per-device.

## Attempt to use the full UMA/GTT budget

With all layers on Vulkan0, `LLAMA_EXPERT_S=auto`, and a 512 MiB reserve, the
autofit logic selected S=181 and allocated about 12.96 GiB of fixed experts.
The warm-up completed at approximately 11.03 decode tok/s and 6.05 prompt
tok/s, but the following measured request stopped making progress and was
manually cancelled. No AMDGPU or kernel OOM message was recorded in the final
15-minute kernel log window.

This is a stop result, not a successful benchmark. Advertising the full Vulkan
heap as cheaply usable expert memory is too aggressive on this UMA system: it
leaves insufficient practical headroom for shared system traffic, Vulkan
allocations, graph workspaces, and/or driver paging behavior. A future generic
iGPU autofit must use both an absolute cap and a conservative percentage of the
reported heap, and it should reject configurations whose first post-warm-up
request misses a progress deadline.

## Parameterized reproduction

The benchmark runner accepts optional backend controls without changing
existing cases:

```bash
DEVICE_LIST=Vulkan0,CUDA0 \
SPLIT_MODE=layer \
TENSOR_SPLIT=1,40 \
N_GPU_LAYERS=all \
./scripts/bench_hybrid.sh L0
```

Do not copy those split numbers to another model. Device names, layer count,
available memory, and the split must be discovered on each host. For a full
iGPU diagnostic use `DEVICE_LIST=Vulkan0`, `SPLIT_MODE=none`, and initially a
small explicit S. Increase S only after a successful repeated-request test.

## Decision

- Keep the CUDA-only GTX 1660 Ti configuration as the production path.
- Do not offload layers to Renoir for throughput; even one layer regressed
  decode by more than the project's stop threshold.
- Retain optional runner controls and the uniform-device allocation fix. They
  make the experiment reproducible on stronger APUs, discrete secondary GPUs,
  and systems with different memory ratios.
- Do not enable iGPU offload or iGPU autofit by default.
- Before revisiting on a newer APU, add a percentage/byte cap for UMA allocation
  and measure repeated requests, prompt processing, and the cross-backend copy
  boundary separately.
