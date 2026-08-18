# TurboLLM hybrid profiles

Out-of-the-box wiring for [TurboLLM](https://github.com/) against this tree.
Two measured stacks:

| Engine in TurboLLM | Source | GPU | Binary | S | prefill/decode | DFlash |
|---|---|---|---|---|---|---|
| Hybrid GTX 1660 Ti | `start1660.sh` | sm_75, 6 GiB | `build-main-sm75/bin/llama-server` | 28 | 1856 / 64 | n_max=2 p_min=0.75 |
| Hybrid GTX 1080 | `start1080.sh` | sm_61, 8 GiB | `build-turbo-opt-sm61/bin/llama-server` | 58 | 1024 / 128 | n_max=8 p_min=0.75 |

TurboLLM does not read `start1660.sh` / `start1080.sh`. The wrapper sources
`env/gtx1660.env` or `env/gtx1080.env` (expert tier, CMoE phase, kernels) and
`apply-profiles.sh` writes the matching LoadProfile CLI flags.

## One-time / every machine

```bash
# 1) build the GPU-native llama-server (sm75 or sm61)
# 2) put the Qwen3.6-35B-A3B GGUF + DFlash sidecar under ~/models/... or set HYBRID_MODEL / HYBRID_DFLASH
# 3) start TurboLLM and register both engines
./tools/turbollm/start.sh
```

On this host `start.sh` detects the GPU, activates the matching engine, and
applies both profiles so you can switch in **Engines** without re-running setup.

Then in the UI: **Models** -> MTP `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` -> **Load**.

Do not run `./start1660.sh` or `./start1080.sh` at the same time. One GPU, one
`llama-server`.

## Override

| Env | Meaning |
|---|---|
| `HYBRID_GPU` | `1660` / `1080` / `auto` |
| `HYBRID_MODEL` / `HYBRID_DFLASH` | GGUF paths |
| `HYBRID_LLAMA_SERVER_1660` / `_1080` | binary paths |
| `TURBOLLM_URL` | default `http://127.0.0.1:6996` |
| `TURBOLLM_HOME` | default `$HOME/src/TurboLLM` |
| `HYBRID_ALLOW_SHARED=1` | skip the second-server guard (debug only) |

```bash
HYBRID_GPU=1080 ./tools/turbollm/setup.sh     # force 1080 activation
./tools/turbollm/apply-profiles.sh --reload   # rewrite profiles and Load
```
