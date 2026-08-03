# Router Lookahead Experiments

Date: 2026-08-03

This file is the append-only experiment record for routing-exact lookahead.
Results are recorded only after a reproducible run has completed.

## Environment discovery

### Repository

```text
path: /home/andi/src/llama-wackMall-hybrid
baseline: f337cf7d9e52f6986814c7923ea96b82223f376c
branch: codex/router-lookahead-prefetch
initial status: clean
```

The requested `/root/llama-wackMall-hybrid` path is for the future GTX 1080
host and does not exist on the current machine.

### Current GTX 1660 Ti host

```text
expected architecture: sm_75
current nvcc: CUDA 12.0
gcc-13 and g++-13: present
nvidia-smi on 2026-08-03: driver communication failed
```

Runtime GPU tests are blocked until the NVIDIA driver is available again. This
is an environment observation, not a model or code failure.

### Future GTX 1080 host

```text
expected architecture: sm_61
expected VRAM: 8 GiB
expected CUDA: 12.4
expected host compiler: GCC/G++ 13
expected repository: /root/llama-wackMall-hybrid
expected model: /root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf
```

These future-host values are supplied by the user and have not been locally
verified.

## Phase 0 source analysis

Status: complete.

Findings are recorded in `ROUTER_LOOKAHEAD_ANALYSIS.md` and
`ROUTER_LOOKAHEAD_DESIGN.md`. No productive prefetch code was added.

## Phase 1 planned matrix

Initial constraints:

```text
MTP: off
parallel sequences: 1
adaptation: off
warm slots: 0
temperature: 0
same seed per comparison
minimum output: 512 tokens
fresh server or CLI process for each serious comparison
```

Predictor matrix:

```text
points: post-attn, post-moe
norms: target, source diagnostic
distances: 1, then 2 only if distance 1 is measurable and useful
Top-M: 8, 12, 16, or the model expert-count clamp
prompts: multiple distinct public or user-provided external prompt files
```

For each trace run:

- compare output and token hashes with trace disabled,
- record prediction metrics per layer and overall,
- record trace overhead,
- record unavailable timing fields as unavailable,
- do not infer transfer readiness from recall alone.

## Phase 1 results

Status: not run.

| Host | Point | Norm | Distance | Top-M | Tokens | Hash match | Weighted cold coverage | Decode TPS | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| GTX 1660 Ti | pending | pending | 1 | pending | pending | pending | pending | pending | Driver unavailable |
| GTX 1080 | pending | pending | 1 | pending | pending | pending | pending | pending | Host available later |

No predictor point has won yet. No layer exclusion has been selected. No H2D
copy has been authorized.

## Phase 2 Oracle and schedule simulation

Status: not started by design.

Phase 2 must use actual routing records plus measured pinned-transfer and
per-layer timing data. Assumed PCIe bandwidth is not acceptable. Required
outputs include useful prefetch ratio, useful bytes ratio, hidden copy ratio,
theoretical CPU saving, theoretical net saving, simultaneous scratch demand,
and an ideal Oracle under the same constraints.

## Productive prefetch

Status: not implemented.

The implementation stops after Phase 1 unless the requested measurement and
Oracle gates are positive. MTP lookahead remains disabled by default even if a
later no-MTP prototype is approved.

## Stop conditions

Any experiment stops on CUDA hang, Xid, OOM, invariant failure, token/hash
regression in observational mode, unexplained VRAM growth, quality regression,
or a sustained throughput regression beyond the configured acceptance limit.
