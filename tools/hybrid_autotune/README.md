# hybrid_autotune

Detect host CPU/RAM/GPU/VRAM, write a baseline `start.sh`, and optionally run a timed knob search.

## Anchors

| Host | File | VRAM | S | n_max | draft | prefill | batch |
|------|------|------|---|-------|-------|---------|-------|
| GTX 1660 Ti | `start1660.sh` | 6 GiB | 33 | 2 | q4_0 | 768 | 64 |
| GTX 1080 | `start1080.sh` | 8 GiB | 58 | 8 | turbo4_k | 1024 | 128 |

Baselines interpolate VRAM knobs between these two measured stacks and switch kernel flags by SM class (Pascal vs Turing+).

## Usage

```bash
# interactive menu
python3 tools/hybrid_autotune/autotune.py

# one-shot
python3 tools/hybrid_autotune/autotune.py detect
python3 tools/hybrid_autotune/autotune.py generate -y
python3 tools/hybrid_autotune/autotune.py optimize --mode quick   # ~10 min
python3 tools/hybrid_autotune/autotune.py optimize --mode deep    # ~60 min
```

Optional env:

- `HYBRID_MODEL` / `HYBRID_DFLASH` — override GGUF paths
- `HYBRID_AUTOTUNE_RESULTS` — results root (default `/root/gtx1080-hybrid-results` or `benchmark-results/`)
- `CUDA_VISIBLE_DEVICES` — GPU index

## What optimize searches

**quick (~10 min):** S ladder, n_max, prefill, optional draft KV A/B; short smoke + one validation.

**deep (~60 min):** wider S/n_max grid, p_min refine, 2-3x512 validation when time remains.

Metrics: sustained decode TPS (primary), e2e proxy, `spec_acceptance`. Winner knobs are written into `start.sh`.

Results: `.../autotune-<utc>/results.csv`, `winner.json`, `ANALYSIS.md`.

## Fixed launchers

- `start1660.sh` — do not overwrite (1660 Ti production reference)
- `start1080.sh` — do not overwrite (1080 production reference)
- `start.sh` — auto-generated / autotuned for the current machine
