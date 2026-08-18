#!/usr/bin/env python3
"""Emit a TurboLLM LoadProfile JSON for the GTX 1660 Ti or GTX 1080 stack."""

from __future__ import annotations

import argparse
import json
import sys


def profile(gpu: str, dflash: str) -> dict:
    common_sampling = {
        "temp": 0.8,
        "topP": 0.95,
        "topK": 40,
        "minP": 0.05,
        "repeatPenalty": 1.0,
        "presencePenalty": 0.0,
        "frequencyPenalty": 0.0,
        "stop": [],
    }
    gpu_block = {
        "splitMode": "layer",
        "tensorSplit": [],
        "mainGpu": -1,
        "tensorParallelSize": 1,
    }
    vllm = {
        "maxModelLen": 0,
        "gpuMemoryUtilization": 0.9,
        "maxNumSeqs": 0,
        "dtype": "auto",
        "kvCacheDtype": "auto",
        "enforceEager": False,
        "trustRemoteCode": False,
    }
    base = {
        "ngl": 99,
        "nCpuMoe": 0,
        "parallel": 1,
        "kvUnified": True,
        "kvTypeK": "turbo4_k",
        "kvTypeV": "turbo4_k",
        "flashAttn": "on",
        "kvOffload": True,
        "useMmproj": False,
        "mmprojGpu": True,
        "imageMaxTokens": 0,
        "cacheReuse": 16,
        "useJinja": True,
        "chatTemplateFile": "",
        "speculative": "off",
        "mtpHeadPath": "",
        "draftModelPath": dflash,
        "sampling": common_sampling,
        "contextOverflow": "shift",
        "nKeep": 0,
        "ropeScalingType": "none",
        "ropeFreqBase": 0,
        "ropeFreqScale": 0,
        "gpu": gpu_block,
        "vllm": vllm,
        "grammar": "",
    }
    if gpu == "1660":
        extra = [
            "--cpu-moe",
            "--spec-type", "draft-dflash",
            "--model-draft", dflash,
            "--spec-draft-n-max", "2",
            "--spec-draft-n-min", "1",
            "--draft-p-min", "0.75",
            "--n-gpu-layers-draft", "99",
            "--override-tensor", r"^blk[.]40[.]=CPU",
            "--cache-type-k-draft", "turbo4_k",
            "--cache-type-v-draft", "turbo4_k",
            "--cache-ram", "2048",
            "--cache-prompt",
            "--ctx-checkpoints", "4",
            "--kv-unified",
            "--reasoning-preserve",
            "--reasoning-budget", "1000",
            "--n-predict", "8192",
            "--backend-sampling",
        ]
        base.update(
            {
                "ctx": 24576,
                "threads": 8,
                "threadsBatch": 8,
                "draftMax": 2,
                "draftMin": 1,
                "batchSize": 64,
                "uBatchSize": 64,
                "extraArgs": extra,
            }
        )
        return base
    if gpu == "1080":
        extra = [
            "--cpu-moe",
            "--spec-type", "draft-dflash",
            "--model-draft", dflash,
            "--spec-draft-n-max", "8",
            "--spec-draft-n-min", "0",
            "--draft-p-min", "0.75",
            "--n-gpu-layers-draft", "99",
            "--override-tensor", r"^blk[.]40[.]=CPU",
            "--cache-type-k-draft", "turbo4_k",
            "--cache-type-v-draft", "turbo4_k",
            "--cache-ram", "1024",
            "--cache-prompt",
            "--ctx-checkpoints", "0",
            "--kv-unified",
            "--reasoning-preserve",
            "--reasoning-budget", "512",
            "--n-predict", "20000",
            "--load-mode", "none",
            "--backend-sampling",
        ]
        base.update(
            {
                "ctx": 32768,
                "threads": 4,
                "threadsBatch": 4,
                "draftMax": 8,
                "draftMin": 0,
                "batchSize": 128,
                "uBatchSize": 128,
                "extraArgs": extra,
            }
        )
        return base
    raise SystemExit(f"unknown gpu {gpu!r} (want 1660 or 1080)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gpu", choices=("1660", "1080"))
    p.add_argument("--dflash", required=True)
    args = p.parse_args()
    json.dump(profile(args.gpu, args.dflash), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
