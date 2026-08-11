#!/usr/bin/env python3
"""Anchors (GTX 1660 Ti / GTX 1080) and VRAM/SM-based baseline presets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .hw import Hardware, sm_tag


# Measured production anchors (start1660.sh / start1080.sh)
ANCHOR_1660 = {
    "vram_gib": 6.0,
    "sm": 7.5,
    "S": 33,
    "n_max": 2,
    "prefill": 768,
    "batch": 64,
    "draft_k": "q4_0",
    "draft_v": "q4_0",
    "load": "mmap",
    "ctx": 24576,
    "vram_reserve": 400,
    "threads": 8,
    "multi_row": "0",
    "reuse_rows": "0",
    "moe_multi_fusion": "1",
    "moe_combine_fusion": "1",
    "mmvq_q8_ncols3": "4",
    "mmvq_q6_ncols1": "2",
    "mmvq_q6_ncols3": "4",
    "concat_block": "128",
    "concat_flat": "1",
    "async_h2d": "1",
    "dedup_dst": "1",
    "reasoning_budget": "1024",
    "ctx_checkpoints": "4",
}

ANCHOR_1080 = {
    "vram_gib": 8.0,
    "sm": 6.1,
    "S": 58,
    "n_max": 8,
    "prefill": 1024,
    "batch": 128,
    "draft_k": "turbo4_k",
    "draft_v": "turbo4_k",
    "load": "none",
    "ctx": 32768,
    "vram_reserve": 512,
    "threads": 5,
    "multi_row": "1",
    "reuse_rows": "1",
    "moe_multi_fusion": "0",
    "moe_combine_fusion": "0",
    "mmvq_q8_ncols3": "0",
    "mmvq_q6_ncols1": "0",
    "mmvq_q6_ncols3": "0",
    "concat_block": "0",
    "concat_flat": "0",
    "async_h2d": "1",
    "dedup_dst": "1",
    "reasoning_budget": "512",
    "ctx_checkpoints": "0",
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def vram_t(vram_gib: float) -> float:
    """0 at 6 GiB (1660), 1 at 8 GiB (1080), soft extrapolate to 1.5."""
    return clamp((vram_gib - 6.0) / (8.0 - 6.0), 0.0, 1.5)


@dataclass
class LauncherConfig:
    """Flat KEY -> value map for start.sh assignments (string values, no quotes)."""

    values: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def find_server(project_root: Path, compute_cap: float) -> Optional[Path]:
    tag = sm_tag(compute_cap)
    candidates: list[Path] = []
    # Prefer opt / turbo builds matching SM
    patterns = [
        f"build-turbo-opt-sm{tag}/bin/llama-server",
        f"build-main-sm{tag}/bin/llama-server",
        f"build-gtx1080/bin/llama-server" if tag == "61" else "",
        f"build-turbo-f16only-sm{tag}/bin/llama-server",
        f"build-*/bin/llama-server",
    ]
    for rel in patterns:
        if not rel:
            continue
        if "*" in rel:
            candidates.extend(sorted(project_root.glob(rel)))
        else:
            candidates.append(project_root / rel)
    # Prefer executable
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    for p in candidates:
        if p.is_file():
            return p
    return None


def find_model(project_root: Path) -> Optional[Path]:
    env = os.environ.get("HYBRID_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    home = Path.home()
    candidates = [
        Path("/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf"),
        home / "models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        home / "models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UDT-Q4_K_XL_MTP.gguf",
    ]
    models_dir = Path("/root/atomic-nextn-good/models")
    if models_dir.is_dir():
        candidates.extend(sorted(models_dir.glob("*MTP*.gguf")))
    return _first_existing(candidates)


def find_dflash(project_root: Path) -> Optional[Path]:
    env = os.environ.get("HYBRID_DFLASH")
    if env and Path(env).is_file():
        return Path(env)
    home = Path.home()
    candidates = [
        Path("/root/atomic-nextn-good/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"),
        home / "models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf",
    ]
    models_dir = Path("/root/atomic-nextn-good/models")
    if models_dir.is_dir():
        candidates.extend(sorted(models_dir.glob("*DFlash*.gguf")))
    return _first_existing(candidates)


def find_profiles(project_root: Path) -> tuple[str, str, str]:
    """Return (profile_kind, general_path, specialist_path) as shell strings."""
    general = project_root / "benchmark-results/profile-corpus-train8-512-20260802T124500Z/general-profile.csv"
    specialist = project_root / "profiles/specialist-benchprompt.csv"
    local1 = project_root / "1"
    g = str(general) if general.is_file() else ""
    s = str(specialist) if specialist.is_file() else ""
    if local1.is_file():
        if not s:
            s = str(local1)
        if not g:
            g = str(local1)
    if s:
        kind = "specialist"
    elif g:
        kind = "general"
    else:
        kind = "specialist"
    # Prefer $PROJECT_ROOT-relative when under project
    def rel(p: str) -> str:
        if not p:
            return ""
        pp = Path(p)
        try:
            return f"$PROJECT_ROOT/{pp.relative_to(project_root)}"
        except ValueError:
            return p

    return kind, rel(g) or f"$PROJECT_ROOT/benchmark-results/profile-corpus-train8-512-20260802T124500Z/general-profile.csv", rel(s) or f"$PROJECT_ROOT/1"


def build_baseline(hw: Hardware, project_root: Path) -> LauncherConfig:
    t = vram_t(hw.vram_gib)
    sm = hw.compute_cap
    pascal = sm < 7.0

    # Continuous knobs between 1660 and 1080
    S = int(round(ANCHOR_1660["S"] + t * (ANCHOR_1080["S"] - ANCHOR_1660["S"])))
    S = int(clamp(S, 24, 80))

    if t < 0.25:
        n_max = 2
    elif t < 0.5:
        n_max = 4
    elif t < 0.85:
        n_max = 6
    else:
        n_max = 8

    prefill = 768 if t < 0.5 else 1024
    batch = 64 if t < 0.4 else 128
    ctx = 24576 if hw.vram_gib <= 6.5 else 32768
    vram_reserve = int(round(400 + t * 112))
    threads = max(2, min(hw.physical_cores if hw.physical_cores <= 8 else 8, 8))
    # Prefer measured physical-core count for small CPUs (1080 host used ~4-5)
    if hw.physical_cores <= 6:
        threads = hw.physical_cores

    if pascal:
        a = ANCHOR_1080
        draft_k = "turbo4_k" if hw.vram_gib >= 7.5 else "q4_0"
        draft_v = draft_k
        load = "none"
        draft_exp = "1" if draft_k == "turbo4_k" else "0"
        gpu_arch = f"{int(sm)}.{int(round((sm - int(sm)) * 10))}"
    else:
        a = ANCHOR_1660
        # Turing+: q4 draft default; turbo4 draft only if lots of VRAM
        draft_k = "turbo4_k" if hw.vram_gib >= 10 else "q4_0"
        draft_v = draft_k
        load = "mmap"
        draft_exp = "1" if draft_k == "turbo4_k" else "0"
        gpu_arch = "auto"

    server = find_server(project_root, sm)
    model = find_model(project_root)
    dflash = find_dflash(project_root)
    kind, prof_g, prof_s = find_profiles(project_root)

    server_s = (
        f"$PROJECT_ROOT/{server.relative_to(project_root)}"
        if server and str(server).startswith(str(project_root))
        else (str(server) if server else f"$PROJECT_ROOT/build-main-sm{sm_tag(sm)}/bin/llama-server")
    )
    model_s = str(model) if model else "$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    dflash_s = str(dflash) if dflash else "$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"

    alias = f"qwen3.6-35b-a3b-hybrid-autotune"
    if "1080" in hw.gpu_name or pascal:
        alias = "qwen3.6-35b-a3b-hybrid-gtx1080"
    elif "1660" in hw.gpu_name:
        alias = "qwen3.6-35b-a3b-hybrid"

    values: dict[str, str] = {
        "SERVER": server_s,
        "MODEL": model_s,
        "SPEC_MODE": "dflash",
        "SPEC_DRAFT_MODEL": dflash_s,
        "DFLASH_TARGET_TENSOR_OVERRIDE": "^blk[.]40[.]=CPU",
        "PROFILE_GENERAL": prof_g,
        "PROFILE_SPECIALIST": prof_s,
        "PROFILE_KIND": kind,
        "PLACEMENT": "",
        "HOST": "0.0.0.0",
        "PORT": "8080",
        "CORS_ORIGINS": "*",
        "API_KEY": "",
        "API_KEY_FILE": "",
        "MODEL_ALIAS": alias,
        "CUDA_VISIBLE_DEVICES_VALUE": str(hw.cuda_index),
        "N_PARALLEL": "1",
        "CPU_MOE": "1",
        "MMPROJ_AUTO": "0",
        "CONTEXT": str(ctx),
        "N_PREDICT": "20000" if pascal else "8192",
        "TARGET_TYPE_K": "turbo4_k",
        "TARGET_TYPE_V": "turbo4_k",
        "DRAFT_TYPE_K": draft_k,
        "DRAFT_TYPE_V": draft_v,
        "LLAMA_KV_Q4_SCALE": "legacy",
        "LLAMA_TURBO4_V_EXPERIMENTAL": "1",
        "LLAMA_TURBO4_MTP_EXPERIMENTAL": "1",
        "LLAMA_TURBO4_DFLASH_EXPERIMENTAL": "1",
        "LLAMA_TURBO4_DRAFT_EXPERIMENTAL": draft_exp,
        "LLAMA_TURBO4_Q8_FALLBACK_LAYERS": "",
        "GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH": "2",
        "GGML_CUDA_TURBO4_FAST_F16_CONVERT": "0",
        "GGML_CUDA_TURBO4_WHT_SHUFFLE": "0",
        "MTP_N": "2",
        "LLAMA_MTP_REQUANTIZE_OUTPUT": "none",
        "LLAMA_MTP_HEAD_TRACE": "0",
        "SPEC_DRAFT_N_MAX": str(n_max),
        "SPEC_DRAFT_N_MIN": "0",
        "SPEC_DRAFT_P_MIN": "0.75",
        "SPEC_DRAFT_BACKEND_SAMPLING": "1",
        "DFLASH_COMBINED": "1",
        "DRAFT_NGL": "all",
        "REASONING": "auto",
        "REASONING_BUDGET": str(a["reasoning_budget"]),
        "REASONING_PRESERVE": "1",
        "LOAD_MODE": load,
        "OFFLINE": "1",
        "JINJA": "1",
        "FLASH_ATTN": "on",
        "KV_OFFLOAD": "1",
        "THREADS": str(threads),
        "THREADS_BATCH": str(threads),
        "DRAFT_THREADS": str(threads),
        "DRAFT_THREADS_BATCH": str(threads),
        "THREADS_HTTP": "",
        "TARGET_BACKEND_SAMPLING": "1",
        "DRAFT_BACKEND_SAMPLING": "1",
        "GPU_ARCH": gpu_arch,
        "CMOE_BATCH": str(batch),
        "CMOE_UBATCH": str(batch),
        "CMOE_PREFILL_BATCH": str(prefill),
        "CMOE_PREFILL_UBATCH": str(prefill),
        "CMOE_DECODE_BATCH": str(batch) if pascal else "",
        "CMOE_DECODE_UBATCH": str(batch) if pascal else "",
        "CTX_CHECKPOINTS": str(a["ctx_checkpoints"]),
        "CACHE_RAM": "1024",
        "CACHE_PROMPT": "1",
        "CACHE_REUSE": "16",
        "KV_UNIFIED": "1",
        "CACHE_IDLE_SLOTS": "1",
        "LLAMA_EXPERT_S": str(S),
        "LLAMA_EXPERT_TMAX": "32",
        "LLAMA_EXPERT_STATS": "0",
        "LLAMA_EXPERT_ADAPT": "0",
        "LLAMA_EXPERT_DECAY": "1.0",
        "LLAMA_EXPERT_USAGE": "0",
        "LLAMA_EXPERT_USAGE_MODE": "session",
        "LLAMA_EXPERT_USAGE_CHECKPOINT": "request",
        "LLAMA_EXPERT_ADAPT_INTERVAL": "request",
        "LLAMA_EXPERT_ADAPT_CUDA_GRAPHS": "0",
        "LLAMA_EXPERT_STATS_JSON": "0",
        "LLAMA_EXPERT_TIMING": "0",
        "LLAMA_EXPERT_CPU_CHUNK": "64",
        "LLAMA_EXPERT_CPU_ACT_PARALLEL": "0",
        "LLAMA_EXPERT_CPU_ASYNC": "0",
        "LLAMA_EXPERT_CPU_DOWN_PREFETCH": "0",
        "LLAMA_EXPERT_CPU_REUSE_ROWS": str(a["reuse_rows"]),
        "LLAMA_EXPERT_CPU_MULTI_ROW": str(a["multi_row"]),
        "LLAMA_EXPERT_CPU_FUSED_GATE_UP": "0",
        "LLAMA_EXPERT_WARM_SLOTS": "0",
        "LLAMA_EXPERT_WARM_AUTO_MAX": "8",
        "LLAMA_EXPERT_WARM_POLICY": "lru",
        "LLAMA_EXPERT_WARM_RESET": "request",
        "LLAMA_EXPERT_WARM_ADMISSION": "immediate",
        "LLAMA_EXPERT_WARM_ADMISSION_WINDOW": "8",
        "LLAMA_EXPERT_WARM_PREFETCH": "0",
        "LLAMA_EXPERT_PREFETCH_STREAMS": "1",
        "LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT": "2",
        "LLAMA_EXPERT_VRAM_RESERVE_MIB": str(vram_reserve),
        "LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL": "0",
        "LLAMA_EXPERT_STATIC_NO_SYNC": "1",
        "LLAMA_EXPERT_LOOKAHEAD": "0",
        "LLAMA_EXPERT_LOOKAHEAD_TRACE": "0",
        "LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON": "0",
        "LLAMA_EXPERT_LOOKAHEAD_DISTANCE": "1",
        "LLAMA_EXPERT_LOOKAHEAD_TOP_M": "12",
        "LLAMA_EXPERT_LOOKAHEAD_POINT": "post-moe",
        "LLAMA_EXPERT_LOOKAHEAD_NORM": "target",
        "LLAMA_EXPERT_LOOKAHEAD_PREFETCH": "0",
        "LLAMA_EXPERT_LOOKAHEAD_STREAMS": "1",
        "LLAMA_EXPERT_LOOKAHEAD_MAX_INFLIGHT": "2",
        "LLAMA_EXPERT_LOOKAHEAD_LAYER_MIN": "0",
        "LLAMA_EXPERT_LOOKAHEAD_LAYER_MAX": "-1",
        "LLAMA_EXPERT_LOOKAHEAD_MTP_EXPERIMENTAL": "0",
        "GGML_CUDA_EXPERT_BRIDGE_LAYERS": "",
        "GGML_CUDA_EXPERT_BRIDGE_K": "2",
        "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE": "0",
        "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_LAYERS": "",
        "GGML_CUDA_EXPERT_BRIDGE_RECURRENCE_K": "1",
        "GGML_CUDA_EXPERT_BRIDGE_CONSUME": "0",
        "GGML_CUDA_EXPERT_BRIDGE_VERIFY": "0",
        "GGML_CUDA_EXPERT_BRIDGE_JSON": "0",
        "GGML_CUDA_EXPERT_BRIDGE_TELEMETRY": "0",
        "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT": "0",
        "GGML_CUDA_EXPERT_BRIDGE_DEVICE_QUANT_RECURRENCE": "0",
        "GGML_CUDA_EXPERT_BRIDGE_HIT_ONLY": "0",
        "GGML_CUDA_EXPERT_BRIDGE_CACHE_LAYERS": "",
        "GGML_CUDA_EXPERT_BRIDGE_CACHE_SLOTS": "2",
        "LLAMA_EXPERT_SHARED_HOT_IDS": "1",
        "LLAMA_EXPERT_SKIP_SENTINEL": "1",
        "GGML_CUDA_MOE_MULTI_FUSION": str(a["moe_multi_fusion"]),
        "GGML_CUDA_MOE_COMBINE_FUSION": str(a["moe_combine_fusion"]),
        "GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS": str(a["mmvq_q8_ncols3"]),
        "GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS": "0",
        "GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS": "0",
        "GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS": str(a["mmvq_q6_ncols1"]),
        "GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARPS": "0",
        "GGML_CUDA_MMVQ_Q6_K_NCOLS1_WARP_ROWS": "0",
        "GGML_CUDA_MMVQ_Q6_K_NCOLS1_REUSE_Y": "0",
        "GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS": str(a["mmvq_q6_ncols3"]),
        "GGML_CUDA_MMVQ_MOE_FUSED_ROWS": "0",
        "GGML_CUDA_MMVQ_MOE_PLAIN_ROWS": "0",
        "GGML_CUDA_CONCAT_NONCONT_BLOCK_SIZE": str(a["concat_block"]),
        "GGML_CUDA_CONCAT_NONCONT_FLAT_DIM0": str(a["concat_flat"]),
        "GGML_CUDA_ASYNC_HOST_COPY": str(a["async_h2d"]),
        "GGML_SCHED_ASYNC_D2H_COPY": "0",
        "GGML_SCHED_DEDUP_DST_SYNC": str(a["dedup_dst"]),
    }

    notes = [
        f"GPU={hw.gpu_name} VRAM={hw.vram_gib:.2f}GiB sm={sm} t={t:.2f}",
        f"S={S} n_max={n_max} prefill={prefill} batch={batch} draft={draft_k} load={load}",
        f"threads={threads} ctx={ctx} pascal={pascal}",
        f"server={server_s}",
        "baseline from 1660/1080 anchors; run optimize to refine",
    ]
    if not server:
        notes.append("WARN: no llama-server binary found for this SM")
    if not model:
        notes.append("WARN: model path not found; set HYBRID_MODEL")
    if not dflash:
        notes.append("WARN: DFlash path not found; set HYBRID_DFLASH")

    return LauncherConfig(
        values=values,
        meta={
            "t": t,
            "S": S,
            "n_max": n_max,
            "prefill": prefill,
            "batch": batch,
            "draft_k": draft_k,
            "pascal": pascal,
            "notes": notes,
            "hw": hw.to_dict(),
            "server_resolved": str(server) if server else "",
            "model_resolved": str(model) if model else "",
            "dflash_resolved": str(dflash) if dflash else "",
        },
    )


def apply_overrides(cfg: LauncherConfig, overrides: dict[str, str]) -> LauncherConfig:
    v = dict(cfg.values)
    v.update(overrides)
    meta = dict(cfg.meta)
    meta["overrides"] = overrides
    return LauncherConfig(values=v, meta=meta)


def config_summary(cfg: LauncherConfig) -> str:
    keys = [
        "SERVER",
        "MODEL",
        "SPEC_DRAFT_MODEL",
        "LLAMA_EXPERT_S",
        "SPEC_DRAFT_N_MAX",
        "SPEC_DRAFT_P_MIN",
        "DRAFT_TYPE_K",
        "CMOE_PREFILL_BATCH",
        "CMOE_BATCH",
        "THREADS",
        "CONTEXT",
        "LOAD_MODE",
        "GPU_ARCH",
        "LLAMA_EXPERT_CPU_MULTI_ROW",
        "GGML_CUDA_MOE_MULTI_FUSION",
    ]
    lines = [f"  {k}={cfg.values.get(k, '')}" for k in keys]
    for n in cfg.meta.get("notes", []):
        lines.append(f"  # {n}")
    return "\n".join(lines)
