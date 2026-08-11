#!/usr/bin/env python3
"""Quick/deep autotune using scripts/bench_hybrid.sh."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .generate import patch_start_sh_knobs, project_root_from, write_start_sh
from .hw import Hardware, detect_hardware
from .presets import LauncherConfig, apply_overrides, build_baseline, config_summary


@dataclass
class RunResult:
    name: str
    status: str  # OK | FAIL | OOM | SKIP | TIMEOUT
    pp: float = 0.0
    dec: float = 0.0
    e2e: float = 0.0
    acc: float = 0.0
    mal: float = 0.0
    s: str = ""
    knobs: dict[str, str] = field(default_factory=dict)
    dir: str = ""


def e2e_score(pp: float, dec: float) -> float:
    if pp <= 0 or dec <= 0:
        return 0.0
    return 3328.0 / (3200.0 / pp + 128.0 / dec)


def kill_llama_servers() -> None:
    subprocess.run(["pkill", "-x", "llama-server"], check=False, capture_output=True)
    time.sleep(1)


def results_root(project_root: Path) -> Path:
    env = os.environ.get("HYBRID_AUTOTUNE_RESULTS")
    if env:
        base = Path(env)
    elif Path("/root/gtx1080-hybrid-results").is_dir():
        base = Path("/root/gtx1080-hybrid-results")
    else:
        base = project_root / "benchmark-results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / f"autotune-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(shell_path: str, project_root: Path) -> str:
    s = shell_path
    s = s.replace("$PROJECT_ROOT", str(project_root))
    s = s.replace("${PROJECT_ROOT}", str(project_root))
    s = s.replace("$HOME", str(Path.home()))
    s = os.path.expanduser(s)
    return s


def cfg_to_bench_env(cfg: LauncherConfig, project_root: Path, port: int = 18085) -> dict[str, str]:
    v = cfg.values
    env = os.environ.copy()

    server = resolve_path(v.get("SERVER", ""), project_root)
    model = resolve_path(v.get("MODEL", ""), project_root)
    dflash = resolve_path(v.get("SPEC_DRAFT_MODEL", ""), project_root)
    profile = ""
    kind = v.get("PROFILE_KIND", "specialist")
    if kind == "general":
        profile = resolve_path(v.get("PROFILE_GENERAL", ""), project_root)
    else:
        profile = resolve_path(v.get("PROFILE_SPECIALIST", ""), project_root)
    if not profile or not Path(profile).is_file():
        for cand in [
            project_root / "1",
            project_root / "profiles/specialist-benchprompt.csv",
        ]:
            if cand.is_file():
                profile = str(cand)
                break

    draft_k = v.get("DRAFT_TYPE_K", "q4_0")
    env.update(
        {
            "SERVER": server,
            "MODEL": model,
            "PROFILE": profile,
            "PORT": str(port),
            "CONTEXT": v.get("CONTEXT", "32768"),
            "CPU_THREADS": v.get("THREADS", "4"),
            "DRAFT_THREADS": v.get("DRAFT_THREADS", v.get("THREADS", "4")),
            "CMOE_BATCH": v.get("CMOE_BATCH", "128"),
            "CMOE_UBATCH": v.get("CMOE_UBATCH", "128"),
            "CMOE_PREFILL_BATCH": v.get("CMOE_PREFILL_BATCH", "1024"),
            "CMOE_PREFILL_UBATCH": v.get("CMOE_PREFILL_UBATCH", v.get("CMOE_PREFILL_BATCH", "1024")),
            "CMOE_DECODE_BATCH": v.get("CMOE_DECODE_BATCH") or v.get("CMOE_BATCH", "128"),
            "CMOE_DECODE_UBATCH": v.get("CMOE_DECODE_UBATCH") or v.get("CMOE_UBATCH", "128"),
            "TARGET_TYPE_K": v.get("TARGET_TYPE_K", "turbo4_k"),
            "TARGET_TYPE_V": v.get("TARGET_TYPE_V", "turbo4_k"),
            "DRAFT_TYPE_K": draft_k,
            "DRAFT_TYPE_V": v.get("DRAFT_TYPE_V", draft_k),
            "SPEC_MODE": v.get("SPEC_MODE", "dflash"),
            "SPEC_DRAFT_MODEL": dflash,
            "SPEC_DRAFT_N_MAX": v.get("SPEC_DRAFT_N_MAX", "6"),
            "SPEC_DRAFT_N_MIN": v.get("SPEC_DRAFT_N_MIN", "0"),
            "SPEC_DRAFT_P_MIN": v.get("SPEC_DRAFT_P_MIN", "0.75"),
            "SPEC_DRAFT_BACKEND_SAMPLING": v.get("SPEC_DRAFT_BACKEND_SAMPLING", "1"),
            "DFLASH_COMBINED": v.get("DFLASH_COMBINED", "1"),
            "MTP_OVERRIDE": "0",
            "TARGET_BACKEND_SAMPLING": v.get("TARGET_BACKEND_SAMPLING", "1"),
            "DRAFT_BACKEND_SAMPLING": v.get("DRAFT_BACKEND_SAMPLING", "1"),
            "REASONING_BUDGET": v.get("REASONING_BUDGET", "512"),
            "LOAD_MODE": v.get("LOAD_MODE", "none"),
            "VRAM_RESERVE_MIB": v.get("LLAMA_EXPERT_VRAM_RESERVE_MIB", "512"),
            "STATIC_FIXED_S": v.get("LLAMA_EXPERT_S", "33"),
            "TURBO4_MTP_EXPERIMENTAL": v.get("LLAMA_TURBO4_MTP_EXPERIMENTAL", "1"),
            "TURBO4_V_EXPERIMENTAL": v.get("LLAMA_TURBO4_V_EXPERIMENTAL", "1"),
            "TURBO4_DFLASH_EXPERIMENTAL": v.get("LLAMA_TURBO4_DFLASH_EXPERIMENTAL", "1"),
            "TURBO4_DRAFT_EXPERIMENTAL": v.get("LLAMA_TURBO4_DRAFT_EXPERIMENTAL", "0"),
            "TURBO4_F16_PREFILL_MIN_BATCH": v.get("GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH", "2"),
            "TURBO4_FAST_F16_CONVERT": v.get("GGML_CUDA_TURBO4_FAST_F16_CONVERT", "0"),
            "TURBO4_WHT_SHUFFLE": v.get("GGML_CUDA_TURBO4_WHT_SHUFFLE", "0"),
            "SHARED_HOT_IDS": v.get("LLAMA_EXPERT_SHARED_HOT_IDS", "1"),
            "SKIP_SENTINEL": v.get("LLAMA_EXPERT_SKIP_SENTINEL", "1"),
            "CUDA_ASYNC_HOST_COPY": v.get("GGML_CUDA_ASYNC_HOST_COPY", "1"),
            "CPU_REUSE_ROWS": v.get("LLAMA_EXPERT_CPU_REUSE_ROWS", "0"),
            "CPU_MULTI_ROW": v.get("LLAMA_EXPERT_CPU_MULTI_ROW", "0"),
            "MOE_MULTI_FUSION": v.get("GGML_CUDA_MOE_MULTI_FUSION", "0"),
            "MOE_COMBINE_FUSION": v.get("GGML_CUDA_MOE_COMBINE_FUSION", "0"),
            "MMVQ_Q8_NCOLS3_ROWS": v.get("GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS", "0"),
            "MMVQ_Q6_K_NCOLS1_ROWS": v.get("GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS", "0"),
            "MMVQ_Q6_K_NCOLS3_ROWS": v.get("GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS", "0"),
            "CASES": "SC",
        }
    )
    if draft_k == "turbo4_k":
        env["TURBO4_DRAFT_EXPERIMENTAL"] = "1"
    return env


def run_bench(
    name: str,
    cfg: LauncherConfig,
    project_root: Path,
    result_dir: Path,
    *,
    n_predict: int = 128,
    prompt_repeat: int = 20,
    repeats: int = 1,
    warmup_tokens: int = 32,
    timeout_s: int = 600,
) -> RunResult:
    run_dir = result_dir / f"r-{name}"
    if (run_dir / "runs.csv").is_file():
        # allow resume
        return _parse_runs(name, run_dir, cfg)

    run_dir.mkdir(parents=True, exist_ok=True)
    env = cfg_to_bench_env(cfg, project_root)
    env["RESULTS_DIR"] = str(run_dir)
    env["REPEATS"] = str(repeats)
    env["N_PREDICT"] = str(n_predict)
    env["PROMPT_REPEAT"] = str(prompt_repeat)
    env["WARMUP_PROMPT_REPEAT"] = "4"
    env["WARMUP_TOKENS"] = str(warmup_tokens)
    env["COOLDOWN_SECONDS"] = "3"
    env["SKIP_SENTINEL"] = env.get("SKIP_SENTINEL", "1")

    # prompt
    prompt = result_dir / "prompt.txt"
    if not prompt.is_file():
        prompt.write_text(
            "Explain hybrid MoE expert offloading between GPU hot experts and CPU cold experts "
            "in three concise paragraphs, then list five optimization tradeoffs.\n",
            encoding="utf-8",
        )
    env["PROMPT_SOURCE"] = str(prompt)

    kill_llama_servers()
    stdout_p = result_dir / f"{name}.stdout"
    stderr_p = result_dir / f"{name}.stderr"
    bench = project_root / "scripts" / "bench_hybrid.sh"
    print(f"==== RUN {name} ====", flush=True)
    try:
        with open(stdout_p, "w", encoding="utf-8") as so, open(stderr_p, "w", encoding="utf-8") as se:
            proc = subprocess.run(
                ["bash", str(bench)],
                env=env,
                stdout=so,
                stderr=se,
                timeout=timeout_s,
                check=False,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        kill_llama_servers()
        return RunResult(name=name, status="TIMEOUT", knobs=_knobs(cfg), dir=str(run_dir))
    finally:
        kill_llama_servers()

    if rc != 0 or not (run_dir / "runs.csv").is_file():
        blob = ""
        for p in (stderr_p, stdout_p):
            try:
                blob += p.read_text(encoding="utf-8", errors="replace")[-4000:]
            except OSError:
                pass
        status = "OOM" if re_oom(blob) else "FAIL"
        print(f"{status} {name}", flush=True)
        return RunResult(name=name, status=status, knobs=_knobs(cfg), dir=str(run_dir))

    return _parse_runs(name, run_dir, cfg)


def re_oom(text: str) -> bool:
    t = text.lower()
    keys = ("out of memory", "cudamalloc failed", "failed to load draft", "exiting due to model")
    return any(k in t for k in keys)


def _knobs(cfg: LauncherConfig) -> dict[str, str]:
    keys = [
        "LLAMA_EXPERT_S",
        "SPEC_DRAFT_N_MAX",
        "SPEC_DRAFT_P_MIN",
        "DRAFT_TYPE_K",
        "DRAFT_TYPE_V",
        "CMOE_PREFILL_BATCH",
        "CMOE_BATCH",
        "THREADS",
        "DFLASH_COMBINED",
        "SPEC_MODE",
    ]
    return {k: cfg.values.get(k, "") for k in keys}


def _parse_runs(name: str, run_dir: Path, cfg: LauncherConfig) -> RunResult:
    rows = list(csv.DictReader(open(run_dir / "runs.csv", encoding="utf-8")))
    if not rows:
        return RunResult(name=name, status="FAIL", knobs=_knobs(cfg), dir=str(run_dir))
    # median across reps if multiple
    pps, decs, accs, mals = [], [], [], []
    for r in rows:
        try:
            pps.append(float(r["prompt_tps"]))
            decs.append(float(r["sustained_decode_tps"]))
            accs.append(float(r.get("spec_acceptance") or r.get("mtp_acceptance") or 0))
            mals.append(float(r.get("mean_accepted_length") or 0))
        except (KeyError, ValueError):
            continue
    if not pps:
        return RunResult(name=name, status="FAIL", knobs=_knobs(cfg), dir=str(run_dir))

    def med(xs: list[float]) -> float:
        xs = sorted(xs)
        return xs[len(xs) // 2]

    pp, dec, acc, mal = med(pps), med(decs), med(accs), med(mals)
    e2e = e2e_score(pp, dec)
    s = rows[-1].get("effective_fixed_s") or cfg.values.get("LLAMA_EXPERT_S", "")
    print(f"OK {name} pp={pp:.1f} dec={dec:.2f} e2e={e2e:.1f} acc={acc:.3f} s={s}", flush=True)
    return RunResult(
        name=name,
        status="OK",
        pp=pp,
        dec=dec,
        e2e=e2e,
        acc=acc,
        mal=mal,
        s=str(s),
        knobs=_knobs(cfg),
        dir=str(run_dir),
    )


def rank_key(r: RunResult) -> tuple:
    # prefer decode, then e2e, then acc
    ok = 1 if r.status == "OK" else 0
    return (ok, r.dec, r.e2e, r.acc)


class Budget:
    def __init__(self, minutes: float):
        self.deadline = time.time() + minutes * 60.0
        self.minutes = minutes

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    def ok(self, need_s: float = 60.0) -> bool:
        return self.remaining() >= need_s


def write_summary_csv(path: Path, results: list[RunResult]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "status",
                "pp",
                "dec",
                "e2e",
                "acc",
                "mal",
                "s",
                "n_max",
                "draft",
                "prefill",
                "p_min",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.name,
                    r.status,
                    f"{r.pp:.3f}" if r.pp else "",
                    f"{r.dec:.3f}" if r.dec else "",
                    f"{r.e2e:.3f}" if r.e2e else "",
                    f"{r.acc:.4f}" if r.acc else "",
                    f"{r.mal:.2f}" if r.mal else "",
                    r.s or r.knobs.get("LLAMA_EXPERT_S", ""),
                    r.knobs.get("SPEC_DRAFT_N_MAX", ""),
                    r.knobs.get("DRAFT_TYPE_K", ""),
                    r.knobs.get("CMOE_PREFILL_BATCH", ""),
                    r.knobs.get("SPEC_DRAFT_P_MIN", ""),
                ]
            )


def optimize(
    mode: str = "quick",
    budget_min: Optional[float] = None,
    apply: bool = True,
    project_root: Optional[Path] = None,
    base_cfg: Optional[LauncherConfig] = None,
    hw: Optional[Hardware] = None,
) -> dict[str, Any]:
    project_root = project_root or project_root_from()
    hw = hw or detect_hardware()
    cfg = base_cfg or build_baseline(hw, project_root)

    if budget_min is None:
        budget_min = 10.0 if mode == "quick" else 60.0
    budget = Budget(budget_min)
    out = results_root(project_root)
    lock = out / "autotune.lock"
    if lock.exists():
        # stale lock older than 2h is ignored
        age = time.time() - lock.stat().st_mtime
        if age < 7200:
            raise RuntimeError(f"autotune already running? lock={lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")

    results: list[RunResult] = []
    log = open(out / "orchestrator.log", "a", encoding="utf-8")

    def logp(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    try:
        logp(f"==== AUTOTUNE {mode} budget={budget_min}m {datetime.now(timezone.utc).isoformat()} ====")
        logp(hw.summary())
        logp("baseline:\n" + config_summary(cfg))

        # validate paths
        server = resolve_path(cfg.values["SERVER"], project_root)
        model = resolve_path(cfg.values["MODEL"], project_root)
        if not Path(server).is_file():
            raise RuntimeError(f"server binary missing: {server}")
        if not Path(model).is_file():
            raise RuntimeError(f"model missing: {model}")

        base_s = int(cfg.values.get("LLAMA_EXPERT_S", "33"))
        base_n = int(cfg.values.get("SPEC_DRAFT_N_MAX", "2"))
        base_pref = int(cfg.values.get("CMOE_PREFILL_BATCH", "768") or "768")
        vram = hw.vram_gib

        def trial(name: str, overrides: dict[str, str], n_predict: int, pr: int, reps: int = 1) -> Optional[RunResult]:
            if not budget.ok(90):
                logp(f"SKIP {name} (budget)")
                return None
            c = apply_overrides(cfg, overrides)
            # keep draft V match K
            if "DRAFT_TYPE_K" in overrides and "DRAFT_TYPE_V" not in overrides:
                c = apply_overrides(c, {"DRAFT_TYPE_V": overrides["DRAFT_TYPE_K"]})
            if c.values.get("DRAFT_TYPE_K") == "turbo4_k":
                c = apply_overrides(c, {"LLAMA_TURBO4_DRAFT_EXPERIMENTAL": "1"})
            r = run_bench(
                name,
                c,
                project_root,
                out,
                n_predict=n_predict,
                prompt_repeat=pr,
                repeats=reps,
                timeout_s=min(900, int(budget.remaining()) + 30),
            )
            results.append(r)
            write_summary_csv(out / "results.csv", results)
            return r

        # ---- smoke phase ----
        smoke_npred = 128 if mode == "quick" else 256
        smoke_pr = 20 if mode == "quick" else 40

        # baseline
        trial("base", {}, smoke_npred, smoke_pr)

        # S ladder
        s_vals = sorted({max(24, base_s + d) for d in (-8, -4, 0, 4, 8)})
        if mode == "deep":
            s_vals = sorted({max(24, base_s + d) for d in (-12, -8, -4, 0, 4, 8, 12)})
        for s in s_vals:
            if s == base_s:
                continue
            r = trial(f"s{s}", {"LLAMA_EXPERT_S": str(s)}, smoke_npred, smoke_pr)
            if r and r.status == "OOM" and s > base_s:
                logp(f"prune higher S after OOM at s={s}")
                break

        # n_max
        n_vals = [2, 4, 6, 8] if mode == "quick" else [2, 3, 4, 6, 8, 10]
        for n in n_vals:
            if n == base_n:
                continue
            r = trial(
                f"n{n}_s{base_s}",
                {"SPEC_DRAFT_N_MAX": str(n), "LLAMA_EXPERT_S": str(base_s)},
                smoke_npred,
                smoke_pr,
            )
            if r and r.status in ("OOM", "FAIL") and n > base_n:
                # still try one more lower? stop climbing
                if n >= base_n + 4:
                    break

        # prefill
        prefs = [768, 1024] if vram >= 6 else [768]
        for pref in prefs:
            if pref == base_pref:
                continue
            trial(
                f"p{pref}",
                {
                    "CMOE_PREFILL_BATCH": str(pref),
                    "CMOE_PREFILL_UBATCH": str(pref),
                },
                smoke_npred,
                smoke_pr,
            )

        # draft kv A/B if enough VRAM
        if vram >= 7.5 and budget.ok(120):
            cur = cfg.values.get("DRAFT_TYPE_K", "q4_0")
            other = "q4_0" if cur == "turbo4_k" else "turbo4_k"
            trial(
                f"draft_{other}",
                {
                    "DRAFT_TYPE_K": other,
                    "DRAFT_TYPE_V": other,
                    "LLAMA_TURBO4_DRAFT_EXPERIMENTAL": "1" if other == "turbo4_k" else "0",
                },
                smoke_npred,
                smoke_pr,
            )

        ok_smoke = [r for r in results if r.status == "OK"]
        ok_smoke.sort(key=rank_key, reverse=True)

        if mode == "deep" and ok_smoke and budget.ok(180):
            logp("==== REFINE top configs ====")
            top = ok_smoke[:3]
            for tcfg in top:
                s = int(tcfg.knobs.get("LLAMA_EXPERT_S", base_s))
                n = int(tcfg.knobs.get("SPEC_DRAFT_N_MAX", base_n))
                pref = tcfg.knobs.get("CMOE_PREFILL_BATCH", str(base_pref))
                draft = tcfg.knobs.get("DRAFT_TYPE_K", "q4_0")
                for pmin in ("0.65", "0.75", "0.85"):
                    if pmin == tcfg.knobs.get("SPEC_DRAFT_P_MIN", "0.75"):
                        continue
                    trial(
                        f"pm_s{s}_n{n}_t{pmin}",
                        {
                            "LLAMA_EXPERT_S": str(s),
                            "SPEC_DRAFT_N_MAX": str(n),
                            "CMOE_PREFILL_BATCH": pref,
                            "CMOE_PREFILL_UBATCH": pref,
                            "DRAFT_TYPE_K": draft,
                            "DRAFT_TYPE_V": draft,
                            "SPEC_DRAFT_P_MIN": pmin,
                        },
                        smoke_npred,
                        smoke_pr,
                    )
                    if not budget.ok(120):
                        break

        # re-rank after refine
        ok_all = [r for r in results if r.status == "OK"]
        ok_all.sort(key=rank_key, reverse=True)

        # validation
        winners_to_val = ok_all[: (1 if mode == "quick" else 2)]
        val_results: list[RunResult] = []
        if winners_to_val and budget.ok(180):
            logp("==== VALIDATE ====")
            n_pred = 256 if mode == "quick" else 512
            reps = 1 if mode == "quick" else (3 if budget.remaining() > 1200 else 2)
            pr = 40 if mode == "deep" else 20
            for wr in winners_to_val:
                r = trial(
                    f"val_{wr.name}",
                    wr.knobs,
                    n_pred,
                    pr,
                    reps=reps,
                )
                if r and r.status == "OK":
                    val_results.append(r)

        # pick winner: prefer validation ranking, else smoke
        pool = val_results if val_results else ok_all
        if not pool:
            raise RuntimeError("no successful bench runs; cannot pick winner")
        pool.sort(key=rank_key, reverse=True)
        winner = pool[0]
        logp(f"WINNER {winner.name} dec={winner.dec:.2f} e2e={winner.e2e:.1f} acc={winner.acc:.3f}")

        winner_path = out / "winner.json"
        winner_payload = {
            "name": winner.name,
            "status": winner.status,
            "pp": winner.pp,
            "dec": winner.dec,
            "e2e": winner.e2e,
            "acc": winner.acc,
            "mal": winner.mal,
            "knobs": winner.knobs,
            "mode": mode,
            "budget_min": budget_min,
            "results_dir": str(out),
            "hw": hw.to_dict(),
        }
        winner_path.write_text(json.dumps(winner_payload, indent=2), encoding="utf-8")

        analysis = out / "ANALYSIS.md"
        lines = [
            f"# Autotune {mode}\n\n",
            f"- budget: {budget_min} min\n",
            f"- remaining at end: {budget.remaining():.0f}s\n",
            f"- GPU: {hw.gpu_name} {hw.vram_gib:.2f} GiB sm={hw.compute_cap}\n",
            f"- winner: **{winner.name}** dec={winner.dec:.2f} e2e={winner.e2e:.1f} acc={winner.acc:.3f}\n\n",
            "## Knobs\n\n```\n",
        ]
        for k, v in winner.knobs.items():
            lines.append(f"{k}={v}\n")
        lines.append("```\n\n## Top OK\n\n")
        for r in sorted(ok_all, key=rank_key, reverse=True)[:15]:
            lines.append(
                f"- {r.name}: dec={r.dec:.2f} pp={r.pp:.1f} e2e={r.e2e:.1f} acc={r.acc:.3f} s={r.s}\n"
            )
        analysis.write_text("".join(lines), encoding="utf-8")

        if apply:
            start_sh = project_root / "start.sh"
            if not start_sh.is_file():
                # generate full baseline then patch
                write_start_sh(cfg, project_root, start_sh)
            patch_start_sh_knobs(
                start_sh,
                {
                    "LLAMA_EXPERT_S": winner.knobs.get("LLAMA_EXPERT_S", cfg.values["LLAMA_EXPERT_S"]),
                    "SPEC_DRAFT_N_MAX": winner.knobs.get("SPEC_DRAFT_N_MAX", cfg.values["SPEC_DRAFT_N_MAX"]),
                    "SPEC_DRAFT_P_MIN": winner.knobs.get("SPEC_DRAFT_P_MIN", "0.75"),
                    "DRAFT_TYPE_K": winner.knobs.get("DRAFT_TYPE_K", cfg.values["DRAFT_TYPE_K"]),
                    "DRAFT_TYPE_V": winner.knobs.get("DRAFT_TYPE_V", cfg.values["DRAFT_TYPE_V"]),
                    "CMOE_PREFILL_BATCH": winner.knobs.get(
                        "CMOE_PREFILL_BATCH", cfg.values["CMOE_PREFILL_BATCH"]
                    ),
                    "CMOE_PREFILL_UBATCH": winner.knobs.get(
                        "CMOE_PREFILL_BATCH", cfg.values["CMOE_PREFILL_UBATCH"]
                    ),
                    "LLAMA_TURBO4_DRAFT_EXPERIMENTAL": (
                        "1" if winner.knobs.get("DRAFT_TYPE_K") == "turbo4_k" else "0"
                    ),
                },
                comment=f"{mode} winner {winner.name} dec={winner.dec:.2f} e2e={winner.e2e:.1f} dir={out}",
            )
            logp(f"applied winner knobs to {start_sh}")

        write_summary_csv(out / "results.csv", results)
        return winner_payload
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass
        log.close()
        kill_llama_servers()
