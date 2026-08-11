#!/usr/bin/env python3
"""CLI: detect hardware, generate start.sh, run quick/deep optimize."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python3 tools/hybrid_autotune/autotune.py` without install
_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.hybrid_autotune.generate import project_root_from, write_start_sh  # noqa: E402
from tools.hybrid_autotune.hw import detect_hardware  # noqa: E402
from tools.hybrid_autotune.optimize import optimize  # noqa: E402
from tools.hybrid_autotune.presets import build_baseline, config_summary  # noqa: E402


def cmd_detect(_: argparse.Namespace) -> int:
    hw = detect_hardware()
    print(hw.summary())
    print(json.dumps(hw.to_dict(), indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root) if args.project_root else project_root_from()
    hw = detect_hardware()
    print("=== Hardware ===")
    print(hw.summary())
    cfg = build_baseline(hw, project_root)
    print("\n=== Proposed baseline ===")
    print(config_summary(cfg))
    out = Path(args.out) if args.out else project_root / "start.sh"
    if out.name in ("start1660.sh", "start1080.sh") and not args.force_ref:
        print(f"refusing to overwrite fixed reference {out}; use --force-ref if intentional", file=sys.stderr)
        return 2
    if not args.yes and not args.dry_run:
        ans = input(f"Write {out}? [Y/n] ").strip().lower()
        if ans in ("n", "no"):
            print("aborted")
            return 1
    path = write_start_sh(cfg, project_root, out, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"wrote {path}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root) if args.project_root else project_root_from()
    hw = detect_hardware()
    cfg = build_baseline(hw, project_root)
    # ensure baseline start.sh exists before optimize apply
    start = project_root / "start.sh"
    if not start.is_file() and args.apply:
        write_start_sh(cfg, project_root, start)
        print(f"created baseline {start}")
    winner = optimize(
        mode=args.mode,
        budget_min=args.budget_min,
        apply=args.apply,
        project_root=project_root,
        base_cfg=cfg,
        hw=hw,
    )
    print("\n=== WINNER ===")
    print(json.dumps(winner, indent=2))
    return 0


def cmd_interactive(_: argparse.Namespace) -> int:
    project_root = project_root_from()
    while True:
        print(
            """
hybrid_autotune
  [1] Detect hardware
  [2] Generate start.sh (baseline from 1660/1080 anchors)
  [3] Optimize quick (~10 min)
  [4] Optimize deep (~60 min)
  [5] Show last winner (if any)
  [q] Quit
"""
        )
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return 0
        if choice == "1":
            print(detect_hardware().summary())
        elif choice == "2":
            hw = detect_hardware()
            print(hw.summary())
            cfg = build_baseline(hw, project_root)
            print(config_summary(cfg))
            ans = input("Write start.sh? [Y/n] ").strip().lower()
            if ans not in ("n", "no"):
                p = write_start_sh(cfg, project_root, project_root / "start.sh")
                print(f"wrote {p}")
                opt = input("Start optimize now? [s]kip / [q]uick / [d]eep: ").strip().lower()
                if opt in ("q", "quick"):
                    print(json.dumps(optimize("quick", apply=True, project_root=project_root, base_cfg=cfg, hw=hw), indent=2))
                elif opt in ("d", "deep"):
                    print(json.dumps(optimize("deep", apply=True, project_root=project_root, base_cfg=cfg, hw=hw), indent=2))
        elif choice == "3":
            print(json.dumps(optimize("quick", apply=True, project_root=project_root), indent=2))
        elif choice == "4":
            print(json.dumps(optimize("deep", apply=True, project_root=project_root), indent=2))
        elif choice == "5":
            winners = sorted(
                Path("/root/gtx1080-hybrid-results").glob("autotune-*/winner.json")
                if Path("/root/gtx1080-hybrid-results").is_dir()
                else (project_root / "benchmark-results").glob("autotune-*/winner.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not winners:
                print("no winner.json found")
            else:
                print(winners[0].read_text(encoding="utf-8"))
        else:
            print("unknown choice")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hybrid_autotune",
        description="Detect host hardware, generate start.sh, and autotune hybrid knobs.",
    )
    p.add_argument("--project-root", default=None, help="repo root (default: auto)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("detect", help="print CPU/RAM/GPU/VRAM")

    g = sub.add_parser("generate", help="write baseline start.sh")
    g.add_argument("--out", default=None, help="output path (default: <root>/start.sh)")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("-y", "--yes", action="store_true", help="overwrite without prompt")
    g.add_argument("--force-ref", action="store_true", help="allow writing start1660/1080")

    o = sub.add_parser("optimize", help="run quick or deep bench search")
    o.add_argument("--mode", choices=("quick", "deep"), default="quick")
    o.add_argument("--budget-min", type=float, default=None, help="wall-clock minutes")
    o.add_argument("--apply", action=argparse.BooleanOptionalAction, default=True)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return cmd_interactive(argparse.Namespace())
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "detect":
        return cmd_detect(args)
    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "optimize":
        return cmd_optimize(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
