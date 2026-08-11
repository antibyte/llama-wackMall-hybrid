#!/usr/bin/env python3
"""Hardware probe: CPU, RAM, GPU, VRAM, compute capability."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class Hardware:
    cpu_model: str
    logical_cpus: int
    physical_cores: int
    ram_gib: float
    gpu_name: str
    vram_mib: float
    vram_gib: float
    compute_cap: float  # e.g. 6.1, 7.5
    cuda_index: int
    nvidia_smi: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"CPU: {self.cpu_model}  (physical={self.physical_cores}, logical={self.logical_cpus})\n"
            f"RAM: {self.ram_gib:.1f} GiB\n"
            f"GPU[{self.cuda_index}]: {self.gpu_name}\n"
            f"VRAM: {self.vram_gib:.2f} GiB ({self.vram_mib:.0f} MiB)\n"
            f"Compute: sm_{int(self.compute_cap * 10):02d} ({self.compute_cap})"
        )


def _read_cpuinfo() -> tuple[str, int]:
    model = "unknown"
    phys: set[tuple[str, str]] = set()
    logical = 0
    try:
        text = open("/proc/cpuinfo", encoding="utf-8", errors="replace").read()
    except OSError:
        n = os.cpu_count() or 1
        return model, max(1, n // 2)

    cur_phys = ""
    cur_core = ""
    for line in text.splitlines():
        if line.startswith("model name") and model == "unknown":
            model = line.split(":", 1)[1].strip()
        elif line.startswith("processor"):
            logical += 1
        elif line.startswith("physical id"):
            cur_phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            cur_core = line.split(":", 1)[1].strip()
            if cur_phys != "" or cur_core != "":
                phys.add((cur_phys, cur_core))
        elif line.strip() == "":
            cur_phys, cur_core = "", ""

    physical = len(phys) if phys else max(1, (logical or (os.cpu_count() or 1)) // 2)
    if logical <= 0:
        logical = os.cpu_count() or 1
    return model, physical


def _read_ram_gib() -> float:
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def _parse_compute_cap(s: str) -> float:
    s = (s or "").strip()
    m = re.match(r"^(\d+)\.(\d+)$", s)
    if m:
        return float(f"{int(m.group(1))}.{int(m.group(2))}")
    # sometimes "6.1" with spaces
    try:
        return float(s)
    except ValueError:
        return 0.0


def detect_hardware(cuda_index: Optional[int] = None) -> Hardware:
    """Detect host hardware. Raises RuntimeError if no NVIDIA GPU is usable."""
    cpu_model, physical = _read_cpuinfo()
    logical = os.cpu_count() or physical
    ram_gib = _read_ram_gib()

    if cuda_index is None:
        env = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        try:
            cuda_index = int(env) if env not in ("", "-1") else 0
        except ValueError:
            cuda_index = 0

    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found; hybrid autotune requires an NVIDIA GPU")

    cmd = [
        "nvidia-smi",
        f"--id={cuda_index}",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"nvidia-smi failed: {e}") from e

    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {line!r}")

    gpu_name = parts[0]
    try:
        vram_mib = float(parts[1])
    except ValueError as e:
        raise RuntimeError(f"bad VRAM value from nvidia-smi: {parts[1]!r}") from e
    compute_cap = _parse_compute_cap(parts[2])
    if compute_cap <= 0:
        raise RuntimeError(f"could not parse compute capability: {parts[2]!r}")

    return Hardware(
        cpu_model=cpu_model,
        logical_cpus=int(logical),
        physical_cores=int(physical),
        ram_gib=float(ram_gib),
        gpu_name=gpu_name,
        vram_mib=vram_mib,
        vram_gib=vram_mib / 1024.0,
        compute_cap=compute_cap,
        cuda_index=int(cuda_index),
        nvidia_smi=True,
    )


def sm_tag(compute_cap: float) -> str:
    """Return e.g. '61' for 6.1, '75' for 7.5."""
    major = int(compute_cap)
    minor = int(round((compute_cap - major) * 10))
    return f"{major}{minor}"
