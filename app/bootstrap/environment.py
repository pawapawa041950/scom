"""Detect the host machine and choose a matching PyTorch build.

We pick the CUDA wheel index based on the installed NVIDIA *driver* version
(reported by nvidia-smi). PyTorch CUDA wheels bundle their own CUDA runtime, so
only the driver needs to be recent enough — a CUDA Toolkit install is not
required. With no usable NVIDIA GPU we fall back to the CPU wheels.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# Installed together from the CUDA (or cpu) index. torchaudio is required by
# ComfyUI (e.g. audio_vae.py), so it must come from the same index as torch —
# never from the PyPI default, which would pull a mismatched/CPU build.
TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]

# Signature of the package set above. Bumped implicitly whenever the list
# changes so an already-provisioned backend re-runs the torch step.
TORCH_PKGSET = ",".join(TORCH_PACKAGES)

# Ordered high -> low. Each entry: (cuda tag, minimum Windows driver version).
# Pick the highest CUDA whose minimum driver <= the installed driver.
CUDA_TABLE = [
    ("cu128", 570.00),
    ("cu126", 560.76),
    ("cu124", 551.61),
    ("cu121", 527.41),
    ("cu118", 452.39),
]

_TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"


@dataclass
class GpuInfo:
    has_nvidia: bool = False
    driver_version: Optional[float] = None
    name: str = ""
    vram_mb: int = 0


@dataclass
class TorchPlan:
    tag: str          # cu124 / cpu / ...
    index_url: str
    is_cuda: bool
    reason: str


def _run(cmd: list[str], timeout: float = 8.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def detect_gpu() -> GpuInfo:
    """Query nvidia-smi for driver version, name, and VRAM."""
    if shutil.which("nvidia-smi") is None:
        return GpuInfo()
    out = _run([
        "nvidia-smi",
        "--query-gpu=driver_version,name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return GpuInfo()
    # First GPU line only.
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return GpuInfo()
    try:
        driver = float(parts[0])
    except ValueError:
        driver = None
    try:
        vram = int(float(parts[2]))
    except ValueError:
        vram = 0
    return GpuInfo(has_nvidia=True, driver_version=driver, name=parts[1], vram_mb=vram)


def choose_torch(gpu: GpuInfo) -> TorchPlan:
    """Map detected GPU/driver onto a torch wheel index."""
    if gpu.has_nvidia and gpu.driver_version:
        for tag, min_driver in CUDA_TABLE:
            if gpu.driver_version >= min_driver:
                return TorchPlan(
                    tag=tag,
                    index_url=_TORCH_INDEX.format(tag=tag),
                    is_cuda=True,
                    reason=(
                        f"NVIDIA {gpu.name} / driver {gpu.driver_version} "
                        f"→ {tag}"
                    ),
                )
        return TorchPlan(
            tag="cpu", index_url=_TORCH_INDEX.format(tag="cpu"), is_cuda=False,
            reason=(
                f"NVIDIA driver {gpu.driver_version} が対応 CUDA に対して"
                "古いため CPU を使用"
            ),
        )
    return TorchPlan(
        tag="cpu", index_url=_TORCH_INDEX.format(tag="cpu"), is_cuda=False,
        reason="NVIDIA GPU 未検出のため CPU を使用（低速）",
    )


def recommended_weight_dtype(gpu: GpuInfo, plan: TorchPlan) -> str:
    """Pick a default UNet dtype from available VRAM."""
    if not plan.is_cuda:
        return "default"
    if gpu.vram_mb and gpu.vram_mb < 12000:
        return "fp8_e4m3fn"  # squeeze the ~4GB diffusion model into less VRAM
    return "default"
