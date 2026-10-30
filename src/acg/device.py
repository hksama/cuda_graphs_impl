from __future__ import annotations

import subprocess
import sys

import torch


def require_cuda() -> torch.device:
    """Return cuda device or exit with actionable diagnostics."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    lines = [
        "CUDA is not available — this project requires an NVIDIA GPU and driver.",
        "",
        "Check in this same shell:",
        "  nvidia-smi",
        "",
        "If nvidia-smi fails:",
        "  • WSL2: install/update the Windows NVIDIA driver (WSL support), then reboot Windows.",
        "  • Confirm you are in WSL2 (not Docker/CI without GPU passthrough).",
        "  • In WSL: ls /dev/dxg  (should exist when GPU is wired up).",
        "",
        f"PyTorch {torch.__version__}, bundled CUDA {torch.version.cuda}",
    ]
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.stdout.strip():
            lines.append("")
            lines.append("nvidia-smi -L output:")
            lines.append(out.stdout.strip())
        elif out.stderr.strip():
            lines.append("")
            lines.append("nvidia-smi error:")
            lines.append(out.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        lines.append("")
        lines.append("nvidia-smi not found or timed out in this environment.")

    print("\n".join(lines), file=sys.stderr)
    sys.exit(1)
