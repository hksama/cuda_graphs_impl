from __future__ import annotations

import torch


def mem_snapshot_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024**2),
        torch.cuda.memory_reserved() / (1024**2),
    )


def mem_delta_mb(before: tuple[float, float], after: tuple[float, float]) -> float:
    return after[0] - before[0]
