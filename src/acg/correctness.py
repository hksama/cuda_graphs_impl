from __future__ import annotations

from typing import Any, Dict

import torch


def outputs_match(
    a: torch.Tensor,
    b: torch.Tensor,
    config: Dict[str, Any],
) -> bool:
    atol = float(config.get("correctness_atol", config.get("correctness_tolerance", 1e-5)))
    rtol = float(config.get("correctness_rtol", 1e-5 if atol < 1e-4 else 0.02))
    return bool(torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol))
