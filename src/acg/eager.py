from __future__ import annotations

import time
from typing import Tuple

import torch

from .model import EncoderModel


@torch.inference_mode()
def eager_forward(
    encoder: EncoderModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return encoder.forward_last_hidden(input_ids, attention_mask)


def timed_eager_forward(
    encoder: EncoderModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eager_forward(encoder, input_ids, attention_mask)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0
