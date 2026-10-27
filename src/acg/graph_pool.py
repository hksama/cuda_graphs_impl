from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .model import EncoderModel
from .types import ShapeKey


@dataclass
class CaptureStats:
    warmup_s: float
    capture_s: float
    mem_delta_mb: float


class CapturedShapeGraph:
    """CUDA graph for one (batch, seq_len) encoder forward."""

    def __init__(
        self,
        encoder: EncoderModel,
        shape: ShapeKey,
        warmup_iters: int = 5,
        mem_pool: Optional[object] = None,
    ):
        self.encoder = encoder
        self.shape = shape
        batch, seq = shape
        device = encoder.device

        self.static_input_ids = torch.zeros(batch, seq, dtype=torch.long, device=device)
        self.static_attention_mask = torch.zeros(batch, seq, dtype=torch.long, device=device)

        self._warmup_stream = torch.cuda.Stream()
        self._graph = torch.cuda.CUDAGraph()
        self._mem_pool = mem_pool if mem_pool is not None else torch.cuda.graph_pool_handle()
        self.static_output: torch.Tensor
        self.capture_stats: Optional[CaptureStats] = None

        self._capture(warmup_iters)

    def _capture(self, warmup_iters: int) -> None:
        from .memory import mem_delta_mb, mem_snapshot_mb

        mem_before = mem_snapshot_mb()
        t_w0 = time.perf_counter()
        with torch.cuda.stream(self._warmup_stream):
            for _ in range(warmup_iters):
                self.encoder.forward_last_hidden(
                    self.static_input_ids, self.static_attention_mask
                )
        self._warmup_stream.synchronize()
        warmup_s = time.perf_counter() - t_w0

        t_c0 = time.perf_counter()
        with torch.cuda.graph(self._graph, pool=self._mem_pool):
            self.static_output = self.encoder.forward_last_hidden(
                self.static_input_ids, self.static_attention_mask
            )
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - t_c0

        mem_after = mem_snapshot_mb()
        self.capture_stats = CaptureStats(
            warmup_s=warmup_s,
            capture_s=capture_s,
            mem_delta_mb=mem_delta_mb(mem_before, mem_after),
        )

    def replay(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Copy inputs, replay graph, return cloned output and end-to-end latency."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.static_input_ids.copy_(input_ids, non_blocking=True)
        self.static_attention_mask.copy_(attention_mask, non_blocking=True)
        self._graph.replay()
        out = self.static_output.clone()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0
