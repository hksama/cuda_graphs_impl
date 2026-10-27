#!/usr/bin/env python3
"""Verify eager vs CUDA graph replay for one shape."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acg.eager import timed_eager_forward
from acg.graph_pool import CapturedShapeGraph
from acg.model import EncoderModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--checks", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    encoder = EncoderModel("sentence-transformers/all-MiniLM-L6-v2", device)
    shape = (args.batch, args.seq)

    graph = CapturedShapeGraph(encoder, shape, warmup_iters=5)
    print(f"Capture stats: warmup={graph.capture_stats.warmup_s:.4f}s "
          f"capture={graph.capture_stats.capture_s:.4f}s "
          f"mem_delta={graph.capture_stats.mem_delta_mb:.2f}MB")

    prev_out = None
    for i in range(args.checks):
        ids, mask = encoder.make_random_batch(shape, seed=i)
        eager_out, eager_lat = timed_eager_forward(encoder, ids, mask)
        replay_out, replay_lat = graph.replay(ids, mask)
        diff = float((eager_out - replay_out).abs().max().item())
        assert diff < 1e-5, f"max abs diff {diff} too large"
        if prev_out is not None:
            alias = replay_out.data_ptr() == prev_out.data_ptr()
            assert not alias, "replay output aliases previous request buffer"
        prev_out = replay_out
        print(f"check {i}: eager={eager_lat*1000:.2f}ms replay={replay_lat*1000:.2f}ms diff={diff:.2e}")

    print("OK: correctness and output ownership verified.")


if __name__ == "__main__":
    main()
