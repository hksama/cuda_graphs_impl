from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .eager import timed_eager_forward
from .graph_pool import CapturedShapeGraph
from .memory import mem_snapshot_mb
from .model import EncoderModel
from .types import ShapeKey, all_shape_keys


def _p50(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def calibrate_device(
    encoder: EncoderModel,
    batch_sizes: List[int],
    seq_lengths: List[int],
    warmup_iters: int = 5,
    timing_iters: int = 20,
) -> Dict[str, Any]:
    """Measure per-shape eager, replay (incl. copy), and capture costs."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    shapes: Dict[str, Any] = {}

    for key in all_shape_keys(batch_sizes, seq_lengths):
        key_str = f"{key[0]}_{key[1]}"
        ids, mask = encoder.make_random_batch(key, seed=hash(key) % (2**31))

        eager_times: List[float] = []
        for i in range(timing_iters):
            ids_i, mask_i = encoder.make_random_batch(key, seed=i)
            _, lat = timed_eager_forward(encoder, ids_i, mask_i)
            eager_times.append(lat)

        graph = CapturedShapeGraph(encoder, key, warmup_iters=warmup_iters)
        replay_times: List[float] = []
        for i in range(timing_iters):
            ids_i, mask_i = encoder.make_random_batch(key, seed=1000 + i)
            _, lat = graph.replay(ids_i, mask_i)
            replay_times.append(lat)

        stats = graph.capture_stats
        shapes[key_str] = {
            "batch": key[0],
            "seq": key[1],
            "eager_latency_p50_s": _p50(eager_times),
            "replay_latency_p50_s": _p50(replay_times),
            "capture_latency_p50_s": stats.capture_s if stats else 0.0,
            "warmup_latency_s": stats.warmup_s if stats else 0.0,
            "mem_delta_mb": stats.mem_delta_mb if stats else 0.0,
        }
        del graph
        torch.cuda.empty_cache()

    return {
        "device_name": device_name,
        "shapes": shapes,
    }


def load_calibration(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def save_calibration(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def get_shape_stats(calibration: Dict[str, Any], key: ShapeKey) -> Optional[Dict[str, float]]:
    key_str = f"{key[0]}_{key[1]}"
    return calibration.get("shapes", {}).get(key_str)
