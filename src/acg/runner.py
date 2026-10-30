from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .eager import timed_eager_forward
from .memory import mem_snapshot_mb
from .model import EncoderModel
from .policies import (
    BasePolicy,
    CalibrationFixedPolicy,
    GraphCache,
    PolicyDecision,
    set_policy_calibration,
)
from .correctness import outputs_match
from .types import ShapeKey, ShapeRequest


def _run_capture(
    cache: GraphCache,
    encoder: EncoderModel,
    key: ShapeKey,
    warmup_iters: int,
) -> tuple[Optional[Any], float, float, float]:
    """Returns graph, capture_s, warmup_s, mem_delta."""
    t0 = time.perf_counter()
    graph = cache.try_capture(encoder, key, warmup_iters)
    elapsed = time.perf_counter() - t0
    if graph is None:
        return None, elapsed, 0.0, 0.0
    stats = graph.capture_stats
    return (
        graph,
        stats.capture_s if stats else elapsed,
        stats.warmup_s if stats else 0.0,
        stats.mem_delta_mb if stats else 0.0,
    )


def run_trace(
    encoder: EncoderModel,
    policy: BasePolicy,
    requests: List[ShapeRequest],
    cache: GraphCache,
    calibration: Dict[str, Any],
    config: dict,
    correctness_sample_every: int = 20,
) -> tuple[List[dict], dict]:
    set_policy_calibration(calibration)
    warmup_iters = config["calibration_warmup_iters"]
    padding_enabled = config.get("padding_enabled", False)

    records: List[dict] = []
    history: List[ShapeRequest] = []
    total_capture_s = 0.0
    workload_t0 = time.perf_counter()

    for request_idx, req in enumerate(requests):
        if (
            isinstance(policy, CalibrationFixedPolicy)
            and request_idx == policy.observe_k
            and len(history) >= policy.observe_k
        ):
            to_capture = policy.finalize_observation(history)
            for cap_key in to_capture:
                if cache.get(cap_key) is None:
                    _, cap_s, warm_s, _ = _run_capture(cache, encoder, cap_key, warmup_iters)
                    total_capture_s += cap_s + warm_s

        decision = policy.decide(request_idx, req, history, cache)

        capture_s = 0.0
        warmup_s = 0.0
        path = "eager"
        latency_s = 0.0
        out: Optional[torch.Tensor] = None
        ref: Optional[torch.Tensor] = None
        check_correct = (request_idx % correctness_sample_every) == 0

        graph_key: ShapeKey = req.key
        if decision.pad_to is not None:
            graph_key = decision.pad_to

        if decision.capture:
            g, capture_s, warmup_s, _ = _run_capture(cache, encoder, graph_key, warmup_iters)
            total_capture_s += capture_s + warmup_s
            if g is None:
                decision = PolicyDecision(use_graph=False, reason="capture_failed")

        ids, mask = encoder.make_random_batch(req.key, seed=request_idx)

        if decision.use_graph:
            graph = cache.get(graph_key)
            if graph is None:
                decision = PolicyDecision(use_graph=False, reason="missing_graph")

        if decision.use_graph:
            graph = cache.get(graph_key)
            if graph is not None:
                if decision.pad_to is not None and decision.pad_to != req.key:
                    pad_seq = decision.pad_to[1]
                    ids, mask = encoder.make_padded_batch(req.key, pad_seq, seed=request_idx)
                    path = "padded_graph"
                else:
                    path = "graph"
                out, latency_s = graph.replay(ids, mask)
                if decision.pad_to is not None and decision.pad_to != req.key:
                    out = out[:, : req.seq, :]
            else:
                path = "eager"
                out, latency_s = timed_eager_forward(encoder, ids, mask)
        else:
            if padding_enabled and not decision.capture:
                from .padding import choose_route

                route, pad_to = choose_route(
                    req.key, calibration, cache.keys(), padding_enabled
                )
                if route == "padded_graph" and pad_to is not None:
                    graph = cache.get(pad_to)
                    if graph is None:
                        g, cs, ws, _ = _run_capture(cache, encoder, pad_to, warmup_iters)
                        total_capture_s += cs + ws
                        graph = cache.get(pad_to)
                    if graph is not None:
                        pad_seq = pad_to[1]
                        ids, mask = encoder.make_padded_batch(req.key, pad_seq, seed=request_idx)
                        out, latency_s = graph.replay(ids, mask)
                        out = out[:, : req.seq, :]
                        path = "padded_graph"
                    else:
                        out, latency_s = timed_eager_forward(encoder, ids, mask)
                else:
                    out, latency_s = timed_eager_forward(encoder, ids, mask)
            else:
                out, latency_s = timed_eager_forward(encoder, ids, mask)

        if check_correct:
            ref_ids, ref_mask = encoder.make_random_batch(req.key, seed=request_idx)
            if path == "padded_graph":
                pad_seq = ids.shape[1]
                ref, _ = timed_eager_forward(encoder, ref_ids[:, : req.seq], ref_mask[:, : req.seq])
            else:
                ref, _ = timed_eager_forward(encoder, ids, mask)
            correct = outputs_match(out, ref, config)
        else:
            correct = True

        alloc_mb, reserved_mb = mem_snapshot_mb()
        records.append(
            {
                "request_id": request_idx,
                "shape": {"batch": req.batch, "seq": req.seq},
                "policy": policy.name,
                "path": path,
                "latency_s": latency_s,
                "capture_s": capture_s,
                "warmup_s": warmup_s,
                "mem_allocated_mb": alloc_mb,
                "mem_reserved_mb": reserved_mb,
                "correct": correct,
                "reason": decision.reason,
            }
        )
        history.append(req)

    total_workload_s = time.perf_counter() - workload_t0
    summary = {
        "policy": policy.name,
        "num_requests": len(requests),
        "total_workload_s": total_workload_s,
        "total_capture_s": total_capture_s,
        "capture_count": cache.capture_count,
        "cache_mem_mb": cache.cache_mem_mb,
        "unsafe_shapes": [list(k) for k in cache.unsafe_shapes],
    }
    return records, summary


def write_jsonl(path: Path, records: List[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        for rec in records:
            f.write(json.dumps({"type": "request", **rec}) + "\n")
