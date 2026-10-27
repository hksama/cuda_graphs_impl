from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

from .calibration import get_shape_stats
from .graph_pool import CapturedShapeGraph
from .model import EncoderModel
from .padding import choose_route
from .types import ShapeKey, ShapeRequest


@dataclass
class PolicyDecision:
    use_graph: bool
    capture: bool = False
    pad_to: Optional[ShapeKey] = None
    reason: str = ""


class GraphCache:
    """LRU cache of captured CUDA graphs with memory accounting."""

    def __init__(self, max_size: int, mem_budget_mb: float):
        self.max_size = max_size
        self.mem_budget_mb = mem_budget_mb
        self._graphs: OrderedDict[ShapeKey, CapturedShapeGraph] = OrderedDict()
        self.cache_mem_mb: float = 0.0
        self.unsafe_shapes: Set[ShapeKey] = set()
        self.capture_count: int = 0
        self._shared_pool = torch.cuda.graph_pool_handle()

    def get(self, key: ShapeKey) -> Optional[CapturedShapeGraph]:
        if key not in self._graphs:
            return None
        self._graphs.move_to_end(key)
        return self._graphs[key]

    def keys(self) -> List[ShapeKey]:
        return list(self._graphs.keys())

    def _evict_lru(self) -> None:
        if not self._graphs:
            return
        # Shared graph pool: freeing one captured graph invalidates all siblings.
        graphs = list(self._graphs.values())
        self._graphs.clear()
        self.cache_mem_mb = 0.0
        for graph in graphs:
            del graph._graph
            del graph
        self._shared_pool = torch.cuda.graph_pool_handle()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def try_capture(
        self,
        encoder: EncoderModel,
        key: ShapeKey,
        warmup_iters: int,
    ) -> Optional[CapturedShapeGraph]:
        if key in self.unsafe_shapes:
            return None
        if key in self._graphs:
            return self.get(key)

        stats_existing = get_shape_stats(get_calibration(), key)
        mem_delta = stats_existing["mem_delta_mb"] if stats_existing else 64.0

        while len(self._graphs) >= self.max_size:
            self._evict_lru()
            break  # one flush clears the entire cache

        if self.cache_mem_mb + mem_delta > self.mem_budget_mb:
            return None

        try:
            graph = CapturedShapeGraph(
                encoder, key, warmup_iters=warmup_iters, mem_pool=self._shared_pool
            )
        except RuntimeError:
            self.unsafe_shapes.add(key)
            torch.cuda.empty_cache()
            return None

        actual_delta = graph.capture_stats.mem_delta_mb if graph.capture_stats else mem_delta
        self._graphs[key] = graph
        self._graphs.move_to_end(key)
        self.cache_mem_mb += actual_delta
        self.capture_count += 1
        return graph


# Module-level calibration ref set by runner (avoids passing through every cache call)
_CALIB: Optional[Dict[str, Any]] = None


def set_policy_calibration(calibration: Dict[str, Any]) -> None:
    global _CALIB
    _CALIB = calibration


def get_calibration() -> Dict[str, Any]:
    return _CALIB or {"shapes": {}}


class BasePolicy(ABC):
    name: str

    @abstractmethod
    def decide(
        self,
        request_idx: int,
        req: ShapeRequest,
        history: List[ShapeRequest],
        cache: GraphCache,
    ) -> PolicyDecision:
        ...


class AlwaysEagerPolicy(BasePolicy):
    name = "always_eager"

    def decide(
        self,
        request_idx: int,
        req: ShapeRequest,
        history: List[ShapeRequest],
        cache: GraphCache,
    ) -> PolicyDecision:
        return PolicyDecision(use_graph=False, reason="policy1")


class LruOnFirstUsePolicy(BasePolicy):
    name = "lru_on_first_use"

    def __init__(self, warmup_iters: int):
        self.warmup_iters = warmup_iters
        self._seen: Set[ShapeKey] = set()

    def decide(
        self,
        request_idx: int,
        req: ShapeRequest,
        history: List[ShapeRequest],
        cache: GraphCache,
    ) -> PolicyDecision:
        key = req.key
        if cache.get(key) is not None:
            return PolicyDecision(use_graph=True, reason="cached")
        if key not in self._seen:
            self._seen.add(key)
        if key in cache.unsafe_shapes:
            return PolicyDecision(use_graph=False, reason="unsafe_shape")
        return PolicyDecision(use_graph=True, capture=True, reason="capture_on_miss")


@dataclass
class CalibrationFixedPolicy(BasePolicy):
    name: str = "calibration_fixed"
    observe_k: int = 40
    precapture_m: int = 3
    warmup_iters: int = 5

    _precapture_shapes: List[ShapeKey] = field(default_factory=list, init=False)
    _observation_finalized: bool = field(default=False, init=False)

    def finalize_observation(self, history: List[ShapeRequest]) -> List[ShapeKey]:
        if self._observation_finalized:
            return []
        self._observation_finalized = True
        counts: Dict[ShapeKey, int] = {}
        for h in history[: self.observe_k]:
            counts[h.key] = counts.get(h.key, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        self._precapture_shapes = [k for k, _ in ranked[: self.precapture_m]]
        return list(self._precapture_shapes)

    def decide(
        self,
        request_idx: int,
        req: ShapeRequest,
        history: List[ShapeRequest],
        cache: GraphCache,
    ) -> PolicyDecision:
        key = req.key

        if request_idx < self.observe_k:
            return PolicyDecision(use_graph=False, reason="observe")

        if cache.get(key) is not None:
            return PolicyDecision(use_graph=True, reason="cached")
        if key in self._precapture_shapes:
            return PolicyDecision(use_graph=False, reason="fixed_set_miss")
        return PolicyDecision(use_graph=False, reason="not_in_precapture_set")


@dataclass
class AdaptiveAdmissionPolicy(BasePolicy):
    name: str = "adaptive_admission"
    window: int = 50
    horizon: int = 20
    margin: float = 1.25
    warmup_iters: int = 5
    padding_enabled: bool = False

    def decide(
        self,
        request_idx: int,
        req: ShapeRequest,
        history: List[ShapeRequest],
        cache: GraphCache,
    ) -> PolicyDecision:
        key = req.key

        if cache.get(key) is not None:
            return PolicyDecision(use_graph=True, reason="cached")

        calib = get_calibration()
        if self.padding_enabled and cache.keys():
            route, pad_to = choose_route(key, calib, cache.keys(), True)
            if route == "padded_graph" and pad_to is not None:
                if cache.get(pad_to) is not None:
                    return PolicyDecision(use_graph=True, pad_to=pad_to, reason="pad_route")
                return PolicyDecision(
                    use_graph=True, capture=True, pad_to=pad_to, reason="pad_capture_target"
                )

        if key in cache.unsafe_shapes:
            return PolicyDecision(use_graph=False, reason="unsafe_shape")

        stats = get_shape_stats(calib, key)
        if stats is None:
            return PolicyDecision(use_graph=False, reason="no_calib")

        recent = [h.key for h in history[-self.window :]]
        if not recent:
            return PolicyDecision(use_graph=False, reason="cold_start")

        counts: Dict[ShapeKey, int] = {}
        for k in recent:
            counts[k] = counts.get(k, 0) + 1
        total = len(recent)
        p_shape = counts.get(key, 0) / total

        t_eager = stats["eager_latency_p50_s"]
        t_replay = stats["replay_latency_p50_s"]
        # Admission uses capture latency; full warmup is still charged at runtime.
        t_capture = stats["capture_latency_p50_s"]
        savings_per_hit = max(0.0, t_eager - t_replay)
        expected_hits = p_shape * self.horizon
        expected_savings = expected_hits * savings_per_hit

        mem_delta = stats["mem_delta_mb"]
        if cache.cache_mem_mb + mem_delta > cache.mem_budget_mb:
            return PolicyDecision(use_graph=False, reason="mem_budget")

        if expected_savings > t_capture * self.margin:
            return PolicyDecision(use_graph=True, capture=True, reason="admission_accept")
        return PolicyDecision(use_graph=False, reason="admission_reject")


POLICY_NAMES = {
    "always_eager": AlwaysEagerPolicy,
    "lru_on_first_use": LruOnFirstUsePolicy,
    "calibration_fixed": CalibrationFixedPolicy,
    "adaptive_admission": AdaptiveAdmissionPolicy,
}


def make_policy(name: str, config: dict) -> BasePolicy:
    if name == "always_eager":
        return AlwaysEagerPolicy()
    if name == "lru_on_first_use":
        return LruOnFirstUsePolicy(warmup_iters=config["calibration_warmup_iters"])
    if name == "calibration_fixed":
        return CalibrationFixedPolicy(
            observe_k=config["policy3_observe_k"],
            precapture_m=config["policy3_precapture_m"],
            warmup_iters=config["calibration_warmup_iters"],
        )
    if name == "adaptive_admission":
        return AdaptiveAdmissionPolicy(
            window=config["policy4_window"],
            horizon=config["policy4_horizon"],
            margin=config["policy4_margin"],
            warmup_iters=config["calibration_warmup_iters"],
            padding_enabled=config.get("padding_enabled", False),
        )
    raise ValueError(f"Unknown policy: {name}")
