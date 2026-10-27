from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .calibration import get_shape_stats
from .types import ShapeKey


def smallest_compatible_captured(
    key: ShapeKey,
    captured: List[ShapeKey],
) -> Optional[ShapeKey]:
    batch, seq = key
    candidates = [k for k in captured if k[0] == batch and k[1] >= seq]
    if not candidates:
        return None
    return min(candidates, key=lambda k: k[1])


def choose_route(
    key: ShapeKey,
    calibration: Dict,
    captured: List[ShapeKey],
    padding_enabled: bool,
) -> Tuple[str, Optional[ShapeKey]]:
    """Return ('eager'|'graph'|'padded_graph', pad_target_shape)."""
    stats = get_shape_stats(calibration, key)
    if stats is None:
        return "eager", None
    t_eager = stats["eager_latency_p50_s"]

    if not padding_enabled:
        return "eager", None

    pad_key = smallest_compatible_captured(key, captured)
    if pad_key is None:
        return "eager", None

    pad_stats = get_shape_stats(calibration, pad_key)
    if pad_stats is None:
        return "eager", None
    t_pad = pad_stats["replay_latency_p50_s"]

    if t_pad < t_eager:
        return "padded_graph", pad_key
    return "eager", None
