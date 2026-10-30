from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from .types import ShapeKey, ShapeRequest, all_shape_keys


def _write_trace(path: Path, seed: int, requests: List[ShapeRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "seed": seed,
        "requests": [r.to_dict() for r in requests],
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_trace(path: Path) -> tuple[int, List[ShapeRequest]]:
    with path.open() as f:
        data = json.load(f)
    seed = int(data["seed"])
    reqs = [ShapeRequest.from_dict(r) for r in data["requests"]]
    return seed, reqs


def _pick_hot(keys: List[ShapeKey], preferred: List[ShapeKey], n: int = 3) -> List[ShapeKey]:
    hot = [k for k in preferred if k in keys]
    if len(hot) >= 2:
        return hot
    return keys[: min(n, len(keys))]


def generate_recurring(
    batch_sizes: List[int],
    seq_lengths: List[int],
    length: int,
    seed: int,
) -> List[ShapeRequest]:
    keys = all_shape_keys(batch_sizes, seq_lengths)
    hot = _pick_hot(keys, [(1, 64), (4, 128), (8, 64)])
    rng = random.Random(seed)
    out: List[ShapeRequest] = []
    for _ in range(length):
        if rng.random() < 0.8:
            b, s = rng.choice(hot)
        else:
            b, s = rng.choice(keys)
        out.append(ShapeRequest(b, s))
    return out


def generate_many_rare(
    batch_sizes: List[int],
    seq_lengths: List[int],
    length: int,
    seed: int,
) -> List[ShapeRequest]:
    keys = all_shape_keys(batch_sizes, seq_lengths)
    rng = random.Random(seed)
    return [ShapeRequest(*rng.choice(keys)) for _ in range(length)]


def generate_shift(
    batch_sizes: List[int],
    seq_lengths: List[int],
    length: int,
    seed: int,
) -> List[ShapeRequest]:
    keys = all_shape_keys(batch_sizes, seq_lengths)
    hot_a = _pick_hot(keys, [(1, 32), (4, 64), (8, 128)])
    hot_b = _pick_hot(keys, [(1, 256), (4, 256), (8, 32)])
    if hot_b == hot_a and len(keys) > 1:
        hot_b = keys[-min(3, len(keys)) :]
    rng = random.Random(seed)
    half = length // 2
    out: List[ShapeRequest] = []
    for i in range(length):
        if i < half:
            b, s = rng.choice(hot_a) if rng.random() < 0.85 else rng.choice(keys)
        else:
            b, s = rng.choice(hot_b) if rng.random() < 0.85 else rng.choice(keys)
        out.append(ShapeRequest(b, s))
    return out


def generate_all_traces(
    traces_dir: Path,
    batch_sizes: List[int],
    seq_lengths: List[int],
    length: int,
    seeds: dict,
) -> None:
    _write_trace(
        traces_dir / "recurring.json",
        seeds["recurring"],
        generate_recurring(batch_sizes, seq_lengths, length, seeds["recurring"]),
    )
    _write_trace(
        traces_dir / "many_rare.json",
        seeds["many_rare"],
        generate_many_rare(batch_sizes, seq_lengths, length, seeds["many_rare"]),
    )
    _write_trace(
        traces_dir / "shift.json",
        seeds["shift"],
        generate_shift(batch_sizes, seq_lengths, length, seeds["shift"]),
    )
