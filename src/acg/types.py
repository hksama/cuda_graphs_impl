from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

ShapeKey = Tuple[int, int]


@dataclass(frozen=True)
class ShapeRequest:
    batch: int
    seq: int

    @property
    def key(self) -> ShapeKey:
        return (self.batch, self.seq)

    def to_dict(self) -> dict:
        return {"batch": self.batch, "seq": self.seq}

    @staticmethod
    def from_dict(d: dict) -> ShapeRequest:
        return ShapeRequest(batch=int(d["batch"]), seq=int(d["seq"]))


def all_shape_keys(batch_sizes: Iterable[int], seq_lengths: Iterable[int]) -> List[ShapeKey]:
    return [(b, s) for b in batch_sizes for s in seq_lengths]
