from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .types import ShapeKey

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower().replace("torch.", "")
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {name!r}; use one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


class EncoderModel:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kw: Dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kw["attn_implementation"] = attn_implementation
        self.model = AutoModel.from_pretrained(model_id, **model_kw)
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: torch.device) -> EncoderModel:
        return cls(
            model_id=config["model_id"],
            device=device,
            dtype=resolve_dtype(config.get("dtype", "float32")),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            attn_implementation=config.get("attn_implementation"),
        )

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return out.last_hidden_state

    def make_random_batch(
        self,
        shape: ShapeKey,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq = shape
        rng = random.Random(seed)
        vocab = max(self.tokenizer.vocab_size, 32000)
        hi = min(vocab, self.tokenizer.vocab_size or vocab)
        ids = torch.tensor(
            [[rng.randrange(1, hi) for _ in range(seq)] for _ in range(batch)],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.ones(batch, seq, dtype=torch.long, device=self.device)
        return ids, mask

    def make_padded_batch(
        self,
        shape: ShapeKey,
        pad_to_seq: int,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq = shape
        if pad_to_seq < seq:
            raise ValueError(f"pad_to_seq {pad_to_seq} < actual seq {seq}")
        ids, mask = self.make_random_batch(shape, seed=seed)
        if pad_to_seq == seq:
            return ids, mask
        pad_len = pad_to_seq - seq
        pad_ids = torch.full(
            (batch, pad_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        pad_mask = torch.zeros(batch, pad_len, dtype=torch.long, device=self.device)
        ids = torch.cat([ids, pad_ids], dim=1)
        mask = torch.cat([mask, pad_mask], dim=1)
        return ids, mask
