from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .types import ShapeKey


class EncoderModel:
    def __init__(self, model_id: str, device: torch.device):
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        self.model.to(device)
        self.model.eval()

    def forward_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state

    def make_random_batch(
        self,
        shape: ShapeKey,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq = shape
        rng = random.Random(seed)
        vocab = self.tokenizer.vocab_size
        ids = torch.tensor(
            [[rng.randrange(1, vocab) for _ in range(seq)] for _ in range(batch)],
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
        pad_ids = torch.zeros(batch, pad_len, dtype=torch.long, device=self.device)
        pad_mask = torch.zeros(batch, pad_len, dtype=torch.long, device=self.device)
        ids = torch.cat([ids, pad_ids], dim=1)
        mask = torch.cat([mask, pad_mask], dim=1)
        return ids, mask
