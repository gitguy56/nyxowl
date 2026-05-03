"""Text dataset utilities for causal language modelling."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import BPETokenizer


class TextDataset(Dataset):
    """
    Sliding-window dataset for next-token prediction.

    Each sample is a (x, y) pair of length *seq_len* where y = x shifted
    one position to the right.
    """

    def __init__(self, tokens: list[int], seq_len: int) -> None:
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Need seq_len + 1 tokens per sample (x and y share all but one token).
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: BPETokenizer,
        seq_len: int,
    ) -> "TextDataset":
        return cls(tokenizer.encode(text), seq_len)

    @classmethod
    def train_val_split(
        cls,
        text: str,
        tokenizer: BPETokenizer,
        seq_len: int,
        val_fraction: float = 0.1,
    ) -> tuple["TextDataset", "TextDataset"]:
        """Tokenise *text* and split into train / validation datasets."""
        tokens = tokenizer.encode(text)
        split = int(len(tokens) * (1.0 - val_fraction))
        return cls(tokens[:split], seq_len), cls(tokens[split:], seq_len)


def build_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
