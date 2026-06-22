"""Text dataset utilities for causal language modelling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

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
    dataset: TextDataset | "MemmapTokenDataset",
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        # Keep workers alive across epochs so we don't pay re-fork cost,
        # and let them prefetch a couple of batches so the GPU never idles.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


# ----------------------------------------------------------------------
# Memory-mapped token dataset — for multi-GB pre-tokenised corpora.
# ----------------------------------------------------------------------


class MemmapTokenDataset(Dataset):
    """
    Reads token ids from a flat ``uint32`` binary file via numpy.memmap.

    Use ``encode_to_memmap`` to produce the file once, then sample chunks
    cheaply at training time without loading the whole corpus into RAM.
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self._memmap: np.memmap | None = None
        # Length is known from file size / 4 bytes per uint32.
        nbytes = self.path.stat().st_size
        if nbytes % 4 != 0:
            raise ValueError(f"Token file size {nbytes} is not a multiple of 4")
        self._length = nbytes // 4

    @property
    def tokens(self) -> np.memmap:
        # Lazily open per-worker so DataLoader workers don't share an FD.
        if self._memmap is None:
            self._memmap = np.memmap(self.path, dtype=np.uint32, mode="r")
        return self._memmap

    def __len__(self) -> int:
        return max(0, self._length - self.seq_len - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = np.asarray(self.tokens[idx : idx + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def encode_to_memmap(
    text_paths: list[str | Path],
    out_path: str | Path,
    tokenizer: BPETokenizer,
    chunk_chars: int = 1_000_000,
    verbose: bool = True,
) -> int:
    """
    Stream-encode a list of text files into a single uint32 binary file.

    Returns the number of tokens written. Each file is read in chunks to
    keep peak memory bounded; the tokenizer's per-line encode path keeps
    individual encode calls fast.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out, "wb") as fout:
        for fp in text_paths:
            fp = Path(fp)
            if verbose:
                print(f"  tokenising {fp} ({fp.stat().st_size / 1e6:.1f} MB)")
            with open(fp, "r", encoding="utf-8", errors="replace") as fin:
                buffer = ""
                while True:
                    chunk = fin.read(chunk_chars)
                    if not chunk:
                        break
                    buffer += chunk
                    # Encode up to the last newline so we don't split a word.
                    cut = buffer.rfind("\n")
                    if cut == -1:
                        continue
                    head, buffer = buffer[: cut + 1], buffer[cut + 1 :]
                    ids = tokenizer.encode(head, allow_special=False)
                    np.asarray(ids, dtype=np.uint32).tofile(fout)
                    total += len(ids)
                if buffer:
                    ids = tokenizer.encode(buffer, allow_special=False)
                    np.asarray(ids, dtype=np.uint32).tofile(fout)
                    total += len(ids)
    if verbose:
        print(f"  wrote {total:,} tokens → {out}")
    return total
