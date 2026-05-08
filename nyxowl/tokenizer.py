"""Byte-Pair Encoding tokenizer trained from scratch."""

import base64
import json
from pathlib import Path


class BPETokenizer:
    """
    BPE tokenizer that operates on raw UTF-8 bytes.

    Training is O(n * num_merges) and encoding is O(t * n) where t is the
    number of merges and n is the token sequence length. Good enough for
    small corpora; swap for a Rust-backed tokenizer for large-scale work.
    """

    def __init__(self) -> None:
        # Maps (id_a, id_b) -> new_id in insertion order (= merge priority).
        self.merges: dict[tuple[int, int], int] = {}
        # Maps token id -> raw bytes.
        self.vocab: dict[int, bytes] = {}
        self._trained: bool = False

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE merges on *text* until *vocab_size* is reached."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte alphabet)")

        # Byte-level base vocabulary.
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        ids: list[int] = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for step in range(num_merges):
            counts: dict[tuple[int, int], int] = {}
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1

            if not counts:
                break

            best_pair = max(counts, key=counts.__getitem__)
            new_id = 256 + step

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids = self._apply_merge(ids, best_pair, new_id)

            if verbose and (step + 1) % 500 == 0:
                pct = 100.0 * (step + 1) / num_merges
                token_str = (
                    self.vocab[new_id].decode("utf-8", errors="replace").replace("\n", "\\n")
                )
                print(f"  [{pct:5.1f}%] merge {step + 1}/{num_merges}: '{token_str}'")

        self._trained = True

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token ids.

        For long inputs we chunk by lines first — each encode pass is O(n * m)
        where n is sequence length and m is the number of applicable merges,
        so keeping per-call sequences short dramatically cuts wall time on
        big corpora. Line breaks rarely sit inside meaningful merges, so the
        resulting token sequence is effectively identical.
        """
        if not self._trained:
            raise RuntimeError("Call train() or load() before encoding.")

        if len(text) > 4096:
            out: list[int] = []
            for line in text.splitlines(keepends=True):
                out.extend(self._encode_chunk(line))
            return out
        return self._encode_chunk(text)

    def _encode_chunk(self, text: str) -> list[int]:
        ids: list[int] = list(text.encode("utf-8"))

        # Greedily apply the highest-priority (earliest) eligible merge.
        while len(ids) >= 2:
            best_rank = float("inf")
            best_pair: tuple[int, int] | None = None

            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # best_rank == new_id for this merge
            ids = self._apply_merge(ids, best_pair, int(best_rank))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string."""
        raw = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save tokenizer state to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
            "vocab": {
                str(k): base64.b64encode(v).decode("ascii")
                for k, v in self.vocab.items()
            },
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer state from a JSON file produced by :meth:`save`."""
        tok = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok.merges = {(a, b): nid for a, b, nid in data["merges"]}
        tok.vocab = {
            int(k): base64.b64decode(v) for k, v in data["vocab"].items()
        }
        tok._trained = True
        return tok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_merge(
        ids: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """Replace every occurrence of *pair* in *ids* with *new_id*."""
        result: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result
