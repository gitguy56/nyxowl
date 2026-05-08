"""Byte-Pair Encoding tokenizer trained from scratch."""

import base64
import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """
    BPE tokenizer that operates on raw UTF-8 bytes.

    Training is O(n * num_merges) and encoding is O(t * n) where t is the
    number of merges and n is the token sequence length. Good enough for
    small corpora; swap for a Rust-backed tokenizer for large-scale work.
    """

    # Default special tokens reserved for chat formatting. They never collide
    # with byte values (0-255) or trained merges; we splice them in above the
    # learned vocab during save/load.
    DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|sys|>",
        "<|user|>",
        "<|nyx|>",
        "<|end|>",
    )

    def __init__(self) -> None:
        # Maps (id_a, id_b) -> new_id in insertion order (= merge priority).
        self.merges: dict[tuple[int, int], int] = {}
        # Maps token id -> raw bytes (or UTF-8 form for special tokens).
        self.vocab: dict[int, bytes] = {}
        # Maps special-token literal string (e.g. "<|user|>") -> token id.
        self.special_tokens: dict[str, int] = {}
        self._trained: bool = False

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def add_special_tokens(self, tokens: list[str] | tuple[str, ...]) -> None:
        """Reserve token ids for non-mergeable literal strings (e.g. <|sys|>)."""
        next_id = len(self.vocab) + len(self.special_tokens)
        for tok in tokens:
            if tok in self.special_tokens:
                continue
            self.special_tokens[tok] = next_id
            next_id += 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train BPE merges on *text* until *vocab_size* is reached.

        Optimisations vs. a naive single-stream implementation:
        - Pre-split into lines and deduplicate. Each unique line is stored
          once with a frequency count, so repeated boilerplate (story
          openings, headers, etc.) is processed in O(1) instead of O(N).
        - Pair counts are weighted by line frequency.
        - Each merge step rewrites only the unique line list, not the
          whole corpus.

        For typical natural-language corpora this is one to two orders of
        magnitude faster than the streaming version.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte alphabet)")

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        # Split into lines (newlines preserved) and dedupe.
        lines = text.splitlines(keepends=True) or [text]
        line_counts: Counter[str] = Counter(lines)

        # Each unique line becomes a mutable list of token ids.
        line_ids: dict[str, list[int]] = {
            line: list(line.encode("utf-8")) for line in line_counts
        }

        num_merges = vocab_size - 256

        for step in range(num_merges):
            # Weighted pair counts across unique lines.
            pair_counts: Counter[tuple[int, int]] = Counter()
            for line, ids in line_ids.items():
                weight = line_counts[line]
                if len(ids) < 2:
                    continue
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += weight

            if not pair_counts:
                break

            best_pair, _ = pair_counts.most_common(1)[0]
            new_id = 256 + step

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge to every unique line in place.
            for line, ids in line_ids.items():
                if len(ids) >= 2:
                    line_ids[line] = self._apply_merge(ids, best_pair, new_id)

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

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Encode *text* to a list of token ids.

        Special tokens (e.g. ``<|sys|>``) are matched literally when
        ``allow_special`` is True and emitted as their reserved ids.
        For long inputs we chunk by lines first — each encode pass is O(n * m)
        where n is sequence length and m is the number of applicable merges,
        so keeping per-call sequences short dramatically cuts wall time on
        big corpora. Line breaks rarely sit inside meaningful merges, so the
        resulting token sequence is effectively identical.
        """
        if not self._trained:
            raise RuntimeError("Call train() or load() before encoding.")

        # Split out special tokens (if any) before BPE so they survive verbatim.
        if allow_special and self.special_tokens:
            segments = self._split_special(text)
        else:
            segments = [(text, False)]

        out: list[int] = []
        for segment, is_special in segments:
            if is_special:
                out.append(self.special_tokens[segment])
                continue
            if len(segment) > 4096:
                for line in segment.splitlines(keepends=True):
                    out.extend(self._encode_chunk(line))
            else:
                out.extend(self._encode_chunk(segment))
        return out

    def _split_special(self, text: str) -> list[tuple[str, bool]]:
        """Split text on every literal special-token occurrence."""
        if not self.special_tokens:
            return [(text, False)]
        # Sort by length desc so longer specials match before any prefix subset.
        specials = sorted(self.special_tokens, key=len, reverse=True)
        parts: list[tuple[str, bool]] = []
        i = 0
        n = len(text)
        while i < n:
            match: str | None = None
            for sp in specials:
                if text.startswith(sp, i):
                    match = sp
                    break
            if match is not None:
                parts.append((match, True))
                i += len(match)
            else:
                # Scan ahead until the next special token start.
                j = i + 1
                while j < n:
                    hit = False
                    for sp in specials:
                        if text.startswith(sp, j):
                            hit = True
                            break
                    if hit:
                        break
                    j += 1
                parts.append((text[i:j], False))
                i = j
        return parts

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
        """Decode a list of token ids back to a string.

        Special tokens are emitted as their literal form (``<|sys|>`` etc.)
        so a chat REPL can split on them.
        """
        if self.special_tokens:
            inv_special = {tid: tok for tok, tid in self.special_tokens.items()}
        else:
            inv_special = {}

        out_pieces: list[bytes] = []
        for i in ids:
            if i in inv_special:
                out_pieces.append(inv_special[i].encode("utf-8"))
            else:
                out_pieces.append(self.vocab.get(i, b"\xef\xbf\xbd"))
        return b"".join(out_pieces).decode("utf-8", errors="replace")

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
            "special_tokens": self.special_tokens,
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
        tok.special_tokens = {
            k: int(v) for k, v in data.get("special_tokens", {}).items()
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
