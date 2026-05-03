"""NyxOwl transformer language model — implemented from scratch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.scale = math.sqrt(self.d_head)

        # Fused QKV projection — one matmul instead of three.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask (not a parameter).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                                   # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=-1)                # each (B, T, C)

        # Reshape to (B, n_heads, T, d_head)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale            # (B, H, T, T)
        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                            # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)     # (B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → sublayer → residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NyxOwl(nn.Module):
    """
    Decoder-only transformer language model.

    Architecture highlights:
    - Learned token + position embeddings
    - Pre-LayerNorm residual connections
    - Causal (masked) multi-head self-attention
    - GELU feed-forward blocks
    - Weight tying between token embedding and LM head
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying reduces parameters and often improves perplexity.
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     (B, T) token ids
            targets: (B, T) next-token ids for computing cross-entropy loss

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy, or None if targets is None
        """
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(T, device=idx.device)
        x = self.emb_dropout(self.token_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Auto-regressively sample *max_new_tokens* tokens appended to *idx*.

        Args:
            idx:            (B, T) seed token ids
            max_new_tokens: number of new tokens to generate
            temperature:    softmax temperature (< 1 = sharper, > 1 = flatter)
            top_k:          if set, only sample from the top-k logits

        Returns:
            (B, T + max_new_tokens) token ids
        """
        for _ in range(max_new_tokens):
            idx_window = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_window)
            logits = logits[:, -1, :] / temperature       # (B, vocab_size)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold, _ = torch.topk(logits, k)
                logits[logits < threshold[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (
            self.parameters() if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)
