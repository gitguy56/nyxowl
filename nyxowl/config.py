"""Model and training hyper-parameter containers."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Hyper-parameters for a NyxOwl transformer.

    Default values produce a ~7 M-parameter model that trains comfortably on a
    laptop GPU (4 GB VRAM) with batch_size=32 and seq_len=256.
    """

    vocab_size: int = 8192
    d_model: int = 256       # embedding / hidden dimension
    n_heads: int = 8         # attention heads  (d_model must be divisible)
    n_layers: int = 6        # transformer blocks
    ffn_dim: int = 1024      # feed-forward inner dimension
    max_seq_len: int = 256   # maximum context length
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


@dataclass
class TrainConfig:
    """Training loop hyper-parameters."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    grad_accum_steps: int = 1   # effective batch = batch_size * grad_accum_steps
    max_steps: int = 5000
    warmup_steps: int = 200     # linear LR warmup before cosine decay
    grad_clip: float = 1.0
    eval_interval: int = 200    # evaluate val loss every N steps
    save_interval: int = 1000   # save checkpoint every N steps
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"        # "auto" | "cuda" | "mps" | "cpu"


# ---------------------------------------------------------------------
# Named presets — edit the train script to opt in.
# ---------------------------------------------------------------------

def model_preset(name: str, vocab_size: int) -> ModelConfig:
    """Return a ModelConfig sized for the given name."""
    if name == "tiny":   # ~7M params, runs anywhere
        return ModelConfig(
            vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=6,
            ffn_dim=1024, max_seq_len=256, dropout=0.1,
        )
    if name == "small":  # ~25M params
        return ModelConfig(
            vocab_size=vocab_size, d_model=384, n_heads=8, n_layers=8,
            ffn_dim=1536, max_seq_len=512, dropout=0.1,
        )
    if name == "medium":  # ~150M params (GPT-2 small style)
        return ModelConfig(
            vocab_size=vocab_size, d_model=768, n_heads=12, n_layers=12,
            ffn_dim=3072, max_seq_len=1024, dropout=0.1,
        )
    if name == "large":   # ~350M params
        return ModelConfig(
            vocab_size=vocab_size, d_model=1024, n_heads=16, n_layers=24,
            ffn_dim=4096, max_seq_len=1024, dropout=0.1,
        )
    raise ValueError(f"Unknown preset: {name!r}")
