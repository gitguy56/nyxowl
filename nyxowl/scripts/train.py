"""CLI entry point for training a NyxOwl model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nyxowl-train",
        description="Train a NyxOwl causal language model from scratch.",
    )
    # Data
    p.add_argument("--data", required=True, help="Path to training text file (.txt)")
    p.add_argument("--output", default="runs/default", help="Run output directory")

    # Model preset (overrides individual arch args if provided)
    p.add_argument(
        "--preset",
        choices=["tiny", "small", "medium", "large"],
        default=None,
        help="Named architecture preset (overrides d-model/n-heads/etc.)",
    )

    # Tokenizer
    p.add_argument("--vocab-size", type=int, default=4096, help="BPE vocabulary size")
    p.add_argument(
        "--with-chat-tokens",
        action="store_true",
        help="Reserve <|sys|> / <|user|> / <|nyx|> / <|end|> special tokens",
    )

    # Model (used if --preset not set)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")

    # Resume
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from nyxowl.config import ModelConfig, TrainConfig, model_preset
    from nyxowl.dataset import TextDataset, build_dataloader
    from nyxowl.model import NyxOwl
    from nyxowl.tokenizer import BPETokenizer
    from nyxowl.trainer import Trainer

    # ------------------------------------------------------------------ data
    text = Path(args.data).read_text(encoding="utf-8")
    print(f"Data: {len(text):,} characters from '{args.data}'")

    # ------------------------------------------------------------ tokenizer
    tok_path = output_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"Loading tokenizer from {tok_path} ...")
        tokenizer = BPETokenizer.load(str(tok_path))
    else:
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) ...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=args.vocab_size, verbose=True)
        if args.with_chat_tokens:
            tokenizer.add_special_tokens(list(BPETokenizer.DEFAULT_SPECIAL_TOKENS))
            print(f"  Added {len(tokenizer.special_tokens)} special tokens")
        tokenizer.save(str(tok_path))
        print(f"Tokenizer saved → {tok_path}")

    print(f"Vocab size: {tokenizer.vocab_size}")

    # -------------------------------------------------------------- datasets
    seq_len = (
        model_preset(args.preset, tokenizer.vocab_size).max_seq_len
        if args.preset
        else args.seq_len
    )
    train_ds, val_ds = TextDataset.train_val_split(text, tokenizer, seq_len=seq_len)
    print(
        f"Train tokens: {len(train_ds.tokens):,} | "
        f"Val tokens: {len(val_ds.tokens):,}"
    )

    train_loader = build_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_ds, args.batch_size, shuffle=False)

    # ----------------------------------------------------------------- model
    if args.preset:
        model_cfg = model_preset(args.preset, tokenizer.vocab_size)
        model_cfg.dropout = args.dropout
    else:
        model_cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
        )
    model = NyxOwl(model_cfg)

    train_cfg = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device,
    )

    trainer = Trainer(model, train_cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
