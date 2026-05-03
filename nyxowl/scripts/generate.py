"""CLI entry point for text generation with a trained NyxOwl model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nyxowl-generate",
        description="Generate text from a trained NyxOwl model.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer.json"
    )
    p.add_argument("--prompt", default="", help="Seed text (empty = unconditional)")
    p.add_argument("--max-tokens", type=int, default=200, help="Tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, help="0 = disabled")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from nyxowl.model import NyxOwl
    from nyxowl.tokenizer import BPETokenizer

    # ----------------------------------------------------------------- device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # --------------------------------------------------------------- tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # ------------------------------------------------------------------ model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NyxOwl(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded model ({model.num_parameters():,} params) from '{args.checkpoint}'")

    # --------------------------------------------------------------- generate
    prompt = args.prompt
    print(f"\n{'─' * 60}")
    if prompt:
        print(f"Prompt: {prompt!r}")
        print("─" * 60)
        ids = tokenizer.encode(prompt)
    else:
        # Start with a null byte as a stand-in for BOS
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )

    generated = tokenizer.decode(output[0].tolist())
    print(generated)
    print("─" * 60)


if __name__ == "__main__":
    main()
