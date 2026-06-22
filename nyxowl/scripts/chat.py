"""Interactive chat REPL for NyxOwl."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nyxowl-chat",
        description="Chat with a trained NyxOwl model.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    p.add_argument(
        "--system",
        default=None,
        help="System prompt / persona description (default: built-in NyxOwl)",
    )
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from nyxowl.chat import DEFAULT_NYXOWL_SYSTEM, ChatTemplate, ChatTurn
    from nyxowl.model import NyxOwl
    from nyxowl.tokenizer import BPETokenizer

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tokenizer = BPETokenizer.load(args.tokenizer)
    template = ChatTemplate()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = NyxOwl(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    system = args.system or DEFAULT_NYXOWL_SYSTEM
    history: list[ChatTurn] = []

    end_id = tokenizer.special_tokens.get("<|end|>")
    stop_ids = [end_id] if end_id is not None else None

    print(f"\nNyxOwl ready. Loaded {model.num_parameters():,} params on {device}.")
    print(f"System: {system}\n")
    print("Type 'exit' to quit, 'reset' to clear history, 'system <text>' to change persona.\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "exit":
            break
        if user_input == "reset":
            history.clear()
            print("(history cleared)")
            continue
        if user_input.startswith("system "):
            system = user_input[len("system "):].strip()
            history.clear()
            print(f"(new system prompt; history cleared)\n  {system}")
            continue

        history.append(ChatTurn(role="user", content=user_input))
        prompt_text = template.render(system, history, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, allow_special=True)

        # Trim if longer than model context.
        max_ctx = model.config.max_seq_len - args.max_tokens
        if len(prompt_ids) > max_ctx:
            prompt_ids = prompt_ids[-max_ctx:]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if 0.0 < args.top_p < 1.0 else None,
            stop_token_ids=stop_ids,
        )

        new_ids = out[0, len(prompt_ids):].tolist()
        new_text = tokenizer.decode(new_ids)
        reply = template.extract_reply(new_text)
        if not reply:
            reply = new_text.strip() or "*whirr* (I went blank for a second)"

        history.append(ChatTurn(role="nyx", content=reply))
        print(f"nyx > {reply}\n")


if __name__ == "__main__":
    main()
