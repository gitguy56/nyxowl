# NyxOwl

A small transformer language model and BPE tokenizer implemented **from scratch** in PyTorch — designed to be readable, hackable, and trainable on a laptop GPU.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org)

---

## Features

- **BPE tokenizer** — pure-Python Byte-Pair Encoding, no external tokenizer libraries
- **Transformer LM** — decoder-only architecture written from scratch (no `nn.Transformer`)
- **Laptop-friendly** — default config is ~7 M parameters, fits in 4 GB VRAM
- **Clean codebase** — designed for learning and experimentation

## Architecture

| Component | Details |
|---|---|
| Tokenizer | BPE on raw UTF-8 bytes |
| Embeddings | Learned token + position embeddings, weight-tied with LM head |
| Attention | Causal multi-head self-attention, fused QKV projection |
| Norm | Pre-LayerNorm residual connections |
| FFN | Linear → GELU → Linear |
| LR schedule | Linear warmup + cosine decay |

### Default model size

```
vocab_size  : 4096   (set by tokenizer)
d_model     : 256
n_heads     : 8      (d_head = 32)
n_layers    : 6
ffn_dim     : 1024
max_seq_len : 256
parameters  : ~7 M
```

Reduce `n_layers` or `d_model` to fit smaller GPUs; increase them for a more capable model.

---

## Installation

```bash
git clone https://github.com/gitguy56/nyxowl.git
cd nyxowl
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1

---

## Quick start

### 1 — Prepare data

Any plain-text file works. A good starting point is
[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt):

```bash
curl -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 2 — Train

```bash
nyxowl-train \
  --data data/shakespeare.txt \
  --output runs/shakespeare \
  --vocab-size 4096 \
  --max-steps 5000 \
  --batch-size 32
```

Or run the script directly:

```bash
python -m nyxowl.scripts.train \
  --data data/shakespeare.txt \
  --output runs/shakespeare
```

Training prints a log line every `--eval-interval` steps:

```
NyxOwl | 7,104,512 parameters | device: cuda | steps: 5000 | batch: 32
step     1/5000 | train 8.3241 | val 8.3189 | lr 1.50e-06 | 0.1s
step   200/5000 | train 3.1402 | val 3.2801 | lr 3.00e-04 | 18.3s
step   400/5000 | train 2.7654 | val 2.8913 | lr 2.94e-04 | 17.9s
...
step  5000/5000 | train 1.8341 | val 2.1057 | lr 3.00e-05 | 18.1s
```

### 3 — Generate

```bash
nyxowl-generate \
  --checkpoint runs/shakespeare/checkpoints/ckpt_final.pt \
  --tokenizer  runs/shakespeare/tokenizer.json \
  --prompt "To be or not to be" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

---

## Project structure

```
nyxowl/
├── nyxowl/
│   ├── __init__.py        public API
│   ├── config.py          ModelConfig and TrainConfig dataclasses
│   ├── tokenizer.py       BPE tokenizer (from scratch)
│   ├── model.py           Transformer LM (from scratch)
│   ├── dataset.py         TextDataset + DataLoader helpers
│   ├── trainer.py         training loop, LR schedule, checkpoints
│   └── scripts/
│       ├── train.py       nyxowl-train entry point
│       └── generate.py    nyxowl-generate entry point
├── tests/
│   ├── test_tokenizer.py
│   └── test_model.py
├── pyproject.toml
├── LICENSE                Apache 2.0
└── README.md
```

---

## Training tips

| Goal | Change |
|---|---|
| Faster training | Increase `--batch-size`, use `--device cuda` |
| Larger context | Increase `--seq-len` (memory scales quadratically) |
| Better quality | Increase `--n-layers`, `--d-model`, more `--max-steps` |
| Less overfitting | Increase `--dropout`, add more data |
| Resume training | Pass `--resume runs/.../checkpoints/ckpt_N.pt` |

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## License

[Apache 2.0](LICENSE)
