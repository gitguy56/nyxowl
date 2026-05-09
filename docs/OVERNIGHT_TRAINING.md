# Overnight Training Guide

For training NyxOwl-Medium (~150 M params) overnight on a laptop / desktop
without keeping you awake.

## Before you start

### 1. Cap GPU power (quieter fans)

Open **Command Prompt as Administrator** and run:

```cmd
nvidia-smi -pl 250
```

The 5070 Ti's allowed range is 250-330 W. Capping to 250 W cuts heat
output noticeably and lets the fans run slower. Performance hit is
typically only 5-10 %. **Resets to default on reboot.**

Verify:

```cmd
nvidia-smi -q -d POWER
```

### 2. Custom fan curve (optional, even quieter)

Install [MSI Afterburner](https://www.msi.com/Landing/afterburner/graphics-cards),
go to Settings → Fan → Enable user defined fan curve, and cap fan speed at
~60 % above 70 °C. Without this the fans will ramp into jet-engine
territory.

### 3. Disable Windows sleep + reboots

- Settings → System → Power → Sleep: **Never** (when plugged in)
- Settings → Windows Update → Advanced options → "Restart this device as
  soon as possible": **Off**
- Set Active Hours to 12 AM–12 AM so updates can't reboot mid-training
- Settings → Sound → System sounds: **No Sounds** (no surprise beeps)

### 4. Unplug or mute speakers

Belt and braces.

## Running training

### Step 1 — Pull the latest code

```cmd
cd C:\Users\chill\Desktop\nyxowl
git pull origin claude/create-nyxowl-transformer-XFV2U
pip install -e .   # picks up new dependencies (numpy)
```

### Step 2 — Download data (once, ~30 min)

```cmd
python -m nyxowl.scripts.prep_data --output data\corpus
```

This downloads ~250 MB of public-domain English text (Project Gutenberg
books + WikiText-103). No HuggingFace account needed.

### Step 3 — Synthesise persona examples (seconds)

```cmd
python -m nyxowl.scripts.synth_persona --output data\persona.txt --count 10000
```

### Step 4 — Train (overnight, ~6-8 hrs)

```cmd
python -m nyxowl.scripts.train ^
  --data data\corpus\merged.txt ^
  --output runs\nyxowl_medium ^
  --preset medium ^
  --vocab-size 16384 ^
  --with-chat-tokens ^
  --batch-size 32 ^
  --grad-accum-steps 1 ^
  --num-workers 2 ^
  --max-steps 60000 ^
  --lr 3e-4 ^
  --warmup-steps 1000 ^
  --eval-interval 500 ^
  --save-interval 2000
```

Effective batch size: 32. The 5070 Ti has 16 GB so you can fit the full
batch directly without gradient accumulation — every "step" you see is a
real optimiser step. If VRAM overflows, drop to `--batch-size 24` first
before resorting to `--grad-accum-steps`.

Speed knobs already on by default:
- `torch.compile` (graph fusion, ~30 % faster). If it errors on first
  step, add `--no-compile`.
- Fused AdamW, cuDNN autotune, TF32 matmul, bf16 mixed precision,
  Flash Attention.
- Tokenised corpus is cached to `runs\nyxowl_medium\tokens_*.bin` after
  the first run. Resuming or re-running skips encoding entirely.

Save every 2000 steps so a crash costs at most ~20 minutes of progress.

### Step 5 — Persona fine-tune (optional, 30-90 min, run after morning coffee)

After step 4 finishes, fine-tune the same model on the persona data so it
actually responds as NyxOwl:

```cmd
python -m nyxowl.scripts.train ^
  --data data\persona.txt ^
  --output runs\nyxowl_medium ^
  --preset medium ^
  --batch-size 16 ^
  --max-steps 3000 ^
  --lr 5e-5 ^
  --warmup-steps 100 ^
  --resume runs\nyxowl_medium\checkpoints\ckpt_final.pt
```

The lower learning rate (5e-5 vs 3e-4) prevents the model from forgetting
its general English skills while it learns the persona format.

### Step 6 — Chat with NyxOwl

```cmd
python -m nyxowl.scripts.chat ^
  --checkpoint runs\nyxowl_medium\checkpoints\ckpt_final.pt ^
  --tokenizer  runs\nyxowl_medium\tokenizer.json
```

Type `exit` to quit, `reset` to clear chat history, or
`system <text>` to switch persona on the fly.

## If something goes wrong overnight

Resume from the most recent checkpoint:

```cmd
python -m nyxowl.scripts.train ^
  --data data\corpus\merged.txt ^
  --output runs\nyxowl_medium ^
  --preset medium ^
  --resume runs\nyxowl_medium\checkpoints\ckpt_<N>.pt ^
  ... (same other flags)
```

The optimiser state and step count are restored, so the LR schedule picks
up where it left off.
