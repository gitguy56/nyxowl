"""Training loop with cosine LR schedule and checkpoint management."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .model import NyxOwl


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Minimal training loop for NyxOwl."""

    def __init__(self, model: NyxOwl, config: TrainConfig) -> None:
        self.config = config
        self.step = 0
        self.device = _resolve_device(config.device)

        self.model = model.to(self.device)

        # Fused AdamW is meaningfully faster on CUDA — single kernel for the
        # whole optimiser step instead of one per param tensor.
        adamw_kwargs = dict(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        if self.device.type == "cuda":
            adamw_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

        # Mixed-precision setup. On modern NVIDIA GPUs (Ampere / Ada / Blackwell)
        # bf16 has fp32-like dynamic range, so no GradScaler is needed.
        self.use_amp = self.device.type == "cuda"
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype: torch.dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )

        if self.device.type == "cuda":
            # TF32 matmul + cuDNN autotuner pick the fastest kernel per shape.
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

            # torch.compile fuses the transformer graph and gives a sizeable
            # speedup on Ada / Blackwell. Falls back silently if compile fails
            # (e.g. unsupported Triton on Windows-old, missing C compiler).
            if getattr(config, "use_compile", True):
                try:
                    self.model = torch.compile(self.model, mode="default")
                    print("torch.compile: enabled (mode=default)")
                except Exception as e:  # noqa: BLE001
                    print(f"torch.compile: disabled ({e})")

    # ------------------------------------------------------------------
    # Learning-rate schedule
    # ------------------------------------------------------------------

    def _lr(self, step: int) -> float:
        cfg = self.config
        if step < cfg.warmup_steps:
            # Linear warmup
            return cfg.learning_rate * max(step, 1) / cfg.warmup_steps
        # Cosine decay to 10 % of peak LR
        progress = (step - cfg.warmup_steps) / max(
            1, cfg.max_steps - cfg.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.learning_rate * (0.1 + 0.9 * cosine)

    def _apply_lr(self, step: int) -> float:
        lr = self._lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_loss(
        self, loader: DataLoader, max_batches: int = 20
    ) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for x, y in loader:
            if count >= max_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                _, loss = self.model(x, y)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        cfg = self.config
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.model.train()
        train_iter = iter(train_loader)
        t0 = time.perf_counter()

        n_params = self.model.num_parameters()
        amp_label = (
            f"amp={str(self.amp_dtype).split('.')[-1]}" if self.use_amp else "amp=off"
        )
        print(
            f"NyxOwl | {n_params:,} parameters | device: {self.device} | "
            f"steps: {cfg.max_steps} | batch: {cfg.batch_size} | {amp_label}"
        )

        accum = max(1, cfg.grad_accum_steps)
        for step in range(cfg.max_steps):
            lr = self._apply_lr(step)
            self.optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation: sum gradients over `accum` micro-batches
            # before stepping. Effective batch = batch_size * grad_accum_steps.
            running_loss = 0.0
            for _ in range(accum):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    _, loss = self.model(x, y)

                loss = loss / accum
                self.scaler.scale(loss).backward()
                running_loss += loss.item()

            if cfg.grad_clip > 0.0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step = step
            loss_value = running_loss

            if (step + 1) % cfg.eval_interval == 0 or step == 0:
                elapsed = time.perf_counter() - t0
                interval = cfg.eval_interval if step > 0 else 1
                steps_per_sec = interval / max(elapsed, 1e-6)
                eta_steps = cfg.max_steps - (step + 1)
                eta_h = eta_steps / max(steps_per_sec * 3600, 1e-6)
                val_loss = (
                    self.estimate_loss(val_loader) if val_loader else float("nan")
                )
                print(
                    f"step {step + 1:5d}/{cfg.max_steps} | "
                    f"train {loss_value:.4f} | "
                    f"val {val_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{steps_per_sec:.2f} steps/s | "
                    f"ETA {eta_h:.1f}h",
                    flush=True,
                )
                t0 = time.perf_counter()

            if (step + 1) % cfg.save_interval == 0:
                self._save(step + 1)

        self._save("final")
        print("Training complete.")

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save(self, tag: int | str) -> None:
        path = Path(self.config.checkpoint_dir) / f"ckpt_{tag}.pt"
        torch.save(
            {
                "step": self.step,
                "model_config": self.model.config,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  Saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # If the model was wrapped by torch.compile, load_state_dict still
        # works because OptimizedModule forwards it to the underlying module.
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = ckpt.get("step", 0)
        print(f"Loaded checkpoint '{path}' at step {self.step}")
