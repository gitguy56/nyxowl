"""Smoke tests for the NyxOwl transformer model."""

import torch
import pytest

from nyxowl.config import ModelConfig
from nyxowl.model import NyxOwl

TINY_CFG = ModelConfig(
    vocab_size=256,
    d_model=64,
    n_heads=4,
    n_layers=2,
    ffn_dim=128,
    max_seq_len=32,
    dropout=0.0,
)


def test_forward_shape():
    model = NyxOwl(TINY_CFG)
    model.eval()
    B, T = 2, 16
    idx = torch.randint(0, TINY_CFG.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, TINY_CFG.vocab_size)
    assert loss is None


def test_forward_with_targets():
    model = NyxOwl(TINY_CFG)
    model.eval()
    B, T = 2, 16
    idx = torch.randint(0, TINY_CFG.vocab_size, (B, T))
    targets = torch.randint(0, TINY_CFG.vocab_size, (B, T))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0


def test_generate():
    model = NyxOwl(TINY_CFG)
    model.eval()
    idx = torch.zeros(1, 1, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 11)


def test_weight_tying():
    model = NyxOwl(TINY_CFG)
    assert model.lm_head.weight is model.token_emb.weight


def test_num_parameters():
    model = NyxOwl(TINY_CFG)
    assert model.num_parameters() > 0


def test_seq_len_overflow():
    model = NyxOwl(TINY_CFG)
    idx = torch.zeros(1, TINY_CFG.max_seq_len + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        model(idx)
