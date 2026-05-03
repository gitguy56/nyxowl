"""Smoke tests for the BPE tokenizer."""

import json
import tempfile
from pathlib import Path

import pytest

from nyxowl.tokenizer import BPETokenizer

CORPUS = "hello world " * 500 + "the quick brown fox jumps over the lazy dog " * 200


def make_tokenizer(vocab_size: int = 300) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=vocab_size)
    return tok


def test_vocab_size():
    tok = make_tokenizer(300)
    assert tok.vocab_size == 300


def test_roundtrip():
    tok = make_tokenizer(300)
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text


def test_encode_produces_fewer_ids_than_bytes():
    tok = make_tokenizer(300)
    text = "hello world hello world"
    ids = tok.encode(text)
    assert len(ids) < len(text.encode("utf-8"))


def test_save_load_roundtrip(tmp_path):
    tok = make_tokenizer(300)
    path = tmp_path / "tok.json"
    tok.save(str(path))

    tok2 = BPETokenizer.load(str(path))
    text = "hello world"
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.decode(tok2.encode(text)) == text


def test_encode_requires_training():
    tok = BPETokenizer()
    with pytest.raises(RuntimeError):
        tok.encode("hello")
