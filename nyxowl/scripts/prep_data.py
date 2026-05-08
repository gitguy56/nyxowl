"""Download and prepare pretraining data for NyxOwl.

Pulls a mix of high-quality general English text without requiring any
external account:

  - Project Gutenberg books   (public domain literature)
  - Simple English Wikipedia  (encyclopedic facts, simpler prose)
  - WikiText-103              (curated long-form Wikipedia, no auth needed)
  - tatoeba sentences         (short well-formed sentences, optional)

We deliberately avoid datasets that require a HuggingFace token. If you
have a HuggingFace account and want better data later, swap in FineWeb-Edu
or RedPajama by editing ``DEFAULT_SOURCES``.

Usage:
    python -m nyxowl.scripts.prep_data --output data\\corpus

Outputs:
    data\\corpus\\raw\\*.txt         downloaded files
    data\\corpus\\merged.txt        single concatenated text file
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Source list — each entry is (label, url, format, post-process)
# ---------------------------------------------------------------------------

DEFAULT_SOURCES: list[dict] = [
    {
        "name": "wikitext-103",
        "url": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
        "kind": "zip",
        "extract": ["wikitext-103/wiki.train.tokens"],
    },
    {
        "name": "simplewiki",
        # Pre-extracted plain text dump of Simple English Wikipedia
        "url": "https://dumps.wikimedia.org/other/cirrussearch/current/simplewiki-20240101-cirrussearch-content.json.gz",
        "kind": "skip",  # skipped by default — needs JSON parsing, see prep_simplewiki()
    },
    {
        "name": "gutenberg-top100",
        # Project Gutenberg "Top 100 ebooks last 30 days" — we'll stream a
        # curated set of common public-domain books.
        "url": None,
        "kind": "gutenberg",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.rename(dest)
    print(f"    saved → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _unzip_select(zip_path: Path, members: list[str], dest_dir: Path) -> list[Path]:
    out: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in members:
            target = dest_dir / Path(member).name
            if target.exists():
                out.append(target)
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
            print(f"    extracted {member} → {target.name}")
    return out


# ---------------------------------------------------------------------------
# Project Gutenberg (curated set of public-domain books)
# ---------------------------------------------------------------------------

GUTENBERG_BOOKS: list[tuple[int, str]] = [
    (1342, "Pride and Prejudice"),
    (84,   "Frankenstein"),
    (1661, "Adventures of Sherlock Holmes"),
    (2701, "Moby Dick"),
    (98,   "A Tale of Two Cities"),
    (174,  "The Picture of Dorian Gray"),
    (76,   "Adventures of Huckleberry Finn"),
    (345,  "Dracula"),
    (1080, "A Modest Proposal"),
    (74,   "Adventures of Tom Sawyer"),
    (43,   "Dr Jekyll and Mr Hyde"),
    (11,   "Alice in Wonderland"),
    (1232, "The Prince"),
    (2641, "A Room with a View"),
    (1400, "Great Expectations"),
    (16328, "Beowulf"),
    (5200, "Metamorphosis"),
    (1497, "The Republic"),
    (4300, "Ulysses"),
    (158,  "Emma"),
    (768,  "Wuthering Heights"),
    (514,  "Little Women"),
    (10,   "King James Bible"),
    (2554, "Crime and Punishment"),
    (28054, "The Brothers Karamazov"),
]

GUTENBERG_TEMPLATES = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = (
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    )
    end_markers = (
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    )
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[text.find("\n", idx) + 1 :]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def fetch_gutenberg(dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for book_id, title in GUTENBERG_BOOKS:
        target = dest_dir / f"pg{book_id}.txt"
        if target.exists():
            out.append(target)
            continue
        ok = False
        for template in GUTENBERG_TEMPLATES:
            url = template.format(id=book_id)
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    raw = r.read().decode("utf-8", errors="replace")
                cleaned = _strip_gutenberg_boilerplate(raw)
                if len(cleaned) < 5000:
                    continue
                target.write_text(cleaned, encoding="utf-8")
                size_kb = target.stat().st_size / 1024
                print(f"    pg{book_id} {title!r} ({size_kb:.0f} KB)")
                out.append(target)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                continue
        if not ok:
            print(f"    pg{book_id} {title!r}: SKIPPED (could not fetch)")
    return out


# ---------------------------------------------------------------------------
# Wikitext-103
# ---------------------------------------------------------------------------

def fetch_wikitext(dest_dir: Path) -> list[Path]:
    src = DEFAULT_SOURCES[0]
    zip_path = dest_dir / "wikitext-103.zip"
    _download(src["url"], zip_path)
    return _unzip_select(zip_path, src["extract"], dest_dir)


# ---------------------------------------------------------------------------
# Concatenation step
# ---------------------------------------------------------------------------

def merge_text_files(files: list[Path], output: Path, sep: str = "\n\n") -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output, "w", encoding="utf-8") as out:
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as e:  # noqa: BLE001
                print(f"  skipping {fp}: {e}")
                continue
            out.write(content)
            out.write(sep)
            total += len(content)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nyxowl-prep-data",
        description="Download and merge pretraining text for NyxOwl.",
    )
    p.add_argument("--output", default="data/corpus", help="Output directory")
    p.add_argument(
        "--skip-gutenberg",
        action="store_true",
        help="Skip the Project Gutenberg downloads",
    )
    p.add_argument(
        "--skip-wikitext",
        action="store_true",
        help="Skip the WikiText-103 download",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_root = Path(args.output)
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    collected: list[Path] = []

    if not args.skip_wikitext:
        print("\n[1/3] WikiText-103 (~180 MB train split)")
        try:
            collected.extend(fetch_wikitext(raw_dir))
        except Exception as e:  # noqa: BLE001
            print(f"  WikiText fetch failed: {e}")

    if not args.skip_gutenberg:
        print("\n[2/3] Project Gutenberg curated books (~50 MB)")
        gb_dir = raw_dir / "gutenberg"
        try:
            collected.extend(fetch_gutenberg(gb_dir))
        except Exception as e:  # noqa: BLE001
            print(f"  Gutenberg fetch failed: {e}")

    print(f"\n[3/3] Merging {len(collected)} files → merged.txt")
    merged = out_root / "merged.txt"
    nchars = merge_text_files(collected, merged)
    size_mb = merged.stat().st_size / 1e6
    print(f"  merged.txt: {size_mb:.1f} MB ({nchars:,} chars)")

    print(
        "\nNext steps:\n"
        f"  1. Train tokenizer + model:\n"
        f"     python -m nyxowl.scripts.train --data {merged} \\\n"
        f"       --output runs/nyxowl --preset medium \\\n"
        f"       --vocab-size 16384 --max-steps 60000\n"
        f"  2. Synthesise persona data:\n"
        f"     python -m nyxowl.scripts.synth_persona --output {out_root}/persona.txt\n"
        f"  3. Fine-tune on persona, then chat with: python -m nyxowl.scripts.chat\n"
    )


if __name__ == "__main__":
    main()
