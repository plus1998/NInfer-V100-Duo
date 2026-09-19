#!/usr/bin/env python3
"""Build a distinct-text (non-tiling) English haystack corpus for the 1M needle suite.

Why this exists
---------------
EvalScope's `needle_haystack` adapter builds a haystack by tokenizing one corpus file and,
if the result is shorter than the requested context length, *concatenating the file to
itself until it is long enough* (`_get_context_tokens` in
`evalscope/benchmarks/needle_haystack/needle_haystack_adapter.py`).  The stock ModelScope
corpus is 644,100 bytes of Paul Graham essays -- roughly 150-160k tokens -- so every
context length above that is served by a **tiled** haystack.  At 1,048,576 tokens the stock
corpus would repeat about 6.5 times.  Retrieval over tiled text is mechanically valid (the
needle is still unique) but it is an easier task than retrieval over novel text, so it is
not an acceptable primary measurement for the 1M gate.

This script downloads a fixed list of Project Gutenberg public-domain English books, strips
the Project Gutenberg header/licence boilerplate, and concatenates them into two disjoint
bundles, each long enough that the adapter never tiles at 1,048,576 tokens.

Adapter file-name constraint
----------------------------
`NeedleHaystackAdapter.load()` hardcodes

    file_structure = {'english': ['PaulGraham_Essays.txt'],
                      'chinese': ['Journey_to_the_West.txt']}

so the two subset names and the two file names are not configurable.  The bundles are
therefore written under those exact file names:

    PaulGraham_Essays.txt     <- bundle A (subset id `english`)
    Journey_to_the_West.txt   <- bundle B (subset id `chinese`)

Neither file contains Paul Graham essays or Journey to the West, and bundle B is English,
not Chinese.  The names are the adapter's, the content is this script's; `provenance.json`
records what is actually in each file.  Using both subsets this way is what gives the suite
two *independent* samples per depth: two disjoint book sets, same needle, same depth.

Usage
-----
    python3 tools/tp2/build_needle_1m_corpus.py \
        --out-dir /home/pc/models/ninfer-38/needle-1m \
        --tokenizer /home/pc/models/ninfer-38/unsloth-nvfp4

The tokenizer is used only to *verify* that each bundle exceeds `--min-tokens`; pass
`--no-verify` to skip it (then the no-tiling property is unproven).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


@dataclass(frozen=True)
class Book:
    gid: int
    title: str
    author: str


# Two disjoint sets of Project Gutenberg public-domain English texts.  Ordering is fixed so
# that a truncation at --max-chars is reproducible.
BUNDLE_A: tuple[Book, ...] = (
    Book(2600, "War and Peace", "Leo Tolstoy"),
    Book(1400, "Great Expectations", "Charles Dickens"),
    Book(1342, "Pride and Prejudice", "Jane Austen"),
    Book(84, "Frankenstein; Or, The Modern Prometheus", "Mary Wollstonecraft Shelley"),
    Book(1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    Book(174, "The Picture of Dorian Gray", "Oscar Wilde"),
    Book(120, "Treasure Island", "Robert Louis Stevenson"),
    Book(2814, "Dubliners", "James Joyce"),
    Book(36, "The War of the Worlds", "H. G. Wells"),
    Book(16, "Peter Pan", "J. M. Barrie"),
    Book(219, "Heart of Darkness", "Joseph Conrad"),
    Book(35, "The Time Machine", "H. G. Wells"),
    Book(11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    Book(43, "The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    Book(5200, "Metamorphosis", "Franz Kafka"),
    Book(1080, "A Modest Proposal", "Jonathan Swift"),
)

BUNDLE_B: tuple[Book, ...] = (
    Book(2554, "Crime and Punishment", "Fyodor Dostoyevsky"),
    Book(1260, "Jane Eyre: An Autobiography", "Charlotte Bronte"),
    Book(514, "Little Women", "Louisa May Alcott"),
    Book(6130, "The Iliad", "Homer"),
    Book(158, "Emma", "Jane Austen"),
    Book(345, "Dracula", "Bram Stoker"),
    Book(98, "A Tale of Two Cities", "Charles Dickens"),
    Book(768, "Wuthering Heights", "Emily Bronte"),
    Book(1727, "The Odyssey", "Homer"),
    Book(205, "Walden, and On The Duty Of Civil Disobedience", "Henry David Thoreau"),
    Book(76, "Adventures of Huckleberry Finn", "Mark Twain"),
    Book(2591, "Grimms' Fairy Tales", "Jacob and Wilhelm Grimm"),
    Book(74, "The Adventures of Tom Sawyer", "Mark Twain"),
    Book(1998, "Thus Spake Zarathustra", "Friedrich Nietzsche"),
    Book(408, "The Souls of Black Folk", "W. E. B. Du Bois"),
    Book(1232, "The Prince", "Nicolo Machiavelli"),
)

# The two bundles must share no book: they are the suite's two independent samples per depth, and
# an overlap would silently make them correlated. Checked at import, not in a test, so no way to
# run this script with a bad list.
_A_IDS = {book.gid for book in BUNDLE_A}
_B_IDS = {book.gid for book in BUNDLE_B}
assert len(_A_IDS) == len(BUNDLE_A), f"bundle A repeats a Gutenberg id: {sorted(_A_IDS)}"
assert len(_B_IDS) == len(BUNDLE_B), f"bundle B repeats a Gutenberg id: {sorted(_B_IDS)}"
assert not (_A_IDS & _B_IDS), f"bundles A and B share Gutenberg id(s) {sorted(_A_IDS & _B_IDS)}"

# bundle key -> (adapter file name, adapter subset id, book list)
BUNDLES = {
    "bundle-a": ("PaulGraham_Essays.txt", "english", BUNDLE_A),
    "bundle-b": ("Journey_to_the_West.txt", "chinese", BUNDLE_B),
}

_START_RE = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
_END_RE = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)


def fetch(book: Book, cache_dir: Path, pause: float) -> str:
    """Download one book (cached on disk) and return its raw text."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pg{book.gid}.txt"
    if not path.exists():
        url = GUTENBERG_URL.format(id=book.gid)
        print(f"  fetching {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "ninfer-needle-corpus/1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        path.write_bytes(raw)
        time.sleep(pause)
    return path.read_text(encoding="utf-8", errors="replace")


def strip_gutenberg(text: str, book: Book) -> str:
    """Remove the Project Gutenberg header and licence footer.

    Both markers are mandatory: a book whose boilerplate cannot be located is rejected
    rather than silently contributing licence text to the haystack.
    """
    start = _START_RE.search(text)
    end = _END_RE.search(text)
    if start is None or end is None:
        raise SystemExit(
            f"pg{book.gid} ({book.title}): Project Gutenberg START/END markers not found; "
            "refusing to include unstripped boilerplate"
        )
    body = text[start.end():end.start()]
    # Gutenberg files use CRLF; normalise so char offsets are stable across platforms.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ blank lines, which are frequent in the transcriptions.
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def truncate_at_paragraph(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    return text[:cut].rstrip()


def build_bundle(
    books: tuple[Book, ...],
    cache_dir: Path,
    max_chars: int,
    pause: float,
) -> tuple[str, list[dict]]:
    parts: list[str] = []
    manifest: list[dict] = []
    used = 0
    for book in books:
        record = dict(asdict(book))
        record["url"] = GUTENBERG_URL.format(id=book.gid)
        if used >= max_chars:
            record["status"] = "not-reached"
            record["chars_contributed"] = 0
            manifest.append(record)
            continue
        body = strip_gutenberg(fetch(book, cache_dir, pause), book)
        record["chars_stripped"] = len(body)
        remaining = max_chars - used
        if len(body) > remaining:
            body = truncate_at_paragraph(body, remaining)
            record["status"] = "truncated"
        else:
            record["status"] = "full"
        record["chars_contributed"] = len(body)
        parts.append(body)
        used += len(body) + 2  # the "\n\n" join
        manifest.append(record)
    return "\n\n".join(parts) + "\n", manifest


def count_tokens(tokenizer_path: str, text: str) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return len(tok.encode(text, add_special_tokens=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/home/pc/models/ninfer-38/needle-1m")
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="where the raw Gutenberg downloads are kept (default: <out-dir>/.gutenberg-cache)",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=6_000_000,
        help="character budget per bundle (default 6,000,000 -- about 1.4M Qwen tokens, "
        "35%% above the 1,048,576-token ceiling)",
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=1_100_000,
        help="each bundle must tokenize to at least this many tokens, otherwise the adapter "
        "would tile it at the 1M tier (default 1,100,000)",
    )
    ap.add_argument("--tokenizer", default="/home/pc/models/ninfer-38/unsloth-nvfp4")
    ap.add_argument("--no-verify", action="store_true", help="skip the token-count verification")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between downloads")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / ".gutenberg-cache"

    provenance = {
        "purpose": "distinct-text (non-tiling) English haystack for the Qwen3.8-27B 1M needle suite",
        "source": "Project Gutenberg (public domain), plain-text UTF-8, "
        "header/licence boilerplate stripped between the *** START/END OF THE PROJECT "
        "GUTENBERG EBOOK *** markers",
        "generator": "tools/tp2/build_needle_1m_corpus.py",
        "max_chars_per_bundle": args.max_chars,
        "min_tokens_per_bundle": args.min_tokens,
        "tokenizer": args.tokenizer,
        "adapter_note": (
            "File names are dictated by evalscope's NeedleHaystackAdapter.load(), which "
            "hardcodes english->PaulGraham_Essays.txt and chinese->Journey_to_the_West.txt. "
            "Both bundles here are English public-domain books; the file names carry no "
            "content meaning."
        ),
        "bundles": {},
    }

    for key, (filename, subset, books) in BUNDLES.items():
        print(f"building {key} -> {filename} (subset id `{subset}`)", flush=True)
        text, manifest = build_bundle(books, cache_dir, args.max_chars, args.pause)
        path = out_dir / filename
        path.write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entry = {
            "file": filename,
            "evalscope_subset": subset,
            "language": "english",
            "chars": len(text),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "books": manifest,
        }
        if not args.no_verify:
            ntok = count_tokens(args.tokenizer, text)
            entry["tokens"] = ntok
            entry["chars_per_token"] = round(len(text) / ntok, 4)
            print(f"  {filename}: {len(text):,} chars, {ntok:,} tokens "
                  f"({len(text)/ntok:.3f} chars/token)", flush=True)
            if ntok < args.min_tokens:
                raise SystemExit(
                    f"{filename} tokenizes to {ntok:,} tokens, below --min-tokens "
                    f"{args.min_tokens:,}: the adapter would TILE this bundle at the 1M "
                    "tier. Raise --max-chars or add books."
                )
        else:
            print(f"  {filename}: {len(text):,} chars (token count not verified)", flush=True)
        provenance["bundles"][key] = entry

    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
