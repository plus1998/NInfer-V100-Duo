#!/usr/bin/env python3
"""Deterministic prompt generator for the tp1 golden-token identity gate.

The three prompts exercise three prefill regimes -- a short chat turn, a
multi-paragraph instruction around 2k tokens, and a document around 30k tokens
that crosses several prefill chunks and many KV pages. They are generated
rather than checked in so the gate's inputs are reproducible from a rule
instead of from a large blob, and so a reader can see there is nothing
adversarial or model-specific in them.

Determinism: no randomness source other than the fixed LCG below, no clock, no
locale-dependent formatting, ASCII only, "\n" line endings. Running this on any
machine with any Python 3 produces byte-identical files.

The prompts are emitted as --messages JSON files rather than passed with
--prompt because the 30k-token case is ~158 KB of text, past the shell's
argument-length limit.

Usage: python3 make_prompts.py <output-directory>
Writes case-1-short.json, case-2-instruction.json, case-3-document.json.
"""

import json
import pathlib
import sys

# Numerical Recipes' LCG constants, evaluated in 32-bit. Chosen only because
# the sequence is short to state and identical everywhere; nothing here depends
# on its statistical quality.
_MULTIPLIER = 1664525
_INCREMENT = 1013904223
_MODULUS = 1 << 32


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed % _MODULUS

    def next(self, bound: int) -> int:
        self.state = (self.state * _MULTIPLIER + _INCREMENT) % _MODULUS
        return (self.state >> 16) % bound


WORDS = [
    "buffer", "cache", "commit", "device", "engine", "frame", "graph", "handle",
    "index", "journal", "kernel", "layer", "mapping", "node", "offset", "page",
    "quantum", "record", "shard", "table", "unit", "vector", "window", "zone",
    "aligned", "bounded", "chunked", "dense", "eager", "fused", "greedy", "hidden",
    "inline", "joint", "keyed", "linear", "masked", "nested", "opaque", "paged",
    "queued", "resident", "sparse", "tiled", "unique", "valid", "warm", "zeroed",
]

SENTENCE_STARTS = [
    "The", "Each", "Every", "This", "That", "Any", "One", "Another",
]


def sentence(rng: Lcg) -> str:
    head = SENTENCE_STARTS[rng.next(len(SENTENCE_STARTS))]
    body = " ".join(WORDS[rng.next(len(WORDS))] for _ in range(rng.next(9) + 8))
    return f"{head} {body}."


def paragraph(rng: Lcg, number: int, sentences: int) -> str:
    text = " ".join(sentence(rng) for _ in range(sentences))
    return f"Section {number}. {text}"


def short_prompt() -> str:
    return "Explain in three sentences why merge sort is stable."


def instruction_prompt() -> str:
    rng = Lcg(20260829)
    lines = [
        "You are reviewing a written specification for a storage subsystem.",
        "The specification is reproduced below as a numbered requirement list.",
        "Read all of it, then answer the question at the end in one paragraph.",
        "",
    ]
    for number in range(1, 61):
        lines.append(f"Requirement {number}: {sentence(rng)} {sentence(rng)}")
    lines += [
        "",
        "Question: which single requirement above most constrains the others, and why?",
    ]
    return "\n".join(lines)


def document_prompt() -> str:
    rng = Lcg(20260830)
    lines = [
        "Below is an internal engineering document. Read it, then answer the",
        "question that follows it.",
        "",
    ]
    for number in range(1, 301):
        lines.append(paragraph(rng, number, 6))
        lines.append("")
    lines += [
        "Question: summarize, in three sentences, what kind of document this is",
        "and how it is organized.",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: make_prompts.py <output-directory>\n")
        return 2
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("case-1-short.json", short_prompt()),
        ("case-2-instruction.json", instruction_prompt()),
        ("case-3-document.json", document_prompt()),
    ):
        # sort_keys + a fixed separator so the serialization is stable across
        # Python versions; the file is compared by sha256 in MANIFEST.md.
        body = json.dumps([{"role": "user", "content": text}],
                          ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"))
        (out / name).write_text(body + "\n", encoding="ascii", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
