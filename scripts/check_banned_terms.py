#!/usr/bin/env python3
"""Fail if banned terminology appears in tracked text.

The revised manuscript retracts the language of statistical independence and
carries no theory-program branding. This check keeps README, docs, config, and
source comments consistent with that decision. Run in CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BANNED = [
    "statistical independence",
    "statistically independent",
    "independent axes",
    "natural orthogonality",
    "orthogonal baseline",
    "quasi-orthogonal",
    "projective theory",
    "projectivity of intelligence",
    "optics of intelligence",
    "relativity of intelligence",
]
# Word-boundary tokens that are too short to match as substrings safely.
BANNED_TOKENS = ["O-1", "O-2", "O-3", "O-4", "PTI"]

SCAN_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".toml", ".cff", ".txt"}
SKIP_DIRS = {".git", "__pycache__", "data", "figures"}
SELF = Path(__file__).name


def iter_files(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in SCAN_SUFFIXES and p.name != SELF:
            yield p


def scan(root: Path) -> list[str]:
    hits = []
    patt = [(re.compile(re.escape(b), re.IGNORECASE), b) for b in BANNED]
    tok = [(re.compile(rf"\b{re.escape(t)}\b"), t) for t in BANNED_TOKENS]
    for f in iter_files(root):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for rx, label in patt + tok:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{f}:{line}: banned term '{label}'")
    return hits


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    hits = scan(root)
    if hits:
        print("Banned terminology found:")
        print("\n".join(hits))
        return 1
    print("Vocabulary clean: no banned terms found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
