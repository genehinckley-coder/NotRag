"""Plan item 1: does the corpus fit in context?

This is the architecture decision gate, so it runs before anything real is
built. Extraction here is deliberately crude -- a bare ``text_content()`` pass.
Structure-preserving extraction (item 3) adds newlines; it does not change the
order of magnitude, so a crude count lands within a few percent of the real one.

Uses ``client.messages.count_tokens``, never ``tiktoken``. tiktoken is installed
on this machine and is the tempting shortcut, but it is an OpenAI tokenizer and
its counts do not apply to Claude.

    python fit_check.py
"""

from __future__ import annotations

import sys

import lxml.html
from lxml import etree

from config import (
    CORPUS_DIR,
    CORPUS_GLOBS,
    DOCUMENT_LABELS,
    MODEL,
    TOKEN_CEILING,
)
from credentials import build_client


def crude_text(path) -> str:
    """Strip script/style and return the document's raw text content."""
    parser = lxml.html.HTMLParser(no_network=True, remove_comments=True)
    # Parse bytes, not str, so the file's own meta charset is honored.
    doc = lxml.html.fromstring(path.read_bytes(), parser=parser)
    etree.strip_elements(doc, "script", "style", with_tail=False)
    return doc.text_content()


def main() -> int:
    paths = sorted(
        {p for glob in CORPUS_GLOBS for p in CORPUS_DIR.glob(glob)},
        key=lambda p: p.name,
    )
    if not paths:
        print(f"No corpus files found in {CORPUS_DIR}", file=sys.stderr)
        return 2

    texts: dict[str, str] = {}
    for path in paths:
        texts[path.name] = crude_text(path)

    corpus = "\n\n".join(texts[p.name] for p in paths)
    total_chars = len(corpus)

    print(f"Corpus: {CORPUS_DIR}")
    print(f"Files : {len(paths)}\n")

    header = f"{'file':<34}{'bytes':>11}{'chars':>11}{'est tok':>10}"
    print(header)
    print("-" * len(header))

    client = build_client()
    total_tokens = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": corpus}],
    ).input_tokens

    for path in paths:
        chars = len(texts[path.name])
        share = chars / total_chars if total_chars else 0.0
        label = DOCUMENT_LABELS.get(path.name, path.name)
        print(
            f"{label.split(':')[0]:<34}"
            f"{path.stat().st_size:>11,}"
            f"{chars:>11,}"
            f"{round(total_tokens * share):>10,}"
        )

    print("-" * len(header))
    print(f"{'TOTAL':<34}{'':>11}{total_chars:>11,}{total_tokens:>10,}")
    print("\n(Per-file token figures are apportioned by character share; only")
    print(" the TOTAL is a measured count_tokens result.)")

    pct = total_tokens / TOKEN_CEILING * 100
    print(f"\nMeasured : {total_tokens:,} tokens")
    print(f"Ceiling  : {TOKEN_CEILING:,} tokens  ({pct:.1f}% used)")
    print(f"Ratio    : {total_chars / total_tokens:.2f} chars/token")

    if total_tokens <= TOKEN_CEILING:
        print("\nGATE: PASS -- proceed with stuff-and-cache (items 2 and 3).")
        return 0

    print(
        "\nGATE: FAIL -- corpus exceeds the ceiling. Do not truncate.\n"
        "The agentic glob/grep/read loop becomes the right architecture and\n"
        "items 5-7 change substantially. Stop and re-plan."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
