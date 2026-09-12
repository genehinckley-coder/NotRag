"""Plan item 3: structure-preserving text extraction from the ILGA HTML corpus.

The subsection structure -- ``a)``, ``1)``, ``A)``, ``i)`` -- is what makes this
corpus answerable, and the obvious one-liner (``" ".join(text.split())``)
collapses every paragraph into one wall of text. So extraction keeps block
structure and collapses whitespace only *within* a line:

* Each paragraph-level element (``<p>``, ``<div>``, headings, ...) becomes one
  paragraph, and paragraphs are separated by exactly one blank line. Empty ones
  are dropped -- including ILGA's U+00A0-only spacer paragraphs, which alternate
  with every content paragraph, so the blank-line rhythm matches the page.
* ``<br>`` is a line break inside a paragraph.
* A table is one paragraph: one line per ``<tr>``, cells joined by `` | ``.
  A cell's content is flattened onto that line even when the cell holds several
  ``<p>`` elements, as ILGA's column headings often do.
* Inside a line, every whitespace run becomes one space. That covers the
  source's CRLF hard wraps and U+00A0: lxml decodes ``&nbsp;`` at parse time, so
  it is the character, not the entity, that has to be handled.

Only ILGA's content container (``div.billtext-scale``) is read. It holds all of
the regulation text; everything around it is site chrome. If the container is
missing, extraction fails rather than falling back to the whole page.

    python extract.py              # per-file summary
    python extract.py --show 175   # print one document's extracted text
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import lxml.html
from lxml import etree

from config import CORPUS_DIR, CORPUS_GLOBS, DOCUMENT_LABELS
from display import escape_controls

#: Class of the one element on an ILGA page that holds the regulation text.
CONTENT_CLASS = "billtext-scale"

#: Elements whose content is never shown to a reader.
_STRIPPED_TAGS = ("script", "style", "template")

#: Elements that open and close a paragraph. Anything else, apart from ``br``
#: and ``table``, is inline: its text joins the surrounding line.
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
        "section", "ul",
        # Table parts found outside a <table> (malformed markup) degrade to
        # paragraphs rather than running together.
        "caption", "tbody", "td", "tfoot", "th", "thead", "tr",
    }
)

_CELL_SEPARATOR = " | "

#: In a str pattern ``\s`` is Unicode-aware: it matches CR, LF, tab, U+00A0 and
#: the other Unicode space characters.
_WHITESPACE = re.compile(r"\s+")


class ExtractionError(ValueError):
    """A corpus file could not be turned into document text."""


@dataclass(frozen=True)
class ExtractedDocument:
    """One corpus file's extracted text, ready for assembly (plan item 4)."""

    filename: str
    #: CACHE-AFFECTING citation title, from ``config.DOCUMENT_LABELS``.
    label: str
    text: str


# --- Corpus discovery --------------------------------------------------------


def discover_corpus_files(
    corpus_dir: Path = CORPUS_DIR, globs: Sequence[str] = CORPUS_GLOBS
) -> list[Path]:
    """Corpus files at the top level of ``corpus_dir``, sorted by filename.

    Top level only: the browser's sibling ``*_files`` folders hold CSS, scripts
    and images, never regulation text.
    """
    found = {path for pattern in globs for path in corpus_dir.glob(pattern) if path.is_file()}
    return sorted(found, key=lambda path: path.name)


def extract_corpus(
    corpus_dir: Path = CORPUS_DIR,
    labels: Mapping[str, str] = DOCUMENT_LABELS,
    globs: Sequence[str] = CORPUS_GLOBS,
) -> list[ExtractedDocument]:
    """Extract every corpus file, in filename order, paired with its label.

    The files on disk and the configured labels must agree exactly. An unlabeled
    file has no acceptable fallback title (labels are cache-affecting, so a
    filename standing in for one would change later), and a label with no file
    means the corpus is incomplete -- which would mean answering confidently from
    a partial source.

    Raises:
        ExtractionError: No files were found, the files and labels disagree, or
            a file could not be extracted.
    """
    paths = discover_corpus_files(corpus_dir, globs)
    if not paths:
        raise ExtractionError(f"No corpus files ({', '.join(globs)}) found in {corpus_dir}")

    names = {path.name for path in paths}
    problems = []
    unlabeled = sorted(names - labels.keys())
    if unlabeled:
        problems.append(f"no label in config.DOCUMENT_LABELS for: {', '.join(unlabeled)}")
    missing = sorted(labels.keys() - names)
    if missing:
        problems.append(f"labeled but not found in {corpus_dir}: {', '.join(missing)}")
    if problems:
        raise ExtractionError("Corpus files and labels disagree -- " + "; ".join(problems))

    return [ExtractedDocument(path.name, labels[path.name], extract_file(path)) for path in paths]


# --- Extraction --------------------------------------------------------------


def extract_file(path: Path) -> str:
    """Extract the regulation text from one saved ILGA page."""
    # Bytes, not str, so the page's own meta charset is honored.
    return extract_html(path.read_bytes(), source=path.name)


def extract_html(html: bytes | str, *, source: str = "<html>") -> str:
    """Extract the regulation text from one ILGA page's HTML.

    Args:
        html: The page. Prefer bytes, so the page's meta charset is honored.
        source: Name used in error messages.

    Returns:
        Paragraphs separated by one blank line, tables as ``" | "``-joined rows.
        Contains no CR and no U+00A0.

    Raises:
        ExtractionError: The page does not have exactly one content container,
            or the container yields no text.
    """
    # lxml's HTML parser does not resolve external entities; no_network also
    # rules out any fetch while parsing.
    parser = lxml.html.HTMLParser(no_network=True, remove_comments=True, remove_pis=True)
    document = lxml.html.document_fromstring(html, parser=parser)
    etree.strip_elements(document, *_STRIPPED_TAGS, with_tail=False)

    containers = document.find_class(CONTENT_CLASS)
    if len(containers) != 1:
        raise ExtractionError(
            f"{source}: expected exactly one element with class {CONTENT_CLASS!r}, found "
            f"{len(containers)}. The ILGA page layout may have changed; extraction does "
            f"not fall back to the whole page, which would pull site chrome into the corpus."
        )

    text = _render_contents(containers[0])
    _check_normalized(text, source)
    return text


def _check_normalized(text: str, source: str) -> None:
    # Explicit checks rather than ``assert``, which ``python -O`` strips.
    if not text:
        raise ExtractionError(f"{source}: the content container yielded no text")
    for char, name in (("\r", "a carriage return"), ("\u00a0", "a U+00A0 no-break space")):
        if char in text:
            raise ExtractionError(
                f"{source}: extracted text still contains {name}; whitespace "
                f"normalization is broken"
            )


class _Paragraphs:
    """Accumulates text as paragraphs made of hard-broken lines."""

    def __init__(self) -> None:
        self._paragraphs: list[str] = []
        self._lines: list[str] = []
        self._fragments: list[str] = []

    def add_text(self, text: str | None) -> None:
        if text:
            self._fragments.append(text)

    def line_break(self) -> None:
        self._lines.append(_WHITESPACE.sub(" ", "".join(self._fragments)).strip())
        self._fragments = []

    def end_paragraph(self) -> None:
        self.line_break()
        lines = [line for line in self._lines if line]
        if lines:
            self._paragraphs.append("\n".join(lines))
        self._lines = []

    def add_paragraph(self, text: str) -> None:
        """Add pre-rendered text, such as a table, as a paragraph of its own."""
        self.end_paragraph()
        if text:
            self._paragraphs.append(text)

    def render(self) -> str:
        self.end_paragraph()
        return "\n\n".join(self._paragraphs)


def _tag(element: etree._Element) -> str | None:
    """The element's tag, or None for comments and processing instructions."""
    return element.tag if isinstance(element.tag, str) else None


def _render_contents(element: etree._Element) -> str:
    """Render everything inside ``element`` (not its tail) as paragraphs."""
    out = _Paragraphs()
    _walk_children(element, out)
    return out.render()


def _walk_children(element: etree._Element, out: _Paragraphs) -> None:
    out.add_text(element.text)
    for child in element:
        _walk(child, out)


def _walk(element: etree._Element, out: _Paragraphs) -> None:
    tag = _tag(element)
    if tag == "br":
        out.line_break()
    elif tag == "table":
        out.add_paragraph(_render_table(element))
    elif tag in _BLOCK_TAGS:
        out.end_paragraph()
        _walk_children(element, out)
        out.end_paragraph()
    elif tag is not None:
        _walk_children(element, out)
    # A tail is text that follows the element inside its parent, so it belongs
    # to the parent's flow -- after the paragraph break for a block element.
    out.add_text(element.tail)


def _flatten(element: etree._Element) -> str:
    """Everything inside ``element`` on one line, for a table cell."""
    return " ".join(_render_contents(element).split())


def _render_table(table: etree._Element) -> str:
    """One line per row, cells joined by ``_CELL_SEPARATOR``.

    Empty cells are kept so later cells stay in their columns; rows with no
    text at all are dropped. ``rowspan``/``colspan`` are not expanded.
    """
    lines: list[str] = []
    for child in table:
        tag = _tag(child)
        if tag == "caption":
            lines.append(_flatten(child))
        elif tag == "tr":
            lines.append(_render_row(child))
        elif tag in ("thead", "tbody", "tfoot"):
            lines.extend(_render_row(row) for row in child if _tag(row) == "tr")
    return "\n".join(line for line in lines if line)


def _render_row(row: etree._Element) -> str:
    cells = [_flatten(cell) for cell in row if _tag(cell) in ("td", "th")]
    if not any(cells):
        return ""
    return _CELL_SEPARATOR.join(cells).rstrip()


# --- Command line ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan item 3: extract the corpus.")
    parser.add_argument(
        "--show",
        metavar="NAME",
        help="print the extracted text of the one file whose name contains NAME",
    )
    args = parser.parse_args(argv)

    try:
        documents = extract_corpus()
    except ExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    if args.show:
        matches = [doc for doc in documents if args.show in doc.filename]
        if len(matches) != 1:
            print(
                f"--show {args.show!r} matched {len(matches)} files; it must match exactly one.",
                file=sys.stderr,
            )
            return 2
        print(escape_controls(matches[0].text))
        return 0

    header = f"{'document':<26}{'chars':>10}{'paragraphs':>12}{'lines':>8}"
    print(header)
    print("-" * len(header))
    for doc in documents:
        print(
            f"{doc.label.split(':')[0]:<26}{len(doc.text):>10,}"
            f"{doc.text.count(chr(10) * 2) + 1:>12,}{doc.text.count(chr(10)) + 1:>8,}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':<26}{sum(len(doc.text) for doc in documents):>10,}")
    return 0


if __name__ == "__main__":
    # Regulation text includes characters such as U+2265 that a cp1252 console
    # cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
