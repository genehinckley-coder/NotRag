"""Plan item 6: the quote-match reporter. No API call.

Native citations return ``cited_text`` plus ``char_location`` offsets into the
exact strings sent, so checking a quote is a local string comparison. Each
distinct cited span gets one of several outcomes rather than a pass/fail flag,
so that offsets pointing outside the text we hold -- a bug in what we compare
against -- are never mistaken for a quote that fails to match its source.

A match proves provenance: the quoted text appears at the cited location. It
does not prove the quote supports the claim attached to it
(``config.MATCH_DISCLAIMER``), so a match is reported as "matched", never with a
stronger word.

Nothing here asserts or raises on a failed quote. Failures are the output, and
``assert`` would vanish under ``python -O``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import Enum

from anthropic.types import CitationCharLocation, ContentBlock, TextCitation

from config import MIN_CITATION_LENGTH, THIN_COVERAGE_FRACTION


class MatchStatus(Enum):
    #: The source text at the cited span is exactly the quoted text.
    MATCHED = "matched"
    #: In range and long enough, but the source text there is not the quote.
    MISMATCH = "mismatch"
    #: The document index or the span falls outside the text sent.
    OUT_OF_RANGE = "out of range"
    #: Empty, or too short to mean anything; would match trivially.
    DEGENERATE = "degenerate"
    #: Not a character-offset citation, so there is nothing to compare.
    NOT_CHECKABLE = "not checkable"


@dataclass(frozen=True)
class QuoteCheck:
    """The outcome for one distinct cited span."""

    citation: TextCitation
    status: MatchStatus
    #: Why the quote did not match; None when it did.
    detail: str | None = None

    @property
    def matched(self) -> bool:
        return self.status is MatchStatus.MATCHED


@dataclass(frozen=True)
class BlockCoverage:
    """How much matched source text stands behind one cited answer block.

    A length proxy, not a measurement: citations attach to the whole block, and
    the block paraphrases what it cites. What it catches is a long block
    resting on a short quote, where uncited assertions can ride along.
    """

    text: str
    #: Numbers of the sources this block cites, each once.
    source_numbers: tuple[int, ...]
    #: Non-whitespace characters in the block. Always positive.
    block_chars: int
    #: Non-whitespace characters in the block's matched spans, overlaps once.
    cited_chars: int

    @property
    def fraction(self) -> float:
        return min(1.0, self.cited_chars / self.block_chars)

    @property
    def thin(self) -> bool:
        return self.fraction < THIN_COVERAGE_FRACTION


@dataclass(frozen=True)
class QuoteReport:
    """Every cited span checked, and how much of the answer is sourced."""

    #: One per distinct span; ``checks[n - 1]`` is source ``[n]``.
    checks: tuple[QuoteCheck, ...]
    #: Citations across every block, repeats of the same span included.
    citation_count: int
    #: Every text block that cites something and has visible text.
    cited_blocks: tuple[BlockCoverage, ...]
    #: Non-whitespace characters across the answer's text blocks.
    answer_chars: int
    #: Of those, the characters in blocks with no citation.
    uncited_chars: int

    @property
    def matched_count(self) -> int:
        return sum(1 for check in self.checks if check.matched)

    @property
    def failures(self) -> list[tuple[int, QuoteCheck]]:
        """``(source number, check)`` for every span that did not match."""
        return [
            (number, check)
            for number, check in enumerate(self.checks, start=1)
            if not check.matched
        ]

    @property
    def thin_blocks(self) -> list[BlockCoverage]:
        return [block for block in self.cited_blocks if block.thin]

    @property
    def uncited_fraction(self) -> float:
        return self.uncited_chars / self.answer_chars if self.answer_chars else 0.0


def citation_key(citation: TextCitation) -> Hashable:
    """What makes two citations the same source span.

    Item 2 saw one answer cite the same span twice; both share one number and
    are checked and counted once.
    """
    if isinstance(citation, CitationCharLocation):
        return (citation.document_index, citation.start_char_index, citation.end_char_index)
    # Not expected for plain-text documents; keyed on the quote itself.
    return (citation.type, citation.cited_text)


# --- One citation ------------------------------------------------------------


def check_citation(citation: TextCitation, documents: Sequence[str]) -> QuoteCheck:
    """Compare one citation's quote with the source text at its span.

    Args:
        citation: A citation from the answer.
        documents: The exact document texts sent, in send order, so that a
            citation's ``document_index`` indexes them.
    """
    if not isinstance(citation, CitationCharLocation):
        return QuoteCheck(
            citation,
            MatchStatus.NOT_CHECKABLE,
            f"a {citation.type!r} citation carries no character offsets",
        )

    index = citation.document_index
    start, end = citation.start_char_index, citation.end_char_index
    # Validate before slicing: Python slicing never raises, and a span running
    # past the end silently yields a short string that could even match.
    if not 0 <= index < len(documents):
        return QuoteCheck(
            citation,
            MatchStatus.OUT_OF_RANGE,
            f"document {index} does not exist; the corpus has {len(documents)} "
            f"(numbered from 0)",
        )
    text = documents[index]
    if start > end:
        return QuoteCheck(
            citation,
            MatchStatus.OUT_OF_RANGE,
            f"the span ends (character {end:,}) before it starts ({start:,})",
        )
    if start < 0 or end > len(text):
        return QuoteCheck(
            citation,
            MatchStatus.OUT_OF_RANGE,
            f"characters {start:,}-{end:,} fall outside document {index}, which has "
            f"{len(text):,}",
        )

    source = text[start:end]
    visible = _visible_length(source)
    if visible < MIN_CITATION_LENGTH:
        reason = (
            "the span is empty"
            if start == end
            else f"the span has {visible} non-whitespace characters, under "
            f"config.MIN_CITATION_LENGTH ({MIN_CITATION_LENGTH})"
        )
        return QuoteCheck(citation, MatchStatus.DEGENERATE, reason)

    if source == citation.cited_text:
        return QuoteCheck(citation, MatchStatus.MATCHED)
    return QuoteCheck(
        citation, MatchStatus.MISMATCH, _describe_mismatch(source, citation.cited_text, text, start)
    )


def _describe_mismatch(source: str, quote: str, text: str, start: int) -> str:
    """Where the quote departs from the source span, and whether it exists elsewhere."""
    differ_at = next(
        (i for i, (a, b) in enumerate(zip(source, quote)) if a != b),
        min(len(source), len(quote)),
    )
    parts = [f"the quote departs from the source at character {differ_at:,} of the span"]
    if len(quote) != len(source):
        parts.append(f"the span is {len(source):,} characters and the quote {len(quote):,}")
    # Tells an offset error apart from a quote that is not in the document at all.
    found = text.find(quote) if quote else -1
    if found >= 0:
        parts.append(
            f"the quote does appear verbatim at character {found:,} "
            f"({found - start:+,} from the cited start)"
        )
    else:
        parts.append("the quote appears nowhere verbatim in that document")
    return "; ".join(parts)


# --- The whole answer --------------------------------------------------------


def build_report(
    sources: Sequence[TextCitation],
    content: Sequence[ContentBlock],
    documents: Sequence[str],
) -> QuoteReport:
    """Check every distinct cited span and measure how much of the answer is sourced.

    Args:
        sources: Distinct citations in source-number order, as numbered in the
            rendered answer (``answer.Answer.sources``).
        content: The response's content blocks. Only text blocks are read.
        documents: The exact document texts sent, in send order.

    A citation in ``content`` that is missing from ``sources`` is numbered
    after them. Normally there are none: ``answer.stream_answer`` numbers every
    citation in the final message.
    """
    ordered: list[TextCitation] = []
    numbers: dict[Hashable, int] = {}

    def number_of(citation: TextCitation) -> int:
        key = citation_key(citation)
        if key not in numbers:
            ordered.append(citation)
            numbers[key] = len(ordered)
        return numbers[key]

    for citation in sources:
        number_of(citation)

    citation_count = answer_chars = uncited_chars = 0
    cited: list[tuple[str, tuple[int, ...], int]] = []
    for block in content:
        if block.type != "text":
            continue
        citations = block.citations or []
        chars = _visible_length(block.text)
        answer_chars += chars
        citation_count += len(citations)
        # Every citation is numbered, and so checked, even in a block too
        # empty to measure.
        block_numbers = tuple(dict.fromkeys(number_of(citation) for citation in citations))
        if not citations:
            uncited_chars += chars
        elif chars:
            cited.append((block.text, block_numbers, chars))

    checks = tuple(check_citation(citation, documents) for citation in ordered)
    cited_blocks = tuple(
        BlockCoverage(
            text=text,
            source_numbers=block_numbers,
            block_chars=chars,
            cited_chars=_matched_chars([checks[n - 1] for n in block_numbers], documents),
        )
        for text, block_numbers, chars in cited
    )
    return QuoteReport(checks, citation_count, cited_blocks, answer_chars, uncited_chars)


def _matched_chars(checks: Sequence[QuoteCheck], documents: Sequence[str]) -> int:
    """Non-whitespace characters under the matched spans, overlapping spans merged.

    Only matched spans count: a span that fails its check sources nothing.
    Whitespace is left out so the blank separator lines a span often absorbs
    do not count as sourced text.
    """
    spans: dict[int, list[tuple[int, int]]] = {}
    for check in checks:
        citation = check.citation
        if check.matched and isinstance(citation, CitationCharLocation):
            spans.setdefault(citation.document_index, []).append(
                (citation.start_char_index, citation.end_char_index)
            )

    total = 0
    for index, intervals in spans.items():
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        total += sum(_visible_length(documents[index][start:end]) for start, end in merged)
    return total


def _visible_length(text: str) -> int:
    """Characters in ``text`` that are not whitespace."""
    return len("".join(text.split()))
