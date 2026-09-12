"""Tests for plan item 6: the quote-match reporter (``quotes.py``). All offline."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from anthropic.types import (
    CitationCharLocation,
    CitationPageLocation,
    TextBlock,
    ThinkingBlock,
)

import quotes
from config import MIN_CITATION_LENGTH, THIN_COVERAGE_FRACTION
from quotes import BlockCoverage, MatchStatus, build_report, check_citation

SECTION = (
    "Section 175.410 Tank Sumps\n\n"
    "a) Sumps shall be liquid tight.\n\n"
    "b) Sumps shall be tested annually.\n\n"
)
RECORDS = "Section 176.430 Records\n\nb) Records shall be kept for five years.\n\n"
DOCUMENTS = [SECTION, RECORDS]

TIGHT = "a) Sumps shall be liquid tight."
TESTED = "b) Sumps shall be tested annually."


def visible(text: str) -> int:
    return len("".join(text.split()))


def cite(
    start: int, end: int, text: str | None = None, document_index: int = 0
) -> CitationCharLocation:
    """A citation whose quote is the source text at the span unless ``text`` is given."""
    return CitationCharLocation(
        type="char_location",
        cited_text=DOCUMENTS[document_index][start:end] if text is None else text,
        document_index=document_index,
        document_title="41 Ill. Adm. Code 175",
        start_char_index=start,
        end_char_index=end,
    )


def cite_phrase(phrase: str, document_index: int = 0, *, extra: int = 0) -> CitationCharLocation:
    """A correct citation of ``phrase``, running ``extra`` characters past it."""
    start = DOCUMENTS[document_index].index(phrase)
    return cite(start, start + len(phrase) + extra, document_index=document_index)


def text_block(text: str, *citations: CitationCharLocation) -> TextBlock:
    return TextBlock(type="text", text=text, citations=list(citations) or None)


# --- One citation: match -----------------------------------------------------


def test_an_exact_quote_matches() -> None:
    check = check_citation(cite_phrase(TIGHT), DOCUMENTS)

    assert check.status is MatchStatus.MATCHED
    assert check.matched and check.detail is None


def test_a_quote_in_a_later_document_matches() -> None:
    citation = cite_phrase("b) Records shall be kept", document_index=1)
    assert check_citation(citation, DOCUMENTS).status is MatchStatus.MATCHED


def test_a_span_across_line_breaks_matches() -> None:
    citation = cite_phrase("tight.\n\nb) Sumps")
    assert check_citation(citation, DOCUMENTS).status is MatchStatus.MATCHED


def test_a_span_that_absorbs_trailing_separators_matches() -> None:
    citation = cite_phrase(TIGHT, extra=2)

    assert citation.cited_text.endswith("\n\n")
    assert check_citation(citation, DOCUMENTS).status is MatchStatus.MATCHED


def test_a_span_at_the_minimum_length_matches_and_one_short_is_degenerate() -> None:
    start = SECTION.index("a)")
    end = start
    while visible(SECTION[start:end]) < MIN_CITATION_LENGTH:
        end += 1

    assert check_citation(cite(start, end), DOCUMENTS).status is MatchStatus.MATCHED
    assert check_citation(cite(start, end - 1), DOCUMENTS).status is MatchStatus.DEGENERATE


# --- One citation: mismatch --------------------------------------------------


def test_an_off_by_one_offset_is_a_mismatch_that_names_the_real_position() -> None:
    start = SECTION.index(TIGHT)
    citation = cite(start + 1, start + len(TIGHT) + 1, text=TIGHT)

    check = check_citation(citation, DOCUMENTS)

    assert check.status is MatchStatus.MISMATCH
    assert check.detail is not None
    assert f"appear verbatim at character {start:,} (-1 from the cited start)" in check.detail


def test_a_quote_not_in_the_document_is_a_mismatch() -> None:
    start = SECTION.index(TIGHT)
    citation = cite(start, start + len(TIGHT), text="a) Sumps shall be liquid proof.")

    check = check_citation(citation, DOCUMENTS)

    assert check.status is MatchStatus.MISMATCH
    assert check.detail is not None
    assert "at character 25 of the span" in check.detail
    assert "appears nowhere verbatim" in check.detail


def test_a_quote_of_another_length_reports_both_lengths() -> None:
    start = SECTION.index(TIGHT)
    citation = cite(start, start + len(TIGHT), text=TIGHT + " Always.")

    check = check_citation(citation, DOCUMENTS)

    assert check.status is MatchStatus.MISMATCH
    assert check.detail is not None
    assert f"the span is {len(TIGHT)} characters and the quote {len(TIGHT) + 8}" in check.detail


# --- One citation: out of range ----------------------------------------------


@pytest.mark.parametrize("document_index", [2, 99, -1])
def test_a_document_index_outside_the_corpus_is_out_of_range(document_index: int) -> None:
    citation = cite(0, 10, text="Section 17", document_index=0).model_copy(
        update={"document_index": document_index}
    )

    check = check_citation(citation, DOCUMENTS)

    assert check.status is MatchStatus.OUT_OF_RANGE
    assert check.detail is not None
    assert f"document {document_index} does not exist; the corpus has 2" in check.detail


def test_a_span_past_the_end_is_out_of_range_even_when_slicing_would_match() -> None:
    start = len(SECTION) - 20
    citation = cite(start, len(SECTION) + 5, text=SECTION[start:])

    # The trap: Python slicing clips silently, so a naive check would pass.
    assert SECTION[start : len(SECTION) + 5] == citation.cited_text
    check = check_citation(citation, DOCUMENTS)
    assert check.status is MatchStatus.OUT_OF_RANGE
    assert check.detail is not None
    assert f"which has {len(SECTION)}" in check.detail


def test_a_negative_start_is_out_of_range() -> None:
    citation = cite(-5, 10, text="whatever")
    assert check_citation(citation, DOCUMENTS).status is MatchStatus.OUT_OF_RANGE


def test_a_span_that_ends_before_it_starts_is_out_of_range() -> None:
    check = check_citation(cite(20, 10, text=""), DOCUMENTS)

    assert check.status is MatchStatus.OUT_OF_RANGE
    assert check.detail is not None and "ends" in check.detail


# --- One citation: degenerate and not checkable ------------------------------


def test_an_empty_span_against_an_empty_quote_is_degenerate() -> None:
    check = check_citation(cite(10, 10, text=""), DOCUMENTS)

    assert check.status is MatchStatus.DEGENERATE
    assert check.detail == "the span is empty"


def test_a_whitespace_only_span_is_degenerate() -> None:
    documents = ["x" + " " * 20 + "y"]
    citation = cite(1, 21, text=" " * 20)

    assert check_citation(citation, documents).status is MatchStatus.DEGENERATE


def test_a_citation_without_character_offsets_is_not_checkable() -> None:
    citation = CitationPageLocation(
        type="page_location",
        cited_text="Sumps shall be liquid tight.",
        document_index=0,
        document_title=None,
        start_page_number=1,
        end_page_number=2,
    )

    check = check_citation(citation, DOCUMENTS)

    assert check.status is MatchStatus.NOT_CHECKABLE
    assert not check.matched


# --- The whole answer --------------------------------------------------------


def test_a_span_cited_twice_is_checked_and_counted_once() -> None:
    tight, tested = cite_phrase(TIGHT), cite_phrase(TESTED)
    content = [text_block("Tight.", tight), text_block(" Tested.", tight, tested, tested)]

    report = build_report([tight, tested], content, DOCUMENTS)

    assert len(report.checks) == 2
    assert report.matched_count == 2
    assert report.citation_count == 4
    assert [block.source_numbers for block in report.cited_blocks] == [(1,), (1, 2)]


def test_failures_carry_their_source_number_and_category() -> None:
    good = cite_phrase(TIGHT)
    past_end = cite(0, len(SECTION) + 1, text="x")
    wrong = cite_phrase(TESTED).model_copy(update={"cited_text": "b) Sumps shall be tested weekly.."})
    content = [text_block("Claims.", good, past_end, wrong)]

    report = build_report([good, past_end, wrong], content, DOCUMENTS)

    assert [(number, check.status) for number, check in report.failures] == [
        (2, MatchStatus.OUT_OF_RANGE),
        (3, MatchStatus.MISMATCH),
    ]
    assert report.matched_count == 1


def test_checks_follow_the_source_order_given() -> None:
    tight, tested = cite_phrase(TIGHT), cite_phrase(TESTED)
    content = [text_block("Tight.", tight), text_block(" Tested.", tested)]

    report = build_report([tested, tight], content, DOCUMENTS)

    assert [check.citation for check in report.checks] == [tested, tight]
    assert [block.source_numbers for block in report.cited_blocks] == [(2,), (1,)]


def test_a_citation_missing_from_the_sources_is_numbered_after_them() -> None:
    tight, tested = cite_phrase(TIGHT), cite_phrase(TESTED)

    report = build_report([tight], [text_block("Both.", tight, tested)], DOCUMENTS)

    assert [check.citation for check in report.checks] == [tight, tested]


def test_only_text_blocks_are_read() -> None:
    content = [
        ThinkingBlock(type="thinking", thinking="Long private reasoning.", signature="sig"),
        text_block("Sumps are tight.", cite_phrase(TIGHT)),
    ]

    assert build_report([], content, DOCUMENTS).answer_chars == visible("Sumps are tight.")


# --- Coverage ----------------------------------------------------------------


def test_coverage_leaves_out_whitespace() -> None:
    report = build_report([], [text_block("Tight.", cite_phrase(TIGHT, extra=2))], DOCUMENTS)

    [block] = report.cited_blocks
    assert block.cited_chars == visible(TIGHT)
    assert block.block_chars == visible("Tight.")


def test_overlapping_spans_are_counted_once() -> None:
    start = SECTION.index(TIGHT)
    whole = cite(start, start + len(TIGHT))
    head = cite(start, start + 15)
    tail = cite(start + 10, start + len(TIGHT) + len("\n\nb) Sumps"))

    report = build_report([], [text_block("Claim.", whole, head, tail)], DOCUMENTS)

    [block] = report.cited_blocks
    assert block.cited_chars == visible(TIGHT + "\n\nb) Sumps")


def test_a_span_cited_in_two_blocks_counts_for_each() -> None:
    tight = cite_phrase(TIGHT)

    report = build_report([], [text_block("One.", tight), text_block(" Two.", tight)], DOCUMENTS)

    assert [block.cited_chars for block in report.cited_blocks] == [visible(TIGHT)] * 2


def test_a_failed_span_sources_nothing() -> None:
    good = cite_phrase(TIGHT)
    wrong = cite_phrase(TESTED).model_copy(update={"cited_text": "fabricated text here"})
    past_end = cite(0, len(SECTION) + 1, text="x")

    report = build_report([], [text_block("Claim.", good, wrong, past_end)], DOCUMENTS)

    [block] = report.cited_blocks
    assert block.cited_chars == visible(TIGHT)


def test_a_long_block_on_a_short_quote_is_thin() -> None:
    claim = "Sumps must be liquid tight, tested every year, and replaced as soon as cracked."
    report = build_report([], [text_block(claim, cite_phrase(TIGHT))], DOCUMENTS)

    [block] = report.cited_blocks
    assert block.fraction == pytest.approx(visible(TIGHT) / visible(claim))
    assert block.thin
    assert report.thin_blocks == [block]


def test_a_quote_longer_than_its_block_caps_at_full() -> None:
    report = build_report([], [text_block("Tight.", cite_phrase(TIGHT))], DOCUMENTS)

    [block] = report.cited_blocks
    assert block.fraction == 1.0
    assert not block.thin


def test_the_thin_threshold_is_exclusive() -> None:
    def coverage(cited: int) -> BlockCoverage:
        return BlockCoverage(text="x", source_numbers=(1,), block_chars=100, cited_chars=cited)

    at_threshold = round(THIN_COVERAGE_FRACTION * 100)
    assert not coverage(at_threshold).thin
    assert coverage(at_threshold - 1).thin


def test_uncited_text_is_measured_across_the_answer() -> None:
    content = [
        text_block("The rules are:"),
        text_block(" Sumps are tight.", cite_phrase(TIGHT)),
        text_block(""),
    ]

    report = build_report([], content, DOCUMENTS)

    assert report.answer_chars == visible("The rules are:") + visible(" Sumps are tight.")
    assert report.uncited_chars == visible("The rules are:")
    assert report.uncited_fraction == pytest.approx(report.uncited_chars / report.answer_chars)
    assert len(report.cited_blocks) == 1


def test_a_cited_block_with_no_visible_text_is_checked_but_not_measured() -> None:
    report = build_report([], [text_block("  \n", cite_phrase(TIGHT))], DOCUMENTS)

    assert len(report.checks) == 1
    assert report.cited_blocks == ()


def test_an_empty_answer_has_no_uncited_share() -> None:
    report = build_report([], [], DOCUMENTS)

    assert report.checks == ()
    assert report.uncited_fraction == 0.0


# --- The guarantee -----------------------------------------------------------


def test_the_reporter_never_relies_on_assert() -> None:
    # assert is stripped under python -O, and would end the turn on a failed
    # quote -- the very case the reporter exists to report.
    tree = ast.parse(Path(quotes.__file__).read_text(encoding="utf-8"))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
