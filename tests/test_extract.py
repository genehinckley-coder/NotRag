"""Tests for plan item 3: structure-preserving extraction (``extract.py``).

All offline -- nothing here calls the API. The real-corpus tests at the bottom
read ``admincode/`` directly and take about a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import DOCUMENT_LABELS
from extract import (
    ExtractedDocument,
    ExtractionError,
    discover_corpus_files,
    extract_corpus,
    extract_file,
    extract_html,
)

FIXTURES = Path(__file__).parent / "fixtures"

#: The first two tables of file 175, byte for byte (CRLF included).
TABLES_175 = FIXTURES / "tables_175.html"

NBSP = "\N{NO-BREAK SPACE}"


def page(content: str, *, chrome: str = "") -> str:
    """A minimal ILGA-shaped page with ``content`` inside the content container."""
    return (
        "<html><head><title>ADMINISTRATIVE CODE</title></head><body>"
        f'{chrome}<div class="billtext-scale">{content}</div></body></html>'
    )


def write_page(directory: Path, name: str, content: str) -> None:
    (directory / name).write_text(page(content), encoding="utf-8")


# --- Paragraph structure -----------------------------------------------------


def test_subsection_markers_survive_as_separate_paragraphs() -> None:
    html = page(
        f"<p>a){NBSP * 7} Applicability. This Section applies.</p>"
        f"<p>1){NBSP * 7} First item;</p>"
        f"<p>A){NBSP * 7} Nested item;</p>"
        f"<p>ii){NBSP * 7} Deepest item.</p>"
    )
    assert extract_html(html) == (
        "a) Applicability. This Section applies.\n\n"
        "1) First item;\n\n"
        "A) Nested item;\n\n"
        "ii) Deepest item."
    )


def test_spacer_paragraphs_leave_exactly_one_blank_line() -> None:
    html = page(f"<p>One</p><p>{NBSP}</p><p></p><p> </p><p>Two</p>")
    assert extract_html(html) == "One\n\nTwo"


def test_hard_wrapped_source_lines_join_into_one_line() -> None:
    html = page("<p>The\r\nshell or head   has any\r\n  seam split</p>")
    assert extract_html(html) == "The shell or head has any seam split"


def test_crlf_input_yields_no_carriage_returns() -> None:
    html = page("<p>First\r\nparagraph</p><p>Second</p>").replace("><", ">\r\n<")
    text = extract_html(html.encode("utf-8"))
    assert "\r" not in text
    assert text == "First paragraph\n\nSecond"


def test_literal_nbsp_and_nbsp_entity_are_both_replaced() -> None:
    # lxml decodes &nbsp; at parse time, so both arrive as the U+00A0 character.
    html = page(f"<p>at least 5{NBSP}gallons and 3.5&nbsp;gallons</p>")
    text = extract_html(html)
    assert NBSP not in text
    assert text == "at least 5 gallons and 3.5 gallons"


def test_br_is_a_line_break_within_a_paragraph() -> None:
    html = page(
        '<p class="jcarheading">TITLE 41:  FIRE PROTECTION<br/>CHAPTER I:  OFFICE OF THE '
        "STATE FIRE MARSHAL<br/>PART 175  TECHNICAL REQUIREMENTS</p><p>Next</p>"
    )
    assert extract_html(html) == (
        "TITLE 41: FIRE PROTECTION\n"
        "CHAPTER I: OFFICE OF THE STATE FIRE MARSHAL\n"
        "PART 175 TECHNICAL REQUIREMENTS\n\n"
        "Next"
    )


def test_paragraph_holding_only_a_br_is_dropped() -> None:
    html = page('<p>One</p><p class="content"><br/></p><p>Two</p>')
    assert extract_html(html) == "One\n\nTwo"


def test_inline_markup_stays_on_the_line() -> None:
    html = page(
        "<p><b>Section 175.410</b> applies; a split of <sup>1</sup>/<sub>6</sub> of the "
        '<a href="#x">circumference</a> is <i>too</i> wide.</p>'
    )
    assert extract_html(html) == (
        "Section 175.410 applies; a split of 1/6 of the circumference is too wide."
    )


def test_text_after_a_block_element_starts_a_new_paragraph() -> None:
    html = page("<div><p>Inside</p>Tail text</div>After")
    assert extract_html(html) == "Inside\n\nTail text\n\nAfter"


def test_script_style_comments_and_page_chrome_are_excluded() -> None:
    html = page(
        "<p>Rule<!-- editor note --> text</p>"
        "<script>var leak = 1;</script><style>p { color: red }</style>",
        chrome="<header>Illinois General Assembly ILGA.GOV</header><h1>ADMINISTRATIVE CODE</h1>",
    )
    assert extract_html(html) == "Rule text"


# --- Content container -------------------------------------------------------


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>No container here</p></body></html>",
        page("<p>One</p>").replace(
            "</body>", '<div class="billtext-scale"><p>Two</p></div></body>'
        ),
    ],
    ids=["missing", "duplicated"],
)
def test_content_container_must_appear_exactly_once(html: str) -> None:
    with pytest.raises(ExtractionError, match="exactly one"):
        extract_html(html, source="page.htm")


def test_container_with_no_text_is_an_error() -> None:
    with pytest.raises(ExtractionError, match="no text"):
        extract_html(page(f"<p>{NBSP}</p><p><br/></p>"))


# --- Tables ------------------------------------------------------------------


def test_table_is_its_own_paragraph_with_one_line_per_row() -> None:
    html = page(
        "<p>Before</p>"
        "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
        "<p>After</p>"
    )
    assert extract_html(html) == "Before\n\na | b\nc | d\n\nAfter"


def test_empty_cells_keep_their_column_and_empty_rows_are_dropped() -> None:
    html = page(
        "<table><tbody>"
        "<tr><th>Type</th><th>Permit</th><th>Notes</th></tr>"
        f"<tr><td>{NBSP}</td><td></td><td> </td></tr>"
        "<tr><td>Removal</td><td></td><td>see (c)</td></tr>"
        "<tr><td>Lining</td><td>Yes</td><td></td></tr>"
        "</tbody></table>"
    )
    assert extract_html(html) == (
        "Type | Permit | Notes\nRemoval |  | see (c)\nLining | Yes |"
    )


def test_tables_from_file_175_keep_row_and_column_structure() -> None:
    # Real Word-derived markup: every cell wraps its text in <p>, some cells hold
    # several, and the source is CRLF with hard wraps inside cells.
    assert b"\r\n" in TABLES_175.read_bytes()
    assert extract_file(TABLES_175) == (
        "Nominal tank capacity | Whether use of manual tank gauging for release "
        "detection is allowed | Time limit on use of manual tank gauging for release "
        "detection\n"
        "600 gallons or less | Allowed as sole method of release detection | "
        "Allowed indefinitely\n"
        "601-2,000 gallons | Not allowed | Not allowed\n"
        "Over 2,000 gallons | Not allowed | Not allowed\n"
        "\n"
        "Nominal tank capacity | Weekly standard (one test) | Monthly standard "
        "(average of 4 tests taken once weekly over a 4-week period)\n"
        "600 gallons or less | 10 gallons | 5 gallons"
    )


# --- Corpus discovery and labels ---------------------------------------------


def test_discovery_reads_the_top_level_only_in_filename_order(tmp_path: Path) -> None:
    for name in ("b.htm", "a.html", "notes.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "a_files").mkdir()
    (tmp_path / "a_files" / "inner.htm").write_text("x", encoding="utf-8")
    (tmp_path / "folder.htm").mkdir()

    found = discover_corpus_files(tmp_path, ("*.htm", "*.html"))

    assert [path.name for path in found] == ["a.html", "b.htm"]


def test_corpus_pairs_each_file_with_its_label_in_filename_order(tmp_path: Path) -> None:
    write_page(tmp_path, "b.htm", "<p>Bravo</p>")
    write_page(tmp_path, "a.htm", "<p>Alpha</p>")

    documents = extract_corpus(tmp_path, {"a.htm": "Part A", "b.htm": "Part B"})

    assert documents == [
        ExtractedDocument("a.htm", "Part A", "Alpha"),
        ExtractedDocument("b.htm", "Part B", "Bravo"),
    ]


def test_unlabeled_file_is_an_error_not_a_fallback(tmp_path: Path) -> None:
    write_page(tmp_path, "a.htm", "<p>Alpha</p>")
    write_page(tmp_path, "new.htm", "<p>New</p>")
    with pytest.raises(ExtractionError, match=r"no label .*new\.htm"):
        extract_corpus(tmp_path, {"a.htm": "Part A"})


def test_labeled_file_missing_from_disk_is_an_error(tmp_path: Path) -> None:
    write_page(tmp_path, "a.htm", "<p>Alpha</p>")
    with pytest.raises(ExtractionError, match=r"not found .*gone\.htm"):
        extract_corpus(tmp_path, {"a.htm": "Part A", "gone.htm": "Part G"})


def test_empty_corpus_folder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="No corpus files"):
        extract_corpus(tmp_path, {})


# --- The real corpus ---------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> dict[str, ExtractedDocument]:
    return {doc.filename: doc for doc in extract_corpus()}


def test_real_corpus_matches_the_configured_labels(corpus: dict[str, ExtractedDocument]) -> None:
    assert list(corpus) == sorted(DOCUMENT_LABELS)


def test_real_corpus_is_normalized(corpus: dict[str, ExtractedDocument]) -> None:
    for name, doc in corpus.items():
        assert "\r" not in doc.text, name
        assert NBSP not in doc.text, name
        assert "\n\n\n" not in doc.text, name
        assert all(line == line.strip() for line in doc.text.split("\n")), name


def test_real_corpus_starts_at_the_regulation_not_the_site_chrome(
    corpus: dict[str, ExtractedDocument],
) -> None:
    for name, doc in corpus.items():
        assert doc.text.startswith(
            "TITLE 41: FIRE PROTECTION\nCHAPTER I: OFFICE OF THE STATE FIRE MARSHAL\nPART "
        ), name


def test_file_175_keeps_the_sump_section_and_its_tables(
    corpus: dict[str, ExtractedDocument],
) -> None:
    text = corpus["ADMINISTRATIVE CODE 41 175.htm"].text
    heading = "Section 175.410 Submersible, Under-dispenser, Transition and Other Containment Sumps"
    # Once in the table of contents, once heading the Section itself.
    assert text.count(f"\n\n{heading}\n\n") >= 2
    assert (
        "\n600 gallons or less | Allowed as sole method of release detection | "
        "Allowed indefinitely\n"
    ) in text
    assert "1/6 of the circumference of the tank" in text
