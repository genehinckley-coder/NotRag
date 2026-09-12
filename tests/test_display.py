"""Tests for ``display.escape_controls``: untrusted text cannot drive the terminal."""

from __future__ import annotations

import pytest

from display import escape_controls


def test_escape_is_shown_not_emitted() -> None:
    # rich 15 passes ESC through, so this is the only thing standing between
    # a quoted passage and the terminal.
    assert escape_controls("a\x1b[31mb") == "a\\x1b[31mb"


@pytest.mark.parametrize("char", ["\x00", "\x07", "\x08", "\r", "\x7f", "\x85", "\x9b"])
def test_c0_and_c1_controls_are_escaped(char: str) -> None:
    assert escape_controls(char) == f"\\x{ord(char):02x}"


def test_newline_tab_and_ordinary_text_are_untouched() -> None:
    text = "Section 175.410\n\ta) Sumps § ≥ 5 gallons \U0001d400"
    assert escape_controls(text) == text
