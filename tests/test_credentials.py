"""Tests for plan item 0: API key loading and client construction (``credentials.py``).

All offline, and none reads the real key: every key file here is written to
``tmp_path``, and the default-path test points ``API_KEY_PATH`` there too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import credentials
from credentials import build_client, load_api_key, redact

KEY = "sk-ant-api03-" + "k" * 40
OTHER_KEY = "sk-ant-api03-" + "o" * 40


def key_file(directory: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """``content`` written byte for byte, so CRLF and a BOM survive."""
    path = directory / "notrag_key.txt"
    path.write_bytes(content.encode(encoding))
    return path


# --- Loading -----------------------------------------------------------------


def test_a_bare_key_is_returned(tmp_path: Path) -> None:
    assert load_api_key(key_file(tmp_path, KEY)) == KEY


def test_a_notepad_bom_and_trailing_crlf_are_stripped(tmp_path: Path) -> None:
    path = key_file(tmp_path, KEY + "\r\n", encoding="utf-8-sig")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert load_api_key(path) == KEY


def test_annotation_lines_around_the_key_are_ignored(tmp_path: Path) -> None:
    # The real key file carries a hand-written expiry note above the key.
    path = key_file(tmp_path, f"Expires 2027-01-31, rotate before then\r\n{KEY}\r\n\r\n")

    assert load_api_key(path) == KEY


def test_the_default_path_is_the_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(credentials, "API_KEY_PATH", key_file(tmp_path, KEY))

    assert load_api_key() == KEY


# --- Refusals ----------------------------------------------------------------


def test_a_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent.txt"):
        load_api_key(tmp_path / "absent.txt")


@pytest.mark.parametrize("content", ["", " \r\n\t\r\n"], ids=["empty", "whitespace"])
def test_an_empty_file_is_refused(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        load_api_key(key_file(tmp_path, content))


def test_a_file_with_no_key_line_is_refused_without_echoing_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No Anthropic API key") as caught:
        load_api_key(key_file(tmp_path, "api03-looks-secret\nanother line\n"))

    assert "looks-secret" not in str(caught.value)


def test_two_key_lines_are_ambiguous_and_neither_is_echoed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2 lines") as caught:
        load_api_key(key_file(tmp_path, f"{KEY}\n{OTHER_KEY}\n"))

    message = str(caught.value)
    assert KEY not in message and OTHER_KEY not in message
    assert "kkkk" not in message and "oooo" not in message


def test_a_key_with_embedded_whitespace_is_refused_without_echoing_it(tmp_path: Path) -> None:
    broken = KEY[:20] + " " + KEY[20:]

    with pytest.raises(ValueError, match="embedded whitespace") as caught:
        load_api_key(key_file(tmp_path, broken))

    assert "kkkk" not in str(caught.value)


# --- Client ------------------------------------------------------------------


def test_the_client_gets_the_key_explicitly_and_the_environment_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Putting the key in os.environ would hand it to every child process.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    before = dict(os.environ)

    client = build_client(key_file(tmp_path, KEY))

    assert client.api_key == KEY
    assert dict(os.environ) == before


# --- Redaction ---------------------------------------------------------------


def test_redact_shows_enough_to_tell_keys_apart_and_no_more() -> None:
    shown = redact(KEY)

    assert shown == f"{KEY[:11]}...{KEY[-4:]}"
    assert redact(OTHER_KEY) != shown
    assert KEY[11:-4] not in shown


def test_redact_hides_a_short_key_entirely() -> None:
    assert redact("sk-ant-x") == "********"
