"""Tests for plan item 4: corpus assembly, fingerprint and size guard (``corpus.py``).

All offline: the token counter is injected, so the size guard is exercised at
its exact boundary without an API call. The real-corpus tests at the bottom
extract ``admincode/`` and take a few seconds.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config import CORPUS_CACHE_TTL, DOCUMENT_LABELS, PRICE_CACHE_WRITE_1H_PER_MTOK, TOKEN_CEILING
from corpus import (
    Corpus,
    CorpusTooLargeError,
    FingerprintStatus,
    api_token_counter,
    assemble_corpus,
    build_document_blocks,
    compare_with_previous,
    describe_check,
    load_corpus,
    record_state,
)
from extract import ExtractedDocument

ALPHA = ExtractedDocument("a.htm", "Part A", "Alpha text.\n\na) First.")
BRAVO = ExtractedDocument("b.htm", "Part B", "Bravo text.")
CHARLIE = ExtractedDocument("c.htm", "Part C", "Charlie.")


class FixedCounter:
    """A token counter that returns a fixed count and records what it was shown."""

    def __init__(self, tokens: int = 1_000) -> None:
        self.tokens = tokens
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, blocks: list[dict[str, Any]]) -> int:
        self.calls.append(blocks)
        return self.tokens


def corpus_of(*documents: ExtractedDocument, tokens: int = 1_000) -> Corpus:
    return assemble_corpus(documents, FixedCounter(tokens))


def titles(blocks: list[dict[str, Any]]) -> list[str]:
    return [block["title"] for block in blocks]


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


# --- Document blocks ---------------------------------------------------------


def test_document_block_has_the_production_shape() -> None:
    assert build_document_blocks([ALPHA, BRAVO])[0] == {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": ALPHA.text},
        "title": "Part A",
        "context": "Source file: a.htm",
        "citations": {"enabled": True},
    }


def test_only_the_last_block_carries_the_corpus_breakpoint() -> None:
    blocks = build_document_blocks([ALPHA, BRAVO])
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral", "ttl": CORPUS_CACHE_TTL}


def test_document_blocks_are_fresh_copies() -> None:
    corpus = corpus_of(ALPHA, BRAVO)
    blocks = corpus.document_blocks()
    blocks[0]["title"] = "tampered"
    blocks[0]["source"]["data"] = "tampered"
    blocks.pop()
    assert corpus.document_blocks() == build_document_blocks([ALPHA, BRAVO])


# --- Assembly ----------------------------------------------------------------


def test_documents_are_assembled_in_filename_order() -> None:
    counter = FixedCounter()
    corpus = assemble_corpus([CHARLIE, ALPHA, BRAVO], counter)
    assert corpus.documents == (ALPHA, BRAVO, CHARLIE)
    assert titles(counter.calls[0]) == ["Part A", "Part B", "Part C"]


def test_identical_input_gives_an_identical_fingerprint() -> None:
    first = corpus_of(ALPHA, BRAVO)
    # Equal but distinct objects, supplied in the other order.
    second = corpus_of(replace(BRAVO), replace(ALPHA))
    assert second.fingerprint == first.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "documents",
    [
        [replace(ALPHA, text=ALPHA.text + " "), BRAVO],
        [replace(ALPHA, label="Part A2"), BRAVO],
        [ALPHA, BRAVO, CHARLIE],
        [ALPHA],
    ],
    ids=["text", "label", "added", "removed"],
)
def test_any_change_to_the_documents_changes_the_fingerprint(
    documents: list[ExtractedDocument],
) -> None:
    assert corpus_of(*documents).fingerprint != corpus_of(ALPHA, BRAVO).fingerprint


def test_context_template_is_part_of_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    before = corpus_of(ALPHA).fingerprint
    monkeypatch.setattr("corpus.DOCUMENT_CONTEXT_TEMPLATE", "File: {filename}")
    assert corpus_of(ALPHA).fingerprint != before


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        assemble_corpus([], FixedCounter())


def test_duplicate_filename_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"more than once: a\.htm"):
        assemble_corpus([ALPHA, replace(ALPHA, label="Other")], FixedCounter())


# --- Size guard --------------------------------------------------------------


def test_size_guard_accepts_a_corpus_exactly_at_the_ceiling() -> None:
    corpus = assemble_corpus([ALPHA, BRAVO], FixedCounter(500), ceiling=500)
    assert corpus.tokens == 500


def test_size_guard_rejects_one_token_over_the_ceiling_without_truncating() -> None:
    counter = FixedCounter(501)
    with pytest.raises(CorpusTooLargeError, match="Nothing was truncated") as caught:
        assemble_corpus([ALPHA, BRAVO], counter, ceiling=500)
    assert (caught.value.tokens, caught.value.ceiling) == (501, 500)
    # One count, of the whole corpus -- no retry with fewer documents.
    assert len(counter.calls) == 1
    assert titles(counter.calls[0]) == ["Part A", "Part B"]


def test_size_guard_message_lists_the_options() -> None:
    message = str(CorpusTooLargeError(401_000, 400_000))
    assert "401,000" in message
    for option in ("Remove documents", "Raise config.TOKEN_CEILING", "Re-plan"):
        assert option in message


# --- Token counter -----------------------------------------------------------


class FakeClient:
    """Stands in for ``anthropic.Anthropic``; records the count_tokens request."""

    def __init__(self, input_tokens: int = 1_234) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(count_tokens=self._count_tokens)
        self._input_tokens = input_tokens

    def _count_tokens(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(input_tokens=self._input_tokens)


def test_api_counter_counts_the_blocks_followed_by_a_question() -> None:
    client = FakeClient()
    blocks = build_document_blocks([ALPHA, BRAVO])

    assert api_token_counter(client, model="some-model")(blocks) == 1_234  # type: ignore[arg-type]

    [request] = client.requests
    assert request["model"] == "some-model"
    assert "system" not in request
    [message] = request["messages"]
    assert message["role"] == "user"
    assert message["content"][:-1] == blocks
    assert message["content"][-1]["type"] == "text"


def test_api_counter_includes_the_system_prompt_when_given() -> None:
    client = FakeClient()
    api_token_counter(client, system="Be exact.")(build_document_blocks([ALPHA]))  # type: ignore[arg-type]
    assert client.requests[0]["system"] == "Be exact."


# --- Fingerprint state -------------------------------------------------------


def test_first_run_has_nothing_to_compare(state_file: Path) -> None:
    check = compare_with_previous(corpus_of(ALPHA), state_file)
    assert check.status is FingerprintStatus.FIRST_RUN


def test_recorded_fingerprint_compares_unchanged(state_file: Path) -> None:
    record_state(corpus_of(ALPHA, BRAVO), state_file, lxml_version="6.1.3")
    check = compare_with_previous(corpus_of(ALPHA, BRAVO), state_file, lxml_version="6.1.3")
    assert check.status is FingerprintStatus.UNCHANGED
    assert check.previous_recorded_at
    assert not state_file.with_name(state_file.name + ".tmp").exists()


def test_changed_text_names_the_document_and_the_token_change(state_file: Path) -> None:
    record_state(corpus_of(ALPHA, BRAVO, tokens=1_000), state_file, lxml_version="6.1.3")
    changed = corpus_of(ALPHA, replace(BRAVO, text="Bravo text, amended."), tokens=1_005)

    check = compare_with_previous(changed, state_file, lxml_version="6.1.3")

    assert check.status is FingerprintStatus.CHANGED
    assert check.details == ("b.htm: text changed (11 -> 20 chars)", "tokens: 1,000 -> 1,005")


def test_changed_label_added_and_removed_files_are_each_named(state_file: Path) -> None:
    record_state(corpus_of(ALPHA, BRAVO), state_file, lxml_version="6.1.3")
    changed = corpus_of(replace(ALPHA, label="Part A2"), CHARLIE)

    check = compare_with_previous(changed, state_file, lxml_version="6.1.3")

    assert check.details == (
        "c.htm: added",
        "b.htm: removed",
        "a.htm: label changed ('Part A' -> 'Part A2')",
    )


def test_lxml_upgrade_is_named_as_the_likely_cause(state_file: Path) -> None:
    record_state(corpus_of(ALPHA), state_file, lxml_version="6.1.3")
    re_extracted = corpus_of(replace(ALPHA, text="Alpha text, re-extracted."))

    check = compare_with_previous(re_extracted, state_file, lxml_version="6.2.0")

    assert any(detail.startswith("lxml 6.1.3 -> 6.2.0: the likely cause") for detail in check.details)


def test_block_setting_change_is_reported_when_no_document_changed(
    state_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_state(corpus_of(ALPHA), state_file, lxml_version="6.1.3")
    monkeypatch.setattr("corpus.DOCUMENT_CONTEXT_TEMPLATE", "File: {filename}")

    check = compare_with_previous(corpus_of(ALPHA), state_file, lxml_version="6.1.3")

    assert check.status is FingerprintStatus.CHANGED
    assert check.details[0].startswith("no document's text or label changed, so a block setting did")


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"version": 99}),
        json.dumps({"version": 1, "fingerprint": "abc"}),
    ],
    ids=["invalid-json", "not-an-object", "unknown-version", "missing-fields"],
)
def test_unreadable_state_is_reported_not_raised(state_file: Path, content: str) -> None:
    state_file.write_text(content, encoding="utf-8")
    check = compare_with_previous(corpus_of(ALPHA), state_file)
    assert check.status is FingerprintStatus.UNREADABLE
    assert check.details


def test_recording_replaces_an_unreadable_state_file(state_file: Path) -> None:
    state_file.write_text("{not json", encoding="utf-8")
    record_state(corpus_of(ALPHA), state_file)
    assert compare_with_previous(corpus_of(ALPHA), state_file).status is FingerprintStatus.UNCHANGED


def test_change_warning_names_the_cause_and_the_consequence(state_file: Path) -> None:
    record_state(corpus_of(ALPHA), state_file, lxml_version="6.1.3")
    changed = corpus_of(replace(ALPHA, text="Alpha text, amended."), tokens=200_000)
    check = compare_with_previous(changed, state_file, lxml_version="6.1.3")

    warning = describe_check(check, changed)

    assert warning.startswith("WARNING: the corpus changed")
    assert "  - a.htm: text changed" in warning
    assert "an lxml upgrade" in warning
    assert f"${200_000 * PRICE_CACHE_WRITE_1H_PER_MTOK / 1_000_000:.2f}" in warning


# --- The real corpus ---------------------------------------------------------


def test_real_corpus_loads_to_a_stable_fingerprint(state_file: Path) -> None:
    # Two independent extractions of admincode/: determinism is load-bearing for
    # the cache, so this compares whole runs, not one run's blocks twice.
    first, first_check = load_corpus(FixedCounter(), state_file=state_file)
    second, second_check = load_corpus(FixedCounter(), state_file=state_file)

    assert first_check.status is FingerprintStatus.FIRST_RUN
    assert second_check.status is FingerprintStatus.UNCHANGED
    assert second.fingerprint == first.fingerprint
    assert [document.filename for document in first.documents] == sorted(DOCUMENT_LABELS)


def test_oversized_real_corpus_records_no_fingerprint(state_file: Path) -> None:
    with pytest.raises(CorpusTooLargeError):
        load_corpus(FixedCounter(TOKEN_CEILING + 1), state_file=state_file)
    assert not state_file.exists()
