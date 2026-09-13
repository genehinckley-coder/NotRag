"""Tests for plan item 7: the REPL loop, caching, and cost display (``repl.py``).

All offline. The message, usage and stream fakes are the ones item 5's tests
already define; pytest puts this directory on ``sys.path``, so they are
imported rather than duplicated, which keeps one definition of what a response
looks like.

The tests that matter most are the ones about conversation state. A turn that
fails must leave ``messages`` exactly as it was: the damage from getting that
wrong shows up on a *later* turn, as a 400 the user has no reason to connect to
the network blip that caused it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

import repl
from config import (
    CACHE_READ_TOLERANCE_FRACTION,
    CONTEXT_WARN_FRACTION,
    CONTEXT_WINDOW,
    CONVERSATION_CACHE_TTL,
    MATCH_DISCLAIMER,
)
from corpus import Corpus
from repl import Session, replayable
from test_answer import (
    ALPHA,
    BRAVO,
    DOCUMENTS,
    FakeStream,
    cite,
    corpus_of,
    make_console,
    message_of,
    output,
    text_block,
    thinking_block,
    usage_of,
)

CITED = message_of(text_block("Sumps.", cite(0, 10, text="Alpha text")))


def api_error(kind: type[anthropic.APIStatusError], status: int, message: str) -> Exception:
    """An SDK status error, built the way the SDK builds one.

    The SDK vendors its HTTP client as ``httpx2``; its exceptions are typed
    against that, not against ``httpx``.
    """
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return kind(message, response=httpx2.Response(status, request=request), body=None)


CONNECTION_ERROR = anthropic.APIConnectionError(
    request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
)


class _RaisingStream:
    """Fails partway through streaming, as a dropped connection would."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.closed = False

    def __enter__(self) -> _RaisingStream:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.closed = True
        return False

    def __iter__(self) -> Any:
        raise self.error

    def get_final_message(self) -> Message:
        raise AssertionError("the stream failed; there is no final message")


class ScriptedClient:
    """Serves one outcome per call: a message to stream, or an error to raise."""

    def __init__(self, *outcomes: Message | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.streams: list[Any] = []
        self.messages = SimpleNamespace(stream=self._stream, count_tokens=self._count_tokens)

    def _stream(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        stream = (
            _RaisingStream(outcome) if isinstance(outcome, BaseException) else FakeStream(outcome)
        )
        self.streams.append(stream)
        return stream

    def _count_tokens(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(input_tokens=1_000)


def session_of(*documents: Any) -> Session:
    return Session(corpus_of(*(documents or (ALPHA, BRAVO))))


def ask(session: Session, client: ScriptedClient, question: str = "Q?") -> str:
    console = make_console()
    repl.ask(client, session, question, console)  # type: ignore[arg-type]
    return output(console)


def roles(session: Session) -> list[str]:
    return [message["role"] for message in session.messages]


def alternates(session: Session) -> bool:
    """The API rejects two messages in a row from the same role."""
    order = roles(session)
    return order[::2] == ["user"] * ((len(order) + 1) // 2) and order[1::2] == ["assistant"] * (
        len(order) // 2
    )


# --- Replay ------------------------------------------------------------------


def test_a_trailing_empty_text_block_is_dropped() -> None:
    content = [
        thinking_block("Checking Part A."),
        text_block("Sumps must be tight.", cite(0, 10)),
        text_block(""),
    ]

    assert replayable(content) == content[:2]


def test_every_other_block_is_replayed_unchanged() -> None:
    thinking = thinking_block("Checking Part A.")
    cited = text_block("Sumps.", cite(0, 10))
    kept = replayable([thinking, cited, text_block(""), text_block(" More.")])

    assert kept[0] is thinking and kept[0].signature == "sig"
    assert kept[1] is cited and kept[1].citations == [cite(0, 10)]
    assert [block.text for block in kept if block.type == "text"] == ["Sumps.", " More."]


# --- Requests ----------------------------------------------------------------


def test_the_first_question_shares_a_message_with_the_documents() -> None:
    session = session_of()

    user_message, request = session.request_for("  What is alpha?\n")

    assert user_message["role"] == "user"
    *documents, question = user_message["content"]
    assert [block["title"] for block in documents] == ["Part A: Alpha Rules", "Part B: Bravo Rules"]
    assert question == {"type": "text", "text": "What is alpha?"}
    assert request["messages"] == [user_message]


def test_a_follow_up_sends_the_conversation_and_only_the_new_question() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "First?")

    user_message, request = session.request_for("Second?")

    assert user_message["content"] == [{"type": "text", "text": "Second?"}]
    assert request["messages"] == [*session.messages, user_message]
    assert len(request["messages"]) == 3


def test_every_request_carries_the_conversation_tail_breakpoint() -> None:
    _, request = session_of().request_for("Q?")

    assert request["cache_control"] == {"type": "ephemeral", "ttl": CONVERSATION_CACHE_TTL}


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_an_empty_question_is_refused(question: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        session_of().request_for(question)


def test_an_empty_question_never_reaches_the_api() -> None:
    client = ScriptedClient()
    ask(session_of(), client, "   ")

    assert client.requests == []


# --- Conversation state ------------------------------------------------------


def test_a_successful_turn_is_committed_in_alternating_order() -> None:
    session = session_of()

    ask(session, ScriptedClient(CITED), "First?")
    ask(session, ScriptedClient(CITED), "Second?")

    assert roles(session) == ["user", "assistant", "user", "assistant"]
    assert session.turns == 2


@pytest.mark.parametrize(
    "error",
    [
        CONNECTION_ERROR,
        api_error(anthropic.RateLimitError, 429, "rate limited"),
        api_error(anthropic.BadRequestError, 400, "bad request"),
        api_error(anthropic.AuthenticationError, 401, "invalid key"),
        api_error(anthropic.InternalServerError, 500, "overloaded"),
        KeyboardInterrupt(),
    ],
)
def test_a_failed_turn_leaves_the_conversation_untouched(error: BaseException) -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "First?")
    before = list(session.messages)

    out = ask(session, ScriptedClient(error), "Second?")

    assert session.messages == before
    assert alternates(session)
    assert session.turns == 1
    assert out.strip(), "a dropped turn must say so"


def test_the_turn_after_a_failure_still_sends_a_valid_conversation() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "First?")
    ask(session, ScriptedClient(CONNECTION_ERROR), "Dropped?")

    client = ScriptedClient(CITED)
    ask(session, client, "Third?")

    [request] = client.requests
    assert [message["role"] for message in request["messages"]] == ["user", "assistant", "user"]
    assert request["messages"][-1]["content"] == [{"type": "text", "text": "Third?"}]


def test_a_failing_stream_is_closed() -> None:
    client = ScriptedClient(CONNECTION_ERROR)
    ask(session_of(), client, "Q?")

    assert client.streams[0].closed


def test_an_answer_ending_in_an_empty_block_is_committed_without_it() -> None:
    session = session_of()
    message = message_of(
        thinking_block("Checking."),
        text_block("Sumps.", cite(0, 10, text="Alpha text")),
        text_block(""),
    )

    ask(session, ScriptedClient(message), "Q?")

    committed = session.messages[1]["content"]
    assert [block.type for block in committed] == ["thinking", "text"]
    assert committed[0] is message.content[0]
    assert committed[1] is message.content[1]


def test_an_answer_with_no_sendable_text_is_not_committed() -> None:
    session = session_of()

    out = ask(session, ScriptedClient(message_of(text_block(""))), "Q?")

    assert session.messages == []
    assert "left out of the conversation" in out


# --- Reset -------------------------------------------------------------------


def test_reset_rebuilds_the_first_message_around_the_next_question() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "The first question")

    session.reset()
    client = ScriptedClient(CITED)
    ask(session, client, "A different question")

    [request] = client.requests
    content = request["messages"][0]["content"]
    texts = [block for block in content if block.get("type") == "text"]
    assert texts == [{"type": "text", "text": "A different question"}]
    assert [block["title"] for block in content[:-1]] == [
        "Part A: Alpha Rules",
        "Part B: Bravo Rules",
    ]


def test_reset_keeps_the_corpus_prefix_byte_identical() -> None:
    session = session_of()
    first = ScriptedClient(CITED)
    ask(session, first, "One?")
    session.reset()
    second = ScriptedClient(CITED)
    ask(session, second, "Two?")

    def prefix(client: ScriptedClient) -> str:
        # Serialized without sort_keys, so key order counts: the API hashes
        # bytes, and two dicts that compare equal can serialize differently.
        return json.dumps(client.requests[0]["messages"][0]["content"][:-1])

    assert prefix(first) == prefix(second)


def test_reset_clears_the_turn_count_but_not_the_bill() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "Q?")

    session.reset()

    assert session.turns == 0 and session.messages == []
    assert session.questions == 1 and session.cost > 0


# --- Cache -------------------------------------------------------------------


def test_the_first_turn_sets_the_expected_read_from_its_billed_write() -> None:
    session = session_of()

    assert session.record(usage_of(write_1h=199_727)) == []
    assert session.cached_prefix_tokens == 199_727


def test_a_prefix_already_warm_sets_the_same_expectation() -> None:
    session = session_of()

    assert session.record(usage_of(read=199_727)) == []
    assert session.cached_prefix_tokens == 199_727


def test_the_tail_write_is_not_counted_as_part_of_the_prefix() -> None:
    session = session_of()

    session.record(usage_of(write_1h=199_727, write_5m=40))

    assert session.cached_prefix_tokens == 199_727


def test_a_later_turn_reading_less_than_the_prefix_is_a_loud_miss() -> None:
    session = session_of()
    session.record(usage_of(write_1h=199_727))

    [warning] = session.record(usage_of(input_tokens=200_000, read=0))

    assert warning.startswith("CACHE MISS")
    assert "199,727" in warning
    assert "full input price" in warning


def test_a_rewritten_prefix_is_reported_as_an_expired_entry() -> None:
    session = session_of()
    session.record(usage_of(write_1h=199_727))

    [warning] = session.record(usage_of(write_1h=199_740))

    assert warning.startswith("CACHE MISS")
    assert "expired or been evicted" in warning


def test_a_later_turn_reading_the_prefix_and_a_tail_is_not_a_miss() -> None:
    session = session_of()
    session.record(usage_of(write_1h=199_727))

    assert session.record(usage_of(read=199_800)) == []


def test_a_first_turn_that_read_a_tail_does_not_make_the_next_turn_a_miss() -> None:
    # An earlier session cached this exact first turn, so the read covers the
    # question after the breakpoint as well as the corpus prefix before it.
    session = session_of()
    session.record(usage_of(read=199_760))

    # A /reset, then a different question: only the prefix is read.
    assert session.record(usage_of(read=199_727)) == []


def test_a_shortfall_past_the_tolerance_is_still_a_miss() -> None:
    session = session_of()
    session.record(usage_of(read=199_727))
    short = int(199_727 * (1 - CACHE_READ_TOLERANCE_FRACTION)) - 1

    [warning] = session.record(usage_of(read=short))

    assert warning.startswith("CACHE MISS")


def test_a_turn_with_no_cache_activity_sets_no_expectation() -> None:
    session = session_of()

    assert session.record(usage_of(input_tokens=199_727)) == []
    assert session.cached_prefix_tokens is None
    # The next turn is measured against nothing rather than against zero, so a
    # session that never cached does not report a miss on every turn.
    assert session.record(usage_of(input_tokens=199_727)) == []


def test_the_cache_miss_is_printed_with_the_answer() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "First?")
    session.cached_prefix_tokens = 199_727

    out = ask(session, ScriptedClient(CITED), "Second?")

    assert "CACHE MISS" in out


# --- Context window ----------------------------------------------------------


def test_a_request_past_the_warning_fraction_suggests_a_reset() -> None:
    session = session_of()
    session.record(usage_of(read=1_000))
    limit = int(CONTEXT_WINDOW * CONTEXT_WARN_FRACTION)

    warnings = session.record(usage_of(input_tokens=limit, read=1_000))

    assert any(warning.startswith("CONTEXT") and "/reset" in warning for warning in warnings)


def test_a_short_conversation_raises_no_context_warning() -> None:
    session = session_of()

    assert session.record(usage_of(input_tokens=10, write_1h=199_727)) == []


def test_the_request_size_is_this_turn_not_the_sum_of_turns() -> None:
    session = session_of()
    session.record(usage_of(input_tokens=10, write_1h=199_727))
    session.record(usage_of(input_tokens=20, read=199_727))

    assert session.request_tokens == 199_747


# --- Disclaimer --------------------------------------------------------------


def test_the_disclaimer_is_shown_once_per_session() -> None:
    session = session_of()

    first = ask(session, ScriptedClient(CITED), "First?")
    second = ask(session, ScriptedClient(CITED), "Second?")

    assert " ".join(first.split()).find(MATCH_DISCLAIMER) >= 0
    assert " ".join(second.split()).find(MATCH_DISCLAIMER) < 0


def test_a_reset_does_not_repeat_the_disclaimer() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "First?")

    session.reset()
    out = ask(session, ScriptedClient(CITED), "Second?")

    assert " ".join(out.split()).find(MATCH_DISCLAIMER) < 0


# --- Commands ----------------------------------------------------------------


def drive(session: Session, client: ScriptedClient, *lines: str) -> str:
    """Feed ``lines`` to the loop, then end input as Ctrl+D would."""
    console = make_console()
    remaining = list(lines)

    def read_line(prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    repl.run(client, session, console, read_line=read_line)  # type: ignore[arg-type]
    return output(console)


def test_questions_and_blank_lines_are_told_apart() -> None:
    session = session_of()
    client = ScriptedClient(CITED)

    drive(session, client, "", "   ", "What is alpha?")

    assert len(client.requests) == 1
    assert session.turns == 1


def test_exit_leaves_without_asking_anything() -> None:
    client = ScriptedClient()

    drive(session_of(), client, "/exit", "never read")

    assert client.requests == []


def test_end_of_input_leaves_the_loop() -> None:
    client = ScriptedClient(CITED)
    session = session_of()

    drive(session, client, "One?")

    assert session.turns == 1


def test_an_interrupt_at_the_prompt_does_not_leave() -> None:
    session = session_of()
    client = ScriptedClient(CITED)
    lines: list[Any] = [KeyboardInterrupt(), "Q?"]
    console = make_console()

    def read_line(prompt: str) -> str:
        if not lines:
            raise EOFError
        item = lines.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    repl.run(client, session, console, read_line=read_line)  # type: ignore[arg-type]

    assert session.turns == 1
    assert "/exit to leave." in output(console)


def test_reset_is_a_command_not_a_question() -> None:
    session = session_of()
    client = ScriptedClient(CITED)

    out = drive(session, client, "First?", "/reset")

    assert len(client.requests) == 1
    assert session.messages == []
    assert "Cleared 1 turn(s)" in out


def test_an_unknown_command_is_not_sent_as_a_question() -> None:
    client = ScriptedClient()

    out = drive(session_of(), client, "/nope")

    assert client.requests == []
    assert "Unknown command /nope" in out


def test_cost_reports_every_bucket() -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "Q?")

    out = drive(session, ScriptedClient(), "/cost")

    assert "1 question(s), $" in out
    assert "cache write" in out and "cache read" in out


def test_cost_before_any_question_says_so() -> None:
    assert "Nothing asked yet" in drive(session_of(), ScriptedClient(), "/cost")


# --- Reload ------------------------------------------------------------------


@pytest.fixture
def fake_reload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``/reload`` wired to an in-memory corpus, or to a failure."""
    state: dict[str, Any] = {"documents": [ALPHA, BRAVO]}

    def load_corpus(count_tokens: Any, **kwargs: Any) -> tuple[Corpus, Any]:
        if isinstance(state["documents"], BaseException):
            raise state["documents"]
        from corpus import FingerprintCheck, FingerprintStatus, assemble_corpus

        corpus = assemble_corpus(state["documents"], count_tokens)
        return corpus, FingerprintCheck(FingerprintStatus.UNCHANGED, "2026-09-13T00:00:00+00:00")

    monkeypatch.setattr(repl, "load_corpus", load_corpus)
    return state


def test_reload_picks_up_edited_files_and_clears_the_conversation(
    fake_reload: dict[str, Any],
) -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "Q?")
    session.cached_prefix_tokens = 199_727
    edited = type(ALPHA)(ALPHA.filename, ALPHA.label, "Alpha text, revised.\n\na) First.")
    fake_reload["documents"] = [edited, BRAVO]

    out = drive(session, ScriptedClient(), "/reload")

    assert session.corpus.documents[0].text == edited.text
    assert session.messages == []
    assert session.cached_prefix_tokens is None
    assert "writes the corpus cache afresh" in out


def test_reload_of_an_unchanged_corpus_keeps_the_cache_expectation(
    fake_reload: dict[str, Any],
) -> None:
    session = session_of()
    ask(session, ScriptedClient(CITED), "Q?")
    session.cached_prefix_tokens = 199_727

    out = drive(session, ScriptedClient(), "/reload")

    assert session.cached_prefix_tokens == 199_727
    assert "byte-identical" in out


def test_a_failed_reload_keeps_the_corpus_and_the_conversation(
    fake_reload: dict[str, Any],
) -> None:
    from extract import ExtractionError

    session = session_of()
    ask(session, ScriptedClient(CITED), "Q?")
    before = list(session.messages)
    fake_reload["documents"] = ExtractionError("admincode is empty")

    out = drive(session, ScriptedClient(), "/reload")

    assert session.messages == before
    assert session.corpus.documents[0].text == ALPHA.text
    assert "Reload failed" in out


# --- Rendering ---------------------------------------------------------------


def test_the_answer_and_its_checked_sources_are_printed() -> None:
    out = ask(session_of(), ScriptedClient(CITED), "Q?")

    assert "Sumps.[1]" in out
    assert "[1] Part A, characters 0-10  matched" in out


def test_documents_are_checked_against_the_corpus_actually_loaded() -> None:
    # The reporter must compare against the texts this session sent, so a
    # citation into BRAVO is checked against BRAVO.
    session = session_of()
    message = message_of(
        text_block("Bravo.", cite(0, 11, text="Bravo text.", document_index=1))
    )

    out = ask(session, ScriptedClient(message), "Q?")

    assert "matched" in out
    assert DOCUMENTS[1] == "Bravo text."


def test_an_error_message_cannot_drive_the_terminal() -> None:
    error = api_error(anthropic.BadRequestError, 400, "bad \x1b[31mrequest")

    out = ask(session_of(), ScriptedClient(error), "Q?")

    assert "\x1b" not in out
    assert "\\x1b[31mrequest" in out


def test_running_out_of_credit_is_named_as_such() -> None:
    error = api_error(
        anthropic.BadRequestError, 400, "Your credit balance is too low to access the API"
    )

    assert "Out of credit" in ask(session_of(), ScriptedClient(error), "Q?")
