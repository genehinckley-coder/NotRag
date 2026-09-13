"""Tests for plan item 5: the single-question answer path (``answer.py``).

All offline except the one test marked ``live``, which asks the sump question
for real and is skipped unless ``NOTRAG_LIVE=1`` is set. The fakes below
produce the stream events the SDK yields, carrying real SDK content types.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import (
    CacheCreation,
    CitationCharLocation,
    Message,
    TextBlock,
    ThinkingBlock,
    Usage,
)
from rich.console import Console

import answer
from answer import (
    build_request,
    cache_warning,
    cost_usd,
    render_summary,
    stop_warning,
    stream_answer,
)
from config import (
    CORPUS_CACHE_TTL,
    MATCH_DISCLAIMER,
    MAX_TOKENS,
    MODEL,
    OUTPUT_CONFIG,
    SYSTEM_PROMPT,
    THINKING,
)
from corpus import Corpus, FingerprintCheck, FingerprintStatus, assemble_corpus
from extract import ExtractedDocument, ExtractionError
from quotes import build_report

#: SHA-256 of ``config.SYSTEM_PROMPT``. The system prompt renders before the
#: documents, inside the cached prefix, so any edit cold-starts the corpus cache
#: (about $2 at the current corpus size). If this fails, the edit should be
#: deliberate: update the hash in the same commit.
SYSTEM_PROMPT_SHA256 = "d62477f1e9365fa5d13c21573612a035a15f8763b6466b3cfbb91e7dcf4e32cb"

SUMP_QUESTION = "what are the sump requirements for a UST in Illinois"

ALPHA = ExtractedDocument("a.htm", "Part A: Alpha Rules", "Alpha text.\n\na) First.")
BRAVO = ExtractedDocument("b.htm", "Part B: Bravo Rules", "Bravo text.")
#: The document texts sent, in send order, for checking quotes.
DOCUMENTS = [ALPHA.text, BRAVO.text]


def corpus_of(*documents: ExtractedDocument) -> Corpus:
    return assemble_corpus(documents, lambda blocks: 1_000)


def cite(start: int, end: int, text: str = "cited text", document_index: int = 0) -> CitationCharLocation:
    return CitationCharLocation(
        type="char_location",
        cited_text=text,
        document_index=document_index,
        document_title="Part A: Alpha Rules",
        start_char_index=start,
        end_char_index=end,
    )


def text_block(text: str, *citations: CitationCharLocation) -> TextBlock:
    return TextBlock(type="text", text=text, citations=list(citations) or None)


def thinking_block(thinking: str) -> ThinkingBlock:
    return ThinkingBlock(type="thinking", thinking=thinking, signature="sig")


def usage_of(
    *,
    input_tokens: int = 10,
    output_tokens: int = 100,
    write_1h: int = 0,
    write_5m: int = 0,
    read: int = 0,
    breakdown: bool = True,
) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=write_1h + write_5m,
        cache_read_input_tokens=read,
        cache_creation=(
            CacheCreation(ephemeral_1h_input_tokens=write_1h, ephemeral_5m_input_tokens=write_5m)
            if breakdown
            else None
        ),
    )


def message_of(
    *content: Any, stop_reason: str = "end_turn", usage: Usage | None = None
) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=MODEL,
        content=list(content),
        stop_reason=stop_reason,
        usage=usage or usage_of(write_1h=1_000),
    )


def api_error(kind: type[anthropic.APIStatusError], status: int, message: str) -> Exception:
    """An SDK status error, built the way the SDK builds one.

    The SDK vendors its HTTP client as ``httpx2``; its exceptions are typed
    against that, not against ``httpx``.
    """
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return kind(message, response=httpx2.Response(status, request=request), body=None)


def events_for(message: Message) -> list[Any]:
    """The stream events ``client.messages.stream`` would yield for ``message``.

    Mirrors ``anthropic.lib.streaming._messages.build_events``: every raw
    event is passed through, and each raw ``content_block_delta`` is followed
    by the ``text``, ``thinking``, ``citation`` or ``signature`` event derived
    from it. A renderer that also printed the raw deltas would print every
    answer twice.

    A text block's citation events come first and its text arrives in two
    deltas: the order in which a marker printed on arrival would be misplaced.
    """

    def delta(index: int, derived: SimpleNamespace, **fields: Any) -> list[SimpleNamespace]:
        raw = SimpleNamespace(
            type="content_block_delta", index=index, delta=SimpleNamespace(**fields)
        )
        return [raw, derived]

    events: list[Any] = [SimpleNamespace(type="message_start", message=message)]
    for index, block in enumerate(message.content):
        events.append(
            SimpleNamespace(
                type="content_block_start", index=index, content_block=SimpleNamespace(type=block.type)
            )
        )
        if block.type == "thinking":
            events += delta(
                index,
                SimpleNamespace(type="thinking", thinking=block.thinking),
                type="thinking_delta",
                thinking=block.thinking,
            )
            events += delta(
                index,
                SimpleNamespace(type="signature", signature=block.signature),
                type="signature_delta",
                signature=block.signature,
            )
        elif block.type == "text":
            for citation in block.citations or []:
                events += delta(
                    index,
                    SimpleNamespace(type="citation", citation=citation),
                    type="citations_delta",
                    citation=citation,
                )
            half = len(block.text) // 2
            for fragment in (block.text[:half], block.text[half:]):
                if fragment:
                    events += delta(
                        index,
                        SimpleNamespace(type="text", text=fragment),
                        type="text_delta",
                        text=fragment,
                    )
        events.append(SimpleNamespace(type="content_block_stop", index=index, content_block=block))
    events.append(SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=message.stop_reason)))
    events.append(SimpleNamespace(type="message_stop", message=message))
    return events


class FakeStream:
    def __init__(self, message: Message, *, interrupt_after: int | None = None) -> None:
        self.message = message
        self.events = events_for(message)
        self.interrupt_after = interrupt_after
        self.closed = False

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.closed = True
        return False

    def __iter__(self) -> Any:
        for index, event in enumerate(self.events):
            if index == self.interrupt_after:
                raise KeyboardInterrupt
            yield event

    def get_final_message(self) -> Message:
        return self.message


class FakeClient:
    """Streams one fixed message and records every request it is given."""

    def __init__(self, message: Message, *, interrupt_after: int | None = None) -> None:
        self.stream = FakeStream(message, interrupt_after=interrupt_after)
        self.stream_requests: list[dict[str, Any]] = []
        self.count_requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(stream=self._stream, count_tokens=self._count_tokens)

    def _stream(self, **kwargs: Any) -> FakeStream:
        self.stream_requests.append(kwargs)
        return self.stream

    def _count_tokens(self, **kwargs: Any) -> SimpleNamespace:
        self.count_requests.append(kwargs)
        return SimpleNamespace(input_tokens=1_000)


def make_console() -> Console:
    return Console(file=io.StringIO(), width=200, color_system=None, highlight=False)


def output(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined]


def run(message: Message) -> tuple[answer.Answer, str]:
    console = make_console()
    result = stream_answer(FakeClient(message), {"model": MODEL}, console)  # type: ignore[arg-type]
    return result, output(console)


# --- System prompt -----------------------------------------------------------


def test_system_prompt_is_pinned() -> None:
    assert hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() == SYSTEM_PROMPT_SHA256


# --- Request -----------------------------------------------------------------


def test_request_sends_every_document_then_the_question() -> None:
    request = build_request(corpus_of(ALPHA, BRAVO), "  What is alpha?\n")

    [message] = request["messages"]
    assert message["role"] == "user"
    *documents, question = message["content"]
    assert [block["title"] for block in documents] == ["Part A: Alpha Rules", "Part B: Bravo Rules"]
    assert question == {"type": "text", "text": "What is alpha?"}


def test_only_the_last_document_carries_the_breakpoint() -> None:
    content = build_request(corpus_of(ALPHA, BRAVO), "Q?")["messages"][0]["content"]

    assert ["cache_control" in block for block in content] == [False, True, False]
    assert content[1]["cache_control"] == {"type": "ephemeral", "ttl": CORPUS_CACHE_TTL}


def test_request_uses_the_configured_model_settings() -> None:
    request = build_request(corpus_of(ALPHA), "Q?")

    assert request["model"] == MODEL
    assert request["system"] == SYSTEM_PROMPT
    assert request["thinking"] == THINKING
    assert request["output_config"] == OUTPUT_CONFIG
    assert request["max_tokens"] == MAX_TOKENS


def test_request_never_asks_for_structured_output() -> None:
    # output_config.format is incompatible with citations.
    assert "format" not in build_request(corpus_of(ALPHA), "Q?")["output_config"]


def test_max_tokens_leaves_room_for_thinking() -> None:
    assert build_request(corpus_of(ALPHA), "Q?")["max_tokens"] > 16_000


def test_different_questions_share_a_byte_identical_prefix() -> None:
    corpus = corpus_of(ALPHA, BRAVO)
    first, second = build_request(corpus, "One?"), build_request(corpus, "Two?")

    def prefix(request: dict[str, Any]) -> str:
        return json.dumps([request["system"], request["messages"][0]["content"][:-1]], sort_keys=True)

    assert prefix(first) == prefix(second)
    assert first["messages"][0]["content"][-1] != second["messages"][0]["content"][-1]


def test_mutating_one_request_cannot_alter_the_next() -> None:
    corpus = corpus_of(ALPHA)
    first = build_request(corpus, "One?")
    first["thinking"]["type"] = "disabled"
    first["output_config"]["effort"] = "low"
    first["messages"][0]["content"][0]["title"] = "tampered"

    second = build_request(corpus, "Two?")
    assert second["thinking"] == THINKING
    assert second["output_config"] == OUTPUT_CONFIG
    assert second["messages"][0]["content"][0]["title"] == "Part A: Alpha Rules"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_an_empty_question_is_refused(question: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        build_request(corpus_of(ALPHA), question)


# --- Stop reason -------------------------------------------------------------


def test_end_turn_carries_no_warning() -> None:
    assert stop_warning("end_turn") is None


def test_max_tokens_is_reported_as_truncation() -> None:
    warning = stop_warning("max_tokens")
    assert warning is not None
    assert warning.startswith("TRUNCATED")
    assert f"{MAX_TOKENS:,}" in warning


def test_refusal_is_reported() -> None:
    warning = stop_warning("refusal")
    assert warning is not None and warning.startswith("REFUSED")


@pytest.mark.parametrize(
    "stop_reason", ["stop_sequence", "pause_turn", "model_context_window_exceeded", None]
)
def test_any_other_stop_reason_is_reported_as_incomplete(stop_reason: str | None) -> None:
    warning = stop_warning(stop_reason)
    assert warning is not None and warning.startswith("INCOMPLETE")


# --- Streaming ---------------------------------------------------------------


def test_the_request_is_sent_unchanged() -> None:
    client = FakeClient(message_of(text_block("Hi.")))
    request = build_request(corpus_of(ALPHA), "Q?")

    stream_answer(client, request, make_console())  # type: ignore[arg-type]

    assert client.stream_requests == [request]


def test_markers_follow_each_cited_block() -> None:
    result, out = run(
        message_of(
            thinking_block("Checking Part A."),
            text_block("Sumps must be tight.", cite(0, 10)),
            text_block(" They are tested yearly.", cite(20, 30)),
            text_block(" Nothing else applies."),
            text_block(""),
        )
    )

    assert "Sumps must be tight.[1] They are tested yearly.[2] Nothing else applies." in out
    assert len(result.sources) == 2


def test_thinking_renders_before_the_answer() -> None:
    _, out = run(message_of(thinking_block("Checking Part A."), text_block("Answer text.")))

    positions = [
        out.index(part)
        for part in ("Thinking (summary)", "Checking Part A.", "Answer", "Answer text.")
    ]
    assert positions == sorted(positions)


def test_a_span_cited_twice_gets_one_number() -> None:
    result, out = run(
        message_of(
            text_block("First.", cite(0, 10)),
            text_block(" Second.", cite(0, 10), cite(20, 30), cite(20, 30)),
        )
    )

    assert "First.[1] Second.[1][2]" in out
    assert [(c.start_char_index, c.end_char_index) for c in result.sources] == [(0, 10), (20, 30)]


def test_the_same_offsets_in_another_document_are_a_different_source() -> None:
    result, _ = run(
        message_of(text_block("A.", cite(0, 10, document_index=0), cite(0, 10, document_index=1)))
    )

    assert len(result.sources) == 2


def test_control_characters_in_the_stream_are_escaped() -> None:
    _, out = run(message_of(thinking_block("t\x1b[2J"), text_block("a\x1b[31mb")))

    assert "\x1b" not in out
    assert "a\\x1b[31mb" in out
    assert "t\\x1b[2J" in out


def test_a_normal_answer_has_no_stop_warning() -> None:
    result, _ = run(message_of(text_block("Done.")))
    assert result.stop_warning is None


def test_a_truncated_answer_carries_the_warning() -> None:
    result, _ = run(message_of(text_block("Half an ans", cite(0, 10)), stop_reason="max_tokens"))

    assert result.stop_warning is not None
    assert result.stop_warning.startswith("TRUNCATED")


def test_an_interrupt_propagates_and_closes_the_stream() -> None:
    client = FakeClient(message_of(text_block("Long answer.")), interrupt_after=2)

    with pytest.raises(KeyboardInterrupt):
        stream_answer(client, {"model": MODEL}, make_console())  # type: ignore[arg-type]
    assert client.stream.closed


def test_a_citation_only_in_the_final_message_still_becomes_a_source() -> None:
    # The stream's stop event shows the block uncited; the final message does not.
    client = FakeClient(message_of(text_block("Sumps.", cite(0, 10))))
    client.stream.events = events_for(message_of(text_block("Sumps.")))

    result = stream_answer(client, {"model": MODEL}, make_console())  # type: ignore[arg-type]

    assert [(c.start_char_index, c.end_char_index) for c in result.sources] == [(0, 10)]


# --- Summary -----------------------------------------------------------------


def summary_of(message: Message, documents: Sequence[str] = DOCUMENTS) -> str:
    result, _ = run(message)
    console = make_console()
    report = build_report(result.sources, result.message.content, documents)
    render_summary(result, report, console)
    return output(console)


def test_the_sources_list_numbers_each_citation() -> None:
    out = summary_of(
        message_of(
            text_block("A.", cite(0, 10, text="Section 175.410 Tank Sumps")),
            text_block(" B.", cite(1_200, 1_350, text="b) Capacity.")),
        )
    )

    assert "[1] Part A, characters 0-10" in out
    assert '"Section 175.410 Tank Sumps"' in out
    assert "[2] Part A, characters 1,200-1,350" in out


def test_quoted_text_is_flattened_truncated_and_escaped() -> None:
    long_quote = "a)  Sumps\n\nshall\x1b[31m be " + "tight " * 60
    out = summary_of(message_of(text_block("A.", cite(0, 400, text=long_quote))))

    assert "\x1b" not in out
    assert '"a) Sumps shall\\x1b[31m be tight' in out
    assert '..."' in out


def test_the_truncation_warning_comes_before_the_sources() -> None:
    out = summary_of(message_of(text_block("Half", cite(0, 10)), stop_reason="max_tokens"))

    assert out.index("TRUNCATED") < out.index("Sources")


def test_an_answer_without_citations_is_flagged() -> None:
    assert "No citations" in summary_of(message_of(text_block("Unsupported claim.")))


def test_each_source_shows_its_match_status_and_why_it_failed() -> None:
    # ALPHA.text is "Alpha text.\n\na) First."
    out = summary_of(
        message_of(
            text_block("Alpha.", cite(0, 10, text="Alpha text")),
            text_block(" First.", cite(13, 22, text="a) Frist.")),
            text_block(" More.", cite(0, 99, text="Alpha text.")),
        )
    )

    assert "[1] Part A, characters 0-10  matched" in out
    assert "[2] Part A, characters 13-22  MISMATCH" in out
    assert "departs from the source at character 4 of the span" in out
    assert "[3] Part A, characters 0-99  OUT OF RANGE" in out
    assert "fall outside document 0, which has 22" in out
    assert (
        "Quotes: 1 of 3 distinct cited spans matched their source exactly. "
        "2 did not: [2] mismatch, [3] out of range." in out
    )


def test_a_repeated_span_is_counted_once_in_the_summary() -> None:
    out = summary_of(
        message_of(
            text_block("A.", cite(0, 10, text="Alpha text")),
            text_block(" B.", cite(0, 10, text="Alpha text")),
        )
    )

    assert "1 of 1 distinct cited spans matched their source exactly (2 citations;" in out


def test_uncited_text_and_thin_blocks_are_reported() -> None:
    claim = "Alpha rules require every tank to be inspected, logged, and certified annually."
    out = summary_of(
        message_of(
            text_block("In short:"),
            text_block(f" {claim}", cite(0, 10, text="Alpha text")),
        )
    )

    assert "Uncited answer text: " in out
    assert "Thinly sourced: 1 cited block(s)" in out
    assert f'[1] "{claim}"' in out


def test_the_disclaimer_is_shown_and_verified_is_never_claimed() -> None:
    out = summary_of(message_of(text_block("Alpha.", cite(0, 10, text="Alpha text"))))

    assert " ".join(out.split()).find(MATCH_DISCLAIMER) >= 0
    assert "verified" not in out.lower()


def test_the_usage_line_shows_every_bucket_and_the_cost() -> None:
    out = summary_of(
        message_of(
            text_block("A."),
            usage=usage_of(input_tokens=12, write_1h=199_700, output_tokens=3_000),
        )
    )

    assert "input 12 | cache write 199,700 | cache read 0 | output 3,000 | $2.07" in out


# --- Cost and cache ----------------------------------------------------------


def test_cost_prices_every_input_bucket_separately() -> None:
    usage = usage_of(
        input_tokens=1_000_000,
        write_1h=1_000_000,
        write_5m=1_000_000,
        read=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost_usd(usage) == pytest.approx(5.00 + 10.00 + 6.25 + 0.50 + 25.00)


def test_a_write_without_a_ttl_breakdown_is_priced_at_the_1h_rate() -> None:
    usage = usage_of(input_tokens=0, output_tokens=0, write_1h=1_000_000, breakdown=False)
    assert cost_usd(usage) == pytest.approx(10.00)


def test_no_cache_activity_is_flagged() -> None:
    assert cache_warning(usage_of()) is not None
    assert cache_warning(usage_of(write_1h=1)) is None
    assert cache_warning(usage_of(read=1)) is None


# --- Command line ------------------------------------------------------------


@pytest.fixture
def fake_app(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``main`` wired to a fake client and an in-memory corpus: no API, no state file."""
    app: dict[str, Any] = {
        "message": message_of(text_block("Sumps.", cite(0, 10, text="Alpha text")))
    }

    def build_client() -> FakeClient:
        app["client"] = FakeClient(app["message"], interrupt_after=app.get("interrupt_after"))
        return app["client"]

    def load_corpus(count_tokens: Any) -> tuple[Corpus, FingerprintCheck]:
        corpus = assemble_corpus([ALPHA, BRAVO], count_tokens)
        return corpus, FingerprintCheck(FingerprintStatus.UNCHANGED, "2026-09-12T00:00:00+00:00")

    monkeypatch.setattr(answer, "build_client", build_client)
    monkeypatch.setattr(answer, "load_corpus", load_corpus)
    return app


def test_main_counts_the_corpus_with_the_system_prompt(fake_app: dict[str, Any]) -> None:
    assert answer.main(["What", "is", "alpha?"]) == 0

    client = fake_app["client"]
    [count_request] = client.count_requests
    assert count_request["system"] == SYSTEM_PROMPT
    [stream_request] = client.stream_requests
    assert stream_request["messages"][0]["content"][-1]["text"] == "What is alpha?"


def test_main_prints_the_answer_and_its_sources(
    fake_app: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    answer.main(["Q?"])

    out = capsys.readouterr().out
    assert "Sumps.[1]" in out
    assert "[1] Part A, characters 0-10  matched" in out


def test_main_exits_nonzero_when_a_quote_fails(fake_app: dict[str, Any]) -> None:
    fake_app["message"] = message_of(text_block("Sumps.", cite(0, 10, text="fabricated")))
    assert answer.main(["Q?"]) == 1


def test_main_exits_nonzero_on_a_truncated_answer(fake_app: dict[str, Any]) -> None:
    fake_app["message"] = message_of(text_block("Half"), stop_reason="max_tokens")
    assert answer.main(["Q?"]) == 1


def test_main_exits_130_on_interrupt(fake_app: dict[str, Any]) -> None:
    fake_app["interrupt_after"] = 1
    assert answer.main(["Q?"]) == 130


def test_main_refuses_an_empty_question_without_calling_the_api(
    fake_app: dict[str, Any],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        answer.main(["   "])
    assert exit_info.value.code == 2
    assert "client" not in fake_app


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (api_error(anthropic.RateLimitError, 429, "slow \x1b[31mdown"), 2),
        (api_error(anthropic.InternalServerError, 500, "slow \x1b[31mdown"), 2),
        (ExtractionError("slow \x1b[31mdown"), 1),
    ],
    ids=["rate-limit", "api-status", "extraction"],
)
def test_main_escapes_error_text(
    fake_app: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    exit_code: int,
) -> None:
    def failing_load(count_tokens: Any) -> None:
        raise error

    monkeypatch.setattr(answer, "load_corpus", failing_load)

    assert answer.main(["Q?"]) == exit_code
    err = capsys.readouterr().err
    assert "\x1b" not in err
    assert "slow \\x1b[31mdown" in err


# --- Live --------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("NOTRAG_LIVE") != "1",
    reason="calls the real API and costs money (about $2 cold); set NOTRAG_LIVE=1 to run",
)
def test_live_sump_question_cites_section_175_410(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = answer.main([SUMP_QUESTION])

    out = capsys.readouterr().out
    with capsys.disabled():
        print(out)
    assert exit_code == 0
    assert "175.410" in out
