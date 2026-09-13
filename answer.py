"""Plan item 5: the single-question answer path.

One streaming call: the system prompt, then every document block (citations
enabled, the 1-hour cache breakpoint on the last), then the question, with
adaptive thinking at effort "high". The corpus comes from item 4's
``load_corpus``, counted together with the system prompt, since both sit in the
cached prefix.

The answer is rendered as it streams -- the thinking summary dimmed, then the
answer text with a ``[n]`` marker after each cited text block -- so a long
answer never looks like a hung terminal. ``stop_reason`` is checked on every
response: an answer cut off at ``max_tokens`` still carries citations, so
without the check it would pass for a complete one.

After the answer, every distinct cited span is checked against the text sent
(``quotes.py``, plan item 6) and its outcome shown in the sources list.

    python answer.py what are the sump requirements for a UST in Illinois
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import CitationCharLocation, Message, TextCitation, Usage
from rich.console import Console
from rich.text import Text

from config import (
    CONVERSATION_CACHE_TTL,
    MATCH_DISCLAIMER,
    MAX_TOKENS,
    MODEL,
    OUTPUT_CONFIG,
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_1H_PER_MTOK,
    PRICE_CACHE_WRITE_5M_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    SYSTEM_PROMPT,
    THIN_COVERAGE_FRACTION,
    THINKING,
    TOKEN_CEILING,
)
from corpus import (
    Corpus,
    CorpusTooLargeError,
    FingerprintCheck,
    FingerprintStatus,
    api_token_counter,
    describe_check,
    load_corpus,
)
from credentials import build_client
from display import escape_controls
from extract import ExtractionError
from quotes import MatchStatus, QuoteReport, build_report, citation_key

#: Longest excerpt of a citation's quoted text shown in the sources list.
_EXCERPT_CHARS = 160

_MARKER_STYLE = "bold cyan"


@dataclass(frozen=True)
class Answer:
    """One completed response and what the renderer learned from it."""

    message: Message
    #: Distinct citations in the order first cited; marker ``[n]`` is ``sources[n - 1]``.
    sources: tuple[TextCitation, ...]
    #: Why the answer cannot be taken as complete, or None when it ended normally.
    stop_warning: str | None
    #: Seconds from sending the request to the final message.
    elapsed: float


# --- Request -----------------------------------------------------------------


def first_turn(corpus: Corpus, question: str) -> dict[str, Any]:
    """``messages[0]``: every document block, then the question.

    The API requires the first message to be a user message, so the documents
    and the first question necessarily share it. The blocks are rebuilt from
    ``corpus`` every time, so nothing a previous turn did can alter the cached
    prefix.

    Raises:
        ValueError: ``question`` is empty or only whitespace. The API rejects
            an empty text block, so this fails before any call is made.
    """
    question = question.strip()
    if not question:
        raise ValueError("The question is empty")
    return {
        "role": "user",
        "content": [*corpus.document_blocks(), {"type": "text", "text": question}],
    }


def request_kwargs(
    messages: list[dict[str, Any]], *, tail_cache: bool = False
) -> dict[str, Any]:
    """The keyword arguments for ``client.messages.stream`` around ``messages``.

    Everything before the question -- the system prompt, then the document
    blocks -- is identical on every call, which is what lets the corpus cache
    hit.

    Args:
        messages: The conversation so far, ending in the new user message.
        tail_cache: Add a top-level breakpoint covering the conversation tail,
            at the shorter ``config.CONVERSATION_CACHE_TTL``. Item 2's spike
            confirmed it produces a tail read alongside the 1-hour document
            breakpoint. Only the REPL needs it; one question has no tail.
    """
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "thinking": dict(THINKING),
        "output_config": dict(OUTPUT_CONFIG),
    }
    if tail_cache:
        # The shorter TTL lands after the 1h document breakpoint, as the API
        # requires.
        kwargs["cache_control"] = {"type": "ephemeral", "ttl": CONVERSATION_CACHE_TTL}
    return kwargs


def build_request(corpus: Corpus, question: str) -> dict[str, Any]:
    """One question against the corpus, as a complete request."""
    return request_kwargs([first_turn(corpus, question)])


# --- Response checks ---------------------------------------------------------


def stop_warning(stop_reason: str | None) -> str | None:
    """Why an answer that ended with ``stop_reason`` is not complete, or None."""
    if stop_reason == "end_turn":
        return None
    if stop_reason == "max_tokens":
        return (
            f"TRUNCATED: the response reached the {MAX_TOKENS:,}-token output limit "
            f"(config.MAX_TOKENS; thinking counts toward it), so the answer above "
            f"stops partway. Its citations point at real text, but the answer is "
            f"incomplete: do not rely on it. Ask a narrower question, or raise "
            f"config.MAX_TOKENS."
        )
    if stop_reason == "refusal":
        return (
            "REFUSED: the model declined to answer (stop_reason 'refusal'). Any text "
            "above is not an answer."
        )
    return (
        f"INCOMPLETE: the response ended with stop_reason {stop_reason!r} rather "
        f"than 'end_turn', so the answer above may stop partway."
    )


def cache_warning(usage: Usage) -> str | None:
    """A warning when the response shows no cache activity at all, else None."""
    if (usage.cache_creation_input_tokens or 0) or (usage.cache_read_input_tokens or 0):
        return None
    return (
        "NOT CACHED: the response reports no cache write and no cache read, so the "
        "corpus breakpoint did not take effect and every question pays the full "
        "input price."
    )


def cost_usd(usage: Usage) -> float:
    """What a response cost, in USD, at config prices.

    ``input_tokens`` excludes cached tokens, so the three input buckets are
    priced separately; a single multiply under-reports a cold call about
    tenfold.
    """
    creation = usage.cache_creation
    written = usage.cache_creation_input_tokens or 0
    # A REPL request carries two breakpoints: the 1-hour corpus prefix and the
    # 5-minute conversation tail. Without a TTL breakdown there is no way to
    # tell them apart, so the whole write is priced at the 1-hour rate -- the
    # higher of the two, and the one the corpus actually pays, so the figure
    # errs high rather than low. Item 2 measured the breakdown as present in
    # practice; this is the defensive path.
    write_1h = creation.ephemeral_1h_input_tokens if creation else written
    write_5m = creation.ephemeral_5m_input_tokens if creation else 0
    return (
        usage.input_tokens * PRICE_INPUT_PER_MTOK
        + write_1h * PRICE_CACHE_WRITE_1H_PER_MTOK
        + write_5m * PRICE_CACHE_WRITE_5M_PER_MTOK
        + (usage.cache_read_input_tokens or 0) * PRICE_CACHE_READ_PER_MTOK
        + usage.output_tokens * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000


# --- Streaming ---------------------------------------------------------------


class _Renderer:
    """Writes stream events to the console as they arrive, and numbers citations."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._section: str | None = None
        self._numbers: dict[Hashable, int] = {}
        self.sources: list[TextCitation] = []

    def handle(self, event: Any) -> None:
        if event.type == "content_block_start":
            self._start(event.content_block.type)
        elif event.type == "thinking":
            self._write(event.thinking, "dim")
        elif event.type == "text":
            self._write(event.text)
        elif event.type == "content_block_stop" and event.content_block.type == "text":
            # Markers go at the end of the block rather than on each citation
            # event: a citation can arrive before the text it cites, and it
            # attaches to the whole block anyway.
            numbers = self.number(event.content_block.citations or [])
            if numbers:
                self._write("".join(f"[{n}]" for n in numbers), _MARKER_STYLE)

    def number(self, citations: Sequence[TextCitation]) -> list[int]:
        """Number each citation not seen before; return these citations' numbers once each."""
        numbers: list[int] = []
        for citation in citations:
            key = citation_key(citation)
            if key not in self._numbers:
                self.sources.append(citation)
                self._numbers[key] = len(self.sources)
            if self._numbers[key] not in numbers:
                numbers.append(self._numbers[key])
        return numbers

    def finish(self) -> None:
        if self._section is not None:
            self._console.print()

    def _start(self, block_type: str) -> None:
        section = "answer" if block_type == "text" else "thinking"
        if section != self._section:
            if self._section is not None:
                self._console.print()
            title = "Answer" if section == "answer" else "Thinking (summary)"
            self._console.rule(title, style="dim", align="left")
            self._section = section
        if block_type == "redacted_thinking":
            self._write("(redacted)", "dim")

    def _write(self, text: str, style: str = "") -> None:
        # soft_wrap: rich cannot know the cursor column between deltas, so
        # wrapping is left to the terminal instead of re-wrapping each fragment.
        self._console.print(Text(escape_controls(text), style=style), end="", soft_wrap=True)


def stream_answer(
    client: anthropic.Anthropic, request: dict[str, Any], console: Console
) -> Answer:
    """Send ``request``, render the answer as it streams, and return it.

    Raises whatever the SDK raises, and ``KeyboardInterrupt`` on Ctrl+C;
    leaving the ``with`` block closes the connection either way.
    """
    renderer = _Renderer(console)
    started = time.monotonic()
    with client.messages.stream(**request) as stream:
        for event in stream:
            renderer.handle(event)
        message = stream.get_final_message()
    renderer.finish()

    # Normally a no-op, since every cited block has already passed through
    # content_block_stop. It keeps the sources list complete if one did not.
    for block in message.content:
        if block.type == "text":
            renderer.number(block.citations or [])

    return Answer(
        message=message,
        sources=tuple(renderer.sources),
        stop_warning=stop_warning(message.stop_reason),
        elapsed=time.monotonic() - started,
    )


# --- Summary -----------------------------------------------------------------


def render_summary(
    answer: Answer, report: QuoteReport, console: Console, *, show_disclaimer: bool = True
) -> None:
    """After the streamed answer: any stop warning, the checked sources, and usage.

    The stop warning cannot go above the answer, which is already on screen by
    the time ``stop_reason`` arrives. It goes first here, before the sources.

    ``report`` is :func:`quotes.build_report` over this answer, so
    ``report.checks`` lines up with ``answer.sources`` and the ``[n]`` markers.

    Args:
        show_disclaimer: Print ``config.MATCH_DISCLAIMER`` below the report.
            The one-question CLI shows it every time; the REPL shows it once
            per session (plan item 7).
    """
    if answer.stop_warning:
        console.print()
        console.print(Text(answer.stop_warning, style="bold red"))

    console.print()
    if report.checks:
        console.rule("Sources", style="dim", align="left")
        for number, check in enumerate(report.checks, start=1):
            console.print(
                Text.assemble(
                    (f"[{number}] ", _MARKER_STYLE),
                    _source_heading(check.citation),
                    "  ",
                    _status_label(check.status),
                )
            )
            console.print(Text(f'    "{_excerpt(check.citation.cited_text)}"', style="dim"))
            if check.detail:
                console.print(Text(f"    {escape_controls(check.detail)}", style="red"))
        console.print()
        _render_quote_report(report, console)
    else:
        console.print(
            Text("No citations: nothing in this answer is tied to a source passage.", style="yellow")
        )
    if show_disclaimer:
        console.print(Text(MATCH_DISCLAIMER, style="dim"))

    usage = answer.message.usage
    warning = cache_warning(usage)
    if warning:
        console.print(Text(warning, style="yellow"))
    console.print()
    console.print(Text(usage_line(answer), style="dim"))


def usage_line(answer: Answer) -> str:
    usage = answer.message.usage
    return (
        f"{answer.elapsed:.0f}s | stop {answer.message.stop_reason} | input "
        f"{usage.input_tokens:,} | cache write {usage.cache_creation_input_tokens or 0:,} | "
        f"cache read {usage.cache_read_input_tokens or 0:,} | output "
        f"{usage.output_tokens:,} | ${cost_usd(usage):.2f}"
    )


def _render_quote_report(report: QuoteReport, console: Console) -> None:
    """Quotes matched, each failure by category, and how much of the answer is sourced."""
    total = len(report.checks)
    line = (
        f"Quotes: {report.matched_count} of {total} distinct cited spans matched their "
        f"source exactly"
    )
    if report.citation_count > total:
        line += f" ({report.citation_count} citations; a span cited more than once counts once)"
    failures = report.failures
    if failures:
        listed = ", ".join(f"[{number}] {check.status.value}" for number, check in failures)
        line += f". {len(failures)} did not: {listed}."
    else:
        line += "."
    console.print(Text(line, style="bold red" if failures else "green"))

    console.print(
        Text(
            f"Uncited answer text: {report.uncited_fraction:.0%} ({report.uncited_chars:,} of "
            f"{report.answer_chars:,} non-whitespace characters are in blocks with no "
            f"citation).",
            style="dim",
        )
    )

    thin = report.thin_blocks
    if thin:
        console.print(
            Text(
                f"Thinly sourced: {len(thin)} cited block(s) whose matched source text is "
                f"under {THIN_COVERAGE_FRACTION:.0%} of the block's length. A length proxy, "
                f"not a check: read these claims against their sources.",
                style="yellow",
            )
        )
        for block in thin:
            markers = "".join(f"[{number}]" for number in block.source_numbers)
            console.print(
                Text.assemble(
                    f"  {block.fraction:>4.0%} ", (markers, _MARKER_STYLE), f' "{_excerpt(block.text)}"'
                )
            )


def _status_label(status: MatchStatus) -> tuple[str, str]:
    if status is MatchStatus.MATCHED:
        return (status.value, "green")
    return (status.value.upper(), "bold red")


def _source_heading(citation: TextCitation) -> str:
    if isinstance(citation, CitationCharLocation):
        title = citation.document_title or f"document {citation.document_index}"
        # "41 Ill. Adm. Code 175: Technical ..." -> "41 Ill. Adm. Code 175"
        return (
            f"{escape_controls(title.split(':')[0])}, characters "
            f"{citation.start_char_index:,}-{citation.end_char_index:,}"
        )
    return f"({escape_controls(citation.type)} citation)"


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > _EXCERPT_CHARS:
        flat = flat[: _EXCERPT_CHARS - 3].rstrip() + "..."
    return escape_controls(flat)


# --- Command line ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan item 5: answer one question from the corpus."
    )
    parser.add_argument("question", nargs="+", help="the question; quoting it is optional")
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    if not question:
        parser.error("the question is empty")

    console = Console(highlight=False)
    try:
        client = build_client()
    except (FileNotFoundError, ValueError) as exc:
        # credentials.py never puts the key itself in an error message.
        print(f"Cannot load the API key: {escape_controls(str(exc))}", file=sys.stderr)
        return 2

    # Exception text can carry an API response body or a file name from disk,
    # so it is escaped like any other untrusted text.
    try:
        corpus, check = load_corpus(api_token_counter(client, system=SYSTEM_PROMPT))
        render_preamble(console, corpus, check)
        console.print()
        console.print(Text(f"Q: {escape_controls(question)}", style="bold"))
        answer = stream_answer(client, build_request(corpus, question), console)
    except KeyboardInterrupt:
        console.print()
        print("Interrupted. The answer is incomplete and was not kept.", file=sys.stderr)
        return 130
    except (ExtractionError, CorpusTooLargeError) as exc:
        print(f"Corpus assembly failed: {escape_controls(str(exc))}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("Authentication failed. Check the key file in config.API_KEY_PATH.", file=sys.stderr)
        return 2
    except anthropic.RateLimitError as exc:
        print(f"Rate limited or out of quota: {escape_controls(exc.message)}", file=sys.stderr)
        return 2
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {escape_controls(exc.message)}", file=sys.stderr)
        return 2
    except anthropic.APIConnectionError:
        print("Network error reaching the API.", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Could not record the corpus fingerprint: {escape_controls(str(exc))}", file=sys.stderr)
        return 1

    report = build_report(
        answer.sources, answer.message.content, [document.text for document in corpus.documents]
    )
    render_summary(answer, report, console)
    return 0 if answer.stop_warning is None and not report.failures else 1


def render_preamble(console: Console, corpus: Corpus, check: FingerprintCheck) -> None:
    """The corpus size, what it costs to cache, and the fingerprint comparison."""
    write = corpus.tokens * PRICE_CACHE_WRITE_1H_PER_MTOK / 1_000_000
    read = corpus.tokens * PRICE_CACHE_READ_PER_MTOK / 1_000_000
    console.print(
        Text(
            f"Corpus: {len(corpus.documents)} documents, {corpus.tokens:,} tokens with the "
            f"system prompt ({corpus.tokens / TOKEN_CEILING:.1%} of the {TOKEN_CEILING:,} "
            f"ceiling). A cold corpus cache costs about ${write:.2f} to write; a warm one "
            f"about ${read:.2f} to read.",
            style="dim",
        )
    )
    warn = check.status in (FingerprintStatus.CHANGED, FingerprintStatus.UNREADABLE)
    console.print(
        Text(f"Fingerprint: {describe_check(check, corpus)}", style="yellow" if warn else "dim")
    )


if __name__ == "__main__":
    # Regulation text and model output include characters such as U+2265 that a
    # cp1252 console cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
