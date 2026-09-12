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

This module lists what each citation points at. Whether the quoted text
matches the source is plan item 6.

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
    MAX_TOKENS,
    MODEL,
    OUTPUT_CONFIG,
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_1H_PER_MTOK,
    PRICE_CACHE_WRITE_5M_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    SYSTEM_PROMPT,
    THINKING,
    TOKEN_CEILING,
)
from corpus import (
    Corpus,
    CorpusTooLargeError,
    FingerprintStatus,
    api_token_counter,
    describe_check,
    load_corpus,
)
from credentials import build_client
from display import escape_controls
from extract import ExtractionError

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


def build_request(corpus: Corpus, question: str) -> dict[str, Any]:
    """The keyword arguments for ``client.messages.stream``.

    Everything before the question -- the system prompt, then the document
    blocks -- is identical on every call, which is what lets the corpus cache
    hit. The question is the only part that varies.

    Raises:
        ValueError: ``question`` is empty or only whitespace. The API rejects
            an empty text block, so this fails before any call is made.
    """
    question = question.strip()
    if not question:
        raise ValueError("The question is empty")
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [*corpus.document_blocks(), {"type": "text", "text": question}],
            }
        ],
        "thinking": dict(THINKING),
        "output_config": dict(OUTPUT_CONFIG),
    }


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
    # Without a TTL breakdown, price the whole write at the 1-hour rate: it is
    # the only breakpoint this request carries, and the higher of the two
    # prices, so the figure is never low.
    write_1h = creation.ephemeral_1h_input_tokens if creation else written
    write_5m = creation.ephemeral_5m_input_tokens if creation else 0
    return (
        usage.input_tokens * PRICE_INPUT_PER_MTOK
        + write_1h * PRICE_CACHE_WRITE_1H_PER_MTOK
        + write_5m * PRICE_CACHE_WRITE_5M_PER_MTOK
        + (usage.cache_read_input_tokens or 0) * PRICE_CACHE_READ_PER_MTOK
        + usage.output_tokens * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000


def citation_key(citation: TextCitation) -> Hashable:
    """What makes two citations the same source span.

    Item 2 saw one answer cite the same span twice; both share one number.
    """
    if isinstance(citation, CitationCharLocation):
        return (citation.document_index, citation.start_char_index, citation.end_char_index)
    # Not expected for plain-text documents; keyed on the quote itself.
    return (citation.type, citation.cited_text)


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


def render_summary(answer: Answer, console: Console) -> None:
    """After the streamed answer: any stop warning, the sources, and usage.

    The stop warning cannot go above the answer, which is already on screen by
    the time ``stop_reason`` arrives. It goes first here, before the sources.
    """
    if answer.stop_warning:
        console.print()
        console.print(Text(answer.stop_warning, style="bold red"))

    console.print()
    if answer.sources:
        console.rule("Sources", style="dim", align="left")
        for number, citation in enumerate(answer.sources, start=1):
            console.print(Text.assemble((f"[{number}] ", _MARKER_STYLE), _source_heading(citation)))
            console.print(Text(f'    "{_excerpt(citation.cited_text)}"', style="dim"))
    else:
        console.print(
            Text("No citations: nothing in this answer is tied to a source passage.", style="yellow")
        )

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
        print(f"Cannot load the API key: {exc}", file=sys.stderr)
        return 2

    try:
        corpus, check = load_corpus(api_token_counter(client, system=SYSTEM_PROMPT))
        _render_preamble(console, corpus, check.status, describe_check(check, corpus), question)
        answer = stream_answer(client, build_request(corpus, question), console)
    except KeyboardInterrupt:
        console.print()
        print("Interrupted. The answer is incomplete and was not kept.", file=sys.stderr)
        return 130
    except (ExtractionError, CorpusTooLargeError) as exc:
        print(f"Corpus assembly failed: {exc}", file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("Authentication failed. Check the key file in config.API_KEY_PATH.", file=sys.stderr)
        return 2
    except anthropic.RateLimitError as exc:
        print(f"Rate limited or out of quota: {exc.message}", file=sys.stderr)
        return 2
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 2
    except anthropic.APIConnectionError:
        print("Network error reaching the API.", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Could not record the corpus fingerprint: {exc}", file=sys.stderr)
        return 1

    render_summary(answer, console)
    return 0 if answer.stop_warning is None else 1


def _render_preamble(
    console: Console, corpus: Corpus, status: FingerprintStatus, check: str, question: str
) -> None:
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
    warn = status in (FingerprintStatus.CHANGED, FingerprintStatus.UNREADABLE)
    console.print(Text(f"Fingerprint: {check}", style="yellow" if warn else "dim"))
    console.print()
    console.print(Text(f"Q: {escape_controls(question)}", style="bold"))


if __name__ == "__main__":
    # Regulation text and model output include characters such as U+2265 that a
    # cp1252 console cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
