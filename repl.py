"""Plan item 7: the REPL loop, caching, and cost display.

The conversation is a list of messages that grows by two entries per answered
turn. Three things about that list are load-bearing, and each is easy to get
wrong in a way that breaks the session much later than the mistake:

* **The first message is rebuilt, never sliced.** The API requires
  ``messages[0]`` to be a user message, so the documents and the first question
  share it. ``/reset`` therefore rebuilds it from the corpus plus the *next*
  question. Slicing ``messages[:1]`` would keep the previous session's question
  and steer every answer after it.
* **State is committed only after a turn succeeds.** A user message left in the
  list with no assistant reply makes the next turn send two user messages in a
  row, which the API rejects for non-alternating roles -- a session broken by a
  network blip three questions earlier.
* **Empty text blocks are dropped before the assistant turn is committed.** A
  cited answer can end in one, and sending it back fails the *next* call with
  "text content blocks must be non-empty".

The economic case rests on the corpus prefix matching, and prefix matching
fails *silently*: a correct answer at several times the price, with no error.
So every turn's cache read is compared with the prefix the first turn billed,
and a shortfall is printed loudly.

    python repl.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic.types import ContentBlock, Message, Usage
from rich.console import Console
from rich.text import Text

from answer import (
    cost_usd,
    first_turn,
    render_preamble,
    render_summary,
    request_kwargs,
    stream_answer,
)
from config import (
    CACHE_READ_TOLERANCE_FRACTION,
    CONTEXT_WARN_FRACTION,
    CONTEXT_WINDOW,
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_1H_PER_MTOK,
    SYSTEM_PROMPT,
)
from corpus import (
    Corpus,
    CorpusTooLargeError,
    api_token_counter,
    load_corpus,
)
from credentials import build_client
from display import escape_controls
from extract import ExtractionError
from quotes import build_report

_PROMPT = "\n> "

_HELP = (
    "/reset   clear the conversation, keeping the cached corpus\n"
    "/reload  re-extract the corpus from disk (invalidates the corpus cache)\n"
    "/cost    what this session has cost so far\n"
    "/help    this list\n"
    "/exit    leave (Ctrl+D also works)"
)


def replayable(content: Sequence[ContentBlock]) -> list[ContentBlock]:
    """An assistant turn in the form the API accepts back.

    A cited answer arrives as many text blocks and can end in an empty one,
    which the API rejects on replay. Only those are dropped: thinking blocks
    keep their signatures and text blocks keep their citations, both of which
    the model needs to see its own previous turn intact.
    """
    return [block for block in content if not (block.type == "text" and not block.text)]


@dataclass
class Session:
    """One REPL session: the conversation, the cache baseline, and the bill."""

    corpus: Corpus
    #: The conversation so far, always ending on an assistant turn.
    messages: list[dict[str, Any]] = field(default_factory=list)
    #: Answered turns in the current conversation. ``/reset`` returns it to 0.
    turns: int = 0
    #: What the first turn billed for the corpus prefix, and so roughly the
    #: least any later turn should read. An upper bound when it came from a
    #: cache read rather than a write, which is why the comparison allows
    #: config.CACHE_READ_TOLERANCE_FRACTION. None until a turn caches anything.
    cached_prefix_tokens: int | None = None
    #: config.MATCH_DISCLAIMER is shown once per session, not once per answer.
    disclaimer_shown: bool = False

    # Totals across the whole session, resets included, for /cost.
    questions: int = 0
    cost: float = 0.0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    #: Total input of the most recent request: what fills the context window.
    request_tokens: int = 0

    # --- Requests ------------------------------------------------------------

    def request_for(self, question: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """The user message asking ``question``, and the request that sends it.

        Neither is committed to :attr:`messages`; :meth:`commit` does that,
        after the answer arrives.

        Raises:
            ValueError: ``question`` is empty or only whitespace.
        """
        question = question.strip()
        if not question:
            raise ValueError("The question is empty")
        if self.messages:
            user_message: dict[str, Any] = {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        else:
            # Rebuilt from the corpus, so a /reset cannot carry the previous
            # conversation's first question into this one.
            user_message = first_turn(self.corpus, question)
        return user_message, request_kwargs([*self.messages, user_message], tail_cache=True)

    def commit(self, user_message: dict[str, Any], message: Message) -> bool:
        """Add the completed turn to the conversation. False if it was unusable.

        Called only after the answer is in hand, so an interrupted or failed
        turn leaves the conversation exactly as it was.
        """
        content = replayable(message.content)
        if not content:
            # Nothing sendable to attribute to the assistant; committing the
            # user message alone would break role alternation next turn.
            return False
        self.messages.append(user_message)
        self.messages.append({"role": "assistant", "content": content})
        self.turns += 1
        return True

    def reset(self) -> None:
        """Drop the conversation. The corpus, and its cache entry, are untouched."""
        self.messages.clear()
        self.turns = 0

    # --- Usage ---------------------------------------------------------------

    def record(self, usage: Usage) -> list[str]:
        """Bill this turn against the session, and return what it warns about.

        Warnings are computed against the previous state before the baseline
        moves, so the first turn establishes the expected cache read and later
        turns are measured against it.
        """
        write = usage.cache_creation_input_tokens or 0
        read = usage.cache_read_input_tokens or 0
        creation = usage.cache_creation
        # The corpus breakpoint is the only 1-hour one; the conversation tail
        # is written at the 5-minute TTL and is not part of the prefix.
        prefix_write = creation.ephemeral_1h_input_tokens if creation else write

        warnings: list[str] = []
        expected = self.cached_prefix_tokens
        if expected is not None:
            if read < expected - _tolerance(expected):
                warnings.append(_cache_miss(expected, read, prefix_write))
        elif prefix_write + read > 0:
            # A first turn can write the prefix, read an entry a previous
            # session left warm, or split between the two. A write is exactly
            # the prefix; a read can also cover the question after the
            # breakpoint, when an earlier session cached the identical first
            # turn, so the sum is an upper bound and _tolerance absorbs it.
            self.cached_prefix_tokens = prefix_write + read

        self.questions += 1
        self.cost += cost_usd(usage)
        self.input_tokens += usage.input_tokens
        self.cache_write_tokens += write
        self.cache_read_tokens += read
        self.output_tokens += usage.output_tokens
        # Not cumulative: every turn resends the whole conversation, so it is
        # this request, not the sum of them, that approaches the window.
        self.request_tokens = usage.input_tokens + write + read

        limit = int(CONTEXT_WINDOW * CONTEXT_WARN_FRACTION)
        if self.request_tokens >= limit:
            warnings.append(
                f"CONTEXT: this request was {self.request_tokens:,} tokens, past "
                f"{CONTEXT_WARN_FRACTION:.0%} of the {CONTEXT_WINDOW:,}-token window. The "
                f"whole conversation is resent every turn, so it will keep growing until "
                f"a call is rejected. /reset clears it; the corpus stays cached."
            )
        return warnings


def _tolerance(expected: int) -> int:
    """How far short of ``expected`` a cache read may fall without being a miss."""
    return int(expected * CACHE_READ_TOLERANCE_FRACTION)


def _cache_miss(expected: int, read: int, prefix_write: int) -> str:
    """Why a turn read less of the corpus prefix than it should have."""
    line = (
        f"CACHE MISS: this turn read {read:,} cached tokens, short of the {expected:,} "
        f"the corpus prefix should give."
    )
    if prefix_write >= expected - _tolerance(expected):
        extra = expected * (PRICE_CACHE_WRITE_1H_PER_MTOK - PRICE_CACHE_READ_PER_MTOK) / 1_000_000
        return (
            f"{line} It wrote the corpus afresh instead ({prefix_write:,} tokens), so the "
            f"1-hour entry had expired or been evicted. That cost about ${extra:.2f} more "
            f"than a read; the turns after it should hit again."
        )
    return (
        f"{line} The prefix no longer matches, so every turn is paying full input price. "
        f"Check whether the corpus or a CACHE-AFFECTING value in config.py changed; "
        f"/reload re-extracts from disk and reports what differs."
    )


# --- One turn ----------------------------------------------------------------


def ask(client: anthropic.Anthropic, session: Session, question: str, console: Console) -> None:
    """Ask one question, render the answer and its checked quotes, and commit the turn.

    Every failure path returns without touching ``session.messages``: a turn is
    either complete or absent, never half-recorded.
    """
    try:
        user_message, request = session.request_for(question)
    except ValueError:
        return

    try:
        answer = stream_answer(client, request, console)
    except KeyboardInterrupt:
        console.print()
        console.print(
            Text(
                "Interrupted. This turn was dropped, so the conversation is as it was "
                "before the question.",
                style="yellow",
            )
        )
        return
    except anthropic.AuthenticationError:
        console.print(
            Text(
                "Authentication failed: the key is no longer accepted. Check the file at "
                "config.API_KEY_PATH, then start a new session.",
                style="bold red",
            )
        )
        return
    except anthropic.RateLimitError as exc:
        console.print(
            Text(
                f"Rate limited or out of quota: {escape_controls(exc.message)} The turn was "
                f"dropped; wait and ask again.",
                style="bold red",
            )
        )
        return
    except anthropic.BadRequestError as exc:
        console.print(Text(_bad_request(exc), style="bold red"))
        return
    except anthropic.APIStatusError as exc:
        console.print(
            Text(
                f"API error {exc.status_code}: {escape_controls(exc.message)} The turn was "
                f"dropped.",
                style="bold red",
            )
        )
        return
    except anthropic.APIConnectionError:
        console.print(
            Text("Network error reaching the API. The turn was dropped; ask again.", style="bold red")
        )
        return

    report = build_report(
        answer.sources,
        answer.message.content,
        [document.text for document in session.corpus.documents],
    )
    render_summary(answer, report, console, show_disclaimer=not session.disclaimer_shown)
    session.disclaimer_shown = True

    for warning in session.record(answer.message.usage):
        console.print(Text(warning, style="bold red"))

    if not session.commit(user_message, answer.message):
        console.print(
            Text(
                "The answer carried no text to send back, so this turn was left out of the "
                "conversation.",
                style="yellow",
            )
        )


def _bad_request(exc: anthropic.BadRequestError) -> str:
    """A 400, with the one cause worth naming separately called out."""
    if "credit balance" in exc.message.lower():
        return (
            f"Out of credit: {escape_controls(exc.message)} Nothing further will succeed "
            f"until the account is topped up."
        )
    return (
        f"The API rejected the request (400): {escape_controls(exc.message)} The turn was "
        f"dropped. If this repeats, /reset clears the conversation."
    )


# --- Commands ----------------------------------------------------------------


def run(
    client: anthropic.Anthropic,
    session: Session,
    console: Console,
    *,
    read_line: Callable[[str], str] = input,
) -> None:
    """Read questions and commands until the user leaves or input ends."""
    console.print()
    console.print(Text(_HELP, style="dim"))
    while True:
        try:
            line = read_line(_PROMPT).strip()
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            console.print()
            console.print(Text("/exit to leave.", style="dim"))
            continue

        if not line:
            continue
        if line.startswith("/"):
            if _command(line, client, session, console):
                return
            continue
        ask(client, session, line, console)


def _command(
    line: str, client: anthropic.Anthropic, session: Session, console: Console
) -> bool:
    """Run one ``/command``. True means leave the REPL."""
    name = line.split()[0].lower()
    if name == "/exit":
        return True
    if name == "/help":
        console.print(Text(_HELP, style="dim"))
    elif name == "/reset":
        turns = session.turns
        session.reset()
        console.print(
            Text(
                f"Cleared {turns} turn(s). The corpus stays cached; the next question "
                f"rebuilds the first message from it.",
                style="dim",
            )
        )
    elif name == "/cost":
        render_cost(session, console)
    elif name == "/reload":
        reload_corpus(client, session, console)
    else:
        console.print(
            Text(
                f"Unknown command {escape_controls(name)}. Try /help.",
                style="yellow",
            )
        )
    return False


def reload_corpus(client: anthropic.Anthropic, session: Session, console: Console) -> None:
    """Re-extract the corpus from disk, replacing it only if extraction succeeds.

    Picking up edited files invalidates the corpus cache, which is correct: the
    cached prefix no longer describes what is on disk. The conversation goes
    with it, since ``messages[0]`` holds the old document blocks.
    """
    try:
        corpus, check = load_corpus(api_token_counter(client, system=SYSTEM_PROMPT))
    except (ExtractionError, CorpusTooLargeError) as exc:
        console.print(
            Text(f"Reload failed, so the loaded corpus is unchanged: {exc}", style="bold red")
        )
        return
    except OSError as exc:
        console.print(
            Text(f"Could not record the corpus fingerprint: {exc}", style="bold red")
        )
        return
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        console.print(
            Text(
                f"Could not count the reloaded corpus: {exc}. The loaded corpus is "
                f"unchanged.",
                style="bold red",
            )
        )
        return

    changed = corpus.fingerprint != session.corpus.fingerprint
    session.corpus = corpus
    session.reset()
    if changed:
        # The old prefix is no longer what we send, so the old baseline would
        # report a miss on every turn from here.
        session.cached_prefix_tokens = None
    console.print()
    render_preamble(console, corpus, check)
    console.print(
        Text(
            "Reloaded; the conversation was cleared. The next question writes the corpus "
            "cache afresh."
            if changed
            else "Reloaded; the corpus is byte-identical, so its cache entry still applies. "
            "The conversation was cleared.",
            style="yellow" if changed else "dim",
        )
    )


def render_cost(session: Session, console: Console) -> None:
    """What the session has cost, by input bucket, since a single multiply misleads."""
    if not session.questions:
        console.print(Text("Nothing asked yet, so nothing billed.", style="dim"))
        return
    console.print(
        Text(
            f"{session.questions} question(s), ${session.cost:.2f} total. Input: "
            f"{session.input_tokens:,} uncached, {session.cache_write_tokens:,} cache "
            f"write, {session.cache_read_tokens:,} cache read. Output: "
            f"{session.output_tokens:,}. Last request: {session.request_tokens:,} tokens "
            f"({session.request_tokens / CONTEXT_WINDOW:.0%} of the window).",
            style="dim",
        )
    )


# --- Command line ------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan item 7: ask questions about the corpus in a REPL."
    )
    parser.parse_args(argv)

    console = Console(highlight=False)
    try:
        client = build_client()
    except (FileNotFoundError, ValueError) as exc:
        # credentials.py never puts the key itself in an error message.
        print(f"Cannot load the API key: {exc}", file=sys.stderr)
        return 2

    try:
        corpus, check = load_corpus(api_token_counter(client, system=SYSTEM_PROMPT))
    except (ExtractionError, CorpusTooLargeError) as exc:
        print(f"Corpus assembly failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not record the corpus fingerprint: {exc}", file=sys.stderr)
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

    render_preamble(console, corpus, check)
    session = Session(corpus)
    run(client, session, console)
    render_cost(session, console)
    return 0


if __name__ == "__main__":
    # Regulation text and model output include characters such as U+2265 that a
    # cp1252 console cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
