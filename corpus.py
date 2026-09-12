"""Plan item 4: corpus assembly, fingerprint, and hard size guard.

The worst failure in this kind of app is answering confidently from part of the
corpus. So assembly never truncates: the assembled document blocks are counted
with ``client.messages.count_tokens`` and, if they exceed
``config.TOKEN_CEILING``, assembly stops with an error listing the options.

The assembled blocks are also fingerprinted (SHA-256), and the fingerprint is
compared with the one the previous run persisted. A change means extraction
output changed -- an edited source file, an lxml upgrade, an edit to
``extract.py`` or to a cache-affecting value in ``config.py`` -- and that any
cached copy of the previous corpus no longer matches, so the next question pays
the cache-write price. A hash that is printed and never compared catches none
of that, so the comparison names what changed.

    python corpus.py    # assemble, count tokens, compare with the previous run
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic
from lxml import etree

from config import (
    CONTEXT_WINDOW,
    CORPUS_CACHE_TTL,
    CORPUS_DIR,
    DOCUMENT_CONTEXT_TEMPLATE,
    MAX_TOKENS,
    MODEL,
    PRICE_CACHE_WRITE_1H_PER_MTOK,
    STATE_FILE,
    SYSTEM_PROMPT,
    TOKEN_CEILING,
)
from credentials import build_client
from extract import ExtractedDocument, ExtractionError, extract_corpus

#: Counts the input tokens of a request whose first user message holds these
#: document blocks. Injectable so the size guard is testable without the API.
TokenCounter = Callable[[list[dict[str, Any]]], int]

#: Stands in for the user's question when counting: a request needs one to be
#: valid, and a single character adds a token or two to the count.
_PLACEHOLDER_QUESTION = "?"

#: Layout version of the state file. Any other version is reported as
#: unreadable rather than guessed at.
_STATE_VERSION = 1

_STATE_DOCUMENT_KEYS = frozenset({"filename", "label", "chars", "text_sha256"})


class CorpusTooLargeError(RuntimeError):
    """The assembled corpus exceeds the token ceiling. It is never truncated."""

    def __init__(self, tokens: int, ceiling: int) -> None:
        self.tokens = tokens
        self.ceiling = ceiling
        super().__init__(
            f"The assembled corpus is {tokens:,} tokens, over the {ceiling:,}-token "
            f"ceiling (config.TOKEN_CEILING). Nothing was truncated, and nothing will "
            f"be: answering from part of the corpus is the failure this guard exists "
            f"to prevent. Options:\n"
            f"  1. Remove documents: take files out of {CORPUS_DIR} and their entries "
            f"out of config.DOCUMENT_LABELS.\n"
            f"  2. Raise config.TOKEN_CEILING, if the {CONTEXT_WINDOW:,}-token context "
            f"window still leaves room for conversation history and up to "
            f"{MAX_TOKENS:,} tokens of output.\n"
            f"  3. Re-plan: at this size the plan's agentic glob/grep/read loop is the "
            f"right architecture (see Sequencing, the item 1 gate)."
        )


@dataclass(frozen=True)
class Corpus:
    """The assembled corpus: documents in send order, fingerprinted and counted."""

    #: In send order. A citation's ``document_index`` indexes this tuple.
    documents: tuple[ExtractedDocument, ...]
    #: SHA-256 of the document blocks' canonical JSON, in hex.
    fingerprint: str
    #: ``count_tokens`` for the document blocks plus a placeholder question, and
    #: the system prompt when the counter was given one.
    tokens: int

    def document_blocks(self) -> list[dict[str, Any]]:
        """The document blocks, built afresh so no caller can alter the prefix."""
        return build_document_blocks(self.documents)


# --- Assembly ----------------------------------------------------------------


def document_block(document: ExtractedDocument) -> dict[str, Any]:
    """One document as the API receives it: plain text, labeled, citable."""
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": document.text},
        "title": document.label,
        "context": DOCUMENT_CONTEXT_TEMPLATE.format(filename=document.filename),
        "citations": {"enabled": True},
    }


def build_document_blocks(documents: Sequence[ExtractedDocument]) -> list[dict[str, Any]]:
    """Every document's block, in the order given.

    The 1-hour cache breakpoint sits on the last block, closing the corpus
    prefix that every question reuses.
    """
    blocks = [document_block(document) for document in documents]
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": CORPUS_CACHE_TTL}
    return blocks


def fingerprint_blocks(blocks: Sequence[dict[str, Any]]) -> str:
    """SHA-256 of the blocks' canonical JSON.

    Covers every field of every block -- text, title, context, citation
    setting, breakpoint -- and their order. Keys are sorted and non-ASCII is
    escaped, so the bytes hashed depend only on the content.
    """
    canonical = json.dumps(list(blocks), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def assemble_corpus(
    documents: Sequence[ExtractedDocument],
    count_tokens: TokenCounter,
    *,
    ceiling: int = TOKEN_CEILING,
) -> Corpus:
    """Order, fingerprint and count the documents; refuse a corpus over the ceiling.

    Args:
        documents: Extracted documents, as from :func:`extract.extract_corpus`.
        count_tokens: Counts the input tokens of the assembled request.
        ceiling: Largest acceptable token count, inclusive.

    Raises:
        ValueError: ``documents`` is empty or names a file twice.
        CorpusTooLargeError: The count exceeds ``ceiling``. The corpus is never
            truncated to fit.
    """
    if not documents:
        raise ValueError("Cannot assemble an empty corpus")
    ordered = tuple(sorted(documents, key=lambda document: document.filename))
    names = [document.filename for document in ordered]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        # Sorting cannot order two documents with one name, so the send order
        # (and with it the fingerprint) would depend on the caller's order.
        raise ValueError(f"Corpus names a file more than once: {', '.join(duplicates)}")

    blocks = build_document_blocks(ordered)
    tokens = count_tokens(blocks)
    if tokens > ceiling:
        raise CorpusTooLargeError(tokens, ceiling)
    return Corpus(ordered, fingerprint_blocks(blocks), tokens)


def api_token_counter(
    client: anthropic.Anthropic, *, system: str | None = None, model: str = MODEL
) -> TokenCounter:
    """A :data:`TokenCounter` backed by ``client.messages.count_tokens``.

    Never ``tiktoken``: it is an OpenAI tokenizer and its counts do not apply
    to Claude.

    Args:
        client: Client to count with.
        system: The system prompt, ``config.SYSTEM_PROMPT`` in production. It
            renders before the documents, inside the cached prefix, so it
            belongs in the count.
        model: Model whose tokenizer counts.
    """

    def count(blocks: list[dict[str, Any]]) -> int:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [*blocks, {"type": "text", "text": _PLACEHOLDER_QUESTION}],
                }
            ],
        }
        if system is not None:
            kwargs["system"] = system
        return client.messages.count_tokens(**kwargs).input_tokens

    return count


# --- Fingerprint state -------------------------------------------------------


class FingerprintStatus(Enum):
    FIRST_RUN = "first run"
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class FingerprintCheck:
    """How the corpus compares with the one the previous run recorded."""

    status: FingerprintStatus
    #: When the previous run recorded its fingerprint (ISO 8601, UTC), if known.
    previous_recorded_at: str | None = None
    #: One line per change when CHANGED; why the state file could not be used
    #: when UNREADABLE.
    details: tuple[str, ...] = ()


def compare_with_previous(
    corpus: Corpus, state_file: Path = STATE_FILE, *, lxml_version: str = etree.__version__
) -> FingerprintCheck:
    """Compare ``corpus`` with the fingerprint the previous run recorded.

    Never raises for a missing, unreadable or malformed state file: the check
    reports it, and :func:`record_state` then replaces the file.
    """
    try:
        raw = state_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return FingerprintCheck(FingerprintStatus.FIRST_RUN)
    except OSError as exc:
        return FingerprintCheck(
            FingerprintStatus.UNREADABLE, details=(f"{state_file}: {exc.strerror or exc}",)
        )

    try:
        previous = _parse_state(raw)
    except ValueError as exc:
        return FingerprintCheck(FingerprintStatus.UNREADABLE, details=(f"{state_file}: {exc}",))

    recorded_at = previous["recorded_at"]
    if previous["fingerprint"] == corpus.fingerprint:
        return FingerprintCheck(FingerprintStatus.UNCHANGED, recorded_at)
    return FingerprintCheck(
        FingerprintStatus.CHANGED,
        recorded_at,
        tuple(_describe_changes(previous, corpus, lxml_version)),
    )


def record_state(
    corpus: Corpus, state_file: Path = STATE_FILE, *, lxml_version: str = etree.__version__
) -> None:
    """Persist ``corpus``'s fingerprint for the next run to compare against.

    Written to a temporary file and moved into place, so an interrupted write
    cannot leave a truncated state file behind.
    """
    state = {
        "version": _STATE_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": corpus.fingerprint,
        "tokens": corpus.tokens,
        "lxml_version": lxml_version,
        "documents": [
            {
                "filename": document.filename,
                "label": document.label,
                "chars": len(document.text),
                "text_sha256": _sha256(document.text),
            }
            for document in corpus.documents
        ],
    }
    temporary = state_file.with_name(state_file.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, state_file)


def describe_check(check: FingerprintCheck, corpus: Corpus) -> str:
    """The fingerprint comparison as text for the user."""
    if check.status is FingerprintStatus.FIRST_RUN:
        return "No previous fingerprint on record; this one is now recorded."
    if check.status is FingerprintStatus.UNCHANGED:
        return f"Unchanged since the run recorded at {check.previous_recorded_at}."
    if check.status is FingerprintStatus.UNREADABLE:
        return (
            f"WARNING: the previous fingerprint could not be read ({'; '.join(check.details)}), "
            f"so a change in extraction output cannot be ruled out. This run's "
            f"fingerprint replaces it."
        )

    write_cost = corpus.tokens * PRICE_CACHE_WRITE_1H_PER_MTOK / 1_000_000
    return "\n".join(
        [
            f"WARNING: the corpus changed since the run recorded at "
            f"{check.previous_recorded_at}:",
            *(f"  - {detail}" for detail in check.details),
            "Likely causes: an edited source file, an lxml upgrade, or an edit to "
            "extract.py or to a CACHE-AFFECTING value in config.py.",
            f"Consequence: any cached copy of the previous corpus no longer matches. "
            f"The next question writes the corpus cache afresh (about ${write_cost:.2f} "
            f"at the 1-hour write price) instead of reading it.",
        ]
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_state(raw: str) -> dict[str, Any]:
    """The previous run's state, validated. Raises ValueError saying what is wrong."""
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc.msg})") from exc
    if not isinstance(state, dict) or state.get("version") != _STATE_VERSION:
        raise ValueError(f"not a version-{_STATE_VERSION} NotRag state file")

    documents = state.get("documents")
    well_formed = (
        isinstance(state.get("fingerprint"), str)
        and isinstance(state.get("recorded_at"), str)
        and isinstance(state.get("tokens"), int)
        and isinstance(state.get("lxml_version"), str)
        and isinstance(documents, list)
        and all(
            isinstance(document, dict) and _STATE_DOCUMENT_KEYS <= document.keys()
            for document in documents
        )
    )
    if not well_formed:
        raise ValueError("missing or malformed fields")
    return state


def _describe_changes(
    previous: dict[str, Any], corpus: Corpus, lxml_version: str
) -> list[str]:
    """What differs between the recorded state and ``corpus``, one line each."""
    before = {document["filename"]: document for document in previous["documents"]}
    after = {document.filename: document for document in corpus.documents}

    changes = [f"{name}: added" for name in sorted(after.keys() - before.keys())]
    changes += [f"{name}: removed" for name in sorted(before.keys() - after.keys())]
    for name in sorted(before.keys() & after.keys()):
        old, new = before[name], after[name]
        if old["label"] != new.label:
            changes.append(f"{name}: label changed ({old['label']!r} -> {new.label!r})")
        if old["text_sha256"] != _sha256(new.text):
            changes.append(
                f"{name}: text changed ({old['chars']:,} -> {len(new.text):,} chars)"
            )
    if not changes:
        changes.append(
            "no document's text or label changed, so a block setting did "
            "(config.DOCUMENT_CONTEXT_TEMPLATE, config.CORPUS_CACHE_TTL, or the block "
            "shape in corpus.py)"
        )

    if previous["lxml_version"] != lxml_version:
        changes.append(
            f"lxml {previous['lxml_version']} -> {lxml_version}: the likely cause of "
            f"text changes if no source file was edited"
        )
    if previous["tokens"] != corpus.tokens:
        changes.append(f"tokens: {previous['tokens']:,} -> {corpus.tokens:,}")
    return changes


# --- Loading -----------------------------------------------------------------


def load_corpus(
    count_tokens: TokenCounter, *, state_file: Path = STATE_FILE
) -> tuple[Corpus, FingerprintCheck]:
    """Extract, assemble and size-check the corpus, then check its fingerprint.

    The fingerprint is recorded only after the corpus passes the size guard,
    and it replaces the previous one, so a change is reported by the first run
    that sees it rather than by every run after.

    Raises:
        ExtractionError: The corpus could not be extracted.
        CorpusTooLargeError: The corpus exceeds ``config.TOKEN_CEILING``.
        OSError: The state file could not be written.
    """
    corpus = assemble_corpus(extract_corpus(), count_tokens)
    check = compare_with_previous(corpus, state_file)
    record_state(corpus, state_file)
    return corpus, check


# --- Command line ------------------------------------------------------------


def main() -> int:
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

    header = f"{'document':<26}{'chars':>10}"
    print(header)
    print("-" * len(header))
    for document in corpus.documents:
        print(f"{document.label.split(':')[0]:<26}{len(document.text):>10,}")
    print("-" * len(header))
    print(f"{'TOTAL':<26}{sum(len(document.text) for document in corpus.documents):>10,}")

    print(
        f"\nTokens      : {corpus.tokens:,} of the {TOKEN_CEILING:,} ceiling "
        f"({corpus.tokens / TOKEN_CEILING:.1%}), counted on {MODEL} with the "
        f"system prompt"
    )
    print(f"Fingerprint : {corpus.fingerprint}")
    print(f"Previous run: {describe_check(check, corpus)}")
    return 0


if __name__ == "__main__":
    # Labels are ASCII, but API error messages are not guaranteed to be; never
    # let a stray character crash the report on a cp1252 console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
