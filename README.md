# NotRag

Ask questions about the Illinois Administrative Code rules for underground storage
tanks (Title 41, Parts 172 and 174–178) and get answers from Claude in which every
quoted passage is checked against the source file it claims to come from.

It is deliberately not RAG: there are no embeddings, no chunking and no vector store.
The whole corpus, six HTML files of about 200,000 tokens, goes into every request as
cached document blocks with citations turned on. The model sees all of it, and each
citation's character offsets are checked, character for character, against the text
that was sent.

## Requirements

- Windows and PowerShell. The paths and commands below assume both.
- Python 3.13. The project was built on 3.13.14.
- An Anthropic API key with credit.

## Setup

```powershell
cd D:\code\NotRag
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Every command below assumes the virtual environment is active.

### API key

The key is read from `D:\claude\apikeys\notrag_key.txt`, outside the repository. To
keep it somewhere else, change `API_KEY_PATH` in `config.py`.

The file must contain exactly one line that starts with `sk-ant-`. Any other lines,
such as an expiry note, are ignored. A byte-order mark and Windows line endings are
both fine.

The key is passed straight to the API client and is never put in an environment
variable, so `ANTHROPIC_API_KEY` is not needed. Never set `ANTHROPIC_LOG=debug`: the
request headers it logs contain the key.

## Running it

### An interactive session (the normal way)

```powershell
python repl.py
```

Type a question at the `>` prompt. Follow-up questions see the conversation so far.

| Command   | What it does |
|-----------|--------------|
| `/reset`  | Start a new conversation. The corpus stays cached. |
| `/reload` | Re-read the HTML files from disk, for example after editing one. Clears the conversation. If the text changed, the next question pays for a fresh cache write. |
| `/cost`   | What the session has cost so far, by token type. |
| `/help`   | List the commands. |
| `/exit`   | Leave. Ctrl+Z then Enter also ends the session on Windows. |

Pressing Ctrl+C while an answer is streaming drops that question and leaves the
conversation as it was before it. Pressing Ctrl+C at the prompt does not exit.

### One question

```powershell
python answer.py what are the sump requirements for a UST in Illinois
```

Quoting the question is optional. The exit code says how it went:

| Code | Meaning |
|------|---------|
| 0    | A complete answer whose quotes all matched. |
| 1    | The answer was cut off or refused, a quote failed its check, or the corpus could not be assembled. |
| 2    | A problem with the key, authentication, rate limits or the API. |
| 130  | Interrupted with Ctrl+C. |

### Checking the corpus without asking anything

```powershell
python extract.py              # size of each extracted document; no API call
python extract.py --show 175   # print the extracted text of Part 175; no API call
python corpus.py               # token count and fingerprint check; one free token-count call
```

## What it costs

Measured with the prices in `config.py`:

- **First question in a session: about $2.** It writes the corpus to a one-hour cache.
- **Each later question: about $0.20**, output included, while the cache is warm.
- **The cache expires after an hour without use.** The next question then pays the
  $2 write again, and the REPL prints a `CACHE MISS` line explaining why.
- **Changing the cached prefix** (the corpus text, a document label, the system prompt
  or the model) also means a fresh $2 write.

Every answer ends with a usage line showing its cost, and `/cost` totals the session.

## Reading an answer

The answer streams in as it is written, sometimes after a dimmed summary of the
model's reasoning. A `[n]` marker follows each cited passage. After the answer come:

- **Sources.** Each cited passage, with its document, its character range, a short
  excerpt and the result of its check:
  - `matched`: the quoted text is exactly the text at those characters in the source.
  - `MISMATCH`, `OUT OF RANGE`, `DEGENERATE` or `NOT CHECKABLE`: the check failed, and
    the line below says why.
- **Summary lines.** How many quotes matched, how much of the answer carries no
  citation, and any *thinly sourced* blocks, meaning long claims that rest on a
  short quote.
- **Warnings**, when they apply:

  | Warning       | Meaning |
  |---------------|---------|
  | `TRUNCATED`   | The answer hit the output limit and stops partway. |
  | `REFUSED`     | The model declined to answer. |
  | `CACHE MISS`  | The cached corpus was not read. |
  | `NOT CACHED`  | Nothing was cached at all. |
  | `CONTEXT`     | The conversation is nearing the context window; use `/reset`. |

**"Matched" is not "verified."** A match proves the quote really appears at the cited
place in the source. It does not prove that the quote supports the claim next to it.
Read the cited text before you rely on a conclusion.

## Tests

```powershell
python -m pytest
```

The suite runs offline in a few seconds and needs no key. One test asks the real
sump question. It is skipped by default because it costs about $2:

```powershell
$env:NOTRAG_LIVE = "1"; python -m pytest -m live; Remove-Item Env:NOTRAG_LIVE
```

## Changing the corpus

Files are read from the top level of `admincode/`, matching `*.htm` and `*.html`.
The `*_files` folders the browser saved beside them are ignored.

Each file needs a label in `DOCUMENT_LABELS` in `config.py`, and each label needs a
file. A mismatch either way stops startup with an error rather than answering from
part of the corpus. Startup also stops if the corpus grows past 400,000 tokens
(`TOKEN_CEILING`). Nothing is ever cut to fit.

On every start, the corpus fingerprint is compared with the previous run's, which is
kept in `.notrag_state.json` (not tracked in git). If the extracted text changed, a
warning says what changed and that the next question pays for a fresh cache write.

## Project layout

| Path | Contents |
|------|----------|
| `admincode/` | The corpus: six pages saved from the ILGA website. |
| `config.py` | Every setting: paths, model, token limits, prices, document labels, system prompt. |
| `repl.py` | The interactive session. |
| `answer.py` | One question: the request, the streamed display and the summary. |
| `quotes.py` | Checks each quote against its source. Makes no API call. |
| `corpus.py` | Assembles the corpus, applies the size limit, fingerprints it. |
| `extract.py` | Turns the HTML into text that keeps its subsection and table structure. |
| `credentials.py` | Loads the API key. |
| `display.py` | Escapes control characters so untrusted text cannot drive the terminal. |
| `tests/` | The pytest suite. |
| `plans/2026-09-11_docqa-notrag.md` | The design plan, with measured results for each build step. |
| `fit_check.py`, `spike_request_shape.py` | One-off scripts from the plan's first two steps, kept as a record. The spike makes billed calls. |
