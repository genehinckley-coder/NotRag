"""API key loading and Anthropic client construction.

The obvious implementations of credential loading are the insecure ones, so the
rules this module enforces are worth stating plainly:

* The key is read from a file outside the repository and passed explicitly to
  ``Anthropic(api_key=...)``.
* The key is **never** assigned to ``os.environ``. Doing so would leak it into
  the environment of every child process the REPL spawns.
* The key is never logged, echoed, or included in an exception message. Failures
  name the *file*, never its contents. ``redact`` exists for the rare case where
  a diagnostic genuinely needs to identify which key is loaded.
* HTTP-level SDK debug logging (``ANTHROPIC_LOG=debug``) is never enabled:
  request headers carry the key in plaintext.
"""

from __future__ import annotations

from pathlib import Path

from anthropic import Anthropic

from config import API_KEY_PATH

#: Expected prefix for an Anthropic API key. Checked so that pointing at the
#: wrong file fails immediately with a useful message, rather than surfacing
#: later as an opaque 401 from the API.
_KEY_PREFIX = "sk-ant-"


def load_api_key(path: Path | None = None) -> str:
    """Read and validate the Anthropic API key from disk.

    The file may contain annotation lines alongside the key -- these key files
    are maintained by hand and commonly carry an expiry note. Exactly one line
    must begin with the Anthropic key prefix; everything else is ignored.

    Args:
        path: Key file to read. Defaults to :data:`config.API_KEY_PATH`.

    Returns:
        The key, stripped of a byte-order mark and surrounding whitespace.

    Raises:
        FileNotFoundError: The key file does not exist.
        ValueError: The file is empty, contains no line that looks like an
            Anthropic API key, or contains more than one such line.
    """
    key_path = Path(path) if path is not None else API_KEY_PATH

    try:
        # utf-8-sig strips a BOM, which Notepad adds silently and which would
        # otherwise ride along into the Authorization header.
        raw = key_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"API key file not found: {key_path}. Create it containing only the "
            f"key, or point config.API_KEY_PATH somewhere else."
        ) from exc

    if not raw.strip():
        raise ValueError(f"API key file is empty: {key_path}")

    # Stripping each line also removes any trailing "\r", which is invalid in an
    # HTTP header value and would surface as a confusing AuthenticationError
    # rather than as a file problem.
    candidates = [
        line.strip()
        for line in raw.splitlines()
        if line.strip().startswith(_KEY_PREFIX)
    ]

    if not candidates:
        raise ValueError(
            f"No Anthropic API key found in {key_path} (expected a line starting "
            f"with {_KEY_PREFIX!r}). Check that config.API_KEY_PATH points at the "
            f"right file."
        )

    if len(candidates) > 1:
        raise ValueError(
            f"{key_path} contains {len(candidates)} lines that look like API keys. "
            f"Leave exactly one so there is no ambiguity about which is current."
        )

    key = candidates[0]

    if any(ch.isspace() for ch in key):
        raise ValueError(
            f"API key in {key_path} contains embedded whitespace. Expected the "
            f"key on a single unbroken line."
        )

    return key


def redact(key: str) -> str:
    """Return a non-secret fingerprint of ``key`` that is safe to display.

    Shows enough to tell two keys apart without revealing either.
    """
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:11]}...{key[-4:]}"


def build_client(path: Path | None = None) -> Anthropic:
    """Construct an :class:`Anthropic` client with an explicitly supplied key.

    Args:
        path: Key file to read. Defaults to :data:`config.API_KEY_PATH`.

    Returns:
        A configured client. The key is passed as a constructor argument and is
        never written to the process environment.
    """
    return Anthropic(api_key=load_api_key(path))
