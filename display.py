"""Terminal-safe display of untrusted text.

Corpus text and model output are both untrusted, and either can carry ESC: an
escape sequence printed raw can recolor, move or clear the terminal. rich's
``Text`` is not enough on its own -- measured on rich 15.0.0, it strips BEL and
CR but passes ESC through -- so every untrusted string goes through
:func:`escape_controls` before it is printed.
"""

from __future__ import annotations

#: C0 and C1 control characters other than newline and tab, shown escaped so
#: untrusted text cannot drive the terminal.
_CONTROL_ESCAPES = {
    code: f"\\x{code:02x}"
    for code in (*range(0x20), 0x7F, *range(0x80, 0xA0))
    if chr(code) not in "\n\t"
}


def escape_controls(text: str) -> str:
    """``text`` with every control character except newline and tab shown as ``\\xNN``."""
    return text.translate(_CONTROL_ESCAPES)
