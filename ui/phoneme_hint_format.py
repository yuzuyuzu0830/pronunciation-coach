"""Locale-specific rendering of the phoneme facts in ui/phoneme_hints.py
(docs/design_ui.md "音素表示のヒント併記" section).

Only this module knows how a hint should read to a human; phoneme_hints.py
knows only the underlying language-neutral facts (example word, grapheme).
Adding a UI display language is a new template function registered here, not
a change to the data table -- e.g. an English-UI template would read
"/θ/ (th as in think)" from the exact same PhonemeHint.
"""

from __future__ import annotations

from typing import Callable

from ui.phoneme_hints import PHONEME_HINTS, PhonemeHint

_NONE_PLACEHOLDER = "-"  # matches the existing dash convention for a missing side


def _format_ja(symbol: str, hint: PhonemeHint) -> str:
    return f"/{symbol}/ ({hint.example} の {hint.grapheme})"


_TEMPLATES: dict[str, Callable[[str, PhonemeHint], str]] = {
    "ja": _format_ja,
}


def format_phoneme_hint(symbol: str | None, locale: str = "ja") -> str:
    """Render one phoneme symbol for display.

    None (a deletion's actual / an insertion's expected) renders as the
    placeholder dash. A symbol not in PHONEME_HINTS renders bare ("/x/"):
    the table is deliberately incomplete, so an unhinted symbol is expected,
    not an error.
    """
    if symbol is None:
        return _NONE_PLACEHOLDER
    hint = PHONEME_HINTS.get(symbol)
    if hint is None:
        return f"/{symbol}/"
    try:
        template = _TEMPLATES[locale]
    except KeyError:
        raise ValueError(f"Unknown locale {locale!r}; available: {sorted(_TEMPLATES)}") from None
    return template(symbol, hint)
