"""Grapheme-to-phoneme conversion and normalization.

Reference and hypothesis phonemes must use the same normalization before
alignment. The length mark ː is preserved.
"""

from __future__ import annotations

import string

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

STRESS_MARKS = frozenset("ˈˌ")

# Only canonicalize notation variants that are phonetically identical in en-us.
# Learner contrasts such as ð/d, l/ɹ, and s/θ must remain detectable.
EQUIVALENCE_CLASSES = {
    # espeak uses both symbols for English /r/.
    "r": "ɹ",
    # The en-us r-colored vowel varies with stress context.
    "ɜː": "ɚ",
}

# phone=" " keeps diphthongs/affricates as single tokens (measured: "high" ->
# "h aɪ", "church" -> "tʃ ɜː tʃ"), matching wav2vec2-espeak's token unit.
_SEPARATOR = Separator(phone=" ", word="|")

# phonemizer initializes espeak on every phonemize() call; reuse one backend
# (and prefer list inputs) as the library docs recommend.
_espeak_backend: EspeakBackend | None = None


def _get_espeak_backend() -> EspeakBackend:
    global _espeak_backend
    if _espeak_backend is None:
        _espeak_backend = EspeakBackend(language="en-us")
    return _espeak_backend


def reset_espeak_backend_for_tests() -> None:
    global _espeak_backend
    _espeak_backend = None


def normalize(phonemes: list[str]) -> list[str]:
    """Remove stress marks and canonicalize equivalent notation."""
    stripped = ("".join(ch for ch in p if ch not in STRESS_MARKS) for p in phonemes)
    return [EQUIVALENCE_CLASSES.get(p, p) for p in stripped if p]


def _phonemize_raw(texts: list[str]) -> list[str]:
    if not texts:
        return []
    return _get_espeak_backend().phonemize(
        texts, separator=_SEPARATOR, strip=True
    )


def _parse_phoneme_groups(phonemized: str) -> list[list[str]]:
    return [group.split() for group in phonemized.split("|") if group.strip()]


def _words_from_text(text: str) -> list[str]:
    words = [w.strip(string.punctuation) for w in text.split()]
    return [w for w in words if w]


def _phonemize_words_individually(words: list[str]) -> list[list[str]]:
    """Guarantee one phoneme group per word when espeak drops separators.

    This occurred in about 17% of tested SpeechOcean762 utterances.
    """
    raw = _phonemize_raw(words)
    groups: list[list[str]] = []
    for word, phonemized in zip(words, raw):
        word_groups = _parse_phoneme_groups(phonemized)
        if len(word_groups) != 1:
            raise ValueError(
                f"Cannot phonemize word {word!r} as a single unit: got "
                f"{len(word_groups)} phoneme groups instead of 1, even after "
                "falling back to individual per-word phonemization. Numbers "
                "or abbreviations may expand to multiple words; spell them "
                "out in the target text."
            )
        groups.append(word_groups[0])
    return groups


def _pair_words_with_groups(
    text: str, phoneme_groups: list[list[str]]
) -> list[tuple[str, list[str]]]:
    """Pair words with groups, retrying per word when their counts differ."""
    words = _words_from_text(text)
    if len(words) != len(phoneme_groups):
        phoneme_groups = _phonemize_words_individually(words)
    return list(zip(words, phoneme_groups))


def _phonemize_words(text: str) -> list[list[str]]:
    return _parse_phoneme_groups(_phonemize_raw([text])[0])


def to_phonemes(text: str) -> list[str]:
    return [phone for word in _phonemize_words(text) for phone in word]


def to_phonemes_by_word(text: str) -> list[tuple[str, list[str]]]:
    """Phonemize text keeping word boundaries for error-to-word attribution."""
    return _pair_words_with_groups(text, _phonemize_words(text))


def to_phonemes_by_word_many(
    texts: list[str],
) -> list[list[tuple[str, list[str]]] | ValueError]:
    """Batch-phonemize texts (one espeak init), pairing each with its words.

    Backend failures propagate. Per-text word/group
    count mismatches are returned as ValueError entries so callers can skip
    one utterance without aborting the rest.
    """
    raw = _phonemize_raw(texts)
    results: list[list[tuple[str, list[str]]] | ValueError] = []
    for text, phonemized in zip(texts, raw):
        try:
            results.append(
                _pair_words_with_groups(text, _parse_phoneme_groups(phonemized))
            )
        except ValueError as e:
            results.append(e)
    return results
