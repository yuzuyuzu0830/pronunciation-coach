"""Grapheme-to-phoneme conversion and phoneme normalization.

Reference (phonemizer/espeak-ng) and hypothesis (wav2vec2-espeak) phonemes
share the espeak IPA inventory but differ in stress marking; both sequences
must pass through the same normalize() before alignment (docs/design.md §4).
The length mark ː is kept so vowel-length errors stay detectable.
"""

from __future__ import annotations

import string

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

STRESS_MARKS = frozenset("ˈˌ")

# Notation variants that espeak/wav2vec2-espeak use interchangeably for the
# same sound, mapped to one canonical symbol. Inclusion criterion: only pairs
# that are phonetically identical in en-us — i.e. no minimal pair exists and
# a learner could never be wrong by producing one instead of the other. Never
# add contrasts we want to detect as learner errors (e.g. ð/d, l/ɹ, s/θ).
# Extend this table as new variants are observed in E2E runs.
EQUIVALENCE_CLASSES = {
    # Alveolar approximant: espeak prints "r" or "ɹ" for the same English /r/
    # (measured: hypothesis flips between them); IPA-strict form ɹ is canonical.
    "r": "ɹ",
    # R-colored vowel: en-us "church"/"water" vowel appears as ɜː or ɚ
    # depending on stress context, but is one phoneme for en-us; ɚ is canonical.
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
    """Drop the cached backend so tests can assert (re)initialization."""
    global _espeak_backend
    _espeak_backend = None


def normalize(phonemes: list[str]) -> list[str]:
    """Strip stress marks and map equivalence-class variants to canonical form.

    Must be applied to BOTH the reference (phonemizer) and hypothesis
    (wav2vec2-espeak) sequences before alignment: the two sources pick
    different members of EQUIVALENCE_CLASSES, and an unnormalized side would
    turn notation variance into false substitutions.
    """
    stripped = ("".join(ch for ch in p if ch not in STRESS_MARKS) for p in phonemes)
    return [EQUIVALENCE_CLASSES.get(p, p) for p in stripped if p]


def _phonemize_raw(texts: list[str]) -> list[str]:
    """Phonemize many texts with a single espeak backend instance."""
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
    """Phonemize each word as its own call, guaranteeing one group per word.

    Fallback for when a single batched phonemize call merges words together:
    espeak-ng sometimes drops the '|' word separator for short function-word
    sequences ("did not" / "there was" / "not a farmer" -> one merged group
    instead of two; measured on ~17% of speechocean762 utterances, confirmed
    against the real corpus 2026-07-22 -- see docs/devlog.md). Phonemizing
    one word at a time forces a boundary between every pair.
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
    """Pair words with phoneme groups, falling back to per-word phonemization
    on a count mismatch (see _phonemize_words_individually). Raises
    ValueError only when that fallback also can't produce one group per word.
    """
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

    Backend failures (RuntimeError/OSError) propagate. Per-text word/group
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
