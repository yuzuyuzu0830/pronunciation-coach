"""Grapheme-to-phoneme conversion and phoneme normalization.

Reference (phonemizer/espeak-ng) and hypothesis (wav2vec2-espeak) phonemes
share the espeak IPA inventory but differ in stress marking; both sequences
must pass through the same normalize() before alignment (docs/design.md §4).
The length mark ː is kept so vowel-length errors stay detectable.
"""

from __future__ import annotations

import string

from phonemizer import phonemize
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


def normalize(phonemes: list[str]) -> list[str]:
    """Strip stress marks and map equivalence-class variants to canonical form.

    Must be applied to BOTH the reference (phonemizer) and hypothesis
    (wav2vec2-espeak) sequences before alignment: the two sources pick
    different members of EQUIVALENCE_CLASSES, and an unnormalized side would
    turn notation variance into false substitutions.
    """
    stripped = ("".join(ch for ch in p if ch not in STRESS_MARKS) for p in phonemes)
    return [EQUIVALENCE_CLASSES.get(p, p) for p in stripped if p]


def _phonemize_words(text: str) -> list[list[str]]:
    result = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        separator=_SEPARATOR,
    )
    return [group.split() for group in result.split("|") if group.strip()]


def to_phonemes(text: str) -> list[str]:
    return [phone for word in _phonemize_words(text) for phone in word]


def to_phonemes_by_word(text: str) -> list[tuple[str, list[str]]]:
    """Phonemize text keeping word boundaries for error-to-word attribution."""
    words = [w.strip(string.punctuation) for w in text.split()]
    words = [w for w in words if w]
    phoneme_groups = _phonemize_words(text)
    if len(words) != len(phoneme_groups):
        raise ValueError(
            f"Word count mismatch between text ({len(words)} words) and "
            f"phonemizer output ({len(phoneme_groups)} groups) for {text!r}. "
            "Numbers or abbreviations may expand to multiple words; "
            "spell them out in the target text."
        )
    return list(zip(words, phoneme_groups))
