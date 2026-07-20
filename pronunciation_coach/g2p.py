"""Grapheme-to-phoneme conversion and phoneme normalization.

Reference (phonemizer/espeak-ng) and hypothesis (wav2vec2-espeak) phonemes
share the espeak IPA inventory but differ in stress marking; both sequences
must pass through the same normalize() before alignment (docs/design.md §4).
The length mark ː is kept so vowel-length errors stay detectable.
"""

from __future__ import annotations

STRESS_MARKS = frozenset("ˈˌ")


def normalize(phonemes: list[str]) -> list[str]:
    stripped = ("".join(ch for ch in p if ch not in STRESS_MARKS) for p in phonemes)
    return [p for p in stripped if p]
