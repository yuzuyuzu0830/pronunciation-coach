"""Typical graphemes used to highlight English phonemes in target words."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhonemeHint:
    grapheme: str


PHONEME_HINTS: dict[str, PhonemeHint] = {
    # Consonants
    "b": PhonemeHint("b"),
    "d": PhonemeHint("d"),
    "dʒ": PhonemeHint("j"),
    "f": PhonemeHint("f"),
    "ɡ": PhonemeHint("g"),
    "h": PhonemeHint("h"),
    "j": PhonemeHint("y"),
    "k": PhonemeHint("k"),
    "l": PhonemeHint("l"),
    "m": PhonemeHint("m"),
    "n": PhonemeHint("n"),
    "ŋ": PhonemeHint("ng"),
    "p": PhonemeHint("p"),
    "ɹ": PhonemeHint("r"),
    "ɾ": PhonemeHint("tt"),  # American flap: t/d between vowels
    "s": PhonemeHint("s"),
    "ʃ": PhonemeHint("sh"),
    "t": PhonemeHint("t"),
    "tʃ": PhonemeHint("ch"),
    "v": PhonemeHint("v"),
    "w": PhonemeHint("w"),
    "z": PhonemeHint("z"),
    "ʒ": PhonemeHint("s"),
    "ð": PhonemeHint("th"),
    "θ": PhonemeHint("th"),
    # Vowels and diphthongs
    "aɪ": PhonemeHint("i"),
    "aʊ": PhonemeHint("ou"),
    "eɪ": PhonemeHint("ay"),
    "i": PhonemeHint("y"),
    "iː": PhonemeHint("ee"),
    "oʊ": PhonemeHint("oa"),
    "uː": PhonemeHint("ue"),
    "æ": PhonemeHint("a"),
    "ə": PhonemeHint("a"),
    "əl": PhonemeHint("le"),  # syllabic l
    "ɚ": PhonemeHint("er"),
    "ɑːɹ": PhonemeHint("ar"),
    "ɔ": PhonemeHint("o"),
    "ɔː": PhonemeHint("aw"),
    "ɔːɹ": PhonemeHint("or"),
    "ɛ": PhonemeHint("e"),
    "ɪ": PhonemeHint("i"),
    "ɪɹ": PhonemeHint("ear"),
    "ʊ": PhonemeHint("oo"),
    "ʌ": PhonemeHint("u"),
}


def locate_grapheme(
    word: str, phoneme: str, phoneme_index_in_word: int, word_phonemes: list[str]
) -> tuple[int, int] | None:
    """Locate a phoneme's likely grapheme range without full G2P alignment.

    The Nth phoneme occurrence maps to the Nth typical grapheme occurrence
    only when their counts agree. Ambiguous or inconsistent input returns
    None instead of guessing.
    """
    # Reject inconsistent caller input and phonemes without a known grapheme.
    if not (0 <= phoneme_index_in_word < len(word_phonemes)):
        return None
    if word_phonemes[phoneme_index_in_word] != phoneme:
        return None
    hint = PHONEME_HINTS.get(phoneme)
    if hint is None:
        return None

    # Determine which occurrence of this phoneme the caller selected.
    occurrence_rank = sum(
        1 for i in range(phoneme_index_in_word + 1) if word_phonemes[i] == phoneme
    )
    total_phoneme_occurrences = sum(1 for p in word_phonemes if p == phoneme)

    grapheme = hint.grapheme.lower()
    word_lower = word.lower()
    # Collect non-overlapping grapheme occurrences from left to right.
    positions: list[tuple[int, int]] = []
    search_from = 0
    while True:
        idx = word_lower.find(grapheme, search_from)
        if idx == -1:
            break
        positions.append((idx, idx + len(grapheme)))
        search_from = idx + len(grapheme)

    # A count mismatch signals an irregular or ambiguous spelling.
    if len(positions) != total_phoneme_occurrences:
        return None
    return positions[occurrence_rank - 1]
