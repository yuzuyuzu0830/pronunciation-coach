"""Language-neutral example words and graphemes for English phonemes.

UI phrasing belongs to `phoneme_hint_format.py`. Coverage is intentionally
limited to trial and common phonemes; unknown symbols render without a hint.
Keys use the canonical symbols produced by `g2p.normalize()`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhonemeHint:
    example: str  # an English word containing the sound
    grapheme: str  # the spelling in that word usually associated with the sound


PHONEME_HINTS: dict[str, PhonemeHint] = {
    # Consonants
    "b": PhonemeHint("bed", "b"),
    "d": PhonemeHint("dog", "d"),
    "dʒ": PhonemeHint("job", "j"),
    "f": PhonemeHint("fish", "f"),
    "ɡ": PhonemeHint("go", "g"),
    "h": PhonemeHint("hat", "h"),
    "j": PhonemeHint("yes", "y"),
    "k": PhonemeHint("key", "k"),
    "l": PhonemeHint("light", "l"),
    "m": PhonemeHint("man", "m"),
    "n": PhonemeHint("nice", "n"),
    "ŋ": PhonemeHint("king", "ng"),
    "p": PhonemeHint("pen", "p"),
    "ɹ": PhonemeHint("red", "r"),
    "ɾ": PhonemeHint("butter", "tt"),  # American flap: t/d between vowels
    "s": PhonemeHint("sun", "s"),
    "ʃ": PhonemeHint("shoe", "sh"),
    "t": PhonemeHint("top", "t"),
    "tʃ": PhonemeHint("chair", "ch"),
    "v": PhonemeHint("van", "v"),
    "w": PhonemeHint("water", "w"),
    "z": PhonemeHint("zoo", "z"),
    "ʒ": PhonemeHint("measure", "s"),
    "ð": PhonemeHint("this", "th"),
    "θ": PhonemeHint("think", "th"),
    # Vowels and diphthongs
    "aɪ": PhonemeHint("time", "i"),
    "aʊ": PhonemeHint("house", "ou"),
    "eɪ": PhonemeHint("day", "ay"),
    "i": PhonemeHint("happy", "y"),
    "iː": PhonemeHint("see", "ee"),
    "oʊ": PhonemeHint("boat", "oa"),
    "uː": PhonemeHint("blue", "ue"),
    "æ": PhonemeHint("cat", "a"),
    "ə": PhonemeHint("about", "a"),
    "əl": PhonemeHint("table", "le"),  # syllabic l
    "ɚ": PhonemeHint("teacher", "er"),
    "ɑːɹ": PhonemeHint("car", "ar"),
    "ɔ": PhonemeHint("dog", "o"),
    "ɔː": PhonemeHint("law", "aw"),
    "ɔːɹ": PhonemeHint("door", "or"),
    "ɛ": PhonemeHint("bed", "e"),
    "ɪ": PhonemeHint("sit", "i"),
    "ɪɹ": PhonemeHint("near", "ear"),
    "ʊ": PhonemeHint("book", "oo"),
    "ʌ": PhonemeHint("cup", "u"),
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
