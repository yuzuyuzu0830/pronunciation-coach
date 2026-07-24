"""Language-neutral facts about English phonemes: an example word and the
spelling (grapheme) usually associated with the sound (per-symbol, not
per-display-string -- docs/design_ui.md "音素表示のヒント併記" section).

This module holds no display strings: it only states, for a given target
language (English) phoneme, what it sounds like and how it's spelled. How
that fact gets phrased for a reader belongs to the localization layer
(ui/phoneme_hint_format.py), not here -- this table is reusable unchanged
for a Spanish-L1 or Chinese-L1 trial, or an English-language UI, since none
of that depends on the learner's L1 or the UI's display language.

Coverage: the phonemes the current trial sentences (ui/sentences.py) can
produce, plus a handful of other common English phonemes (verified against
real g2p output, not guessed) -- not the full espeak IPA inventory.
Symbols not covered here render without a hint (docs/design_ui.md
"音素表示のヒント併記"): this table is deliberately incomplete, not a
correctness-critical lookup.

Post-normalize() notation variants (docs/design.md §4) are never separate
entries here: e.g. "r"/"ɜː" never appear in reference/hypothesis phonemes by
the time they reach the UI (g2p.EQUIVALENCE_CLASSES already canonicalizes
them to "ɹ"/"ɚ"), so only the canonical symbol needs a hint.
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
    """Best-effort (start, end) character range in `word` for one occurrence
    of `phoneme` -- an approximation, not a true G2P alignment (design_ui.md
    §11 records the trade-off). Returns None whenever the heuristic can't
    confidently disambiguate; callers must treat that as "no highlight", not
    an error.

    Heuristic: PHONEME_HINTS gives `phoneme`'s typical spelling (grapheme).
    If that grapheme substring appears in `word` exactly as many times as
    `phoneme` appears in `word_phonemes`, the Nth phonetic occurrence (in
    phoneme-sequence order) is assumed to line up with the Nth spelled
    occurrence (left to right, non-overlapping). This breaks for irregular
    spellings (silent letters, a digraph not covered by the table, two
    different phonemes sharing one typical grapheme in the same word) --
    the counts won't match in those cases, and this returns None rather
    than guessing which occurrence is which.
    """
    if not (0 <= phoneme_index_in_word < len(word_phonemes)):
        return None
    if word_phonemes[phoneme_index_in_word] != phoneme:
        return None
    hint = PHONEME_HINTS.get(phoneme)
    if hint is None:
        return None

    occurrence_rank = sum(
        1 for i in range(phoneme_index_in_word + 1) if word_phonemes[i] == phoneme
    )
    total_phoneme_occurrences = sum(1 for p in word_phonemes if p == phoneme)

    grapheme = hint.grapheme.lower()
    word_lower = word.lower()
    positions: list[tuple[int, int]] = []
    search_from = 0
    while True:
        idx = word_lower.find(grapheme, search_from)
        if idx == -1:
            break
        positions.append((idx, idx + len(grapheme)))
        search_from = idx + len(grapheme)  # non-overlapping matches only

    if len(positions) != total_phoneme_occurrences:
        return None
    return positions[occurrence_rank - 1]
