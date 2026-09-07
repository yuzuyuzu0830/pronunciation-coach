"""Map SpeechOcean762 ARPAbet positions to the system's espeak IPA reference.

Unlike `g2p.EQUIVALENCE_CLASSES`, this evaluation-only mapping accepts
canonical-pronunciation differences between transcription systems.
The table is provisional pending validation against corpus samples.
"""

from __future__ import annotations

from dataclasses import dataclass

from pronunciation_coach.aligner import align_phonemes

# Bare ARPAbet phone to equivalent espeak IPA symbols for reference alignment.
ACCEPTED_ESPEAK: dict[str, frozenset[str]] = {
    # Consonants
    "B": frozenset({"b"}),
    "CH": frozenset({"tʃ"}),
    "D": frozenset({"d"}),
    "DH": frozenset({"ð"}),
    "F": frozenset({"f"}),
    "G": frozenset({"ɡ", "g"}),
    "HH": frozenset({"h"}),
    "JH": frozenset({"dʒ"}),
    "K": frozenset({"k"}),
    "L": frozenset({"l"}),
    "M": frozenset({"m"}),
    "N": frozenset({"n"}),
    "NG": frozenset({"ŋ"}),
    "P": frozenset({"p"}),
    "R": frozenset({"ɹ", "r"}),
    "S": frozenset({"s"}),
    "SH": frozenset({"ʃ"}),
    "T": frozenset({"t", "ɾ", "ʔ"}),  # flap and glottal-stop allophones observed in espeak output
    "TH": frozenset({"θ"}),
    "V": frozenset({"v"}),
    "W": frozenset({"w"}),
    "Y": frozenset({"j"}),
    "Z": frozenset({"z"}),
    "ZH": frozenset({"ʒ"}),
    # Vowels
    "AA": frozenset({"ɑː", "ɑ"}),
    "AE": frozenset({"æ"}),
    "AH": frozenset({"ʌ", "ə"}),  # stressed AH1 -> ʌ, unstressed AH0 -> ə; stress stripped before lookup
    "AO": frozenset({"ɔː", "ɔ"}),
    "AW": frozenset({"aʊ"}),
    "AY": frozenset({"aɪ"}),
    "EH": frozenset({"ɛ"}),
    "ER": frozenset({"ɚ", "ɜː"}),  # matches g2p.EQUIVALENCE_CLASSES' r-colored-vowel pair
    "EY": frozenset({"eɪ"}),
    "IH": frozenset({"ɪ"}),
    "IY": frozenset({"iː"}),
    "OW": frozenset({"oʊ"}),
    "OY": frozenset({"ɔɪ"}),
    "UH": frozenset({"ʊ"}),
    "UW": frozenset({"uː"}),
}

_ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)

# Used only by the broad-class alignment fallback. This remains separate from
# knowledge._VOWELS but must track changes to the espeak vowel inventory.
ESPEAK_VOWELS = frozenset(
    {
        "i", "iː", "ɪ", "e", "ɛ", "æ", "ɑ", "ɑː", "ɒ", "ɔ", "ɔː",
        "o", "oʊ", "ʊ", "u", "uː", "ʌ", "ɜ", "ɜː", "ɚ", "ə",
        "aɪ", "aʊ", "ɔɪ", "eɪ",
    }
)

COST_EQUIVALENT = 0.0
COST_SAME_BROAD_CLASS = 0.5
COST_UNRELATED = 1.0


def strip_stress(phone: str) -> str:
    return phone[:-1] if phone and phone[-1] in "012" else phone


def equivalence_cost(gt_phone: str, espeak_phone: str) -> float:
    """Return the substitution cost for ARPAbet-to-espeak alignment.

    Equivalent pairs cost 0, phones in the same broad class cost 0.5 to favor
    diagonal alignment, and unrelated phones cost 1.
    """
    bare = strip_stress(gt_phone)
    if espeak_phone in ACCEPTED_ESPEAK.get(bare, frozenset()):
        return COST_EQUIVALENT
    gt_is_vowel = bare in _ARPABET_VOWELS
    espeak_is_vowel = espeak_phone in ESPEAK_VOWELS
    if gt_is_vowel == espeak_is_vowel:
        return COST_SAME_BROAD_CLASS
    return COST_UNRELATED


@dataclass(frozen=True)
class WordPositionMap:
    # Reference index for each ground-truth phone; None marks coverage loss.
    gt_to_reference: list[int | None]
    # Reference phones with no ground-truth counterpart, unrelated to learner errors.
    unmapped_reference_indices: list[int]


def map_word_positions(gt_phones: list[str], reference_phones: list[str]) -> WordPositionMap:
    """Map one word's ground-truth phones to the espeak reference.

    Reusing the production aligner preserves its substitution, deletion, and
    insertion tie-break order while applying the evaluation-only cost.
    """
    ops = align_phonemes(gt_phones, reference_phones, substitution_cost=equivalence_cost)

    gt_to_reference: list[int | None] = []
    unmapped_reference_indices: list[int] = []
    ref_i = 0
    for op in ops:
        if op.op in ("match", "substitution"):
            gt_to_reference.append(ref_i)
            ref_i += 1
        elif op.op == "deletion":
            gt_to_reference.append(None)
        else:  # Reference insertion, not a learner insertion.
            unmapped_reference_indices.append(ref_i)
            ref_i += 1
    return WordPositionMap(gt_to_reference, unmapped_reference_indices)
