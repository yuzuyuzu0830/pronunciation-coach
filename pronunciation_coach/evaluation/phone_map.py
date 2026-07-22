"""ARPAbet (speechocean762 ground truth) <-> espeak IPA (system reference)
position mapping, for evaluation only (docs/design_eval.md §2).

This is NOT g2p.EQUIVALENCE_CLASSES: that table canonicalizes notation
variants the detector must never treat as a learner error (dh/d must stay
distinguishable). Here both sides are always the *canonical* pronunciation
of the same target text (ground truth vs. this system's espeak reference),
so the acceptance criterion is broader by design: "same underlying phoneme,
different transcription convention" (e.g. flapped /t/, schwa vs. stressed
vowel). Provisional first pass; to be corrected against real corpus/espeak
samples during data acquisition (§7).
"""

from __future__ import annotations

from dataclasses import dataclass

from pronunciation_coach.aligner import align_phonemes

# Bare ARPAbet phone (stress digit stripped) -> espeak IPA symbols accepted
# as the same underlying phoneme when aligning ground-truth canonical phones
# against this system's espeak reference.
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

# espeak vowel inventory for the coarse-class DP fallback below. Kept
# independent of knowledge.py's _VOWELS (different purpose: alignment-cost
# fallback here, not ANY_VOWEL matching there) but should stay in sync if
# espeak's vowel set changes.
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
    """DP substitution cost for aligning ground-truth ARPAbet against espeak IPA.

    Three tiers (§2.2): an accepted transcription-convention pair costs
    nothing; phones from the same broad vowel/consonant class cost 0.5 so an
    unrecognized pair still aligns diagonally instead of fragmenting into a
    deletion+insertion pair; anything else costs 1.
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
    # index i -> local index into `reference_phones` for gt_phones[i], or None
    # when no reference phone could be paired with it (§2.3 coverage loss).
    gt_to_reference: list[int | None]
    # local indices into `reference_phones` with no ground-truth counterpart
    # (informational: the espeak reference's own "insertion" relative to
    # ground truth, not the learner's — e.g. an epenthetic schwa espeak adds
    # that the corpus doesn't transcribe as a separate phone).
    unmapped_reference_indices: list[int]


def map_word_positions(gt_phones: list[str], reference_phones: list[str]) -> WordPositionMap:
    """Align one word's ground-truth ARPAbet phones to this system's espeak reference.

    Reuses the detection DP (aligner.align_phonemes) with the
    evaluation-only equivalence_cost, so the tie-break rules (substitution >
    deletion > insertion) match the production alignment exactly.
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
        else:  # insertion: reference phone with no ground-truth counterpart
            unmapped_reference_indices.append(ref_i)
            ref_i += 1
    return WordPositionMap(gt_to_reference, unmapped_reference_indices)
