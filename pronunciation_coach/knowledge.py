"""Structured knowledge lookup for pronunciation error explanations.

The LLM rephrases these facts but does not detect or re-evaluate errors.
Phoneme inputs and `MatchSpec.context_next` must
use the normalized symbol inventory produced by `g2p.normalize()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pronunciation_coach.types import KnowledgeRecord, PhonemeError, PracticeWord

_DATA_DIR = Path(__file__).parent / "knowledge_data"
_L1_RULES_DIR = _DATA_DIR / "l1_rules"
_PHONEME_FALLBACK_PATH = _DATA_DIR / "phoneme_fallback.json"

# Normalized vowel inventory used by the ANY_VOWEL pattern. Keep it in sync
# with g2p.EQUIVALENCE_CLASSES. Without word boundaries, non-linguistic vowel
# insertions may also match.
_VOWELS = frozenset(
    {
        "i", "iː", "ɪ", "e", "ɛ", "æ", "ɑ", "ɑː", "ɒ", "ɔ", "ɔː",
        "o", "oʊ", "ʊ", "u", "uː", "ʌ", "ɜ", "ɚ", "ə",
        "aɪ", "aʊ", "ɔɪ", "eɪ",
    }
)


@dataclass(frozen=True)
class MatchSpec:
    op: Literal["substitution", "deletion", "insertion"]
    expected: str | list[str] | Literal["ANY_VOWEL", "ANY"] | None = None
    actual: str | list[str] | Literal["ANY_VOWEL", "ANY"] | None = None
    context_next: list[str] | None = None  # Checked at error.position + 1.
    relation: Literal["length_mismatch"] | None = None  # Skips expected/actual matching.


# Match specifications stay in Python rather than the content-oriented JSON
# files. Every JSON rule id must have an entry here;
# match_knowledge fails fast when one is missing.
_L1_MATCH_SPECS: dict[str, MatchSpec] = {
    "dh_stopping": MatchSpec(op="substitution", expected="ð", actual="d"),
    "th_sibilant_substitution": MatchSpec(op="substitution", expected="θ", actual="s"),
    "liquid_confusion": MatchSpec(
        op="substitution", expected=["l", "ɹ"], actual=["l", "ɹ"]
    ),
    "si_palatalization": MatchSpec(
        op="substitution", expected="s", actual="ʃ", context_next=["i", "ɪ", "iː"]
    ),
    "vowel_epenthesis": MatchSpec(op="insertion", actual="ANY_VOWEL"),
    "vowel_length_mismatch": MatchSpec(op="substitution", relation="length_mismatch"),
}


def _phoneme_matches(
    pattern: str | list[str] | Literal["ANY_VOWEL", "ANY"] | None, value: str | None
) -> bool:
    if pattern is None:
        return True
    if value is None:
        return False
    if pattern == "ANY":
        return True
    if pattern == "ANY_VOWEL":
        return value in _VOWELS
    if isinstance(pattern, list):
        return value in pattern
    return value == pattern


def _is_length_mismatch(expected: str, actual: str) -> bool:
    """Check whether two vowels differ only by the length mark.

    Quality differences such as /ɪ/ versus /iː/ do not match.
    """
    if expected == actual:
        return False
    base_expected = expected.rstrip("ː")
    base_actual = actual.rstrip("ː")
    return base_expected == base_actual and expected.endswith("ː") != actual.endswith("ː")


def _matches(spec: MatchSpec, error: PhonemeError, reference_phonemes: list[str]) -> bool:
    if error.op != spec.op:
        return False

    if spec.relation == "length_mismatch":
        if error.expected is None or error.actual is None:
            return False
        return _is_length_mismatch(error.expected, error.actual)

    if not _phoneme_matches(spec.expected, error.expected):
        return False
    if not _phoneme_matches(spec.actual, error.actual):
        return False

    if spec.context_next is not None:
        next_index = error.position + 1
        if next_index >= len(reference_phonemes):
            return False
        if reference_phonemes[next_index] not in spec.context_next:
            return False

    return True


def _record_from_json(raw: dict, tier: Literal["l1_specific", "phoneme_fallback"], cause: str | None) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=raw["id"],
        phenomenon=raw["phenomenon"],
        cause=cause,
        articulation_tip=raw["articulation_tip"],
        practice_words=[PracticeWord(**w) for w in raw["practice_words"]],
        citation=raw["citation"],
        tier=tier,
    )


@lru_cache(maxsize=None)
def load_l1_rules(l1: str) -> list[KnowledgeRecord]:
    """Load L1 rules in match-priority order.

    Missing languages return no rules and fall through to the phoneme
    fallback layer.
    """
    path = _L1_RULES_DIR / f"{l1.strip().lower()}.json"
    if not path.exists():
        return []
    raw_records = json.loads(path.read_text())
    return [_record_from_json(raw, "l1_specific", raw["cause"]) for raw in raw_records]


@lru_cache(maxsize=None)
def load_phoneme_fallback() -> dict[str, KnowledgeRecord]:
    """Load L1-independent fallback records keyed by target phoneme."""
    raw_records = json.loads(_PHONEME_FALLBACK_PATH.read_text())
    return {
        raw["phoneme"]: _record_from_json(raw, "phoneme_fallback", cause=None)
        for raw in raw_records
    }


def match_knowledge(
    error: PhonemeError, reference_phonemes: list[str], l1: str
) -> KnowledgeRecord | None:
    """Classify one already-detected error into a knowledge tier.

    Priority is L1-specific rule, phoneme fallback, then no match. JSON record
    order resolves multiple L1 matches.
    """
    for record in load_l1_rules(l1):
        spec = _L1_MATCH_SPECS.get(record.id)
        if spec is None:
            raise KeyError(
                f"No MatchSpec registered for l1 rule id {record.id!r}; "
                "add an entry to _L1_MATCH_SPECS in knowledge.py"
            )
        if _matches(spec, error, reference_phonemes):
            return record

    # Insertions have no target phoneme (nothing was "supposed to be" there),
    # so the fallback dictionary cannot apply.
    if error.op in ("substitution", "deletion") and error.expected is not None:
        return load_phoneme_fallback().get(error.expected)

    return None
