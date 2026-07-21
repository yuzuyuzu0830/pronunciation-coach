"""Structured knowledge lookup for pronunciation error explanations.

Supplies the LLM with pre-written facts (cause / articulation tip / practice
words) so it only has to rephrase, never invent (docs/design_3c.md §0-§3).
Detection stays untouched: this module only classifies already-detected
PhonemeError values, it never adds, removes, or re-judges errors.

Input contract: `reference_phonemes` and `PhonemeError.expected` / `.actual`
are assumed to already be normalized via `g2p.normalize()` (stress marks
stripped, equivalence classes collapsed). `MatchSpec.context_next` sets must
be written in that same normalized symbol inventory (docs/design_3c.md §2,
§8 item 4). Passing raw phonemizer output or a different transcription
scheme (e.g. ARPAbet) will silently fail to match.
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

# Vowel symbols in this project's normalized (post g2p.normalize) inventory,
# used to test MatchSpec.actual == "ANY_VOWEL" for vowel_epenthesis. Known
# limitation (docs/design_3c.md §7): this also matches non-linguistic vowel
# insertions (e.g. an utterance-initial breath), since word-boundary info is
# out of scope for this module (§2).
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
    context_next: list[str] | None = None  # reference_phonemes[position + 1] must be in this set
    relation: Literal["length_mismatch"] | None = None  # overrides expected/actual comparison


# MatchSpec is an implementation detail of this module's matching logic, not
# part of the JSON data (docs/design_3c.md §6), so each l1 rule id is wired to
# its spec here. New ids added to an l1_rules/*.json file must get an entry
# here too, or match_knowledge raises (fail fast on a typo'd id rather than
# silently never matching).
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
    # No expected/actual: matched purely by the length_mismatch relation.
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
    """True iff expected/actual are the same base vowel and differ only in ː.

    Deliberately narrow: /ɪ/ vs /iː/ differ in quality as well as length and
    must NOT match here (docs/design_3c.md §2) — only pairs like /ɑː/-/ɑ/
    (identical base symbol, one has the length mark) count.
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
    """Load the L1 rule table for `l1`, in file order (= match priority).

    An L1 with no rule file returns an empty list, so match_knowledge falls
    through to the phoneme fallback layer naturally (docs/design_3c.md §1).
    Cached: match_knowledge() calls this once per PhonemeError, and the JSON
    files don't change during a process's lifetime, so re-parsing per error
    is pure waste.
    """
    path = _L1_RULES_DIR / f"{l1.strip().lower()}.json"
    if not path.exists():
        return []
    raw_records = json.loads(path.read_text())
    return [_record_from_json(raw, "l1_specific", raw["cause"]) for raw in raw_records]


@lru_cache(maxsize=None)
def load_phoneme_fallback() -> dict[str, KnowledgeRecord]:
    """Load the L1-independent fallback dictionary, keyed by target phoneme.

    Cached for the same reason as load_l1_rules: called once per unmatched
    error, and the underlying file is static for the process's lifetime.
    """
    raw_records = json.loads(_PHONEME_FALLBACK_PATH.read_text())
    return {
        raw["phoneme"]: _record_from_json(raw, "phoneme_fallback", cause=None)
        for raw in raw_records
    }


def match_knowledge(
    error: PhonemeError, reference_phonemes: list[str], l1: str
) -> KnowledgeRecord | None:
    """Classify one already-detected error into a knowledge tier.

    Tier order (docs/design_3c.md §3): L1-specific rule > phoneme fallback >
    no match (caller renders facts-only for None). Only the first matching
    L1 rule is used — record order in the JSON file is the priority order
    (docs/design_3c.md §2).
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
    # so the fallback dictionary — keyed by target phoneme — cannot apply.
    if error.op in ("substitution", "deletion") and error.expected is not None:
        return load_phoneme_fallback().get(error.expected)

    return None
