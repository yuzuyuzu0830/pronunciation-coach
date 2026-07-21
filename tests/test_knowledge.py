"""Tests for pronunciation_coach.knowledge (docs/design_3c.md §1, §2, §3)."""

from __future__ import annotations

import json

from pronunciation_coach import g2p
from pronunciation_coach.knowledge import (
    _L1_MATCH_SPECS,
    _L1_RULES_DIR,
    _matches,
    load_l1_rules,
    load_phoneme_fallback,
    match_knowledge,
)
from pronunciation_coach.types import PhonemeError

_L1_RECORD_IDS = {
    "dh_stopping",
    "th_sibilant_substitution",
    "liquid_confusion",
    "si_palatalization",
    "vowel_epenthesis",
    "vowel_length_mismatch",
}


# --- loading ---


def test_load_l1_rules_returns_the_six_patterns_tagged_l1_specific():
    records = load_l1_rules("Japanese")
    assert {r.id for r in records} == _L1_RECORD_IDS
    assert all(r.tier == "l1_specific" for r in records)
    assert all(r.cause is not None for r in records)


def test_load_l1_rules_unknown_l1_returns_empty_list():
    assert load_l1_rules("Klingon") == []


def test_l1_rule_ids_and_matchspecs_are_in_bidirectional_agreement():
    """Regression test for the 2026-07-21 incident: japanese.json's ids were
    hand-edited independently of knowledge.py's hardcoded _L1_MATCH_SPECS
    table (parenthetical suffixes added, one id reverted to an old name) and
    silently diverged, so match_knowledge() raised KeyError for 4 of 6
    patterns. _L1_MATCH_SPECS stays a hardcoded Python table for now rather
    than JSON-driven (see docs/devlog.md 2026-07-21 backlog note); this test
    is the cheap substitute for a compile-time link between the two files —
    it reads every l1_rules/*.json directly, independent of match_knowledge's
    own fail-fast check, so it fails on a mismatch instead of relying on some
    other test to happen to exercise the broken id.
    """
    json_ids: set[str] = set()
    for path in _L1_RULES_DIR.glob("*.json"):
        json_ids.update(record["id"] for record in json.loads(path.read_text()))

    spec_ids = set(_L1_MATCH_SPECS)

    assert json_ids <= spec_ids, (
        f"ids present in l1_rules/*.json with no _L1_MATCH_SPECS entry: "
        f"{json_ids - spec_ids}"
    )
    assert spec_ids <= json_ids, (
        f"_L1_MATCH_SPECS entries unused by any l1_rules/*.json id: "
        f"{spec_ids - json_ids}"
    )


def test_load_phoneme_fallback_tagged_phoneme_fallback_with_no_cause():
    fallback = load_phoneme_fallback()
    assert len(fallback) >= 10
    assert all(r.tier == "phoneme_fallback" for r in fallback.values())
    assert all(r.cause is None for r in fallback.values())


# --- practice word factuality: regression test for 3b's fabricated practice
# words (thin/theme used for ð, red used for flap) — the pattern only lives
# in this JSON table now, so a machine check replaces manual review ---


def _all_records():
    yield from load_l1_rules("Japanese")
    yield from load_phoneme_fallback().values()


def test_practice_words_contain_their_target_phoneme():
    for record in _all_records():
        assert record.practice_words, f"{record.id} has no practice words"
        for pw in record.practice_words:
            phonemes = g2p.normalize(g2p.to_phonemes(pw.word))
            assert pw.target_phoneme in phonemes, (
                f"{record.id}: practice word {pw.word!r} does not contain "
                f"target phoneme {pw.target_phoneme!r} (g2p gave {phonemes})"
            )


# --- MatchSpec: positive matches for each of the 6 initial patterns ---


def test_dh_stopping_matches():
    error = PhonemeError("substitution", "ð", "d", 0, "this")
    record = match_knowledge(error, reference_phonemes=["ð", "ɪ", "s"], l1="Japanese")
    assert record is not None and record.id == "dh_stopping"


def test_th_sibilant_substitution_matches():
    error = PhonemeError("substitution", "θ", "s", 0, "think")
    record = match_knowledge(
        error, reference_phonemes=["θ", "ɪ", "ŋ", "k"], l1="Japanese"
    )
    assert record is not None and record.id == "th_sibilant_substitution"


def test_liquid_confusion_matches_both_directions():
    l_to_r = PhonemeError("substitution", "l", "ɹ", 0, "light")
    r_to_l = PhonemeError("substitution", "ɹ", "l", 0, "right")
    for error in (l_to_r, r_to_l):
        record = match_knowledge(
            error, reference_phonemes=["l", "aɪ", "t"], l1="Japanese"
        )
        assert record is not None and record.id == "liquid_confusion"


def test_si_palatalization_matches_before_i_like_vowels():
    for next_phoneme in ("i", "ɪ", "iː"):
        error = PhonemeError("substitution", "s", "ʃ", 0, "sit")
        record = match_knowledge(
            error, reference_phonemes=["s", next_phoneme, "t"], l1="Japanese"
        )
        assert record is not None and record.id == "si_palatalization"


def test_vowel_epenthesis_matches_any_inserted_vowel():
    for vowel in ("ɪ", "ɑː", "ə"):
        error = PhonemeError("insertion", None, vowel, 0, "word")
        record = match_knowledge(error, reference_phonemes=["w"], l1="Japanese")
        assert record is not None and record.id == "vowel_epenthesis"


def test_vowel_length_mismatch_matches_same_base_length_only_difference():
    error = PhonemeError("substitution", "ɑː", "ɑ", 0, "father")
    record = match_knowledge(error, reference_phonemes=["ɑː"], l1="Japanese")
    assert record is not None and record.id == "vowel_length_mismatch"


# --- MatchSpec: near-neighbor cases that must NOT match ---


def test_si_palatalization_does_not_match_before_a_consonant():
    """Explicit near-neighbor case from docs/design_3c.md §2: the same
    substitution is not the palatalization pattern when nothing palatalizing
    follows."""
    error = PhonemeError("substitution", "s", "ʃ", 0, "word")
    record = match_knowledge(error, reference_phonemes=["s", "t"], l1="Japanese")
    assert record is None or record.id != "si_palatalization"


def test_si_palatalization_does_not_match_at_end_of_reference():
    """No phoneme follows position 0, so context_next has nothing to check;
    this must not match rather than raise or match by default."""
    error = PhonemeError("substitution", "s", "ʃ", 0, "word")
    record = match_knowledge(error, reference_phonemes=["s"], l1="Japanese")
    assert record is None or record.id != "si_palatalization"


def test_dh_stopping_does_not_match_a_different_substitution_of_dh():
    error = PhonemeError("substitution", "ð", "v", 0, "this")
    record = match_knowledge(error, reference_phonemes=["ð"], l1="Japanese")
    assert record is None or record.id != "dh_stopping"


def test_liquid_confusion_does_not_match_an_unrelated_consonant():
    error = PhonemeError("substitution", "l", "n", 0, "light")
    record = match_knowledge(error, reference_phonemes=["l"], l1="Japanese")
    assert record is None or record.id != "liquid_confusion"


def test_vowel_epenthesis_does_not_match_an_inserted_consonant():
    error = PhonemeError("insertion", None, "p", 0, "word")
    record = match_knowledge(error, reference_phonemes=["w"], l1="Japanese")
    assert record is None or record.id != "vowel_epenthesis"


def test_vowel_length_mismatch_does_not_match_different_vowel_quality():
    """Near-neighbor case: /ɪ/ vs /iː/ differ in quality, not just in the
    presence of ː, so the length_mismatch relation must reject this pair
    (docs/design_3c.md §2)."""
    error = PhonemeError("substitution", "ɪ", "iː", 0, "word")
    record = match_knowledge(error, reference_phonemes=["ɪ"], l1="Japanese")
    assert record is None or record.id != "vowel_length_mismatch"


# --- mutual exclusivity: exhaustive pairwise check (docs/design_3c.md §2, §8 item 5) ---

_CANONICAL_ERRORS = {
    "dh_stopping": (PhonemeError("substitution", "ð", "d", 0, "this"), ["ð", "ɪ", "s"]),
    "th_sibilant_substitution": (
        PhonemeError("substitution", "θ", "s", 0, "think"),
        ["θ", "ɪ", "ŋ", "k"],
    ),
    "liquid_confusion": (
        PhonemeError("substitution", "l", "ɹ", 0, "light"),
        ["l", "aɪ", "t"],
    ),
    "si_palatalization": (
        PhonemeError("substitution", "s", "ʃ", 0, "sit"),
        ["s", "ɪ", "t"],
    ),
    "vowel_epenthesis": (
        PhonemeError("insertion", None, "ɪ", 0, "book"),
        ["b", "ʊ", "k"],
    ),
    "vowel_length_mismatch": (
        PhonemeError("substitution", "ɑː", "ɑ", 0, "father"),
        ["ɑː"],
    ),
}


def test_canonical_errors_match_exactly_one_pattern():
    """No synthetic error crafted for one pattern accidentally satisfies
    another pattern's MatchSpec. None of the 6 patterns turned out to have a
    genuinely ambiguous overlap (docs/design_3c.md §8 item 5 allows resolving
    such cases by record order if one is ever found; record order in
    japanese.json is the tie-breaker going forward)."""
    for expected_id, (error, reference_phonemes) in _CANONICAL_ERRORS.items():
        hits = [
            record_id
            for record_id, spec in _L1_MATCH_SPECS.items()
            if _matches(spec, error, reference_phonemes)
        ]
        assert hits == [expected_id], (
            f"{expected_id}'s canonical error also matched {hits}; resolve "
            "via record order in japanese.json and document why (docs/design_3c.md §2)"
        )


# --- fallback tier and no-match tier ---


def test_match_knowledge_falls_back_to_phoneme_dictionary_when_l1_table_misses():
    error = PhonemeError("substitution", "v", "b", 0, "van")
    record = match_knowledge(error, reference_phonemes=["v", "æ", "n"], l1="Japanese")
    assert record is not None and record.tier == "phoneme_fallback"


def test_match_knowledge_returns_none_when_nothing_matches():
    error = PhonemeError("substitution", "x", "y", 0, "word")
    record = match_knowledge(error, reference_phonemes=["x"], l1="Japanese")
    assert record is None


def test_match_knowledge_insertion_never_uses_the_fallback_dictionary():
    """Insertions have no `expected` phoneme, so there is no target sound to
    look up in the fallback dictionary (docs/design_3c.md §3)."""
    error = PhonemeError("insertion", None, "p", 0, "word")
    record = match_knowledge(error, reference_phonemes=["w"], l1="Japanese")
    assert record is None
