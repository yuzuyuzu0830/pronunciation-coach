from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from pronunciation_coach import g2p
from pronunciation_coach.knowledge import (
    _L1_MATCH_SPECS,
    _L1_RULES_DIR,
    _matches,
    load_l1_rules,
    load_phoneme_fallback,
    match_knowledge,
)
from pronunciation_coach.types import KnowledgeRecord, PhonemeError

_L1_RECORD_IDS: set[str] = {
    "dh_stopping",
    "th_sibilant_substitution",
    "liquid_confusion",
    "si_palatalization",
    "vowel_epenthesis",
    "vowel_length_mismatch",
}


# Loading


def test_load_l1_rules_returns_the_six_patterns_tagged_l1_specific() -> None:
    records = load_l1_rules("Japanese")
    assert {r.id for r in records} == _L1_RECORD_IDS
    assert all(r.tier == "l1_specific" for r in records)
    assert all(r.cause is not None for r in records)


def test_load_l1_rules_unknown_l1_returns_empty_list() -> None:
    assert load_l1_rules("Klingon") == []


def test_l1_rule_ids_and_matchspecs_are_in_bidirectional_agreement() -> None:
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


def test_load_phoneme_fallback_tagged_phoneme_fallback_with_no_cause() -> None:
    fallback = load_phoneme_fallback()
    assert len(fallback) >= 10
    assert all(r.tier == "phoneme_fallback" for r in fallback.values())
    assert all(r.cause is None for r in fallback.values())


# Practice words


def _all_records() -> Iterator[KnowledgeRecord]:
    yield from load_l1_rules("Japanese")
    yield from load_phoneme_fallback().values()


def test_practice_words_contain_their_target_phoneme() -> None:
    for record in _all_records():
        assert record.practice_words, f"{record.id} has no practice words"
        for pw in record.practice_words:
            phonemes = g2p.normalize(g2p.to_phonemes(pw.word))
            assert pw.target_phoneme in phonemes, (
                f"{record.id}: practice word {pw.word!r} does not contain "
                f"target phoneme {pw.target_phoneme!r} (g2p gave {phonemes})"
            )


# Positive matches


def test_dh_stopping_matches() -> None:
    error = PhonemeError("substitution", "ð", "d", 0, "this")
    record = match_knowledge(error, reference_phonemes=["ð", "ɪ", "s"], l1="Japanese")
    assert record is not None and record.id == "dh_stopping"


def test_th_sibilant_substitution_matches() -> None:
    error = PhonemeError("substitution", "θ", "s", 0, "think")
    record = match_knowledge(
        error, reference_phonemes=["θ", "ɪ", "ŋ", "k"], l1="Japanese"
    )
    assert record is not None and record.id == "th_sibilant_substitution"


@pytest.mark.parametrize(
    ("expected", "actual"),
    [("l", "ɹ"), ("ɹ", "l")],
    ids=["l-to-r", "r-to-l"],
)
def test_liquid_confusion_matches_both_directions(expected: str, actual: str) -> None:
    error = PhonemeError("substitution", expected, actual, 0, "word")
    record = match_knowledge(
        error, reference_phonemes=[expected, "aɪ", "t"], l1="Japanese"
    )
    assert record is not None and record.id == "liquid_confusion"


@pytest.mark.parametrize("next_phoneme", ["i", "ɪ", "iː"])
def test_si_palatalization_matches_before_i_like_vowels(next_phoneme: str) -> None:
    error = PhonemeError("substitution", "s", "ʃ", 0, "sit")
    record = match_knowledge(
        error, reference_phonemes=["s", next_phoneme, "t"], l1="Japanese"
    )
    assert record is not None and record.id == "si_palatalization"


@pytest.mark.parametrize("vowel", ["ɪ", "ɑː", "ə"])
def test_vowel_epenthesis_matches_any_inserted_vowel(vowel: str) -> None:
    error = PhonemeError("insertion", None, vowel, 0, "word")
    record = match_knowledge(error, reference_phonemes=["w"], l1="Japanese")
    assert record is not None and record.id == "vowel_epenthesis"


def test_vowel_length_mismatch_matches_same_base_length_only_difference() -> None:
    error = PhonemeError("substitution", "ɑː", "ɑ", 0, "father")
    record = match_knowledge(error, reference_phonemes=["ɑː"], l1="Japanese")
    assert record is not None and record.id == "vowel_length_mismatch"


# Non-matches


def test_si_palatalization_does_not_match_before_a_consonant() -> None:
    error = PhonemeError("substitution", "s", "ʃ", 0, "word")
    record = match_knowledge(error, reference_phonemes=["s", "t"], l1="Japanese")
    assert record is None or record.id != "si_palatalization"


def test_si_palatalization_does_not_match_at_end_of_reference() -> None:
    error = PhonemeError("substitution", "s", "ʃ", 0, "word")
    record = match_knowledge(error, reference_phonemes=["s"], l1="Japanese")
    assert record is None or record.id != "si_palatalization"


def test_dh_stopping_does_not_match_a_different_substitution_of_dh() -> None:
    error = PhonemeError("substitution", "ð", "v", 0, "this")
    record = match_knowledge(error, reference_phonemes=["ð"], l1="Japanese")
    assert record is None or record.id != "dh_stopping"


def test_liquid_confusion_does_not_match_an_unrelated_consonant() -> None:
    error = PhonemeError("substitution", "l", "n", 0, "light")
    record = match_knowledge(error, reference_phonemes=["l"], l1="Japanese")
    assert record is None or record.id != "liquid_confusion"


def test_vowel_epenthesis_does_not_match_an_inserted_consonant() -> None:
    error = PhonemeError("insertion", None, "p", 0, "word")
    record = match_knowledge(error, reference_phonemes=["w"], l1="Japanese")
    assert record is None or record.id != "vowel_epenthesis"


def test_vowel_length_mismatch_does_not_match_different_vowel_quality() -> None:
    error = PhonemeError("substitution", "ɪ", "iː", 0, "word")
    record = match_knowledge(error, reference_phonemes=["ɪ"], l1="Japanese")
    assert record is None or record.id != "vowel_length_mismatch"


# Mutual exclusivity

_CANONICAL_ERRORS: dict[str, tuple[PhonemeError, list[str]]] = {
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


def test_canonical_errors_match_exactly_one_pattern() -> None:
    for expected_id, (error, reference_phonemes) in _CANONICAL_ERRORS.items():
        hits = [
            record_id
            for record_id, spec in _L1_MATCH_SPECS.items()
            if _matches(spec, error, reference_phonemes)
        ]
        assert hits == [expected_id], (
            f"{expected_id}'s canonical error matched unexpected rules: {hits}"
        )


# Fallback and no-match tiers


def test_match_knowledge_uses_fallback_after_l1_miss() -> None:
    error = PhonemeError("substitution", "v", "b", 0, "van")
    record = match_knowledge(error, reference_phonemes=["v", "æ", "n"], l1="Japanese")
    assert record is not None and record.tier == "phoneme_fallback"


def test_match_knowledge_returns_none_when_nothing_matches() -> None:
    error = PhonemeError("substitution", "x", "y", 0, "word")
    record = match_knowledge(error, reference_phonemes=["x"], l1="Japanese")
    assert record is None


def test_match_knowledge_insertion_never_uses_the_fallback_dictionary() -> None:
    error = PhonemeError("insertion", None, "p", 0, "word")
    record = match_knowledge(error, reference_phonemes=["w"], l1="Japanese")
    assert record is None
