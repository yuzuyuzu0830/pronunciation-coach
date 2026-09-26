import pytest

from pronunciation_coach.evaluation.phone_map import (
    WordPositionMap,
    equivalence_cost,
    map_word_positions,
    strip_stress,
)


@pytest.mark.parametrize(
    "phone, expected",
    [
        pytest.param("IH1", "IH", id="primary-stress"),
        pytest.param("AH0", "AH", id="unstressed"),
        pytest.param("ER2", "ER", id="secondary-stress"),
        pytest.param("S", "S", id="no-stress"),
        pytest.param("", "", id="empty"),
    ],
)
def test_strip_stress(phone: str, expected: str) -> None:
    assert strip_stress(phone) == expected


@pytest.mark.parametrize(
    "gt_phone, espeak_phone, expected_cost",
    [
        pytest.param("DH", "ð", 0.0, id="equivalent-consonant"),
        pytest.param("IH1", "ɪ", 0.0, id="equivalent-stressed-vowel"),
        pytest.param("T", "ɾ", 0.0, id="flap-allophone"),
        pytest.param("ER0", "ɚ", 0.0, id="rhotic-vowel"),
        pytest.param("IH0", "ə", 0.5, id="different-vowels"),
        pytest.param("T", "p", 0.5, id="different-consonants"),
        pytest.param("T", "ɪ", 1.0, id="consonant-to-vowel"),
        pytest.param("IH1", "s", 1.0, id="vowel-to-consonant"),
    ],
)
def test_equivalence_cost(
    gt_phone: str, espeak_phone: str, expected_cost: float
) -> None:
    assert equivalence_cost(gt_phone, espeak_phone) == expected_cost


def test_map_word_positions_one_to_one() -> None:
    result = map_word_positions(["DH", "IH1", "S"], ["ð", "ɪ", "s"])
    assert result == WordPositionMap(
        gt_to_reference=[0, 1, 2], unmapped_reference_indices=[]
    )


def test_map_word_positions_prefers_equivalent_phone_after_gt_deletion() -> None:
    result = map_word_positions(["T", "IH1"], ["ɪ"])
    assert result == WordPositionMap(
        gt_to_reference=[None, 0], unmapped_reference_indices=[]
    )


def test_map_word_positions_prefers_match_after_reference_insertion() -> None:
    result = map_word_positions(["IH1"], ["ə", "ɪ"])
    assert result == WordPositionMap(
        gt_to_reference=[1], unmapped_reference_indices=[0]
    )


def test_map_word_positions_empty_inputs() -> None:
    assert map_word_positions([], []) == WordPositionMap([], [])
