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
        ("IH1", "IH"),
        ("AH0", "AH"),
        ("ER2", "ER"),
        ("S", "S"),  # no stress digit: unchanged
        ("", ""),
    ],
)
def test_strip_stress(phone, expected):
    assert strip_stress(phone) == expected


@pytest.mark.parametrize(
    "gt_phone, espeak_phone, expected_cost",
    [
        # Accepted transcription-convention pair: no cost.
        ("DH", "ð", 0.0),
        ("IH1", "ɪ", 0.0),  # stress digit stripped before lookup
        ("T", "ɾ", 0.0),  # flap allophone
        ("ER0", "ɚ", 0.0),
        # Same broad class (vowel/consonant) but not an accepted pair: half cost.
        ("IH0", "ə", 0.5),
        ("T", "p", 0.5),
        # Different broad class: full cost.
        ("T", "ɪ", 1.0),
        ("IH1", "s", 1.0),
    ],
)
def test_equivalence_cost(gt_phone, espeak_phone, expected_cost):
    assert equivalence_cost(gt_phone, espeak_phone) == expected_cost


def test_map_word_positions_one_to_one():
    result = map_word_positions(["DH", "IH1", "S"], ["ð", "ɪ", "s"])
    assert result == WordPositionMap(gt_to_reference=[0, 1, 2], unmapped_reference_indices=[])


def test_map_word_positions_unmapped_gt_phone():
    """A ground-truth phone with no matching reference phone maps to None.

    gt=[T, IH1] vs reference=[ɪ]: deleting T and matching IH1->ɪ (cost 1.0)
    beats substituting T->ɪ and deleting IH1 (cost 2.0).
    """
    result = map_word_positions(["T", "IH1"], ["ɪ"])
    assert result == WordPositionMap(gt_to_reference=[None, 0], unmapped_reference_indices=[])


def test_map_word_positions_unmapped_reference_phone():
    """A reference phone with no ground-truth counterpart is reported separately.

    gt=[IH1] vs reference=[ə, ɪ]: inserting the leading ə and matching
    IH1->ɪ (cost 1.0) beats substituting IH1->ə and inserting ɪ (cost 1.5).
    """
    result = map_word_positions(["IH1"], ["ə", "ɪ"])
    assert result == WordPositionMap(gt_to_reference=[1], unmapped_reference_indices=[0])


def test_map_word_positions_empty_inputs():
    assert map_word_positions([], []) == WordPositionMap([], [])
