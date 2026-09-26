import pytest

from pronunciation_coach.aligner import align_phonemes
from pronunciation_coach.types import AlignmentOp


def match(phone: str) -> AlignmentOp:
    return AlignmentOp("match", phone, phone)


def sub(ref: str, hyp: str) -> AlignmentOp:
    return AlignmentOp("substitution", ref, hyp)


def dele(ref: str) -> AlignmentOp:
    return AlignmentOp("deletion", ref, None)


def ins(hyp: str) -> AlignmentOp:
    return AlignmentOp("insertion", None, hyp)


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        pytest.param([], [], [], id="empty"),
        pytest.param(["ð", "ɪ"], [], [dele("ð"), dele("ɪ")], id="deletions"),
        pytest.param([], ["ð", "ɪ"], [ins("ð"), ins("ɪ")], id="insertions"),
        pytest.param(
            ["ð", "ɪ", "s"],
            ["ð", "ɪ", "s"],
            [match("ð"), match("ɪ"), match("s")],
            id="matches",
        ),
        pytest.param(
            ["ð", "æ"],
            ["d", "ʌ"],
            [sub("ð", "d"), sub("æ", "ʌ")],
            id="substitutions",
        ),
        pytest.param(
            ["k", "æ", "t"],
            ["k", "æ"],
            [match("k"), match("æ"), dele("t")],
            id="final-deletion",
        ),
        pytest.param(
            ["k", "æ"],
            ["k", "æ", "ɯ"],
            [match("k"), match("æ"), ins("ɯ")],
            id="vowel-epenthesis",
        ),
        pytest.param(
            ["ð", "ɪ", "s"],
            ["d", "ɪ"],
            [sub("ð", "d"), match("ɪ"), dele("s")],
            id="substitution-and-deletion",
        ),
    ],
)
def test_align_phonemes(
    reference: list[str], hypothesis: list[str], expected: list[AlignmentOp]
) -> None:
    assert align_phonemes(reference, hypothesis) == expected


def test_tie_break_prefers_substitution_over_gap_pair() -> None:
    assert align_phonemes(["a", "b"], ["b", "c"]) == [sub("a", "b"), sub("b", "c")]


def test_tie_break_prefers_substitution_over_deletion() -> None:
    assert align_phonemes(["a", "b"], ["c"]) == [dele("a"), sub("b", "c")]


def test_tie_break_prefers_deletion_over_insertion() -> None:
    assert align_phonemes(["a", "b", "a"], ["b", "a", "b"]) == [
        ins("b"),
        match("a"),
        match("b"),
        dele("a"),
    ]
