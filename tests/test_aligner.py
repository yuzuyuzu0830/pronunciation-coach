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
        # Empty inputs produce an empty alignment.
        ([], [], []),
        # Empty hypothesis: every reference phone is deleted.
        (["ð", "ɪ"], [], [dele("ð"), dele("ɪ")]),
        # Empty reference: every hypothesis phone is inserted.
        ([], ["ð", "ɪ"], [ins("ð"), ins("ɪ")]),
        # Identical sequences: every phone is a match.
        (["ð", "ɪ", "s"], ["ð", "ɪ", "s"], [match("ð"), match("ɪ"), match("s")]),
        # Substitutions only.
        (["ð", "æ"], ["d", "ʌ"], [sub("ð", "d"), sub("æ", "ʌ")]),
        # Deletion only: final phone missing from the hypothesis.
        (["k", "æ", "t"], ["k", "æ"], [match("k"), match("æ"), dele("t")]),
        # Insertion only: extra phone in the hypothesis (vowel epenthesis).
        (["k", "æ"], ["k", "æ", "ɯ"], [match("k"), match("æ"), ins("ɯ")]),
        # Mixed: substitution then deletion (this -> "di").
        (["ð", "ɪ", "s"], ["d", "ɪ"], [sub("ð", "d"), match("ɪ"), dele("s")]),
    ],
)
def test_align_phonemes(reference, hypothesis, expected):
    """align_phonemes should return the expected edit operations."""
    assert align_phonemes(reference, hypothesis) == expected


def test_tie_break_prefers_substitution_over_gap_pair():
    """Equal-cost paths must resolve to substitutions, not delete+insert.

    ref=[a,b] vs hyp=[b,c] costs 2 either as two substitutions or as
    delete(a)+match(b)+insert(c); the substitution-first rule picks the former.
    """
    assert align_phonemes(["a", "b"], ["b", "c"]) == [sub("a", "b"), sub("b", "c")]


def test_tie_break_is_deterministic_for_sub_vs_deletion():
    """When substitution and deletion tie, backtrace resolves deterministically.

    ref=[a,b] vs hyp=[c] admits [sub(a,c), del(b)] and [del(a), sub(b,c)] at
    equal cost; the substitution-first backtrace always yields the latter.
    """
    assert align_phonemes(["a", "b"], ["c"]) == [dele("a"), sub("b", "c")]


def test_tie_break_prefers_deletion_over_insertion():
    """When deletion and insertion tie, deletion must win.

    ref=[a,b,a] vs hyp=[b,a,b] costs 2 either as [ins(b), match(a), match(b),
    del(a)] or [del(a), match(b), match(a), ins(b)]; at the final DP cell the
    diagonal is strictly worse (cost 3), so only deletion and insertion tie.
    The deletion-first rule closes the alignment with the deletion.
    """
    assert align_phonemes(["a", "b", "a"], ["b", "a", "b"]) == [
        ins("b"),
        match("a"),
        match("b"),
        dele("a"),
    ]
