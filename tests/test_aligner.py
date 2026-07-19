import pytest

from aligner import align_phonemes


# Each case provides (reference, hypothesis, expected alignment ops).
# An op is (type, ref_phone | None, hyp_phone | None).
@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        # Empty inputs produce an empty alignment.
        ([], [], []),
        # Identical sequences: every phone is a match.
        (
            ["AH", "B"],
            ["AH", "B"],
            [("match", "AH", "AH"), ("match", "B", "B")],
        ),
        # Differing phone at the same position: substitution.
        (
            ["AH", "B"],
            ["AH", "P"],
            [("match", "AH", "AH"), ("substitution", "B", "P")],
        ),
        # Phone present in reference but missing in hypothesis: deletion.
        (
            ["AH", "B"],
            ["AH"],
            [("match", "AH", "AH"), ("deletion", "B", None)],
        ),
        # Phone present in hypothesis but missing in reference: insertion.
        (
            ["AH"],
            ["AH", "B"],
            [("match", "AH", "AH"), ("insertion", None, "B")],
        ),
    ],
)
def test_align_phonemes(reference, hypothesis, expected):
    """align_phonemes should return the expected edit operations."""
    assert align_phonemes(reference, hypothesis) == expected
