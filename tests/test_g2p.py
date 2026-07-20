import pytest

from pronunciation_coach.g2p import normalize


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        # Empty input produces empty output.
        ([], []),
        # Plain phonemes pass through unchanged.
        (["ð", "ɪ", "s"], ["ð", "ɪ", "s"]),
        # Primary and secondary stress marks are stripped from elements.
        (["ˈaɪ", "ˌb"], ["aɪ", "b"]),
        # Stress marks embedded mid-element are stripped too.
        (["ɹˈaɪt"], ["ɹaɪt"]),
        # The length mark ː must be preserved (vowel-length errors matter
        # for Japanese learners; docs/design.md §4).
        (["uː", "ɑː"], ["uː", "ɑː"]),
        # Length mark survives while stress on the same element is stripped.
        (["ˈuː"], ["uː"]),
        # Elements consisting only of stress marks are dropped entirely.
        (["ˈ", "ə", "ˌ"], ["ə"]),
        (["ˈ"], []),
    ],
)
def test_normalize(phonemes, expected):
    """normalize should strip stress marks, keep ː, and drop emptied elements."""
    assert normalize(phonemes) == expected
