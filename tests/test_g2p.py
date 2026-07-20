import pytest

from pronunciation_coach.g2p import normalize, to_phonemes, to_phonemes_by_word

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except Exception:
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng is not installed"
)


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


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        # r and ɹ are notation variants of the same alveolar approximant in
        # the espeak inventory; hypothesis output flips between them (measured
        # in E2E), so both must map to the canonical ɹ.
        (["r"], ["ɹ"]),
        (["r", "aɪ", "t"], ["ɹ", "aɪ", "t"]),
        # ɜː and ɚ are notation variants of the r-colored vowel (en-us);
        # reference emits ɜː for e.g. "church" while hypothesis may emit ɚ.
        (["ɜː"], ["ɚ"]),
        (["tʃ", "ɜː", "tʃ"], ["tʃ", "ɚ", "tʃ"]),
        # The canonical symbols themselves pass through unchanged.
        (["ɹ", "ɚ"], ["ɹ", "ɚ"]),
        # Symbols outside the equivalence classes must not be touched —
        # especially learner-error contrasts like ð/d.
        (["ð", "d", "ɪ", "s", "uː"], ["ð", "d", "ɪ", "s", "uː"]),
    ],
)
def test_normalize_equivalence_classes(phonemes, expected):
    """Phonetically equivalent notation variants map to one canonical symbol."""
    assert normalize(phonemes) == expected


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        # Stress stripping and equivalence mapping must compose: ˈr → r → ɹ.
        (["ˈr"], ["ɹ"]),
        (["ˌɜː"], ["ɚ"]),
        (["ˈr", "ˌɜː", "ð"], ["ɹ", "ɚ", "ð"]),
    ],
)
def test_normalize_combines_stress_and_equivalence(phonemes, expected):
    """Equivalence mapping applies after stress marks are stripped."""
    assert normalize(phonemes) == expected


@requires_espeak
def test_to_phonemes_keeps_diphthong_as_one_token():
    """Diphthongs must stay single tokens, matching wav2vec2-espeak output."""
    assert to_phonemes("high") == ["h", "aɪ"]


@requires_espeak
def test_to_phonemes_keeps_affricate_as_one_token():
    assert to_phonemes("church") == ["tʃ", "ɜː", "tʃ"]


@requires_espeak
def test_to_phonemes_flattens_multiple_words():
    assert to_phonemes("this high") == ["ð", "ɪ", "s", "h", "aɪ"]


@requires_espeak
def test_to_phonemes_by_word_returns_per_word_phonemes():
    """Word boundaries must be preserved for error-to-word attribution."""
    assert to_phonemes_by_word("this high water") == [
        ("this", ["ð", "ɪ", "s"]),
        ("high", ["h", "aɪ"]),
        ("water", ["w", "ɔː", "ɾ", "ɚ"]),
    ]


@requires_espeak
def test_to_phonemes_composes_with_normalize():
    """normalize must keep the length mark ː and leave default output intact."""
    phonemes = to_phonemes("water")
    assert normalize(phonemes) == phonemes
    assert "ɔː" in phonemes
