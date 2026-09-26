import re

import pytest

import pronunciation_coach.g2p as g2p_module
from pronunciation_coach.g2p import (
    normalize,
    reset_espeak_backend_for_tests,
    to_phonemes,
    to_phonemes_by_word,
    to_phonemes_by_word_many,
)

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except (ImportError, OSError, RuntimeError):
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng is not installed"
)


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        pytest.param([], [], id="empty"),
        pytest.param(["ð", "ɪ", "s"], ["ð", "ɪ", "s"], id="unchanged"),
        pytest.param(["ˈaɪ", "ˌb"], ["aɪ", "b"], id="stress-marks"),
        pytest.param(["ɹˈaɪt"], ["ɹaɪt"], id="embedded-stress"),
        pytest.param(["uː", "ɑː"], ["uː", "ɑː"], id="length-marks"),
        pytest.param(["ˈuː"], ["uː"], id="stress-and-length"),
        pytest.param(["ˈ", "ə", "ˌ"], ["ə"], id="empty-elements"),
        pytest.param(["ˈ"], [], id="only-stress"),
    ],
)
def test_normalize(phonemes: list[str], expected: list[str]) -> None:
    assert normalize(phonemes) == expected


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        pytest.param(["r"], ["ɹ"], id="r"),
        pytest.param(["r", "aɪ", "t"], ["ɹ", "aɪ", "t"], id="r-in-word"),
        pytest.param(["ɜː"], ["ɚ"], id="rhotic-vowel"),
        pytest.param(["tʃ", "ɜː", "tʃ"], ["tʃ", "ɚ", "tʃ"], id="rhotic-vowel-in-word"),
        pytest.param(["ɹ", "ɚ"], ["ɹ", "ɚ"], id="canonical-symbols"),
        pytest.param(
            ["ð", "d", "ɪ", "s", "uː"],
            ["ð", "d", "ɪ", "s", "uː"],
            id="unrelated-symbols",
        ),
    ],
)
def test_normalize_equivalence_classes(
    phonemes: list[str], expected: list[str]
) -> None:
    assert normalize(phonemes) == expected


@pytest.mark.parametrize(
    "phonemes, expected",
    [
        pytest.param(["ˈr"], ["ɹ"], id="stressed-r"),
        pytest.param(["ˌɜː"], ["ɚ"], id="stressed-rhotic-vowel"),
        pytest.param(["ˈr", "ˌɜː", "ð"], ["ɹ", "ɚ", "ð"], id="mixed"),
    ],
)
def test_normalize_combines_stress_and_equivalence(
    phonemes: list[str], expected: list[str]
) -> None:
    assert normalize(phonemes) == expected


@requires_espeak
def test_to_phonemes_keeps_diphthong_as_one_token() -> None:
    assert to_phonemes("high") == ["h", "aɪ"]


@requires_espeak
def test_to_phonemes_keeps_affricate_as_one_token() -> None:
    assert to_phonemes("church") == ["tʃ", "ɜː", "tʃ"]


@requires_espeak
def test_to_phonemes_flattens_multiple_words() -> None:
    assert to_phonemes("this high") == ["ð", "ɪ", "s", "h", "aɪ"]


@requires_espeak
def test_to_phonemes_by_word_returns_per_word_phonemes() -> None:
    assert to_phonemes_by_word("this high water") == [
        ("this", ["ð", "ɪ", "s"]),
        ("high", ["h", "aɪ"]),
        ("water", ["w", "ɔː", "ɾ", "ɚ"]),
    ]


@requires_espeak
def test_to_phonemes_composes_with_normalize() -> None:
    phonemes = to_phonemes("water")
    assert normalize(phonemes) == phonemes
    assert "ɔː" in phonemes


@requires_espeak
def test_espeak_backend_is_reused_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_espeak_backend_for_tests()
    created = {"n": 0}
    real_backend = g2p_module.EspeakBackend

    def tracking_backend(language: str) -> EspeakBackend:
        created["n"] += 1
        return real_backend(language=language)

    monkeypatch.setattr(g2p_module, "EspeakBackend", tracking_backend)

    try:
        to_phonemes("high")
        to_phonemes("water")
        to_phonemes_by_word("this is high")
        assert created["n"] == 1
    finally:
        reset_espeak_backend_for_tests()


@requires_espeak
def test_to_phonemes_by_word_many_matches_single_calls() -> None:
    texts = ["this is high", "water"]
    batched = to_phonemes_by_word_many(texts)
    assert batched == [to_phonemes_by_word(t) for t in texts]


@requires_espeak
def test_to_phonemes_by_word_many_returns_value_error_per_bad_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_raw(texts: list[str]) -> list[str]:
        if texts == ["this water", "water"]:
            return ["w ɔː ɾ ɚ", "w ɔː ɾ ɚ"]
        if texts == ["this", "water"]:
            return ["x|y", "w ɔː ɾ ɚ"]
        raise AssertionError(f"unexpected phonemize call: {texts}")

    monkeypatch.setattr(g2p_module, "_phonemize_raw", fake_raw)
    results = to_phonemes_by_word_many(["this water", "water"])
    assert isinstance(results[0], ValueError)
    assert results[1] == [("water", ["w", "ɔː", "ɾ", "ɚ"])]


# Word-boundary fallback


def test_to_phonemes_by_word_falls_back_when_separator_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_raw(texts: list[str]) -> list[str]:
        if texts == ["did not"]:
            return ["d ɪ d n ɑː t"]
        if texts == ["did", "not"]:
            return ["d ɪ d", "n ɑː t"]
        raise AssertionError(f"unexpected phonemize call: {texts}")

    monkeypatch.setattr(g2p_module, "_phonemize_raw", fake_raw)
    assert to_phonemes_by_word("did not") == [
        ("did", ["d", "ɪ", "d"]),
        ("not", ["n", "ɑː", "t"]),
    ]


def test_to_phonemes_by_word_raises_when_fallback_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_raw(texts: list[str]) -> list[str]:
        if texts == ["a b"]:
            return ["x"]
        if texts == ["a", "b"]:
            return ["x|y", "z"]
        raise AssertionError(f"unexpected phonemize call: {texts}")

    monkeypatch.setattr(g2p_module, "_phonemize_raw", fake_raw)
    with pytest.raises(ValueError, match="'a'"):
        to_phonemes_by_word("a b")


@requires_espeak
@pytest.mark.parametrize(
    "text",
    [
        "did not",
        "there was",
        "mark is not a farmer",
        "where was the knife",
    ],
)
def test_to_phonemes_by_word_recovers_real_word_boundary_loss(text: str) -> None:
    result = to_phonemes_by_word(text)
    assert [word for word, _ in result] == text.split()
    assert all(phones for _, phones in result)


@requires_espeak
@pytest.mark.parametrize(
    "word",
    [
        "blicket",
        "sprindle",
        "wug",
        "Yuki",
        "Tokyo",
        "NASA",
        "FBI",
    ],
)
def test_to_phonemes_by_word_resolves_out_of_vocabulary_words(word: str) -> None:
    assert to_phonemes_by_word(word) == [(word, to_phonemes(word))]
    assert to_phonemes(word)


@requires_espeak
@pytest.mark.parametrize(
    "text, bad_token",
    [
        ("I paid 250 dollars", "250"),
        ("It was 1999", "1999"),
        ("Meet me at 3:30", "3:30"),
        ("e.g. this", "e.g"),
        ("same.i something", "same.i"),
    ],
)
def test_to_phonemes_by_word_rejects_multi_word_expansions(
    text: str, bad_token: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(repr(bad_token))):
        to_phonemes_by_word(text)


@requires_espeak
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Total group counts can hide compensating expansion and boundary loss; "
        "remove after validating group counts per input token"
    ),
)
def test_to_phonemes_by_word_rejects_compensating_group_count_errors() -> None:
    with pytest.raises(ValueError, match=re.escape(repr("e.g"))):
        to_phonemes_by_word("e.g. this one")
