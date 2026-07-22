import pytest

from pronunciation_coach.g2p import (
    normalize,
    reset_espeak_backend_for_tests,
    to_phonemes,
    to_phonemes_by_word,
    to_phonemes_by_word_many,
)
import pronunciation_coach.g2p as g2p_module

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


@requires_espeak
def test_espeak_backend_is_reused_across_calls(monkeypatch):
    """phonemizer docs: avoid re-initializing espeak on every phonemize call."""
    reset_espeak_backend_for_tests()
    created = {"n": 0}
    real_backend = g2p_module.EspeakBackend

    class TrackingBackend(real_backend):
        def __init__(self, *args, **kwargs):
            created["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(g2p_module, "EspeakBackend", TrackingBackend)

    to_phonemes("high")
    to_phonemes("water")
    to_phonemes_by_word("this is high")
    assert created["n"] == 1


@requires_espeak
def test_to_phonemes_by_word_many_matches_single_calls():
    texts = ["this is high", "water"]
    batched = to_phonemes_by_word_many(texts)
    assert batched == [to_phonemes_by_word(t) for t in texts]


@requires_espeak
def test_to_phonemes_by_word_many_returns_value_error_per_bad_text(monkeypatch):
    """One text whose fallback also fails must not prevent pairing the others."""

    def fake_raw(texts: list[str]):
        if texts == ["this water", "water"]:
            # Batched primary call: "this water" merges into one group
            # (mismatch, triggers the per-word fallback); "water" is fine.
            return ["w ɔː ɾ ɚ", "w ɔː ɾ ɚ"]
        if texts == ["this", "water"]:
            # Fallback for "this water": "this" itself yields two groups
            # (unresolvable) -- ValueError must still be scoped to that text.
            return ["x|y", "w ɔː ɾ ɚ"]
        raise AssertionError(f"unexpected phonemize call: {texts}")

    monkeypatch.setattr(g2p_module, "_phonemize_raw", fake_raw)
    results = to_phonemes_by_word_many(["this water", "water"])
    assert isinstance(results[0], ValueError)
    assert results[1] == [("water", ["w", "ɔː", "ɾ", "ɚ"])]


# --- word-boundary-loss fallback (docs/devlog.md 2026-07-22) ---


def test_to_phonemes_by_word_falls_back_when_separator_is_dropped(monkeypatch):
    """espeak-ng sometimes merges short function-word pairs into one group
    (e.g. "did not" observed as a single group instead of two); falling back
    to per-word phonemization must recover the correct word boundary."""

    def fake_raw(texts: list[str]):
        if texts == ["did not"]:
            return ["d ɪ d n ɑː t"]  # merged: one group for two words
        if texts == ["did", "not"]:
            return ["d ɪ d", "n ɑː t"]  # per-word: correct
        raise AssertionError(f"unexpected phonemize call: {texts}")

    monkeypatch.setattr(g2p_module, "_phonemize_raw", fake_raw)
    assert to_phonemes_by_word("did not") == [
        ("did", ["d", "ɪ", "d"]),
        ("not", ["n", "ɑː", "t"]),
    ]


def test_to_phonemes_by_word_raises_when_fallback_also_fails(monkeypatch):
    """ValueError is raised only when even per-word phonemization can't
    produce one group per word (not merely on the initial mismatch)."""

    def fake_raw(texts: list[str]):
        if texts == ["a b"]:
            return ["x"]  # 1 group for 2 words -> triggers fallback
        if texts == ["a", "b"]:
            return ["x|y", "z"]  # "a" alone still yields 2 groups
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
def test_to_phonemes_by_word_recovers_real_word_boundary_loss(text):
    """Real espeak-ng integration check: these phrases were confirmed
    (2026-07-22, against speechocean762) to merge word boundaries in a
    single batched call; the fallback must still return one entry per word."""
    result = to_phonemes_by_word(text)
    assert [word for word, _ in result] == text.split()
    assert all(phones for _, phones in result)
