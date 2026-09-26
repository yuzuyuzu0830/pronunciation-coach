import pytest

from pronunciation_coach.g2p import normalize, to_phonemes_by_word
from ui.phoneme_hints import PHONEME_HINTS, locate_grapheme
from ui.sentences import TRIAL_SENTENCES

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except (ImportError, OSError, RuntimeError):
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng is not installed"
)


@requires_espeak
def test_all_trial_sentence_phonemes_have_a_hint() -> None:
    missing: set[str] = set()
    for sentence in TRIAL_SENTENCES:
        phonemes = normalize(
            [p for _, phones in to_phonemes_by_word(sentence.text) for p in phones]
        )
        missing |= {p for p in phonemes if p not in PHONEME_HINTS}
    assert not missing, f"phonemes with no hint: {sorted(missing)}"


def test_phoneme_hints_have_non_empty_graphemes() -> None:
    for symbol, hint in PHONEME_HINTS.items():
        assert hint.grapheme.strip(), f"{symbol!r} has an empty grapheme"


def test_locate_grapheme_basic_consonant_digraph() -> None:
    word_phonemes = ["θ", "æ", "ŋ", "k"]
    assert locate_grapheme("Thank", "θ", 0, word_phonemes) == (0, 2)


def test_locate_grapheme_basic_single_letter_vowel() -> None:
    word_phonemes = ["θ", "æ", "ŋ", "k"]
    assert locate_grapheme("Thank", "æ", 1, word_phonemes) == (2, 3)


def test_locate_grapheme_ng_digraph() -> None:
    word_phonemes = ["k", "ɪ", "ŋ"]
    assert locate_grapheme("king", "ŋ", 2, word_phonemes) == (2, 4)


def test_locate_grapheme_ee_digraph() -> None:
    word_phonemes = ["s", "iː"]
    assert locate_grapheme("see", "iː", 1, word_phonemes) == (1, 3)


def test_locate_grapheme_disambiguates_repeated_same_phoneme_by_order() -> None:
    word = "banana"
    word_phonemes = ["b", "ə", "n", "æ", "n", "ə"]
    first_n = locate_grapheme(word, "n", 2, word_phonemes)
    second_n = locate_grapheme(word, "n", 4, word_phonemes)
    assert first_n == (2, 3)
    assert second_n == (4, 5)
    assert first_n != second_n


def test_locate_grapheme_returns_none_for_irregular_spelling() -> None:
    word_phonemes = ["ð", "oʊ"]
    assert locate_grapheme("though", "oʊ", 1, word_phonemes) is None


def test_locate_grapheme_returns_none_when_phoneme_has_no_hint() -> None:
    assert locate_grapheme("x", "ʔ", 0, ["ʔ"]) is None


def test_locate_grapheme_returns_none_for_index_out_of_range() -> None:
    assert locate_grapheme("thank", "θ", 5, ["θ", "æ", "ŋ", "k"]) is None
    assert locate_grapheme("thank", "θ", -1, ["θ", "æ", "ŋ", "k"]) is None


def test_locate_grapheme_returns_none_when_index_does_not_match_phoneme() -> None:
    assert locate_grapheme("thank", "æ", 0, ["θ", "æ", "ŋ", "k"]) is None


@requires_espeak
def test_locate_grapheme_never_raises_for_any_trial_word_and_phoneme() -> None:
    for sentence in TRIAL_SENTENCES:
        for word, phones in to_phonemes_by_word(sentence.text):
            word_phonemes = normalize(phones)
            for i, phoneme in enumerate(word_phonemes):
                locate_grapheme(word, phoneme, i, word_phonemes)
