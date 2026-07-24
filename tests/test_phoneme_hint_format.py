import pytest

from ui.phoneme_hint_format import format_phoneme_hint


def test_format_covered_symbol_ja():
    assert format_phoneme_hint("θ", locale="ja") == "/θ/ (think の th)"
    assert format_phoneme_hint("ð", locale="ja") == "/ð/ (this の th)"
    assert format_phoneme_hint("ŋ", locale="ja") == "/ŋ/ (king の ng)"


def test_format_defaults_to_ja_locale():
    assert format_phoneme_hint("æ") == "/æ/ (cat の a)"


def test_format_uncovered_symbol_passes_through_bare():
    assert format_phoneme_hint("ʔ", locale="ja") == "/ʔ/"


def test_format_none_renders_as_dash():
    """None is what a deletion's actual / an insertion's expected carry."""
    assert format_phoneme_hint(None, locale="ja") == "-"
    assert format_phoneme_hint(None) == "-"


def test_format_unknown_locale_raises():
    with pytest.raises(ValueError, match="Unknown locale"):
        format_phoneme_hint("θ", locale="en")
