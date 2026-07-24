import pytest

from pronunciation_coach.validator import validate_reading


def test_exact_match_passes_with_zero_wer():
    result = validate_reading("This is high water.", "this is high water")
    assert result.wer == 0.0
    assert result.passed
    assert result.target_words == ["this", "is", "high", "water"]
    assert result.transcript_words == ["this", "is", "high", "water"]


def test_case_and_punctuation_are_ignored():
    result = validate_reading("Hello, world!", "hello world")
    assert result.wer == 0.0
    assert result.passed


def test_minor_mistranscription_passes():
    """One wrong word out of five (WER 0.2) stays under the 0.5 threshold.

    Mispronunciations distort the transcript; the gate must not reject them.
    """
    result = validate_reading(
        "the quick brown fox jumps", "the quick brown fog jumps"
    )
    assert result.wer == pytest.approx(0.2)
    assert result.passed


def test_completely_different_sentence_fails():
    result = validate_reading(
        "the quick brown fox jumps", "good morning everyone today"
    )
    assert result.wer > 0.5
    assert not result.passed


def test_empty_transcript_fails_against_nonempty_target():
    result = validate_reading("this is high water", "")
    assert result.wer == 1.0
    assert not result.passed


def test_both_empty_passes():
    result = validate_reading("", "")
    assert result.wer == 0.0
    assert result.passed


def test_threshold_is_adjustable():
    result = validate_reading(
        "the quick brown fox jumps", "the quick brown fog jumps", threshold=0.1
    )
    assert result.threshold == 0.1
    assert not result.passed


def test_result_records_default_threshold():
    result = validate_reading("this", "this")
    assert result.threshold == 0.5


# --- word_mismatches: which target words disagreed with the transcript ---


def test_word_mismatches_empty_on_exact_match():
    result = validate_reading("this is high water", "this is high water")
    assert result.word_mismatches == {}


def test_word_mismatches_records_substituted_word_with_what_was_read():
    """A misread word maps to the transcript word actually heard."""
    result = validate_reading(
        "please check the feature list", "please check the future list"
    )
    assert result.word_mismatches == {"feature": "future"}
    assert result.passed  # under threshold: mismatch info must survive passing


def test_word_mismatches_records_omitted_word_as_none():
    result = validate_reading("the quick brown fox", "the quick fox")
    assert result.word_mismatches == {"brown": None}


def test_word_mismatches_ignores_extra_transcript_words():
    """An inserted word has no target-side word to flag."""
    result = validate_reading("this is high", "this is very high")
    assert result.word_mismatches == {}


# --- number normalization (docs/devlog.md 2026-07-24: Whisper transcribing
# a spoken number as digits caused a false word-mismatch / misread flag) ---


def test_digit_transcript_matches_spelled_out_target():
    result = validate_reading("three thin books", "3 thin books")
    assert result.wer == 0.0
    assert result.passed
    assert result.target_words == ["three", "thin", "books"]
    assert result.transcript_words == ["three", "thin", "books"]


def test_digit_target_matches_spelled_out_transcript():
    """Normalization must apply to both sides, not just the transcript."""
    result = validate_reading("i have 3 books", "i have three books")
    assert result.wer == 0.0
    assert result.passed


def test_number_normalization_covers_zero_through_twenty_and_bare_tens():
    for digits, word in [
        ("0", "zero"), ("7", "seven"), ("11", "eleven"), ("19", "nineteen"),
        ("20", "twenty"), ("30", "thirty"), ("90", "ninety"), ("100", "hundred"),
    ]:
        result = validate_reading(word, digits)
        assert result.wer == 0.0, f"{digits!r} should normalize to {word!r}"


def test_number_normalization_does_not_cover_ordinals():
    """"3rd" is out of scope by design; it must not silently become "three"."""
    result = validate_reading("third", "3rd")
    assert result.wer == 1.0
    assert not result.passed


def test_number_normalization_does_not_cover_compound_numbers():
    """"23" is out of scope (only 0-20 and the bare tens are covered)."""
    result = validate_reading("twenty three", "23")
    assert result.wer == 1.0


def test_word_mismatches_no_longer_false_flags_digit_transcription():
    """Reproduces the exact reported bug: a participant reads "three"
    correctly, Whisper transcribes "3" -- this must not be recorded as a
    possible misread (docs/devlog.md 2026-07-24)."""
    result = validate_reading("three thin books", "3 thin books")
    assert result.word_mismatches == {}
