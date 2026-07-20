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
