from __future__ import annotations

import pytest

from pronunciation_coach.validator import validate_reading


def test_exact_match_passes_with_zero_wer() -> None:
    result = validate_reading("This is high water.", "this is high water")
    assert result.wer == 0.0
    assert result.passed
    assert result.target_words == ["this", "is", "high", "water"]
    assert result.transcript_words == ["this", "is", "high", "water"]


def test_case_and_punctuation_are_ignored() -> None:
    result = validate_reading("Hello, world!", "hello world")
    assert result.wer == 0.0
    assert result.passed


def test_minor_mistranscription_passes() -> None:
    result = validate_reading("the quick brown fox jumps", "the quick brown fog jumps")
    assert result.wer == pytest.approx(0.2)
    assert result.passed


def test_completely_different_sentence_fails() -> None:
    result = validate_reading(
        "the quick brown fox jumps", "good morning everyone today"
    )
    assert result.wer > 0.5
    assert not result.passed


def test_empty_transcript_fails_against_nonempty_target() -> None:
    result = validate_reading("this is high water", "")
    assert result.wer == 1.0
    assert not result.passed


def test_both_empty_passes() -> None:
    result = validate_reading("", "")
    assert result.wer == 0.0
    assert result.passed


def test_threshold_is_adjustable() -> None:
    result = validate_reading(
        "the quick brown fox jumps", "the quick brown fog jumps", threshold=0.1
    )
    assert result.threshold == 0.1
    assert not result.passed


def test_result_records_default_threshold() -> None:
    result = validate_reading("this", "this")
    assert result.threshold == 0.5


def test_word_mismatches_empty_on_exact_match() -> None:
    result = validate_reading("this is high water", "this is high water")
    assert result.word_mismatches == {}


def test_word_mismatches_records_substituted_word_with_what_was_read() -> None:
    result = validate_reading(
        "please check the feature list", "please check the future list"
    )
    assert result.word_mismatches == {"feature": "future"}
    assert result.passed


def test_word_mismatches_records_omitted_word_as_none() -> None:
    result = validate_reading("the quick brown fox", "the quick fox")
    assert result.word_mismatches == {"brown": None}


def test_word_mismatches_ignores_extra_transcript_words() -> None:
    result = validate_reading("this is high", "this is very high")
    assert result.word_mismatches == {}


def test_digit_transcript_matches_spelled_out_target() -> None:
    result = validate_reading("three thin books", "3 thin books")
    assert result.wer == 0.0
    assert result.passed
    assert result.target_words == ["three", "thin", "books"]
    assert result.transcript_words == ["three", "thin", "books"]


def test_digit_target_matches_spelled_out_transcript() -> None:
    result = validate_reading("i have 3 books", "i have three books")
    assert result.wer == 0.0
    assert result.passed


@pytest.mark.parametrize(
    "digits, word",
    [
        ("0", "zero"),
        ("7", "seven"),
        ("11", "eleven"),
        ("19", "nineteen"),
        ("20", "twenty"),
        ("30", "thirty"),
        ("90", "ninety"),
        ("100", "hundred"),
    ],
    ids=[
        "zero",
        "seven",
        "eleven",
        "nineteen",
        "twenty",
        "thirty",
        "ninety",
        "100-as-hundred",
    ],
)
def test_number_normalization_handles_representative_supported_tokens(
    digits: str, word: str
) -> None:
    result = validate_reading(word, digits)
    assert result.wer == 0.0
    assert result.passed
    assert result.target_words == [word]
    assert result.transcript_words == [word]


def test_number_normalization_does_not_cover_ordinals() -> None:
    result = validate_reading("third", "3rd")
    assert result.wer == 1.0
    assert not result.passed


def test_number_normalization_does_not_cover_compound_numbers() -> None:
    result = validate_reading("twenty three", "23")
    assert result.wer == 1.0


def test_word_mismatches_empty_for_digit_transcription() -> None:
    result = validate_reading("three thin books", "3 thin books")
    assert result.word_mismatches == {}
