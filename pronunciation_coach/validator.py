"""Reading validation: gate phoneme evaluation on transcript/target agreement.

The threshold is deliberately lenient (default 0.5): mispronunciations
themselves distort the transcript, so this gate only rejects wrong-sentence
reads and recording failures, not pronunciation errors.
"""

from __future__ import annotations

import string

from pronunciation_coach.aligner import align_phonemes
from pronunciation_coach.types import ReadingValidation

DEFAULT_WER_THRESHOLD = 0.5

# Whisper may transcribe spoken numbers as digits, which would inflate WER.
# Trial sentences only require cardinals 0-20 and bare tens up to 100,
# so ordinals and compound numbers remain unchanged.
_NUMBER_WORDS: dict[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
    "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty",
    "70": "seventy", "80": "eighty", "90": "ninety", "100": "hundred",
}


def normalize_word(token: str) -> str:
    """Normalize a token for transcript comparison."""
    stripped = token.strip(string.punctuation).lower()
    if not stripped:
        return ""
    return _NUMBER_WORDS.get(stripped, stripped)


def _normalize_words(text: str) -> list[str]:
    return [w for tok in text.split() if (w := normalize_word(tok))]


def validate_reading(
    target_text: str,
    transcript: str,
    threshold: float = DEFAULT_WER_THRESHOLD,
) -> ReadingValidation:
    target_words = _normalize_words(target_text)
    transcript_words = _normalize_words(transcript)

    # The aligner's edit-distance DP works on any token list, including words.
    ops = align_phonemes(target_words, transcript_words)
    edit_count = sum(1 for op in ops if op.op != "match")
    wer = edit_count / max(len(target_words), 1)

    # Target words the transcript disagreed on (ref side of substitution or
    # deletion). Inserted transcript words have no target word to flag.
    word_mismatches: dict[str, str | None] = {
        op.ref_phone: op.hyp_phone
        for op in ops
        if op.op != "match" and op.ref_phone is not None
    }

    return ReadingValidation(
        target_words=target_words,
        transcript_words=transcript_words,
        wer=wer,
        threshold=threshold,
        passed=wer <= threshold,
        word_mismatches=word_mismatches,
    )
