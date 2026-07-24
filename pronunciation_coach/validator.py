"""Reading validation: gate phoneme evaluation on transcript/target agreement.

The threshold is deliberately lenient (default 0.5): mispronunciations
themselves distort the transcript, so this gate only rejects wrong-sentence
reads and recording failures, not pronunciation errors (docs/design.md §6).
"""

from __future__ import annotations

import string

from pronunciation_coach.aligner import align_phonemes
from pronunciation_coach.types import ReadingValidation

DEFAULT_WER_THRESHOLD = 0.5

# Whisper sometimes transcribes a spoken number as digits ("three" -> "3"),
# which the un-normalized target text never does, producing a false
# word-mismatch / inflated WER for a correctly-read word (observed in UI
# E2E testing, 2026-07-24: a participant reading "three" correctly got
# flagged as a possible misread because the transcript said "3").
# Deliberately limited to plain cardinals 0-20 and the bare tens 30-100:
# ordinals ("3rd") and compounds ("twenty-three") are NOT covered -- they
# don't match these dict keys and pass through unchanged. Trial sentences
# should avoid them (docs/design_ui.md); this table isn't a general
# number-to-words converter.
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
    """Normalize one token the same way validate_reading keys word_mismatches.

    Lowercase, strip punctuation, map bare digit cardinals via _NUMBER_WORDS.
    Empty after stripping returns "" (callers that build word lists filter it).
    """
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
