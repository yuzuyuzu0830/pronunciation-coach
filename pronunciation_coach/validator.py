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


def _normalize_words(text: str) -> list[str]:
    words = (w.strip(string.punctuation) for w in text.lower().split())
    return [w for w in words if w]


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

    return ReadingValidation(
        target_words=target_words,
        transcript_words=transcript_words,
        wer=wer,
        threshold=threshold,
        passed=wer <= threshold,
    )
