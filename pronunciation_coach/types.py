"""Shared data contracts for the pronunciation coaching pipeline.

See docs/design.md §3. Detection stays on the acoustic-model side; the LLM
receives only these structured results and never performs detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class AlignmentOp:
    op: Literal["match", "substitution", "deletion", "insertion"]
    ref_phone: str | None  # None for insertion
    hyp_phone: str | None  # None for deletion


@dataclass(frozen=True)
class PhonemeError:
    op: Literal["substitution", "deletion", "insertion"]
    expected: str | None  # None for insertion
    actual: str | None  # None for deletion
    position: int  # index into the reference phonemes (insertion: preceding index)
    word: str | None  # word the error belongs to, when attributable
    # Set when reading validation saw a different word here: the learner likely
    # read another word, so phoneme-level coaching would mislead (design.md §6).
    possibly_misread: bool = False
    misread_as: str | None = None  # transcript word actually read (None: omitted)


@dataclass(frozen=True)
class DiagnosisReport:
    transcript: str
    target_text: str | None
    reference_phonemes: list[str]
    hypothesis_phonemes: list[str]
    errors: list[PhonemeError]
    learner_l1: str


@dataclass(frozen=True)
class ReadingValidation:
    target_words: list[str]
    transcript_words: list[str]
    wer: float
    threshold: float
    passed: bool
    # Target words the transcript disagreed on, mapped to the word actually
    # read (None: omitted). Keyed by word string, so a repeated target word is
    # flagged at every occurrence — accepted while targets are single sentences.
    word_mismatches: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadingMismatch:
    """Expected pipeline outcome: the learner should re-read the target text.

    Not an error — modelled as a value so callers must branch on the result
    type before accessing coaching output.
    """

    validation: ReadingValidation


@dataclass(frozen=True)
class CoachingResult:
    report: DiagnosisReport
    explanation: str
    validation: ReadingValidation | None  # present only when target_text was given


# Pipeline outcome: coaching feedback, or a re-read request after validation.
PipelineResult = CoachingResult | ReadingMismatch
