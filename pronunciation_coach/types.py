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
class Diagnosis:
    """Pipeline.diagnose()'s success case: detection without an explanation yet.

    validation travels alongside the report (not embedded in it) so callers
    that only need detection -- e.g. the UI's staged display, or trial
    logging that records WER even on a passing read -- don't have to go
    through the explainer to see it (docs/design_ui.md §3).
    """

    report: DiagnosisReport
    validation: ReadingValidation | None  # present only when target_text was given


# Pipeline.diagnose() outcome: a diagnosis ready for explanation, or a
# re-read request after validation.
DiagnoseResult = Diagnosis | ReadingMismatch


@dataclass(frozen=True)
class CoachingResult:
    report: DiagnosisReport
    explanation: str
    validation: ReadingValidation | None  # present only when target_text was given


# Pipeline outcome: coaching feedback, or a re-read request after validation.
PipelineResult = CoachingResult | ReadingMismatch


@dataclass(frozen=True)
class PracticeWord:
    word: str
    target_phoneme: str  # verified by test against g2p.to_phonemes (docs/design_3c.md §1)


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    phenomenon: str
    cause: str | None  # None for phoneme_fallback tier: no L1-transfer claim made
    articulation_tip: str
    practice_words: list[PracticeWord]
    citation: str  # "" means not yet recorded
    tier: Literal["l1_specific", "phoneme_fallback"]
