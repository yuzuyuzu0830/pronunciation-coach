"""Pipeline orchestrator: validation gate, alignment, error extraction, explanation.

Depends only on the module Protocols so concrete models can be swapped for
comparison experiments.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

from pronunciation_coach.aligner import align_phonemes
from pronunciation_coach.g2p import normalize, to_phonemes_by_word
from pronunciation_coach.types import (
    AlignmentOp,
    CoachingResult,
    Diagnosis,
    DiagnoseResult,
    DiagnosisReport,
    PhonemeError,
    PipelineResult,
    ReadingMismatch,
)
from pronunciation_coach.validator import (
    DEFAULT_WER_THRESHOLD,
    normalize_word,
    validate_reading,
)


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class PhonemeRecognizer(Protocol):
    def recognize(self, audio_path: Path) -> list[str]: ...


class Explainer(Protocol):
    def explain(self, report: DiagnosisReport) -> str: ...


def extract_errors(
    ops: list[AlignmentOp], word_spans: list[tuple[str, list[str]]]
) -> list[PhonemeError]:
    """Convert non-match alignment ops into PhonemeErrors with word attribution.

    Positions index the reference sequence. An insertion is attributed to the
    preceding reference position's word (a leading insertion clamps to the first word).
    """
    # Expand word spans into a lookup parallel to the reference phone sequence.
    word_at: list[str] = [
        word for word, phones in word_spans for _ in phones
    ]

    def word_for(position: int) -> str | None:
        return word_at[position] if 0 <= position < len(word_at) else None

    errors: list[PhonemeError] = []
    ref_pos = 0
    # Matches, substitutions, and deletions consume a reference phone;
    # insertions stay anchored to the preceding reference position.
    for op in ops:
        if op.op == "match":
            ref_pos += 1
        elif op.op == "substitution":
            errors.append(
                PhonemeError("substitution", op.ref_phone, op.hyp_phone, ref_pos, word_for(ref_pos))
            )
            ref_pos += 1
        elif op.op == "deletion":
            errors.append(
                PhonemeError("deletion", op.ref_phone, None, ref_pos, word_for(ref_pos))
            )
            ref_pos += 1
        else:
            position = max(ref_pos - 1, 0)
            errors.append(
                PhonemeError("insertion", None, op.hyp_phone, position, word_for(position))
            )
    return errors


def flag_misread_errors(
    errors: list[PhonemeError], word_mismatches: dict[str, str | None]
) -> list[PhonemeError]:
    """Mark errors in words that failed reading validation as possible misreads.

    Such errors reflect a different word being read, not a pronunciation habit,
    so the explainer must switch to word-level guidance for them.

    Lookup uses validator.normalize_word so G2P's case-preserving error.
    word matches the lowercased (and digit-normalized) word_mismatches keys.
    """
    flagged: list[PhonemeError] = []
    for error in errors:
        key = normalize_word(error.word) if error.word is not None else ""
        if key and key in word_mismatches:
            flagged.append(
                replace(
                    error,
                    possibly_misread=True,
                    misread_as=word_mismatches[key],
                )
            )
        else:
            flagged.append(error)
    return flagged


class Pipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        phoneme_recognizer: PhonemeRecognizer,
        explainer: Explainer,
        g2p_by_word: Callable[[str], list[tuple[str, list[str]]]] = to_phonemes_by_word,
        wer_threshold: float = DEFAULT_WER_THRESHOLD,
    ) -> None:
        self._transcriber = transcriber
        self._phoneme_recognizer = phoneme_recognizer
        self._explainer = explainer
        self._g2p_by_word = g2p_by_word
        self._wer_threshold = wer_threshold

    def diagnose(
        self,
        audio_path: Path,
        target_text: str | None = None,
        learner_l1: str = "Japanese",
    ) -> DiagnoseResult:
        """Run transcription, validation, and detection without explanation.

        This boundary lets callers display or log detection before waiting for
        the explainer.
        """
        # Reject a wrong sentence before running phoneme recognition.
        transcript = self._transcriber.transcribe(audio_path)

        validation = None
        if target_text is not None:
            validation = validate_reading(target_text, transcript, self._wer_threshold)
            if not validation.passed:
                return ReadingMismatch(validation)

        # Build normalized reference and hypothesis sequences in the same
        # symbol inventory while retaining reference word boundaries.
        base_text = target_text if target_text is not None else transcript
        word_spans = [
            (word, normalize(phones)) for word, phones in self._g2p_by_word(base_text)
        ]
        reference = [phone for _, phones in word_spans for phone in phones]
        hypothesis = normalize(self._phoneme_recognizer.recognize(audio_path))

        # Convert alignment differences into word-attributed detector output.
        ops = align_phonemes(reference, hypothesis)
        errors = extract_errors(ops, word_spans)
        # Keep word-reading mistakes separate from pronunciation habits.
        if validation is not None and validation.word_mismatches:
            errors = flag_misread_errors(errors, validation.word_mismatches)

        report = DiagnosisReport(
            transcript=transcript,
            target_text=target_text,
            reference_phonemes=reference,
            hypothesis_phonemes=hypothesis,
            errors=errors,
            learner_l1=learner_l1,
        )
        return Diagnosis(report=report, validation=validation)

    def explain(self, diagnosis: Diagnosis) -> CoachingResult:
        """Generate coaching for an existing diagnosis."""
        explanation = self._explainer.explain(diagnosis.report)
        return CoachingResult(
            report=diagnosis.report, explanation=explanation, validation=diagnosis.validation
        )

    def run(
        self,
        audio_path: Path,
        target_text: str | None = None,
        learner_l1: str = "Japanese",
    ) -> PipelineResult:
        """Run diagnosis and, when validation passes, generate coaching."""
        diagnosis = self.diagnose(audio_path, target_text, learner_l1)
        if isinstance(diagnosis, ReadingMismatch):
            return diagnosis
        return self.explain(diagnosis)
