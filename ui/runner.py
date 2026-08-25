"""One trial's execution: pre-checks -> detection -> explanation -> log.

run_trial is a generator so the caller (ui/interface.py) can update the UI
as each stage completes instead of blocking silently for ~30s on the explainer.
It is deliberately decoupled from Gradio: it yields plain
TrialState snapshots, never gr.* objects, so it's testable with fakes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal

import torchaudio

from pronunciation_coach.pipeline import Pipeline
from pronunciation_coach.types import Diagnosis, ReadingMismatch
from ui.trial_logging import (
    TargetSource,
    append_trial_record,
    build_trial_record,
    copy_trial_audio,
)

MIN_RECORDING_SECONDS = 0.5
SILENCE_RMS_THRESHOLD = 0.005

Stage = Literal[
    "running", "precheck_failed", "reading_mismatch", "detected", "done", "explain_failed", "error"
]


@dataclass(frozen=True)
class TrialState:
    stage: Stage
    status_message: str  # for the status area
    diagnosis: Diagnosis | None = None
    explanation: str | None = None
    reading_mismatch: ReadingMismatch | None = None
    error_detail: str | None = None
    logged: bool = False
    log_error: str | None = None  # set if writing the trial record itself failed


def _default_audio_duration_seconds(path: Path) -> float:
    # torchaudio.info() is unavailable with this project's torchcodec backend;
    # load() works for both browser WAV recordings and M4A samples.
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.shape[-1] / sample_rate


def _default_is_silent(path: Path, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    waveform, _ = torchaudio.load(str(path))
    return waveform.abs().mean().item() < threshold


def run_trial(
    *,
    pipeline: Pipeline,
    audio_path: Path | None,
    participant_id: str,
    learner_l1: str,
    target_text: str | None,
    target_source: TargetSource,
    app_session_id: str,
    trial_logs_dir: Path,
    config: dict,
    audio_duration_seconds: Callable[[Path], float] = _default_audio_duration_seconds,
    is_silent: Callable[[Path], bool] = _default_is_silent,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Iterator[TrialState]:
    """Run one trial, yielding a TrialState at each stage.

    The last yielded state is always terminal (stage != "running"/"detected")
    and has already been logged (or logged=False with log_error set, if
    logging itself failed -- the trial's own outcome is never hidden behind
    a logging failure).
    """
    participant_id = participant_id.strip()
    timings: dict[str, float] = {}
    total_start = time.monotonic()
    audio_copy_error: str | None = None

    def finish(
        stage: Stage,
        status_message: str,
        *,
        outcome: str,
        diagnosis: Diagnosis | None = None,
        explanation: str | None = None,
        reading_mismatch: ReadingMismatch | None = None,
        error_detail: str | None = None,
        audio_file: str | None = None,
        transcript: str | None = None,
    ) -> TrialState:
        """Persist the outcome and return its terminal UI state."""
        timings["total"] = time.monotonic() - total_start
        validation = diagnosis.validation if diagnosis else (
            reading_mismatch.validation if reading_mismatch else None
        )
        report = diagnosis.report if diagnosis else None
        details = [detail for detail in (error_detail,) if detail]
        if audio_copy_error is not None:
            details.append(f"audio copy failed: {audio_copy_error}")
        recorded_error_detail = "; ".join(details) or None

        # Keep going without a saved copy rather than losing the trial outcome
        record = build_trial_record(
            app_session_id=app_session_id,
            participant_id=participant_id,
            learner_l1=learner_l1,
            target_text=target_text,
            target_source=target_source,
            audio_file=audio_file,
            outcome=outcome,  # type: ignore[arg-type]
            transcript=transcript or (report.transcript if report else None),
            validation=validation,
            report=report,
            explanation=explanation,
            error_detail=recorded_error_detail,
            timings_sec=dict(timings),
            config=config,
            timestamp=now(),
        )

        try:
            append_trial_record(record, trial_logs_dir / "trial_log.jsonl")
            logged, log_error = True, None
        except OSError as e:
            logged, log_error = False, str(e)
        return TrialState(
            stage=stage,
            status_message=status_message,
            diagnosis=diagnosis,
            explanation=explanation,
            reading_mismatch=reading_mismatch,
            error_detail=recorded_error_detail,
            logged=logged,
            log_error=log_error,
        )

    # --- pre-checks ---
    if not participant_id:
        yield finish(
            "precheck_failed", "Please enter a participant ID",
            outcome="precheck_failed", error_detail="empty participant_id",
        )
        return

    if audio_path is None:
        yield finish(
            "precheck_failed", "Please record audio first",
            outcome="precheck_failed", error_detail="no audio",
        )
        return

    audio_file: str | None = None
    try:
        dest = copy_trial_audio(audio_path, trial_logs_dir / "audio", participant_id, timestamp=now())
        audio_file = str(dest.relative_to(trial_logs_dir))
    except OSError as e:
        # Analysis can continue from the original recording even if archival fails.
        audio_copy_error = str(e)

    try:
        duration = audio_duration_seconds(audio_path)
    except Exception as e:
        yield finish(
            "precheck_failed", "Could not read the recording. Please record again",
            outcome="precheck_failed", error_detail=str(e), audio_file=audio_file,
        )
        return
    if duration < MIN_RECORDING_SECONDS:
        yield finish(
            "precheck_failed", "Recording is too short. Please record again",
            outcome="precheck_failed", error_detail=f"duration={duration:.3f}s", audio_file=audio_file,
        )
        return
    if is_silent(audio_path):
        yield finish(
            "precheck_failed",
            "No speech detected. Check your microphone settings and try again",
            outcome="precheck_failed", error_detail="silent recording", audio_file=audio_file,
        )
        return

    # --- detection ---
    yield TrialState(stage="running", status_message="Transcribing…")
    diagnose_start = time.monotonic()
    try:
        diagnosis = pipeline.diagnose(audio_path, target_text=target_text, learner_l1=learner_l1)
    except ValueError as e:
        # e.g. Whisper/wav2vec2's own empty-output guards, or a custom
        # target_text that fails g2p.
        yield finish(
            "precheck_failed", "Could not analyze. Please check the sentence and try again",
            outcome="precheck_failed", error_detail=str(e), audio_file=audio_file,
        )
        return
    except Exception as e:
        yield finish(
            "error", "An error occurred. Please call a staff member",
            outcome="error", error_detail=str(e), audio_file=audio_file,
        )
        return
    timings["diagnose"] = time.monotonic() - diagnose_start

    if isinstance(diagnosis, ReadingMismatch):
        yield finish(
            "reading_mismatch", "Does not match the target sentence. Please read again",
            outcome="reading_mismatch", reading_mismatch=diagnosis, audio_file=audio_file,
        )
        return

    yield TrialState(stage="detected", status_message="Generating explanation…", diagnosis=diagnosis)

    # --- explanation ---
    explain_start = time.monotonic()
    try:
        coaching = pipeline.explain(diagnosis)
    except ConnectionError as e:
        yield finish(
            "explain_failed",
            "Failed to generate explanation (detection results are shown above). Please call a staff member",
            outcome="explain_failed", diagnosis=diagnosis, error_detail=str(e), audio_file=audio_file,
        )
        return
    except Exception as e:
        yield finish(
            "error", "An error occurred. Please call a staff member",
            outcome="error", diagnosis=diagnosis, error_detail=str(e), audio_file=audio_file,
        )
        return
    timings["explain"] = time.monotonic() - explain_start

    yield finish(
        "done", "Done",
        outcome="ok", diagnosis=diagnosis, explanation=coaching.explanation, audio_file=audio_file,
    )
