"""Trial session logging: one JSONL record per trial (docs/design_ui.md §5).

build_trial_record is pure (dict in/out); append_trial_record and
copy_trial_audio are the only I/O in this module, kept thin so the record
shape stays independently testable.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pronunciation_coach.types import DiagnosisReport, ReadingValidation

Outcome = Literal["ok", "reading_mismatch", "precheck_failed", "explain_failed", "error"]
TargetSource = Literal["preset", "custom"]


def build_trial_record(
    *,
    app_session_id: str,
    participant_id: str,
    learner_l1: str,
    target_text: str | None,
    target_source: TargetSource,
    audio_file: str | None,
    outcome: Outcome,
    transcript: str | None = None,
    validation: ReadingValidation | None = None,
    report: DiagnosisReport | None = None,
    explanation: str | None = None,
    error_detail: str | None = None,
    timings_sec: dict[str, float] | None = None,
    config: dict | None = None,
    timestamp: datetime | None = None,
) -> dict:
    """Assemble one trial_log.jsonl record.

    app_session_id identifies the app launch (a fresh UUID per process
    start), not the participant or the trial -- if the app is restarted
    mid-trial-day, records before and after the restart carry different
    values, so an environment-caused behavior change can be isolated to a
    specific launch during analysis.
    """
    ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
    return {
        "timestamp_utc": ts.isoformat(),
        "app_session_id": app_session_id,
        "participant_id": participant_id,
        "learner_l1": learner_l1,
        "target_text": target_text,
        "target_source": target_source,
        "audio_file": audio_file,
        "outcome": outcome,
        "transcript": transcript,
        "validation": asdict(validation) if validation is not None else None,
        "report": asdict(report) if report is not None else None,
        "explanation": explanation,
        "error_detail": error_detail,
        "timings_sec": timings_sec or {},
        "config": config or {},
    }


def append_trial_record(record: dict, jsonl_path: Path) -> None:
    """Append one record as a JSON line, flushing immediately (mirrors
    scripts/evaluate_detection.py's run_stage1: a crash loses at most the
    in-flight record, not prior ones)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def sanitize_participant_id(participant_id: str) -> str:
    """Keep only filesystem-safe characters, for use in a filename."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in participant_id)
    return safe or "unknown"


def copy_trial_audio(
    source_path: Path,
    audio_dir: Path,
    participant_id: str,
    timestamp: datetime | None = None,
) -> Path:
    """Copy the recorded audio into audio_dir before it can be cleaned up
    (Gradio's own temp file), named by UTC timestamp + participant id so
    files sort chronologically and stay attributable at a glance."""
    ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix or ".wav"
    dest_path = audio_dir / f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{sanitize_participant_id(participant_id)}{suffix}"
    shutil.copy2(source_path, dest_path)
    return dest_path
