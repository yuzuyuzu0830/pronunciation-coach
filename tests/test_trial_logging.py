from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pronunciation_coach.types import DiagnosisReport, PhonemeError, ReadingValidation
from ui.trial_logging import (
    append_trial_record,
    build_trial_record,
    copy_trial_audio,
    sanitize_participant_id,
)

FIXED_TS = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


def _make_report() -> DiagnosisReport:
    return DiagnosisReport(
        transcript="this is high",
        target_text="this is high",
        reference_phonemes=["ð", "ɪ", "s", "ɪ", "z", "h", "aɪ"],
        hypothesis_phonemes=["d", "ɪ", "s", "ɪ", "z", "h", "aɪ"],
        errors=[PhonemeError("substitution", "ð", "d", 0, "this")],
        learner_l1="Japanese",
    )


def _make_validation(passed: bool = True) -> ReadingValidation:
    return ReadingValidation(
        target_words=["this", "is", "high"],
        transcript_words=["this", "is", "high"],
        wer=0.0,
        threshold=0.5,
        passed=passed,
    )


def test_build_trial_record_ok_outcome_includes_all_fields() -> None:
    record = build_trial_record(
        app_session_id="session-123",
        participant_id="P01",
        learner_l1="Japanese",
        target_text="this is high",
        target_source="preset",
        audio_file="audio/20260724T120000Z_P01.wav",
        outcome="ok",
        transcript="this is high",
        validation=_make_validation(),
        report=_make_report(),
        explanation="Great job overall...",
        timings_sec={
            "transcribe": 1.2,
            "recognize": 0.3,
            "explain": 28.9,
            "total": 30.4,
        },
        config={"ollama_model": "llama3.1:8b", "prompt_version": "v3"},
        timestamp=FIXED_TS,
    )

    assert record["timestamp_utc"] == "2026-07-24T12:00:00+00:00"
    assert record["app_session_id"] == "session-123"
    assert record["participant_id"] == "P01"
    assert record["learner_l1"] == "Japanese"
    assert record["target_text"] == "this is high"
    assert record["target_source"] == "preset"
    assert record["audio_file"] == "audio/20260724T120000Z_P01.wav"
    assert record["outcome"] == "ok"
    assert record["transcript"] == "this is high"
    assert record["validation"] == {
        "target_words": ["this", "is", "high"],
        "transcript_words": ["this", "is", "high"],
        "wer": 0.0,
        "threshold": 0.5,
        "passed": True,
        "word_mismatches": {},
    }
    assert record["report"]["transcript"] == "this is high"
    assert record["report"]["errors"][0]["op"] == "substitution"
    assert record["explanation"] == "Great job overall..."
    assert record["error_detail"] is None
    assert record["timings_sec"]["explain"] == 28.9
    assert record["config"] == {"ollama_model": "llama3.1:8b", "prompt_version": "v3"}


def test_build_trial_record_is_json_serializable() -> None:
    record = build_trial_record(
        app_session_id="session-123",
        participant_id="P01",
        learner_l1="Japanese",
        target_text="this is high",
        target_source="preset",
        audio_file="audio/x.wav",
        outcome="ok",
        report=_make_report(),
        validation=_make_validation(),
        timestamp=FIXED_TS,
    )
    json.dumps(record, ensure_ascii=False)


def test_build_trial_record_defaults_missing_optional_fields_to_none_or_empty() -> None:
    record = build_trial_record(
        app_session_id="session-123",
        participant_id="P02",
        learner_l1="Japanese",
        target_text=None,
        target_source="custom",
        audio_file=None,
        outcome="precheck_failed",
        error_detail="recording too short",
        timestamp=FIXED_TS,
    )
    assert record["transcript"] is None
    assert record["validation"] is None
    assert record["report"] is None
    assert record["explanation"] is None
    assert record["timings_sec"] == {}
    assert record["config"] == {}
    assert record["error_detail"] == "recording too short"


def test_build_trial_record_uses_current_time_when_timestamp_omitted() -> None:
    before = datetime.now(timezone.utc)
    record = build_trial_record(
        app_session_id="s",
        participant_id="P01",
        learner_l1="Japanese",
        target_text=None,
        target_source="custom",
        audio_file=None,
        outcome="error",
    )
    after = datetime.now(timezone.utc)
    ts = datetime.fromisoformat(record["timestamp_utc"])
    assert before <= ts <= after


def test_append_trial_record_writes_one_json_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "trial_log.jsonl"
    record1 = {"outcome": "ok", "participant_id": "P01"}
    record2 = {"outcome": "error", "participant_id": "P02"}

    append_trial_record(record1, path)
    append_trial_record(record2, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == record1
    assert json.loads(lines[1]) == record2


def test_append_trial_record_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "trial_log.jsonl"
    append_trial_record({"outcome": "ok"}, path)
    assert path.exists()


def test_copy_trial_audio_copies_content_to_named_destination(tmp_path: Path) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"fake-audio-bytes")
    audio_dir = tmp_path / "audio"

    dest = copy_trial_audio(source, audio_dir, "P01", timestamp=FIXED_TS)

    assert dest == audio_dir / "20260724T120000Z_P01.wav"
    assert dest.read_bytes() == b"fake-audio-bytes"
    assert source.exists()


def test_copy_trial_audio_creates_audio_dir(tmp_path: Path) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    audio_dir = tmp_path / "nested" / "audio"

    dest = copy_trial_audio(source, audio_dir, "P01", timestamp=FIXED_TS)
    assert dest.exists()


def test_copy_trial_audio_sanitizes_unsafe_participant_id(tmp_path: Path) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"x")
    audio_dir = tmp_path / "audio"

    dest = copy_trial_audio(source, audio_dir, "P01/../etc", timestamp=FIXED_TS)
    assert dest.name == "20260724T120000Z_P01____etc.wav"
    assert dest.parent == audio_dir


def test_copy_trial_audio_preserves_source_suffix(tmp_path: Path) -> None:
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"x")
    dest = copy_trial_audio(source, tmp_path / "audio", "P01", timestamp=FIXED_TS)
    assert dest.suffix == ".mp3"


def test_sanitize_participant_id_keeps_alnum_dash_underscore() -> None:
    assert sanitize_participant_id("P01") == "P01"
    assert sanitize_participant_id("P-01_a") == "P-01_a"


def test_sanitize_participant_id_replaces_unsafe_characters() -> None:
    assert sanitize_participant_id("P01/../etc") == "P01____etc"
    assert sanitize_participant_id("P01 (retry)") == "P01__retry_"


def test_sanitize_participant_id_empty_becomes_unknown() -> None:
    assert sanitize_participant_id("") == "unknown"
