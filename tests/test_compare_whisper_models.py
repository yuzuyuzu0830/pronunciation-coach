"""Unit tests for scripts/compare_whisper_models.py helpers.

No real Whisper model is loaded
and format_report so a single bad recording cannot regress into aborting
the whole multi-model run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compare_whisper_models import evaluate_case, format_report  # noqa: E402


class _StubTranscriber:
    def __init__(self, behavior):
        self._behavior = behavior

    def transcribe(self, audio_path: Path) -> str:
        return self._behavior(audio_path)


def _case(audio_file: str = "audio/x.wav", target: str = "hello world") -> dict:
    return {
        "audio_file": audio_file,
        "outcome": "ok",
        "target_text": target,
    }


def test_evaluate_case_scores_successful_transcription(tmp_path: Path):
    audio = tmp_path / "audio" / "x.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"fake")
    transcriber = _StubTranscriber(lambda _p: "hello world")

    result = evaluate_case(transcriber, _case(), tmp_path)

    assert result["error"] is None
    assert result["transcript"] == "hello world"
    assert result["wer"] == 0.0
    assert result["gate_passed"] is True


def test_evaluate_case_records_file_not_found_without_raising(tmp_path: Path):
    transcriber = _StubTranscriber(
        lambda p: (_ for _ in ()).throw(FileNotFoundError(f"Audio file not found: {p}"))
    )

    result = evaluate_case(transcriber, _case(), tmp_path)

    assert result["transcript"] is None
    assert result["wer"] is None
    assert result["gate_passed"] is None
    assert result["error"] is not None
    assert "FileNotFoundError" in result["error"]


def test_evaluate_case_records_empty_transcript_without_raising(tmp_path: Path):
    transcriber = _StubTranscriber(
        lambda p: (_ for _ in ()).throw(
            ValueError(f"Whisper produced an empty transcript for {p}.")
        )
    )

    result = evaluate_case(transcriber, _case(), tmp_path)

    assert result["error"] is not None
    assert "ValueError" in result["error"]


def test_evaluate_case_records_model_runtime_error_without_raising(tmp_path: Path):
    transcriber = _StubTranscriber(
        lambda _p: (_ for _ in ()).throw(RuntimeError("inference failed"))
    )

    result = evaluate_case(transcriber, _case(), tmp_path)

    assert result["transcript"] is None
    assert result["error"] == "RuntimeError: inference failed"


def test_format_report_includes_error_rows_and_skips_them_in_wer():
    ok = {
        "audio_file": "audio/ok.wav",
        "logged_outcome": "ok",
        "target_text": "hello",
        "transcript": "hello",
        "wer": 0.0,
        "gate_passed": True,
        "error": None,
        "elapsed_sec": 0.1,
    }
    bad = {
        "audio_file": "audio/bad.wav",
        "logged_outcome": "ok",
        "target_text": "hello",
        "transcript": None,
        "wer": None,
        "gate_passed": None,
        "error": "FileNotFoundError: missing",
        "elapsed_sec": 0.0,
    }
    report = format_report("P01", {"base": (1.0, [ok, bad])})

    assert "ERROR: FileNotFoundError: missing" in report
    assert "1 scored, 1 failed" in report
    assert "Recordings that would still fail the WER gate: 0/1" in report
