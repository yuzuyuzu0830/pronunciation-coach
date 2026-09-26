from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NoReturn, TypedDict, cast

import pytest

from pronunciation_coach.pipeline import Pipeline
from pronunciation_coach.types import DiagnosisReport, PhonemeError
from ui.runner import (
    TrialState,
    _default_audio_duration_seconds,
    _default_is_silent,
    run_trial,
)
from ui.trial_logging import TargetSource

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

PHONEMES_BY_WORD = {
    "this": ["ð", "ɪ", "s"],
    "is": ["ɪ", "z"],
    "high": ["h", "aɪ"],
}


class _ErrorRecord(TypedDict):
    op: str


class _ReportRecord(TypedDict):
    errors: list[_ErrorRecord]


class _TrialLogRecord(TypedDict, total=False):
    app_session_id: str
    audio_file: str | None
    config: dict[str, object]
    error_detail: str | None
    explanation: str | None
    outcome: str
    report: _ReportRecord | None
    target_source: str
    timings_sec: dict[str, float]


def _fake_g2p_by_word(text: str) -> list[tuple[str, list[str]]]:
    words = text.lower().split()
    for word in words:
        if word == "unphonemizable":
            raise ValueError(f"Word count mismatch for {text!r}")
    return [(word, PHONEMES_BY_WORD[word]) for word in words]


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, _audio_path: Path) -> str:
        return self.text


class _FakeRecognizer:
    def __init__(self, phonemes: list[str]) -> None:
        self.phonemes = phonemes

    def recognize(self, _audio_path: Path) -> list[str]:
        return self.phonemes


class _FakeExplainer:
    def __init__(
        self, text: str = "FAKE EXPLANATION", raises: Exception | None = None
    ) -> None:
        self.text = text
        self.raises = raises

    def explain(self, _report: DiagnosisReport) -> str:
        if self.raises is not None:
            raise self.raises
        return self.text


def _make_pipeline(
    transcript: str,
    hyp_phonemes: list[str],
    explainer: _FakeExplainer | None = None,
) -> Pipeline:
    return Pipeline(
        transcriber=_FakeTranscriber(transcript),
        phoneme_recognizer=_FakeRecognizer(hyp_phonemes),
        explainer=explainer if explainer is not None else _FakeExplainer(),
        g2p_by_word=_fake_g2p_by_word,
    )


AUDIO = Path("dummy.wav")


def _valid_audio_duration(_path: Path) -> float:
    return 2.0


def _has_no_silence(_path: Path) -> bool:
    return False


def _run_and_collect(
    pipeline: Pipeline,
    tmp_path: Path,
    *,
    audio_path: Path | None = AUDIO,
    participant_id: str = "P01",
    target_text: str | None = "this",
    target_source: TargetSource = "preset",
    app_session_id: str = "session-abc",
    config: dict[str, str] | None = None,
    audio_duration_seconds: Callable[[Path], float] = _valid_audio_duration,
    is_silent: Callable[[Path], bool] = _has_no_silence,
) -> list[TrialState]:
    return list(
        run_trial(
            pipeline=pipeline,
            audio_path=audio_path,
            participant_id=participant_id,
            learner_l1="Japanese",
            target_text=target_text,
            target_source=target_source,
            app_session_id=app_session_id,
            trial_logs_dir=tmp_path,
            config=config if config is not None else {"ollama_model": "llama3.1:8b"},
            audio_duration_seconds=audio_duration_seconds,
            is_silent=is_silent,
        )
    )


def _read_log(tmp_path: Path) -> list[_TrialLogRecord]:
    path = tmp_path / "trial_log.jsonl"
    if not path.exists():
        return []
    records: list[_TrialLogRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        assert isinstance(value, dict), "Trial log entry must be a JSON object"
        records.append(cast(_TrialLogRecord, value))
    return records


def test_empty_participant_id_is_precheck_failed_and_logged(tmp_path: Path) -> None:
    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    states = _run_and_collect(pipeline, tmp_path, participant_id="   ")

    assert len(states) == 1
    assert states[0].stage == "precheck_failed"
    assert "participant ID" in states[0].status_message
    assert states[0].logged is True
    logs = _read_log(tmp_path)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "precheck_failed"
    assert logs[0]["app_session_id"] == "session-abc"


def test_missing_audio_is_precheck_failed(tmp_path: Path) -> None:
    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    states = _run_and_collect(pipeline, tmp_path, audio_path=None)

    assert len(states) == 1
    assert states[0].stage == "precheck_failed"
    assert "record audio first" in states[0].status_message
    assert _read_log(tmp_path)[0]["audio_file"] is None


def test_recording_too_short_is_precheck_failed(tmp_path: Path) -> None:
    audio_source = tmp_path / "recording.wav"
    audio_source.write_bytes(b"fake")
    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    states = _run_and_collect(
        pipeline,
        tmp_path,
        audio_path=audio_source,
        audio_duration_seconds=lambda _: 0.1,
    )

    assert states[-1].stage == "precheck_failed"
    assert "too short" in states[-1].status_message
    assert _read_log(tmp_path)[0]["audio_file"] is not None


def test_silent_recording_is_precheck_failed(tmp_path: Path) -> None:
    audio_source = tmp_path / "recording.wav"
    audio_source.write_bytes(b"fake")
    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    states = _run_and_collect(
        pipeline, tmp_path, audio_path=audio_source, is_silent=lambda _: True
    )

    assert states[-1].stage == "precheck_failed"
    assert "No speech detected" in states[-1].status_message


def test_unphonemizable_custom_text_is_precheck_failed(tmp_path: Path) -> None:
    pipeline = _make_pipeline("unphonemizable", ["ð", "ɪ", "s"])
    states = _run_and_collect(
        pipeline, tmp_path, target_text="unphonemizable", target_source="custom"
    )

    assert states[-1].stage == "precheck_failed"
    assert "Could not analyze" in states[-1].status_message


def test_reading_mismatch_stage(tmp_path: Path) -> None:
    pipeline = _make_pipeline("completely different words", ["ð"])
    states = _run_and_collect(pipeline, tmp_path, target_text="this is high")

    stages = [s.stage for s in states]
    assert stages[-1] == "reading_mismatch"
    assert states[-1].reading_mismatch is not None
    assert not states[-1].reading_mismatch.validation.passed
    logs = _read_log(tmp_path)
    assert logs[-1]["outcome"] == "reading_mismatch"


def test_successful_trial_yields_detected_then_done(tmp_path: Path) -> None:
    audio_source = tmp_path / "recording.wav"
    audio_source.write_bytes(b"fake")
    pipeline = _make_pipeline("this", ["d", "ɪ", "s"])

    states = _run_and_collect(pipeline, tmp_path, audio_path=audio_source)

    stages = [s.stage for s in states]
    assert stages == ["running", "detected", "done"]

    detected = states[1]
    assert detected.diagnosis is not None
    assert detected.diagnosis.report.errors == [
        PhonemeError("substitution", "ð", "d", 0, "this")
    ]

    done = states[2]
    assert done.explanation == "FAKE EXPLANATION"
    assert done.logged is True

    logs = _read_log(tmp_path)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "ok"
    assert logs[0]["explanation"] == "FAKE EXPLANATION"
    report = logs[0]["report"]
    assert report is not None
    assert report["errors"][0]["op"] == "substitution"
    assert logs[0]["audio_file"] is not None
    assert (tmp_path / logs[0]["audio_file"]).exists()
    assert logs[0]["target_source"] == "preset"
    assert logs[0]["config"] == {"ollama_model": "llama3.1:8b"}
    assert "diagnose" in logs[0]["timings_sec"]
    assert "explain" in logs[0]["timings_sec"]
    assert "total" in logs[0]["timings_sec"]


def test_perfect_pronunciation_has_no_errors(tmp_path: Path) -> None:
    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    states = _run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "done"
    diagnosis = states[1].diagnosis
    assert diagnosis is not None
    assert diagnosis.report.errors == []


def test_explainer_connection_error_is_explain_failed_but_keeps_diagnosis(
    tmp_path: Path,
) -> None:
    explainer = _FakeExplainer(raises=ConnectionError("cannot connect to Ollama"))
    pipeline = _make_pipeline("this", ["d", "ɪ", "s"], explainer=explainer)

    states = _run_and_collect(pipeline, tmp_path)

    stages = [s.stage for s in states]
    assert stages == ["running", "detected", "explain_failed"]
    assert states[-1].diagnosis is not None
    assert states[-1].explanation is None
    assert "staff member" in states[-1].status_message

    logs = _read_log(tmp_path)
    assert logs[-1]["outcome"] == "explain_failed"
    assert logs[-1]["report"] is not None
    assert logs[-1]["explanation"] is None


def test_explainer_unexpected_error_is_error_stage(tmp_path: Path) -> None:
    explainer = _FakeExplainer(raises=RuntimeError("boom"))
    pipeline = _make_pipeline("this", ["d", "ɪ", "s"], explainer=explainer)

    states = _run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "error"
    assert "An error occurred" in states[-1].status_message
    assert _read_log(tmp_path)[-1]["outcome"] == "error"


def test_log_write_failure_still_returns_final_trial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ui.runner as runner_module

    def fail_append(_record: dict[str, object], _path: Path) -> NoReturn:
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "append_trial_record", fail_append)

    pipeline = _make_pipeline("this", ["d", "ɪ", "s"])
    states = _run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "done"
    assert states[-1].explanation == "FAKE EXPLANATION"
    assert states[-1].logged is False
    assert states[-1].log_error == "disk full"


def test_audio_copy_failure_is_recorded_without_hiding_trial_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ui.runner as runner_module

    def fail_copy(
        source_path: Path,
        audio_dir: Path,
        participant_id: str,
        timestamp: datetime | None = None,
    ) -> NoReturn:
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "copy_trial_audio", fail_copy)

    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    states = _run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "done"
    assert states[-1].logged is True
    assert states[-1].error_detail == "audio copy failed: disk full"

    logs = _read_log(tmp_path)
    assert logs[-1]["outcome"] == "ok"
    assert logs[-1]["audio_file"] is None
    assert logs[-1]["error_detail"] == "audio copy failed: disk full"


def test_app_session_id_and_config_are_recorded_verbatim(tmp_path: Path) -> None:
    pipeline = _make_pipeline("this", ["ð", "ɪ", "s"])
    _run_and_collect(
        pipeline,
        tmp_path,
        app_session_id="my-session-xyz",
        config={"prompt_version": "v3"},
    )

    logs = _read_log(tmp_path)
    assert logs[0]["app_session_id"] == "my-session-xyz"
    assert logs[0]["config"] == {"prompt_version": "v3"}


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples/ not available")
def test_default_audio_duration_seconds_reads_real_audio() -> None:
    duration = _default_audio_duration_seconds(SAMPLES_DIR / "sample2.m4a")
    assert 1.0 < duration < 30.0


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples/ not available")
def test_default_is_silent_is_false_for_real_speech() -> None:
    assert _default_is_silent(SAMPLES_DIR / "sample2.m4a") is False
