import json
from pathlib import Path

import pytest

from pronunciation_coach.pipeline import Pipeline
from pronunciation_coach.types import DiagnosisReport, PhonemeError
from ui.runner import TrialState, _default_audio_duration_seconds, _default_is_silent, run_trial

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

PHONEMES_BY_WORD = {
    "this": ["ð", "ɪ", "s"],
    "is": ["ɪ", "z"],
    "high": ["h", "aɪ"],
}


def fake_g2p_by_word(text: str) -> list[tuple[str, list[str]]]:
    words = text.lower().split()
    for w in words:
        if w == "unphonemizable":
            raise ValueError(f"Word count mismatch for {text!r}")
    return [(w, PHONEMES_BY_WORD[w]) for w in words]


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio_path: Path) -> str:
        return self.text


class FakeRecognizer:
    def __init__(self, phonemes: list[str]) -> None:
        self.phonemes = phonemes

    def recognize(self, audio_path: Path) -> list[str]:
        return self.phonemes


class FakeExplainer:
    def __init__(self, text: str = "FAKE EXPLANATION", raises: Exception | None = None) -> None:
        self.text = text
        self.raises = raises

    def explain(self, report: DiagnosisReport) -> str:
        if self.raises is not None:
            raise self.raises
        return self.text


def make_pipeline(transcript: str, hyp_phonemes: list[str], explainer=None) -> Pipeline:
    return Pipeline(
        transcriber=FakeTranscriber(transcript),
        phoneme_recognizer=FakeRecognizer(hyp_phonemes),
        explainer=explainer or FakeExplainer(),
        g2p_by_word=fake_g2p_by_word,
    )


AUDIO = Path("dummy.wav")


def run_and_collect(pipeline, tmp_path, **overrides) -> list[TrialState]:
    kwargs = dict(
        pipeline=pipeline,
        audio_path=AUDIO,
        participant_id="P01",
        learner_l1="Japanese",
        target_text="this",
        target_source="preset",
        app_session_id="session-abc",
        trial_logs_dir=tmp_path,
        config={"ollama_model": "llama3.1:8b"},
        audio_duration_seconds=lambda p: 2.0,
        is_silent=lambda p: False,
    )
    kwargs.update(overrides)
    return list(run_trial(**kwargs))


def read_log(tmp_path: Path) -> list[dict]:
    path = tmp_path / "trial_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- pre-checks ---


def test_empty_participant_id_is_precheck_failed_and_logged(tmp_path):
    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    states = run_and_collect(pipeline, tmp_path, participant_id="   ")

    assert len(states) == 1
    assert states[0].stage == "precheck_failed"
    assert "participant ID" in states[0].status_message
    assert states[0].logged is True
    logs = read_log(tmp_path)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "precheck_failed"
    assert logs[0]["app_session_id"] == "session-abc"


def test_missing_audio_is_precheck_failed(tmp_path):
    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    states = run_and_collect(pipeline, tmp_path, audio_path=None)

    assert len(states) == 1
    assert states[0].stage == "precheck_failed"
    assert "record audio first" in states[0].status_message
    # No audio to copy, so no audio_file recorded.
    assert read_log(tmp_path)[0]["audio_file"] is None


def test_recording_too_short_is_precheck_failed(tmp_path):
    audio_source = tmp_path / "recording.wav"
    audio_source.write_bytes(b"fake")
    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    states = run_and_collect(
        pipeline, tmp_path, audio_path=audio_source, audio_duration_seconds=lambda p: 0.1
    )

    assert states[-1].stage == "precheck_failed"
    assert "too short" in states[-1].status_message
    # Audio is still copied even though the trial doesn't proceed.
    assert read_log(tmp_path)[0]["audio_file"] is not None


def test_silent_recording_is_precheck_failed(tmp_path):
    audio_source = tmp_path / "recording.wav"
    audio_source.write_bytes(b"fake")
    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    states = run_and_collect(pipeline, tmp_path, audio_path=audio_source, is_silent=lambda p: True)

    assert states[-1].stage == "precheck_failed"
    assert "No speech detected" in states[-1].status_message


def test_unphonemizable_custom_text_is_precheck_failed(tmp_path):
    # Transcript must match the target closely enough to pass the WER gate,
    # so the failure actually comes from g2p on the target text, not the gate.
    pipeline = make_pipeline("unphonemizable", ["ð", "ɪ", "s"])
    states = run_and_collect(
        pipeline, tmp_path, target_text="unphonemizable", target_source="custom"
    )

    assert states[-1].stage == "precheck_failed"
    assert "Could not analyze" in states[-1].status_message


# --- WER gate ---


def test_reading_mismatch_stage(tmp_path):
    pipeline = make_pipeline("completely different words", ["ð"])
    states = run_and_collect(pipeline, tmp_path, target_text="this is high")

    stages = [s.stage for s in states]
    assert stages[-1] == "reading_mismatch"
    assert states[-1].reading_mismatch is not None
    assert not states[-1].reading_mismatch.validation.passed
    logs = read_log(tmp_path)
    assert logs[-1]["outcome"] == "reading_mismatch"


# --- success path (staged) ---


def test_successful_trial_yields_detected_then_done(tmp_path):
    audio_source = tmp_path / "recording.wav"
    audio_source.write_bytes(b"fake")
    pipeline = make_pipeline("this", ["d", "ɪ", "s"])  # ð->d substitution

    states = run_and_collect(pipeline, tmp_path, audio_path=audio_source)

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

    logs = read_log(tmp_path)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "ok"
    assert logs[0]["explanation"] == "FAKE EXPLANATION"
    assert logs[0]["report"]["errors"][0]["op"] == "substitution"
    assert logs[0]["audio_file"] is not None
    assert (tmp_path / logs[0]["audio_file"]).exists()
    assert logs[0]["target_source"] == "preset"
    assert logs[0]["config"] == {"ollama_model": "llama3.1:8b"}
    assert "diagnose" in logs[0]["timings_sec"]
    assert "explain" in logs[0]["timings_sec"]
    assert "total" in logs[0]["timings_sec"]


def test_perfect_pronunciation_has_no_errors(tmp_path):
    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    states = run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "done"
    assert states[1].diagnosis.report.errors == []


# --- explainer failures ---


def test_explainer_connection_error_is_explain_failed_but_keeps_diagnosis(tmp_path):
    explainer = FakeExplainer(raises=ConnectionError("cannot connect to Ollama"))
    pipeline = make_pipeline("this", ["d", "ɪ", "s"], explainer=explainer)

    states = run_and_collect(pipeline, tmp_path)

    stages = [s.stage for s in states]
    assert stages == ["running", "detected", "explain_failed"]
    assert states[-1].diagnosis is not None  # detection results must stay visible
    assert states[-1].explanation is None
    assert "staff member" in states[-1].status_message

    logs = read_log(tmp_path)
    assert logs[-1]["outcome"] == "explain_failed"
    assert logs[-1]["report"] is not None  # detection was still logged
    assert logs[-1]["explanation"] is None


def test_explainer_unexpected_error_is_error_stage(tmp_path):
    explainer = FakeExplainer(raises=RuntimeError("boom"))
    pipeline = make_pipeline("this", ["d", "ɪ", "s"], explainer=explainer)

    states = run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "error"
    assert "An error occurred" in states[-1].status_message
    assert read_log(tmp_path)[-1]["outcome"] == "error"


# --- logging failure must not swallow the trial's own outcome ---


def test_log_write_failure_still_returns_final_trial_state(tmp_path, monkeypatch):
    import ui.runner as runner_module

    def boom(record, path):
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "append_trial_record", boom)

    pipeline = make_pipeline("this", ["d", "ɪ", "s"])
    states = run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "done"  # the trial itself still succeeded
    assert states[-1].explanation == "FAKE EXPLANATION"
    assert states[-1].logged is False
    assert states[-1].log_error == "disk full"


def test_audio_copy_failure_is_recorded_without_hiding_trial_outcome(
    tmp_path, monkeypatch
):
    import ui.runner as runner_module

    def fail_copy(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runner_module, "copy_trial_audio", fail_copy)

    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    states = run_and_collect(pipeline, tmp_path)

    assert states[-1].stage == "done"
    assert states[-1].logged is True
    assert states[-1].error_detail == "audio copy failed: disk full"

    logs = read_log(tmp_path)
    assert logs[-1]["outcome"] == "ok"
    assert logs[-1]["audio_file"] is None
    assert logs[-1]["error_detail"] == "audio copy failed: disk full"


# --- app_session_id / config pass-through ---


def test_app_session_id_and_config_are_recorded_verbatim(tmp_path):
    pipeline = make_pipeline("this", ["ð", "ɪ", "s"])
    run_and_collect(
        pipeline, tmp_path, app_session_id="my-session-xyz", config={"prompt_version": "v3"}
    )

    logs = read_log(tmp_path)
    assert logs[0]["app_session_id"] == "my-session-xyz"
    assert logs[0]["config"] == {"prompt_version": "v3"}


# --- default precheck implementations against real audio (regression:
# torchaudio.info() doesn't exist in this project's torchaudio/torchcodec
# setup -- caught only by the manual E2E check, docs/design_ui.md §9 step 6,
# because every test above injects a fake audio_duration_seconds/is_silent) ---


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples/ not available")
def test_default_audio_duration_seconds_reads_real_audio():
    duration = _default_audio_duration_seconds(SAMPLES_DIR / "sample2.m4a")
    assert 1.0 < duration < 30.0  # sample2.m4a is a short spoken sentence


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples/ not available")
def test_default_is_silent_is_false_for_real_speech():
    assert _default_is_silent(SAMPLES_DIR / "sample2.m4a") is False
