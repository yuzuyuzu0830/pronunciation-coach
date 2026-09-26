from __future__ import annotations

import pytest

from pronunciation_coach.types import PhonemeError
from scripts import run_pipeline
from ui import config


def test_build_pipeline_uses_current_system_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    transcriber = object()
    recognizer = object()
    explainer = object()

    def make_transcriber(*, model_size: str) -> object:
        calls["whisper"] = model_size
        return transcriber

    def make_recognizer(*, model_name: str) -> object:
        calls["wav2vec2"] = model_name
        return recognizer

    def make_explainer(*, model: str, base_url: str, prompt_version: str) -> object:
        calls["ollama"] = (model, base_url, prompt_version)
        return explainer

    def make_pipeline(**kwargs: object) -> object:
        calls["pipeline"] = kwargs
        return object()

    monkeypatch.setattr(run_pipeline, "WhisperTranscriber", make_transcriber)
    monkeypatch.setattr(run_pipeline, "Wav2Vec2PhonemeRecognizer", make_recognizer)
    monkeypatch.setattr(run_pipeline, "OllamaExplainer", make_explainer)
    monkeypatch.setattr(run_pipeline, "Pipeline", make_pipeline)

    run_pipeline._build_pipeline("small")

    assert calls["whisper"] == "small"
    assert calls["wav2vec2"] == config.WAV2VEC2_MODEL_NAME
    assert calls["ollama"] == (
        config.OLLAMA_MODEL,
        config.OLLAMA_BASE_URL,
        config.PROMPT_VERSION,
    )
    assert calls["pipeline"] == {
        "transcriber": transcriber,
        "phoneme_recognizer": recognizer,
        "explainer": explainer,
        "wer_threshold": config.WER_THRESHOLD,
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            PhonemeError("substitution", "ð", "d", 0, "this"),
            'substitution in "this": expected /ð/, actual /d/',
        ),
        (
            PhonemeError("deletion", "k", None, 3, "desk"),
            'deletion in "desk": expected /k/, actual (missing)',
        ),
        (
            PhonemeError("insertion", None, "ə", 2, "desk"),
            'insertion in "desk": expected -, actual /ə/',
        ),
    ],
)
def test_format_error_handles_missing_phonemes(
    error: PhonemeError, expected: str
) -> None:
    assert run_pipeline._format_error(error) == expected


def test_format_error_marks_possible_misread() -> None:
    error = PhonemeError(
        "substitution", "h", "b", 4, "high", possibly_misread=True, misread_as="buy"
    )

    assert run_pipeline._format_error(error).endswith(
        '  [possible misread: read as "buy"]'
    )
