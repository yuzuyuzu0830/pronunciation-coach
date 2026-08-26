from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ui import config

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "smoke_test_whisper.py"
_SPEC = importlib.util.spec_from_file_location("smoke_test_whisper", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_SCRIPT_PATH}")
smoke_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_test)


def test_main_uses_production_transcriber_and_current_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    calls: dict[str, object] = {}

    class FakeTranscriber:
        def __init__(self, *, model_size: str):
            calls["model_size"] = model_size

        def transcribe(self, path: Path) -> str:
            calls["audio_path"] = path
            return "This is a test."

    monkeypatch.setattr(smoke_test, "WhisperTranscriber", FakeTranscriber, raising=False)
    monkeypatch.setattr(
        smoke_test,
        "transcribe",
        lambda *_: pytest.fail("legacy direct Whisper call must not run"),
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_whisper.py", str(audio_path)])

    smoke_test.main()

    assert calls == {
        "model_size": config.WHISPER_MODEL_SIZE,
        "audio_path": audio_path,
    }
    assert capsys.readouterr().out.splitlines() == [
        f"[model={config.WHISPER_MODEL_SIZE}] {audio_path}",
        "This is a test.",
    ]


def test_main_rejects_missing_audio_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    audio_path = tmp_path / "missing.wav"

    class UnexpectedTranscriber:
        def __init__(self, *, model_size: str):
            pytest.fail("model must not load for a missing audio file")

    monkeypatch.setattr(
        smoke_test,
        "WhisperTranscriber",
        UnexpectedTranscriber,
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_whisper.py", str(audio_path)])

    with pytest.raises(SystemExit) as exc_info:
        smoke_test.main()

    assert exc_info.value.code == 1
    assert f"Audio file not found: {audio_path}" in capsys.readouterr().err
