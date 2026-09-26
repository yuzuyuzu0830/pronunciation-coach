from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from scripts import smoke_test_phoneme as smoke_test


def test_main_uses_production_recognizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    device = torch.device("cpu")
    calls: dict[str, object] = {}

    class FakeRecognizer:
        def __init__(self, *, device: torch.device) -> None:
            calls["device"] = device

        def recognize(self, path: Path) -> list[str]:
            calls["audio_path"] = path
            return ["ð", "ɪ", "s"]

    def fake_default_device() -> torch.device:
        calls["default_device_called"] = True
        return device

    monkeypatch.setattr(smoke_test, "default_device", fake_default_device)
    monkeypatch.setattr(
        smoke_test,
        "Wav2Vec2PhonemeRecognizer",
        FakeRecognizer,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_phoneme.py", str(audio_path)])

    smoke_test.main()

    assert calls == {
        "default_device_called": True,
        "device": device,
        "audio_path": audio_path,
    }
    assert capsys.readouterr().out.splitlines() == [
        f"[device={device}] {audio_path}",
        "ð ɪ s",
    ]


def test_main_rejects_missing_audio_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "missing.wav"

    class UnexpectedRecognizer:
        def __init__(self, *, device: torch.device) -> None:
            pytest.fail("model must not load for a missing audio file")

    monkeypatch.setattr(
        smoke_test,
        "Wav2Vec2PhonemeRecognizer",
        UnexpectedRecognizer,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_phoneme.py", str(audio_path)])

    with pytest.raises(SystemExit) as exc_info:
        smoke_test.main()

    assert exc_info.value.code == 1
    assert f"Audio file not found: {audio_path}" in capsys.readouterr().err
