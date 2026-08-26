from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "smoke_test_phoneme.py"
_SPEC = importlib.util.spec_from_file_location("smoke_test_phoneme", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_SCRIPT_PATH}")
smoke_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_test)


def test_main_uses_production_recognizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    device = torch.device("cpu")
    calls: dict[str, object] = {}

    class FakeRecognizer:
        def __init__(self, *, device: torch.device):
            calls["device"] = device

        def recognize(self, path: Path) -> list[str]:
            calls["audio_path"] = path
            return ["ð", "ɪ", "s"]

    monkeypatch.setattr(smoke_test, "default_device", lambda: device, raising=False)
    monkeypatch.setattr(
        smoke_test,
        "Wav2Vec2PhonemeRecognizer",
        FakeRecognizer,
        raising=False,
    )
    monkeypatch.setattr(
        smoke_test,
        "get_device",
        lambda: pytest.fail("legacy device selection must not run"),
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_phoneme.py", str(audio_path)])

    smoke_test.main()

    assert calls == {"device": device, "audio_path": audio_path}
    assert capsys.readouterr().out.splitlines() == [
        f"[device={device}] {audio_path}",
        "ð ɪ s",
    ]


def test_main_rejects_missing_audio_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    audio_path = tmp_path / "missing.wav"

    class UnexpectedRecognizer:
        def __init__(self, *, device: torch.device):
            pytest.fail("model must not load for a missing audio file")

    monkeypatch.setattr(
        smoke_test,
        "Wav2Vec2PhonemeRecognizer",
        UnexpectedRecognizer,
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_phoneme.py", str(audio_path)])

    with pytest.raises(SystemExit) as exc_info:
        smoke_test.main()

    assert exc_info.value.code == 1
    assert f"Audio file not found: {audio_path}" in capsys.readouterr().err
