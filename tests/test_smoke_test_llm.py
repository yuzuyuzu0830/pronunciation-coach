from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from pronunciation_coach.types import DiagnosisReport
from ui import config

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "smoke_test_llm.py"
_SPEC = importlib.util.spec_from_file_location("smoke_test_llm", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_SCRIPT_PATH}")
smoke_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_test)


def test_main_uses_production_explainer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls: dict[str, object] = {}

    class FakeExplainer:
        def __init__(self, *, model: str, base_url: str, prompt_version: str):
            calls["config"] = (model, base_url, prompt_version)

        def explain(self, report: DiagnosisReport) -> str:
            calls["report"] = report
            return "Place your tongue lightly between your teeth."

    monkeypatch.setattr(smoke_test, "OllamaExplainer", FakeExplainer, raising=False)
    monkeypatch.setattr(
        smoke_test,
        "generate",
        lambda *_: pytest.fail("legacy raw API call must not run"),
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test_llm.py", "--model", "test-model"])

    smoke_test.main()

    assert calls["config"] == (
        "test-model",
        config.OLLAMA_BASE_URL,
        config.PROMPT_VERSION,
    )
    report = calls["report"]
    assert isinstance(report, DiagnosisReport)
    assert report.errors[0].expected == "ð"
    assert report.errors[0].actual == "d"
    assert capsys.readouterr().out.splitlines() == [
        "[model=test-model]",
        "Place your tongue lightly between your teeth.",
    ]


def test_main_reports_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class OfflineExplainer:
        def __init__(self, *, model: str, base_url: str, prompt_version: str):
            pass

        def explain(self, report: DiagnosisReport) -> str:
            raise ConnectionError("Cannot connect to Ollama")

    monkeypatch.setattr(smoke_test, "OllamaExplainer", OfflineExplainer, raising=False)
    monkeypatch.setattr(sys, "argv", ["smoke_test_llm.py"])

    with pytest.raises(SystemExit) as exc_info:
        smoke_test.main()

    assert exc_info.value.code == 1
    assert "Cannot connect to Ollama" in capsys.readouterr().err
