from __future__ import annotations

import sys

import pytest

from pronunciation_coach.types import DiagnosisReport
from scripts import smoke_test_llm as smoke_test
from ui import config


def test_main_uses_production_explainer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    class FakeExplainer:
        def __init__(self, *, model: str, base_url: str, prompt_version: str) -> None:
            calls["config"] = (model, base_url, prompt_version)

        def explain(self, report: DiagnosisReport) -> str:
            calls["report"] = report
            return "Place your tongue lightly between your teeth."

    monkeypatch.setattr(smoke_test, "OllamaExplainer", FakeExplainer)
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
) -> None:
    class OfflineExplainer:
        def __init__(self, *, model: str, base_url: str, prompt_version: str) -> None:
            pass

        def explain(self, _report: DiagnosisReport) -> str:
            raise ConnectionError("Cannot connect to Ollama")

    monkeypatch.setattr(smoke_test, "OllamaExplainer", OfflineExplainer)
    monkeypatch.setattr(sys, "argv", ["smoke_test_llm.py"])

    with pytest.raises(SystemExit) as exc_info:
        smoke_test.main()

    assert exc_info.value.code == 1
    assert "Cannot connect to Ollama" in capsys.readouterr().err
