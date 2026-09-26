from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest
import requests

from ui.models import (
    StartupError,
    _git_commit_short,
    check_espeak_and_trial_sentences,
    check_ollama,
)
from ui.sentences import TRIAL_SENTENCES


class _FakeResponse:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, list[dict[str, str]]]:
        return {"models": [{"name": model} for model in self._models]}


@pytest.mark.parametrize(
    "exc",
    [OSError("git unavailable"), subprocess.CalledProcessError(128, ["git"])],
    ids=["os-error", "nonzero-exit"],
)
def test_git_commit_short_returns_none_for_expected_git_failures(
    exc: Exception,
) -> None:
    with patch("ui.models.subprocess.run", side_effect=exc):
        assert _git_commit_short() is None


def test_git_commit_short_does_not_hide_unexpected_errors() -> None:
    with (
        patch("ui.models.subprocess.run", side_effect=RuntimeError("bug")),
        pytest.raises(RuntimeError, match="bug"),
    ):
        _git_commit_short()


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad sentence"),
        RuntimeError("espeak died"),
        OSError(2, "no espeak"),
    ],
    ids=["invalid-phonemes", "backend-failure", "missing-espeak"],
)
def test_check_espeak_wraps_g2p_backend_failures_as_startup_error(
    exc: Exception,
) -> None:
    with (
        patch("phonemizer.backend.EspeakBackend.is_available", return_value=True),
        patch("ui.models.to_phonemes_by_word", side_effect=exc),
        pytest.raises(StartupError, match="failed g2p") as raised,
    ):
        check_espeak_and_trial_sentences()
    assert TRIAL_SENTENCES[0].text in str(raised.value)
    assert raised.value.__cause__ is exc


def test_check_ollama_raises_on_connection_error() -> None:
    with (
        patch(
            "ui.models.requests.get", side_effect=requests.exceptions.ConnectionError()
        ),
        pytest.raises(StartupError, match="Cannot connect to Ollama"),
    ):
        check_ollama()


def test_check_ollama_raises_when_model_not_pulled() -> None:
    with (
        patch("ui.models.requests.get", return_value=_FakeResponse(["llama3.2:3b"])),
        pytest.raises(StartupError, match="not pulled"),
    ):
        check_ollama(model="llama3.1:8b")


def test_check_ollama_accepts_exact_model_name() -> None:
    with patch("ui.models.requests.get", return_value=_FakeResponse(["llama3.1:8b"])):
        check_ollama(model="llama3.1:8b")


def test_check_ollama_accepts_default_tag_for_an_untagged_config_name() -> None:
    with patch(
        "ui.models.requests.get", return_value=_FakeResponse(["llama3.1:latest"])
    ):
        check_ollama(model="llama3.1")
