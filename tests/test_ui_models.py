"""Only the pure precheck logic is unit-tested here (mocked network/espeak);
load_models() itself needs real Whisper/wav2vec2/Ollama and is exercised by
the manual E2E check, matching how
phoneme_recognizer.py's model-loading constructor is untested elsewhere.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from ui.models import (
    StartupError,
    _git_commit_short,
    check_espeak_and_trial_sentences,
    check_ollama,
)
from ui.sentences import TRIAL_SENTENCES


def _fake_response(models: list[str]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"models": [{"name": m} for m in models]}
    return response


def _patch_espeak_available():
    backend = MagicMock()
    backend.is_available.return_value = True
    return patch("phonemizer.backend.EspeakBackend", backend)


@pytest.mark.parametrize(
    "exc",
    [OSError("git unavailable"), subprocess.CalledProcessError(128, ["git"])],
)
def test_git_commit_short_returns_none_for_expected_git_failures(exc):
    with patch("ui.models.subprocess.run", side_effect=exc):
        assert _git_commit_short() is None


def test_git_commit_short_does_not_hide_unexpected_errors():
    with patch("ui.models.subprocess.run", side_effect=RuntimeError("bug")):
        with pytest.raises(RuntimeError, match="bug"):
            _git_commit_short()


@pytest.mark.parametrize("exc", [ValueError("bad sentence"), RuntimeError("espeak died"), OSError(2, "no espeak")])
def test_check_espeak_wraps_g2p_backend_failures_as_startup_error(exc):
    """g2p lets RuntimeError/OSError propagate; startup must still raise
    StartupError so app.py's fail-fast handler catches them."""
    with _patch_espeak_available(), patch(
        "ui.models.to_phonemes_by_word", side_effect=exc
    ):
        with pytest.raises(StartupError, match="failed g2p") as raised:
            check_espeak_and_trial_sentences()
    assert TRIAL_SENTENCES[0].text in str(raised.value)
    assert raised.value.__cause__ is exc


def test_check_ollama_raises_on_connection_error():
    with patch("ui.models.requests.get", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(StartupError, match="Cannot connect to Ollama"):
            check_ollama()


def test_check_ollama_raises_when_model_not_pulled():
    with patch("ui.models.requests.get", return_value=_fake_response(["llama3.2:3b"])):
        with pytest.raises(StartupError, match="not pulled"):
            check_ollama(model="llama3.1:8b")


def test_check_ollama_accepts_exact_model_name():
    with patch("ui.models.requests.get", return_value=_fake_response(["llama3.1:8b"])):
        check_ollama(model="llama3.1:8b")  # must not raise


def test_check_ollama_accepts_default_tag_for_an_untagged_config_name():
    """A config value without a tag (e.g. 'llama3.1') should still match
    Ollama reporting the pulled model as 'llama3.1:latest'."""
    with patch("ui.models.requests.get", return_value=_fake_response(["llama3.1:latest"])):
        check_ollama(model="llama3.1")  # must not raise
