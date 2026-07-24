"""Only the pure precheck logic is unit-tested here (mocked network/espeak);
load_models() itself needs real Whisper/wav2vec2/Ollama and is exercised by
the manual E2E check (docs/design_ui.md §9 step 6), matching how
phoneme_recognizer.py's model-loading constructor is untested elsewhere.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ui.models import StartupError, check_ollama


def _fake_response(models: list[str]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"models": [{"name": m} for m in models]}
    return response


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
