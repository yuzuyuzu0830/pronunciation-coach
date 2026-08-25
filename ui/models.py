"""Startup model loading + connectivity checks.

Fail-fast: any check failing raises StartupError with a message telling the
operator how to fix it. app.py catches this and exits before opening the UI.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

from pronunciation_coach.explainer import OllamaExplainer
from pronunciation_coach.g2p import to_phonemes_by_word
from pronunciation_coach.phoneme_recognizer import Wav2Vec2PhonemeRecognizer
from pronunciation_coach.pipeline import Pipeline
from pronunciation_coach.transcriber import WhisperTranscriber
from ui import config
from ui.sentences import TRIAL_SENTENCES

_REPO_ROOT = Path(__file__).resolve().parent.parent


class StartupError(RuntimeError):
    """A required model or service isn't usable; the app must not open."""


def _git_commit_short() -> str | None:
    """Return the commit for run metadata, or None when Git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_espeak_and_trial_sentences() -> None:
    """Verify espeak and every fixed trial sentence before UI startup."""
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError as e:
        raise StartupError(f"phonemizer is not installed: {e}") from e
    if not EspeakBackend.is_available():
        raise StartupError(
            "espeak-ng was not found. Install it (e.g. `brew install espeak-ng`) "
            "and try again."
        )
    for sentence in TRIAL_SENTENCES:
        try:
            to_phonemes_by_word(sentence.text)
        # Convert expected G2P/backend failures into an operator-facing error.
        except (ValueError, RuntimeError, OSError) as e:
            raise StartupError(
                f"Trial sentence {sentence.text!r} failed g2p: {e}. "
                "Fix or remove it from ui/sentences.py."
            ) from e


def check_ollama(base_url: str = config.OLLAMA_BASE_URL, model: str = config.OLLAMA_MODEL) -> None:
    """Verify Ollama connectivity and the configured model availability."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise StartupError(
            f"Cannot connect to Ollama at {base_url}. Start it with `ollama serve` "
            "(or `brew services start ollama`) and try again."
        ) from e
    available = [m.get("name", "") for m in response.json().get("models", [])]
    # Ollama tags a pulled model "name:tag" (often ":latest"); the config
    # value is the pull target without a tag, so match on either.
    if not any(name == model or name.startswith(f"{model}:") for name in available):
        raise StartupError(
            f"Model {model!r} is not pulled into Ollama. Run `ollama pull {model}` "
            "and try again."
        )


@dataclass(frozen=True)
class AppModels:
    pipeline: Pipeline
    app_session_id: str
    trial_logs_dir: Path
    run_config: dict


def load_models(trial_logs_dir: Path | None = None) -> AppModels:
    """Load every model/service the trial needs, in order, printing progress
    to the console. Raises StartupError on the first failure."""
    print("[1/4] espeak-ng + trial sentences...", end=" ", flush=True)
    check_espeak_and_trial_sentences()
    print("OK")

    print("[2/4] Whisper...", end=" ", flush=True)
    transcriber = WhisperTranscriber(model_size=config.WHISPER_MODEL_SIZE)
    print("OK")

    print("[3/4] wav2vec2...", end=" ", flush=True)
    phoneme_recognizer = Wav2Vec2PhonemeRecognizer(model_name=config.WAV2VEC2_MODEL_NAME)
    print("OK")

    print("[4/4] Ollama...", end=" ", flush=True)
    check_ollama()
    print("OK")

    explainer = OllamaExplainer(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        prompt_version=config.PROMPT_VERSION,
    )
    pipeline = Pipeline(
        transcriber=transcriber,
        phoneme_recognizer=phoneme_recognizer,
        explainer=explainer,
        wer_threshold=config.WER_THRESHOLD,
    )

    return AppModels(
        pipeline=pipeline,
        app_session_id=str(uuid.uuid4()),
        trial_logs_dir=trial_logs_dir or (_REPO_ROOT / config.TRIAL_LOGS_DIR),
        run_config=config.build_run_config(_git_commit_short()),
    )
