"""Trial-fixed configuration (docs/design_ui.md §1). Not user-editable from
the UI -- change these constants and restart the app to change a trial's
model/prompt setup.
"""

from __future__ import annotations

WHISPER_MODEL_SIZE = "small"
# 2026-07-27: switched from base to small because base produced frequent
# mis-transcriptions in the P01 trial (false gate failures / false misread flags).
# medium did not beat small on accuracy and only increased load/inference cost
# (docs/experiments/whisper_model_comparison_2026-07-27.md).
# Current detection-pipeline default; update once the model comparison
# (docs/experiments/phoneme_model_comparison_2026-07-23.md) reaches a decision.
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
# v3 (structured-knowledge injection) is the adopted prompt version
# (docs/design_3c.md); explainer.py's own default stays v1 for the model
# comparison experiments, so the trial UI sets this explicitly.
PROMPT_VERSION = "v3"
WER_THRESHOLD = 0.5

TRIAL_LOGS_DIR = "results/trial_logs"

L1_CHOICES = ("Japanese",)  # l1_rules/ only has japanese.json today (docs/design_ui.md §10)
DEFAULT_L1 = "Japanese"


def build_run_config(git_commit: str | None) -> dict:
    """Config snapshot embedded verbatim in every trial log record."""
    return {
        "whisper": WHISPER_MODEL_SIZE,
        "wav2vec2": WAV2VEC2_MODEL_NAME,
        "ollama_model": OLLAMA_MODEL,
        "prompt_version": PROMPT_VERSION,
        "wer_threshold": WER_THRESHOLD,
        "git_commit": git_commit,
    }
