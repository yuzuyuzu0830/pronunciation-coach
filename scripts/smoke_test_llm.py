"""Smoke-test the production Ollama explainer with a known phoneme error."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pronunciation_coach.explainer import OllamaExplainer  # noqa: E402
from pronunciation_coach.types import DiagnosisReport, PhonemeError  # noqa: E402
from ui import config  # noqa: E402

SMOKE_REPORT = DiagnosisReport(
    transcript="this",
    target_text="this",
    reference_phonemes=["ð", "ɪ", "s"],
    hypothesis_phonemes=["d", "ɪ", "s"],
    errors=[PhonemeError("substitution", "ð", "d", 0, "this")],
    learner_l1=config.DEFAULT_L1,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Ollama explanation generation for MDD feedback."
    )
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    args = parser.parse_args()

    explainer = OllamaExplainer(
        model=args.model,
        base_url=config.OLLAMA_BASE_URL,
        prompt_version=config.PROMPT_VERSION,
    )
    try:
        text = explainer.explain(SMOKE_REPORT)
    except ConnectionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[model={args.model}]")
    print(text)


if __name__ == "__main__":
    main()
