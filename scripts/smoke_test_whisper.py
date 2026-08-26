"""Smoke-test the production Whisper transcriber on one audio file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pronunciation_coach.transcriber import WhisperTranscriber  # noqa: E402
from ui import config  # noqa: E402

DEFAULT_AUDIO_PATH = Path("data/samples/sample.m4a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH, type=Path)
    parser.add_argument(
        "--model",
        default=config.WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium"],
    )
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    transcriber = WhisperTranscriber(model_size=args.model)
    text = transcriber.transcribe(args.audio_path)
    print(f"[model={args.model}] {args.audio_path}")
    print(text)


if __name__ == "__main__":
    main()
