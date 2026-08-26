"""Smoke-test the production wav2vec2 phoneme recognizer on one audio file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pronunciation_coach.phoneme_recognizer import (  # noqa: E402
    Wav2Vec2PhonemeRecognizer,
    default_device,
)

DEFAULT_AUDIO_PATH = Path("data/samples/sample.m4a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH, type=Path)
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    device = default_device()
    recognizer = Wav2Vec2PhonemeRecognizer(device=device)
    phonemes = recognizer.recognize(args.audio_path)
    print(f"[device={device}] {args.audio_path}")
    print(" ".join(phonemes))


if __name__ == "__main__":
    main()
