import argparse
import sys
from pathlib import Path

import whisper

DEFAULT_AUDIO_PATH = Path("data/samples/sample.m4a")
DEFAULT_MODEL = "tiny"


def transcribe(audio_path: Path, model_name: str) -> str:
    """Transcribe an audio file with Whisper and return English text.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language="en")
    return result["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["tiny", "base"])
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    text = transcribe(args.audio_path, args.model)
    print(f"[model={args.model}] {args.audio_path}")
    print(text)


if __name__ == "__main__":
    main()
