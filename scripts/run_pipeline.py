"""CLI entry point: run the full pronunciation coaching pipeline on one file.

Requires the real models (Whisper, wav2vec2, Ollama server running).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pronunciation_coach.explainer import OllamaExplainer
from pronunciation_coach.phoneme_recognizer import Wav2Vec2PhonemeRecognizer
from pronunciation_coach.pipeline import Pipeline
from pronunciation_coach.transcriber import WhisperTranscriber
from pronunciation_coach.types import PhonemeError, ReadingMismatch
from ui import config


def _build_pipeline(whisper_model: str) -> Pipeline:
    return Pipeline(
        transcriber=WhisperTranscriber(model_size=whisper_model),
        phoneme_recognizer=Wav2Vec2PhonemeRecognizer(
            model_name=config.WAV2VEC2_MODEL_NAME
        ),
        explainer=OllamaExplainer(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            prompt_version=config.PROMPT_VERSION,
        ),
        wer_threshold=config.WER_THRESHOLD,
    )


def _format_error(error: PhonemeError) -> str:
    word = f'"{error.word}"' if error.word else f"position {error.position}"
    expected = f"/{error.expected}/" if error.expected else "-"
    if error.op == "deletion":
        actual = "(missing)"
    else:
        actual = f"/{error.actual}/" if error.actual else "-"
    misread = ""
    if error.possibly_misread:
        read_as = (
            f'read as "{error.misread_as}"' if error.misread_as else "word skipped"
        )
        misread = f"  [possible misread: {read_as}]"
    return f"{error.op} in {word}: expected {expected}, actual {actual}{misread}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pronunciation coaching pipeline")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--target-text",
        default=None,
        help="expected reading text (recommended)",
    )
    parser.add_argument(
        "--l1",
        default=config.DEFAULT_L1,
        help="learner's first language",
    )
    parser.add_argument(
        "--whisper-model",
        default=config.WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium"],
    )
    args = parser.parse_args()

    pipeline = _build_pipeline(args.whisper_model)
    result = pipeline.run(
        args.audio_path,
        target_text=args.target_text,
        learner_l1=args.l1,
    )

    if isinstance(result, ReadingMismatch):
        v = result.validation
        print("Reading does not match the target text. Please try again.", file=sys.stderr)
        print(f"  WER: {v.wer:.2f} (threshold {v.threshold:.2f})", file=sys.stderr)
        print(f"  target:     {' '.join(v.target_words)}", file=sys.stderr)
        print(f"  transcript: {' '.join(v.transcript_words)}", file=sys.stderr)
        sys.exit(1)

    report = result.report
    print(f"transcript : {report.transcript}")
    if result.validation is not None:
        print(f"validation : WER {result.validation.wer:.2f} (passed)")
    print(f"reference  : {' '.join(report.reference_phonemes)}")
    print(f"hypothesis : {' '.join(report.hypothesis_phonemes)}")
    print()
    if report.errors:
        print(f"detected errors ({len(report.errors)}):")
        for error in report.errors:
            print(f"  - {_format_error(error)}")
    else:
        print("detected errors: none")
    print()
    print("=== explanation ===")
    print(result.explanation)


if __name__ == "__main__":
    main()
