"""CLI entry point: run the full pronunciation coaching pipeline on one file.

Requires the real models (Whisper, wav2vec2, Ollama server running).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pronunciation_coach.explainer import OllamaExplainer
from pronunciation_coach.phoneme_recognizer import Wav2Vec2PhonemeRecognizer
from pronunciation_coach.pipeline import Pipeline
from pronunciation_coach.transcriber import WhisperTranscriber
from pronunciation_coach.types import ReadingMismatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Pronunciation coaching pipeline")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--target-text", default=None, help="読み上げ課題文(推奨)")
    parser.add_argument("--l1", default="Japanese", help="学習者の母語")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base"])
    args = parser.parse_args()

    pipeline = Pipeline(
        transcriber=WhisperTranscriber(model_size=args.whisper_model),
        phoneme_recognizer=Wav2Vec2PhonemeRecognizer(),
        explainer=OllamaExplainer(),
    )
    result = pipeline.run(args.audio_path, target_text=args.target_text, learner_l1=args.l1)

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
        for e in report.errors:
            word = f'"{e.word}"' if e.word else f"position {e.position}"
            misread = ""
            if e.possibly_misread:
                read_as = f'read as "{e.misread_as}"' if e.misread_as else "word skipped"
                misread = f"  [possible misread: {read_as}]"
            print(f"  - {e.op} in {word}: expected /{e.expected}/, actual /{e.actual}/{misread}")
    else:
        print("detected errors: none")
    print()
    print("=== explanation ===")
    print(result.explanation)


if __name__ == "__main__":
    main()
