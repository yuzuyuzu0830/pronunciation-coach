"""Compare Whisper model sizes (base/small/medium) on real trial recordings.

Replays saved trial audio to compare transcription accuracy and inference
cost, and WER-gate behavior. The trial log is the source of truth for
audio/target pairs, including reading-mismatch outcomes.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pronunciation_coach.transcriber import WhisperTranscriber
from pronunciation_coach.validator import validate_reading

DEFAULT_TRIAL_LOGS_DIR = REPO_ROOT / "results" / "trial_logs"
OUTPUT_DIR = REPO_ROOT / "docs" / "experiments"
DEFAULT_MODEL_SIZES = ["base", "small", "medium"]


def load_trial_cases(trial_log_path: Path, participant_id: str) -> list[dict]:
    """Load a participant's saved trials that have target text for WER."""
    cases = []
    with trial_log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("participant_id") != participant_id:
                continue
            if not record.get("audio_file") or not record.get("target_text"):
                continue
            cases.append(record)
    return cases


def evaluate_case(
    transcriber: WhisperTranscriber, case: dict, trial_logs_dir: Path
) -> dict:
    """Transcribe and score one recording without aborting on case failures."""
    audio_path = trial_logs_dir / case["audio_file"]
    started = time.perf_counter()
    try:
        transcript = transcriber.transcribe(audio_path)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        return {
            "audio_file": case["audio_file"],
            "logged_outcome": case["outcome"],
            "target_text": case["target_text"],
            "transcript": None,
            "wer": None,
            "gate_passed": None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        }
    elapsed = time.perf_counter() - started
    validation = validate_reading(case["target_text"], transcript)
    return {
        "audio_file": case["audio_file"],
        "logged_outcome": case["outcome"],
        "target_text": case["target_text"],
        "transcript": transcript,
        "wer": validation.wer,
        "gate_passed": validation.passed,
        "error": None,
        "elapsed_sec": elapsed,
    }


def run_model(
    model_size: str, cases: list[dict], trial_logs_dir: Path
) -> tuple[float, list[dict]]:
    """Returns (load_time_sec, per-case results with transcript/WER/elapsed)."""
    load_started = time.perf_counter()
    transcriber = WhisperTranscriber(model_size=model_size)
    load_time = time.perf_counter() - load_started

    results = [evaluate_case(transcriber, case, trial_logs_dir) for case in cases]
    return load_time, results


def format_report(
    participant_id: str, model_reports: dict[str, tuple[float, list[dict]]]
) -> str:
    """Render model summaries and case details as Markdown.

    Failed cases remain visible but do not contribute to mean WER or gate counts.
    """
    lines = [
        f"# Whisper model comparison on real recordings (participant {participant_id})",
        "",
        (
            "Each recording is a saved trial take replayed unchanged through "
            "every model size to compare transcription and WER-gate behavior."
        ),
        "",
    ]

    for model_size, (load_time, results) in model_reports.items():
        # Aggregate accuracy only from successful transcriptions.
        scored = [r for r in results if r["error"] is None]
        errors = [r for r in results if r["error"] is not None]
        total_time = sum(r["elapsed_sec"] for r in results)
        avg_wer = (
            sum(r["wer"] for r in scored) / len(scored) if scored else 0.0
        )
        gate_failures = sum(1 for r in scored if not r["gate_passed"])
        lines += [
            f"## {model_size}",
            "",
            f"- Load time: {load_time:.1f}s",
            f"- Total inference time: {total_time:.1f}s over {len(results)} "
            f"recordings ({total_time / len(results):.1f}s/recording avg)"
            if results
            else "- Total inference time: n/a (no recordings)",
            f"- Mean WER vs. target sentence: {avg_wer:.3f}"
            + (f" ({len(scored)} scored, {len(errors)} failed)" if errors else ""),
            (
                "- Recordings that would still fail the WER gate: "
                f"{gate_failures}/{len(scored)}"
            ),
            "",
            "| audio_file | logged outcome | target_text | transcript | WER | gate |",
            "|---|---|---|---|---|---|",
        ]
        # Preserve every attempted recording in the detailed comparison table.
        for r in results:
            if r["error"] is not None:
                lines.append(
                    f"| {r['audio_file']} | {r['logged_outcome']} | "
                    f"{r['target_text']} | ERROR: {r['error']} | — | — |"
                )
                continue
            gate = "pass" if r["gate_passed"] else "FAIL"
            lines.append(
                f"| {r['audio_file']} | {r['logged_outcome']} | {r['target_text']} "
                f"| {r['transcript']} | {r['wer']:.2f} | {gate} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Whisper model sizes on real trial recordings"
    )
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--model-size", nargs="+", default=DEFAULT_MODEL_SIZES)
    parser.add_argument("--trial-logs-dir", type=Path, default=DEFAULT_TRIAL_LOGS_DIR)
    parser.add_argument(
        "--tag", default=None, help="suffix for the output filename"
    )
    args = parser.parse_args()

    trial_log_path = args.trial_logs_dir / "trial_log.jsonl"
    cases = load_trial_cases(trial_log_path, args.participant)
    if not cases:
        print(f"No cases found for participant {args.participant!r} in {trial_log_path}")
        sys.exit(1)
    print(f"{len(cases)} recordings found for participant {args.participant!r}")

    model_reports = {}
    for model_size in args.model_size:
        print(f"[{model_size}] loading model...", flush=True)
        load_time, results = run_model(model_size, cases, args.trial_logs_dir)
        n_err = sum(1 for r in results if r["error"] is not None)
        err_note = f", {n_err} errors" if n_err else ""
        print(
            f"[{model_size}] load={load_time:.1f}s, "
            f"{len(results)} recordings done{err_note}"
        )
        model_reports[model_size] = (load_time, results)

    report = format_report(args.participant, model_reports)

    today = datetime.datetime.now().astimezone().date().isoformat()
    suffix = f"_{args.tag}" if args.tag else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"whisper_model_comparison_{today}{suffix}.md"
    out_path.write_text(report)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
