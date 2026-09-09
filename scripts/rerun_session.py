"""Replay a logged session through the current pronunciation pipeline.

Trials use ui.runner.run_trial() to preserve live pre-check and WER-gate
behavior. Original timestamps are retained for one-to-one comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ui import config as ui_config  # noqa: E402
from ui.models import StartupError, load_models  # noqa: E402
from ui.runner import TrialState, run_trial  # noqa: E402
from ui.trial_logging import Outcome, TargetSource  # noqa: E402

DEFAULT_LOG = REPO_ROOT / "results" / "trial_logs" / "trial_log.jsonl"


class ReportRecord(TypedDict):
    errors: list[object]


class TrialRecord(TypedDict):
    timestamp_utc: str
    app_session_id: str
    participant_id: str
    learner_l1: str
    target_text: str | None
    target_source: TargetSource
    audio_file: str | None
    outcome: Outcome
    transcript: str | None
    report: ReportRecord | None
    config: dict[str, object]


def load_records(log_path: Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    with log_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {log_path} at line {line_number}: {e.msg}"
                ) from e
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {log_path} at line {line_number}"
                )
            records.append(cast(TrialRecord, value))
    return records


def select_session(
    records: list[TrialRecord], session_prefix: str, participant: str | None
) -> list[TrialRecord]:
    selected = [r for r in records if r["app_session_id"].startswith(session_prefix)]
    if participant is not None:
        selected = [r for r in selected if r["participant_id"] == participant]
    return selected


def replayable(record: TrialRecord, source_dir: Path) -> Path | None:
    """Return the saved audio path when it is available for replay."""
    audio_file = record.get("audio_file")
    if not audio_file:
        return None
    path = source_dir / audio_file
    return path if path.exists() else None


def error_count(record: TrialRecord) -> int | None:
    report = record.get("report")
    return len(report["errors"]) if report else None


def summarize(record: TrialRecord) -> str:
    parts: list[str] = [record["outcome"]]
    n = error_count(record)
    if n is not None:
        parts.append(f"{n} errors")
    transcript = record.get("transcript")
    if transcript:
        parts.append(repr(transcript))
    return " | ".join(parts)


def rerun(
    session_records: list[TrialRecord],
    *,
    source_dir: Path,
    out_dir: Path,
    session_prefix: str,
) -> list[TrialRecord]:
    """Replay available recordings and return the newly written records."""
    out_log = out_dir / "trial_log.jsonl"
    if out_log.exists():
        print(f"Refusing to append to an existing log: {out_log}", file=sys.stderr)
        print("Move or delete it first (re-run records must not be mixed).", file=sys.stderr)
        sys.exit(1)

    try:
        models = load_models(trial_logs_dir=out_dir)
    except StartupError as e:
        print(f"\nStartup failed: {e}", file=sys.stderr)
        sys.exit(1)

    run_config = dict(models.run_config)
    run_config["rerun_of_session"] = session_prefix
    run_config["rerun_at"] = datetime.now().astimezone().isoformat()

    for i, record in enumerate(session_records, start=1):
        audio_path = replayable(record, source_dir)
        label = f"[{i}/{len(session_records)}] {record['timestamp_utc'][:19]}"
        if audio_path is None:
            print(f"{label} skipped (no replayable audio, outcome={record['outcome']})")
            continue

        original_ts = datetime.fromisoformat(record["timestamp_utc"])
        print(f"{label} {audio_path.name} -> ", end="", flush=True)
        state: TrialState | None = None
        # run_trial guarantees that its final state is terminal and logged.
        for state in run_trial(
            pipeline=models.pipeline,
            audio_path=audio_path,
            participant_id=record["participant_id"],
            learner_l1=record["learner_l1"],
            target_text=record["target_text"],
            target_source=record["target_source"],
            app_session_id=models.app_session_id,
            trial_logs_dir=out_dir,
            config=run_config,
            now=lambda ts=original_ts: ts,
        ):
            pass
        print(state.stage if state else "no state")
        if state is not None and state.log_error:
            print(f"    log_error: {state.log_error}", file=sys.stderr)

    produced = load_records(out_log) if out_log.exists() else []
    print(f"\nWrote {len(produced)} records to {out_log}")
    return produced


def compare(
    original: list[TrialRecord], rerun_records: list[TrialRecord], title: str
) -> None:
    by_ts = {r["timestamp_utc"]: r for r in rerun_records}
    print(f"\n=== {title} ===")
    changed = 0
    for old in original:
        new = by_ts.get(old["timestamp_utc"])
        if new is None:
            continue
        mark = " " if old["outcome"] == new["outcome"] else "*"
        if mark == "*":
            changed += 1
        print(f"{mark} {old['timestamp_utc'][11:19]} {old.get('target_text')!r}")
        print(f"    before: {summarize(old)}")
        print(f"    after : {summarize(new)}")
    print(f"\noutcome changed on {changed} of {len(by_ts)} replayed trials")


def outcome_counts(records: list[TrialRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="app_session_id prefix to replay")
    parser.add_argument("--participant", default=None, help="restrict to this participant_id")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--label", default=None, help="output dir name under results/rerun/")
    parser.add_argument(
        "--compare-with",
        default=None,
        help="app_session_id prefix of a later live session, for a 3-way console summary",
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be replayed")
    args = parser.parse_args()

    records = load_records(args.log)
    session_records = select_session(records, args.session, args.participant)
    if not session_records:
        print(f"No records for session {args.session!r}", file=sys.stderr)
        sys.exit(1)

    source_dir = args.log.parent
    label = args.label or args.session
    out_dir = REPO_ROOT / "results" / "rerun" / label

    print(f"session {args.session}: {len(session_records)} logged trials")
    print(f"original config: {json.dumps(session_records[0]['config'], ensure_ascii=False)}")
    print(
        f"current config : whisper={ui_config.WHISPER_MODEL_SIZE} "
        f"prompt={ui_config.PROMPT_VERSION} ollama={ui_config.OLLAMA_MODEL}"
    )
    replay = [r for r in session_records if replayable(r, source_dir)]
    print(
        f"replayable: {len(replay)} "
        f"(skipping {len(session_records) - len(replay)} without audio)"
    )
    print(f"output: {out_dir}\n")

    if args.dry_run:
        for r in replay:
            print(f"  {r['timestamp_utc'][:19]} {r['audio_file']} {r.get('target_text')!r}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    produced = rerun(
        session_records, source_dir=source_dir, out_dir=out_dir, session_prefix=args.session
    )

    compare(session_records, produced, f"session {args.session} : before vs after (same audio)")
    print(f"\nbefore: {outcome_counts(session_records)}")
    print(f"after : {outcome_counts(produced)}")

    if args.compare_with:
        later = select_session(records, args.compare_with, args.participant)
        print(f"\nlive session {args.compare_with}: {outcome_counts(later)}")


if __name__ == "__main__":
    main()
