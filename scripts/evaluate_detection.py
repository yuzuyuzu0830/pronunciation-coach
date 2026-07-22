"""CLI: speechocean762 detection evaluation (docs/design_eval.md §4, §8).

Three stages, run separately so a failed/changed run doesn't force redoing
model inference (§5):

  sample     -- pick a reproducible utterance subset from scores.json
  recognize  -- stage1: run phoneme recognition, append to a resumable JSONL
                (NOT YET IMPLEMENTED -- data acquisition is step 5 of §8)
  score      -- stage2: pure scoring against ground truth, no model needed
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pronunciation_coach.aligner import align_phonemes
from pronunciation_coach.evaluation.metrics import (
    InsertionRecord,
    MetricsResult,
    PhonemeJudgement,
    build_utterance_judgements,
    compute_metrics,
    to_json_dict,
)
from pronunciation_coach.evaluation.report import render_report
from pronunciation_coach.evaluation.so762 import (
    ACCURACY_THRESHOLD_DEFAULT,
    UtteranceAnnotation,
    parse_scores,
    stratified_sample,
)
from pronunciation_coach.g2p import (
    normalize,
    to_phonemes_by_word,
    to_phonemes_by_word_many,
)
from pronunciation_coach.pipeline import extract_errors


def _git_commit_short() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip()
    except Exception:
        return None


def load_hyp_phonemes(path: Path) -> dict[str, list[str]]:
    """Load stage1 output: one JSON object per line, {"utt_id": ..., "phonemes": [...]}."""
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result[record["utt_id"]] = record["phonemes"]
    return result


def build_reference_word_spans(text: str) -> list[tuple[str, list[str]]]:
    return [(word, normalize(phones)) for word, phones in to_phonemes_by_word(text.lower())]


@dataclass(frozen=True)
class UtteranceScoreOutcome:
    utt_id: str
    judgements: list[PhonemeJudgement]
    insertions: list[InsertionRecord]
    skipped_reason: str | None


def score_utterance(
    utt: UtteranceAnnotation,
    hyp_phonemes: list[str],
    threshold: float,
    reference_word_spans: list[tuple[str, list[str]]] | None = None,
) -> UtteranceScoreOutcome:
    """Run the detection pipeline's alignment logic (not the full Pipeline --
    no transcriber/explainer needed) and match it against ground truth.

    Prefer passing precomputed ``reference_word_spans`` from a batched g2p
    call (see run_stage2); the per-utterance path is kept for single-utt use.
    """
    if reference_word_spans is None:
        try:
            reference_word_spans = build_reference_word_spans(utt.text)
        except ValueError as e:
            # Word/group count mismatch from to_phonemes_by_word, or ValueError
            # raised inside phonemizer itself.
            return UtteranceScoreOutcome(
                utt.utt_id, [], [], f"g2p_word_count_mismatch: {e}"
            )
        except (RuntimeError, OSError) as e:
            # phonemizer/espeak-ng commonly surfaces backend failures this way;
            # skip the utterance rather than aborting the whole stage2 batch.
            return UtteranceScoreOutcome(utt.utt_id, [], [], f"g2p_failed: {e}")

    reference = [p for _, phones in reference_word_spans for p in phones]
    hypothesis = normalize(hyp_phonemes)
    ops = align_phonemes(reference, hypothesis)
    errors = extract_errors(ops, reference_word_spans)

    try:
        judgements, insertions = build_utterance_judgements(
            utt, reference_word_spans, errors, threshold
        )
    except ValueError as e:
        return UtteranceScoreOutcome(utt.utt_id, [], [], f"word_count_mismatch: {e}")
    return UtteranceScoreOutcome(utt.utt_id, judgements, insertions, None)


def run_stage2(
    utterances: list[UtteranceAnnotation],
    hyp_by_utt: dict[str, list[str]],
    threshold: float,
) -> tuple[MetricsResult, list[PhonemeJudgement], list[str]]:
    all_judgements: list[PhonemeJudgement] = []
    all_insertions: list[InsertionRecord] = []
    skipped: list[str] = []
    scored_count = 0

    pending: list[UtteranceAnnotation] = []
    for utt in utterances:
        if utt.utt_id not in hyp_by_utt:
            skipped.append(f"{utt.utt_id}: missing_from_hyp_jsonl")
            continue
        pending.append(utt)

    # One espeak backend + one phonemize(list) for all texts (phonemizer docs
    # discourage per-line calls that re-init the backend each time).
    g2p_results: list[list[tuple[str, list[str]]] | ValueError] | None
    try:
        g2p_results = to_phonemes_by_word_many([utt.text.lower() for utt in pending])
    except (RuntimeError, OSError) as e:
        for utt in pending:
            skipped.append(f"{utt.utt_id}: g2p_failed: {e}")
        result = compute_metrics([], [], 0)
        return result, [], skipped

    for utt, g2p_result in zip(pending, g2p_results):
        if isinstance(g2p_result, ValueError):
            skipped.append(f"{utt.utt_id}: g2p_word_count_mismatch: {g2p_result}")
            continue
        spans = [(word, normalize(phones)) for word, phones in g2p_result]
        outcome = score_utterance(
            utt, hyp_by_utt[utt.utt_id], threshold, reference_word_spans=spans
        )
        if outcome.skipped_reason is not None:
            skipped.append(f"{outcome.utt_id}: {outcome.skipped_reason}")
            continue
        all_judgements.extend(outcome.judgements)
        all_insertions.extend(outcome.insertions)
        scored_count += 1
    result = compute_metrics(all_judgements, all_insertions, scored_count)
    return result, all_judgements, skipped


def _cmd_sample(args: argparse.Namespace) -> None:
    raw = json.loads(args.scores_json.read_text(encoding="utf-8"))
    utterances = parse_scores(raw)
    selected = stratified_sample(utterances, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(u.utt_id for u in selected) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} utt ids to {args.out}")


def _cmd_recognize(args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "stage1 (phoneme recognition + resumable JSONL) is step 5 of docs/design_eval.md §8, "
        "pending speechocean762 data acquisition. Not implemented yet."
    )


def _cmd_score(args: argparse.Namespace) -> None:
    raw = json.loads(args.scores_json.read_text(encoding="utf-8"))
    utterances = parse_scores(raw)
    if args.utt_ids is not None:
        wanted = set(args.utt_ids.read_text(encoding="utf-8").split())
        utterances = [u for u in utterances if u.utt_id in wanted]

    hyp_by_utt = load_hyp_phonemes(args.hyp_jsonl)
    result, judgements, skipped = run_stage2(utterances, hyp_by_utt, args.threshold)

    if utterances and result.insertion_stats.utterance_count == 0:
        # e.g. espeak missing: every utt skipped as g2p_failed. Refuse to
        # write empty FAR/FRR that look like a successful run.
        msg = (
            f"stage2 scored 0 utterances ({len(skipped)} skipped); "
            "refusing empty metrics"
        )
        print(msg, file=sys.stderr)
        if skipped:
            skipped_path = args.out_dir / "skipped.txt"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
            print(f"see {skipped_path}", file=sys.stderr)
        raise SystemExit(msg)

    run_metadata = {
        "run_id": args.run_id,
        "subset_description": args.subset_description,
        "n_utterances_requested": len(utterances),
        "n_utterances_scored": result.insertion_stats.utterance_count,
        "n_utterances_skipped": len(skipped),
        "seed": args.seed,
        "phoneme_recognizer_model": args.model_name,
        "accuracy_threshold": args.threshold,
        "git_commit": _git_commit_short(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    judgements_path = args.out_dir / "judgements.jsonl"
    with judgements_path.open("w", encoding="utf-8") as f:
        for j in judgements:
            f.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(to_json_dict(result, run_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = args.out_dir / f"detection_eval_{args.run_id}.md"
    report_path.write_text(render_report(result, run_metadata), encoding="utf-8")

    if skipped:
        skipped_path = args.out_dir / "skipped.txt"
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"skipped {len(skipped)} utterances, see {skipped_path}", file=sys.stderr)

    print(f"FAR={result.far:.3f} FRR={result.frr:.3f} DER={result.der:.3f}")
    print(f"wrote {judgements_path}, {metrics_path}, {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="pick a reproducible utterance subset")
    p_sample.add_argument("--scores-json", type=Path, required=True)
    p_sample.add_argument("--n", type=int, default=300)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument("--out", type=Path, required=True)
    p_sample.set_defaults(func=_cmd_sample)

    p_recognize = sub.add_parser("recognize", help="stage1: phoneme recognition (not yet implemented)")
    p_recognize.set_defaults(func=_cmd_recognize)

    p_score = sub.add_parser("score", help="stage2: score stage1 output against ground truth")
    p_score.add_argument("--scores-json", type=Path, required=True)
    p_score.add_argument("--hyp-jsonl", type=Path, required=True)
    p_score.add_argument("--utt-ids", type=Path, default=None, help="restrict to these utt ids")
    p_score.add_argument("--out-dir", type=Path, required=True)
    p_score.add_argument("--threshold", type=float, default=ACCURACY_THRESHOLD_DEFAULT)
    p_score.add_argument("--seed", type=int, default=None, help="for metadata only")
    p_score.add_argument("--run-id", default="run")
    p_score.add_argument("--subset-description", default="?")
    p_score.add_argument("--model-name", default="?")
    p_score.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
