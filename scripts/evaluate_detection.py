"""CLI: speechocean762 detection evaluation (docs/design_eval.md §4, §8).

Three stages, run separately so a failed/changed run doesn't force redoing
model inference (§5):

  sample     -- pick a reproducible utterance subset from scores.json
  recognize  -- stage1: run phoneme recognition, append to a resumable JSONL
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
    audio_path,
    is_child,
    parse_scores,
    parse_spk2age,
    parse_utt2spk,
    stratified_sample,
)
from pronunciation_coach.g2p import (
    normalize,
    to_phonemes_by_word,
    to_phonemes_by_word_many,
)
from pronunciation_coach.pipeline import PhonemeRecognizer, extract_errors


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


def run_stage1(
    utterances: list[UtteranceAnnotation],
    wave_root: Path,
    recognizer: PhonemeRecognizer,
    out_path: Path,
) -> tuple[int, list[str]]:
    """Recognize phonemes for utterances not already in out_path.

    Appends one JSON line per utterance and flushes immediately, so an
    interruption loses at most the in-flight utterance (§5): rerunning with
    the same out_path skips whatever's already recorded there.
    """
    already_done = set(load_hyp_phonemes(out_path).keys()) if out_path.exists() else set()
    skipped: list[str] = []
    newly_processed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for utt in utterances:
            if utt.utt_id in already_done:
                continue
            path = audio_path(utt, wave_root)
            if not path.exists():
                skipped.append(f"{utt.utt_id}: audio_missing: {path}")
                continue
            try:
                phonemes = recognizer.recognize(path)
            except Exception as e:
                # Model/audio failures vary widely (torch runtime errors, a
                # corrupt WAV); skip this utterance rather than losing every
                # already-recognized one still pending in the batch.
                skipped.append(f"{utt.utt_id}: recognize_failed: {e}")
                continue
            f.write(json.dumps({"utt_id": utt.utt_id, "phonemes": phonemes}, ensure_ascii=False) + "\n")
            f.flush()
            newly_processed += 1
    return newly_processed, skipped


def _load_two_column_mapping(paths: list[Path], parse) -> dict:
    mapping: dict = {}
    for p in paths:
        mapping.update(parse(p.read_text(encoding="utf-8")))
    return mapping


def _load_utterances(
    scores_json: Path, utt2spk_paths: list[Path], utt_ids: Path | None
) -> list[UtteranceAnnotation]:
    """Shared loader for all three subcommands: scores.json + speaker lookup
    (utt id does not embed speaker id -- see so762.parse_utt2spk) + optional
    utt-id restriction (e.g. a test-split or sampled subset list)."""
    raw = json.loads(scores_json.read_text(encoding="utf-8"))
    speaker_by_utt = _load_two_column_mapping(utt2spk_paths, parse_utt2spk)
    utterances = parse_scores(raw, speaker_by_utt)
    if utt_ids is not None:
        wanted = set(utt_ids.read_text(encoding="utf-8").split())
        utterances = [u for u in utterances if u.utt_id in wanted]
    return utterances


def age_group_breakdown(
    utterances: list[UtteranceAnnotation],
    hyp_by_utt: dict[str, list[str]],
    threshold: float,
    spk2age_paths: list[Path],
) -> dict:
    """Score child/adult subsets separately (docs/design_eval.md §1.3 DEI note).

    Reuses run_stage2 on each speaker-age partition of `utterances`, against
    the same already-recognized hyp_by_utt -- no stage1 rerun needed.
    """
    age_by_speaker = _load_two_column_mapping(spk2age_paths, parse_spk2age)
    groups: dict[str, list[UtteranceAnnotation]] = {"child": [], "adult": []}
    unknown_age = 0
    for utt in utterances:
        age = age_by_speaker.get(utt.speaker_id)
        if age is None:
            unknown_age += 1
            continue
        groups["child" if is_child(age) else "adult"].append(utt)

    breakdown: dict = {}
    for group_name, group_utterances in groups.items():
        result, _, skipped = run_stage2(group_utterances, hyp_by_utt, threshold)
        breakdown[group_name] = {
            "n_utterances": len(group_utterances),
            "n_scored": result.insertion_stats.utterance_count,
            "n_skipped": len(skipped),
            "far": result.far,
            "frr": result.frr,
            "confusion": asdict(result.confusion),
        }
    if unknown_age:
        breakdown["n_utterances_unknown_age"] = unknown_age
    return breakdown


def _cmd_sample(args: argparse.Namespace) -> None:
    # e.g. a test-split utt-id list extracted from the corpus's test/utt2spk,
    # so sampling draws from test only (docs/design_eval.md §1.3), not
    # scores.json's combined train+test 5000.
    utterances = _load_utterances(args.scores_json, args.utt2spk, args.utt_ids)
    selected = stratified_sample(utterances, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(u.utt_id for u in selected) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} utt ids to {args.out}")


def _cmd_recognize(args: argparse.Namespace) -> None:
    utterances = _load_utterances(args.scores_json, args.utt2spk, args.utt_ids)

    # Imported lazily: recognize is the only subcommand needing torch/transformers;
    # sample/score stay usable without loading that heavy dependency.
    from pronunciation_coach.phoneme_recognizer import Wav2Vec2PhonemeRecognizer

    kwargs = {"model_name": args.model_name} if args.model_name else {}
    recognizer = Wav2Vec2PhonemeRecognizer(**kwargs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    newly, skipped = run_stage1(utterances, args.wave_root, recognizer, args.out)
    already = len(utterances) - newly - len(skipped)
    print(f"recognized {newly} new utterances ({already} already done, {len(skipped)} skipped)")
    if skipped:
        skipped_path = args.out.parent / "recognize_skipped.txt"
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
        print(f"see {skipped_path}", file=sys.stderr)


def _cmd_score(args: argparse.Namespace) -> None:
    utterances = _load_utterances(args.scores_json, args.utt2spk, args.utt_ids)

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

    metrics_payload = to_json_dict(result, run_metadata)
    if args.spk2age:
        metrics_payload["age_breakdown"] = age_group_breakdown(
            utterances, hyp_by_utt, args.threshold, args.spk2age
        )

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
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
    p_sample.add_argument(
        "--utt2spk", type=Path, nargs="+", required=True,
        help="one or more Kaldi utt2spk files (e.g. train/utt2spk test/utt2spk)",
    )
    p_sample.add_argument("--utt-ids", type=Path, default=None, help="restrict pool to these utt ids")
    p_sample.add_argument("--n", type=int, default=300)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument("--out", type=Path, required=True)
    p_sample.set_defaults(func=_cmd_sample)

    p_recognize = sub.add_parser("recognize", help="stage1: phoneme recognition (resumable JSONL)")
    p_recognize.add_argument("--scores-json", type=Path, required=True)
    p_recognize.add_argument("--utt2spk", type=Path, nargs="+", required=True)
    p_recognize.add_argument("--utt-ids", type=Path, default=None, help="restrict to these utt ids")
    p_recognize.add_argument("--wave-root", type=Path, required=True, help="speechocean762 WAVE/ dir")
    p_recognize.add_argument("--out", type=Path, required=True, help="resumable hyp_phonemes.jsonl path")
    p_recognize.add_argument("--model-name", default=None, help="override the wav2vec2 model name")
    p_recognize.set_defaults(func=_cmd_recognize)

    p_score = sub.add_parser("score", help="stage2: score stage1 output against ground truth")
    p_score.add_argument("--scores-json", type=Path, required=True)
    p_score.add_argument("--utt2spk", type=Path, nargs="+", required=True)
    p_score.add_argument("--hyp-jsonl", type=Path, required=True)
    p_score.add_argument("--utt-ids", type=Path, default=None, help="restrict to these utt ids")
    p_score.add_argument("--out-dir", type=Path, required=True)
    p_score.add_argument(
        "--spk2age", type=Path, nargs="+", default=None,
        help="optional: one or more spk2age files, adds a child/adult metrics breakdown",
    )
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
