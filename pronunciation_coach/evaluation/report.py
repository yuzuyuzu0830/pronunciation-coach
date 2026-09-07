"""Markdown report rendering for a detection evaluation run.
Pure string formatting only.
"""

from __future__ import annotations

from pronunciation_coach.evaluation.metrics import DER_ADAPTATION_NOTE, MetricsResult, PhonePairCount


def _pair_rows(pairs: list[PhonePairCount]) -> list[str]:
    return [
        f"| {p.reference_phone} | {p.hyp_phone if p.hyp_phone is not None else '(deleted)'} | {p.count} |"
        for p in pairs
    ]


def render_report(result: MetricsResult, run_metadata: dict) -> str:
    c = result.confusion
    dc = result.der_counts
    ins = result.insertion_stats

    lines = [
        f"# Detection evaluation report ({run_metadata.get('run_id', '?')})",
        "",
        "## Run conditions",
        "",
        f"- Subset: {run_metadata.get('subset_description', '?')}",
        f"- Utterances (scored): {ins.utterance_count}",
        f"- Sampling seed: {run_metadata.get('seed', '?')}",
        f"- Phoneme recognizer model: {run_metadata.get('phoneme_recognizer_model', '?')}",
        f"- Mispronunciation threshold (accuracy <): {run_metadata.get('accuracy_threshold', '?')}",
        f"- git commit: {run_metadata.get('git_commit', '?')}",
        "",
        "## Metrics",
        "",
        f"- FAR (miss rate): {result.far:.3f}  [FA={c.false_accept}, TR={c.true_reject}]",
        f"- FRR (false alarm rate): {result.frr:.3f}  [FR={c.false_reject}, TA={c.true_accept}]",
        f"- DER (diagnostic error rate): {result.der:.3f}  [mismatched={dc.mismatched}, "
        f"eligible={dc.eligible}] -- out of scope for this corpus, see note below",
        "",
        "### DER is out of scope for this corpus",
        "",
        DER_ADAPTATION_NOTE,
        "",
        "## Exclusions & coverage",
        "",
        f"- Phones excluded from primary metrics (FAR/FRR) due to unmappable position: {c.excluded} / {c.total}",
        f"- Excluded from DER (ambiguous pronounced-phone such as `<unk>`/`*`): {dc.excluded_ambiguous}",
        f"- Excluded from DER (deletion with no content): {dc.excluded_non_substitution}",
        "",
        "## Insertions (auxiliary; not counted in primary metrics)",
        "",
        f"- Insertions: {ins.total} ({ins.per_utterance_rate:.2f}/utt over {ins.utterance_count} utterances)",
        f"- Of which vowel insertions: {ins.vowel_count}",
        "",
        "## Top false-reject (reference, hyp) pairs",
        "",
        "Most frequent (reference phone, hypothesis phone) pairs behind false rejects "
        "(ground truth says correct, system flagged anyway). Candidates for manual review "
        "against g2p.EQUIVALENCE_CLASSES' inclusion criterion -- not classified here.",
        "",
        "| reference | hyp | count |",
        "|---|---|---|",
        *_pair_rows(result.top_false_rejects),
        "",
        "## Top true-reject (reference, hyp) pairs",
        "",
        "Most frequent (reference phone, hypothesis phone) pairs behind true rejects "
        "(ground truth says mispronounced, system correctly flagged it). Compare against "
        "the false-reject table above for the same pair: appearing on both sides suggests "
        "rater leniency on a borderline case; appearing almost only as a false reject "
        "suggests the model is systematically too strict for that pair.",
        "",
        "| reference | hyp | count |",
        "|---|---|---|",
        *_pair_rows(result.top_true_rejects),
        "",
        "## Known limitations",
        "",
        "- Insertion-type errors (e.g. vowel epenthesis) are not reflected in the primary "
        "metrics (FAR/FRR/DER) (docs/design_eval.md §3.3; same class of position-based "
        "evaluation limit noted in Preliminary Report §4.5).",
        "- The ARPAbet–espeak acceptance table (phone_map.ACCEPTED_ESPEAK) is provisional; "
        "the exclusion counts above and the judgement detail (judgements.jsonl) need visual "
        "review (docs/design_eval.md §7).",
    ]
    return "\n".join(lines)
