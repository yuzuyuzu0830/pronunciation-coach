"""Ground-truth/system matching and FAR/FRR/DER computation.

This module matches corpus annotations to detected errors, 
then aggregates the resulting per-phone judgements. 
It performs no file or model I/O.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

from pronunciation_coach.evaluation.phone_map import (
    ACCEPTED_ESPEAK,
    ESPEAK_VOWELS,
    map_word_positions,
    strip_stress,
)
from pronunciation_coach.evaluation.so762 import (
    ACCURACY_THRESHOLD_DEFAULT,
    UtteranceAnnotation,
    is_mispronounced,
)
from pronunciation_coach.types import PhonemeError

# SpeechOcean762 lacks the pronounced-phone annotations required by the
# adapted DER, so its eligible count is always zero. Keep the computation for
# corpora such as L2-ARCTIC and expose this note with every result.
DER_ADAPTATION_NOTE = (
    "DER is out of scope for speechocean762: the corpus provides no "
    "pronounced-phone (or equivalent) annotation of what a mispronounced "
    "phone was actually replaced with, so there is nothing to compare a "
    "detected substitution's content against (confirmed against the real "
    "corpus, 2026-07-22 -- docs/devlog.md, docs/design_eval.md §7). This "
    "computation is retained for a possible future L2-ARCTIC extension where "
    "such annotations may exist. Diagnostic quality on speechocean762 should "
    "instead be assessed through user trials, not this metric."
)


@dataclass(frozen=True)
class PhonemeJudgement:
    utt_id: str
    word: str
    gt_phone: str  # ARPAbet, as given (may carry a stress digit)
    gt_accuracy: float
    ground_truth_mispronounced: bool
    system_flagged: bool  # True: system emitted a substitution/deletion here
    system_op: Literal["substitution", "deletion"] | None
    system_actual: str | None  # hyp phone (espeak), for DER; None for deletion/accept
    reference_phone: str | None  # espeak reference phone this GT phone mapped to; None if excluded
    pronounced_phone: str | None  # ground truth's reported mispronunciation, if any
    excluded: bool  # True: excluded from FAR/FRR (position could not be mapped)
    exclusion_reason: str | None


@dataclass(frozen=True)
class InsertionRecord:
    utt_id: str
    hyp_phone: str
    is_vowel: bool


def build_utterance_judgements(
    utt: UtteranceAnnotation,
    reference_word_spans: list[tuple[str, list[str]]],
    errors: list[PhonemeError],
    threshold: float = ACCURACY_THRESHOLD_DEFAULT,
) -> tuple[list[PhonemeJudgement], list[InsertionRecord]]:
    """Match one utterance's ground truth against detected errors.

    `reference_word_spans` must be normalized per word and ordered like
    `utt.words`; the caller owns the phonemizer dependency.
    """
    if len(utt.words) != len(reference_word_spans):
        raise ValueError(
            f"{utt.utt_id}: ground truth has {len(utt.words)} words but the "
            f"system reference has {len(reference_word_spans)}"
        )

    # Insertions use the preceding position without consuming it, so only
    # substitutions and deletions can represent a per-position decision.
    flaggable_by_position = {e.position: e for e in errors if e.op in ("substitution", "deletion")}
    insertions = [e for e in errors if e.op == "insertion" and e.actual is not None]

    judgements: list[PhonemeJudgement] = []
    ref_offset = 0
    # Align each annotated word to its espeak reference before comparing
    # corpus phone labels with detector positions.
    for word_annotation, (_, ref_phones) in zip(utt.words, reference_word_spans):
        position_map = map_word_positions(word_annotation.phones, ref_phones)
        for local_i, gt_phone in enumerate(word_annotation.phones):
            # The accuracy score supplies the expected accept/reject decision;
            # pronounced_phone, when present, is used separately for DER.
            accuracy = word_annotation.phones_accuracy[local_i]
            mispronounced = is_mispronounced(accuracy, threshold)
            gt_mispron = next(
                (m for m in word_annotation.mispronunciations if m.index == local_i), None
            )
            pronounced_phone = gt_mispron.pronounced_phone if gt_mispron else None

            local_ref_pos = position_map.gt_to_reference[local_i]
            if local_ref_pos is None:
                # A corpus phone with no espeak counterpart has no detector position,
                # so it cannot contribute to FAR or FRR.
                judgements.append(
                    PhonemeJudgement(
                        utt_id=utt.utt_id,
                        word=word_annotation.text,
                        gt_phone=gt_phone,
                        gt_accuracy=accuracy,
                        ground_truth_mispronounced=mispronounced,
                        system_flagged=False,
                        system_op=None,
                        system_actual=None,
                        reference_phone=None,
                        pronounced_phone=pronounced_phone,
                        excluded=True,
                        exclusion_reason="unmapped_position",
                    )
                )
                continue

            # Detector positions span the full utterance; convert the local
            # word position with ref_offset before looking up a flagged error.
            error = flaggable_by_position.get(ref_offset + local_ref_pos)
            flagged = error is not None
            judgements.append(
                PhonemeJudgement(
                    utt_id=utt.utt_id,
                    word=word_annotation.text,
                    gt_phone=gt_phone,
                    gt_accuracy=accuracy,
                    ground_truth_mispronounced=mispronounced,
                    system_flagged=flagged,
                    system_op=error.op if error else None,  # type: ignore[arg-type]
                    system_actual=error.actual if error else None,
                    reference_phone=ref_phones[local_ref_pos],
                    pronounced_phone=pronounced_phone,
                    excluded=False,
                    exclusion_reason=None,
                )
            )
        ref_offset += len(ref_phones)

    # Insertions have no ground-truth phone position, so report their rate
    # separately rather than forcing them into the confusion counts.
    insertion_records = [
        InsertionRecord(utt.utt_id, e.actual, e.actual in ESPEAK_VOWELS) for e in insertions
    ]
    return judgements, insertion_records


@dataclass(frozen=True)
class ConfusionCounts:
    true_accept: int
    false_reject: int
    false_accept: int
    true_reject: int
    excluded: int
    total: int


def compute_confusion(judgements: list[PhonemeJudgement]) -> ConfusionCounts:
    """Count system decisions against ground truth, excluding unmapped phones."""
    ta = fr = fa = tr = excluded = 0
    for j in judgements:
        if j.excluded:
            excluded += 1
        elif j.ground_truth_mispronounced:
            tr += 1 if j.system_flagged else 0
            fa += 0 if j.system_flagged else 1
        else:
            fr += 1 if j.system_flagged else 0
            ta += 0 if j.system_flagged else 1
    return ConfusionCounts(ta, fr, fa, tr, excluded, len(judgements))


def compute_far(counts: ConfusionCounts) -> float:
    """Return the fraction of mispronounced phones the system accepted."""
    denom = counts.false_accept + counts.true_reject
    return counts.false_accept / denom if denom else 0.0


def compute_frr(counts: ConfusionCounts) -> float:
    """Return the fraction of correct phones the system rejected."""
    denom = counts.false_reject + counts.true_accept
    return counts.false_reject / denom if denom else 0.0


_AMBIGUOUS_PRONOUNCED_MARKERS = frozenset({"<unk>"})


def _is_ambiguous_pronounced_phone(value: str) -> bool:
    # "R*" etc.: corpus marks a near-miss it couldn't pin to one phone.
    return value in _AMBIGUOUS_PRONOUNCED_MARKERS or value.endswith("*")


@dataclass(frozen=True)
class DerCounts:
    eligible: int  # true-reject substitutions with an unambiguous ground-truth phone
    mismatched: int
    excluded_ambiguous: int  # pronounced-phone was "<unk>"/"R*"-style
    excluded_non_substitution: int  # true-reject deletions: no "content" to compare


def compute_der(judgements: list[PhonemeJudgement]) -> tuple[float, DerCounts]:
    """Measure diagnosis mismatches among unambiguous true-reject substitutions."""
    eligible = mismatched = excluded_ambiguous = excluded_non_sub = 0
    for j in judgements:
        if j.excluded or not j.ground_truth_mispronounced or not j.system_flagged:
            continue
        if j.system_op != "substitution":
            excluded_non_sub += 1
            continue
        if j.pronounced_phone is None or _is_ambiguous_pronounced_phone(j.pronounced_phone):
            excluded_ambiguous += 1
            continue
        eligible += 1
        expected_espeak = ACCEPTED_ESPEAK.get(strip_stress(j.pronounced_phone), frozenset())
        if j.system_actual not in expected_espeak:
            mismatched += 1
    der = mismatched / eligible if eligible else 0.0
    return der, DerCounts(eligible, mismatched, excluded_ambiguous, excluded_non_sub)


@dataclass(frozen=True)
class PhonePairCount:
    reference_phone: str  # espeak reference phone
    hyp_phone: str | None  # system's hypothesis phone; None for a deletion
    count: int


def _top_phone_pairs(
    judgements: list[PhonemeJudgement], predicate, n: int
) -> list[PhonePairCount]:
    counts: Counter[tuple[str, str | None]] = Counter(
        (j.reference_phone, j.system_actual)
        for j in judgements
        if not j.excluded and j.system_flagged and predicate(j)
    )
    return [
        PhonePairCount(reference_phone=ref, hyp_phone=hyp, count=count)
        for (ref, hyp), count in counts.most_common(n)
    ]


def top_false_reject_pairs(judgements: list[PhonemeJudgement], n: int = 20) -> list[PhonePairCount]:
    """Return the most frequent phone pairs behind false rejects.

    Frequent pairs require manual review: they may be missing notation
    equivalences or genuine recognizer over-detection.
    """
    return _top_phone_pairs(judgements, lambda j: not j.ground_truth_mispronounced, n)


def top_true_reject_pairs(judgements: list[PhonemeJudgement], n: int = 20) -> list[PhonePairCount]:
    """Return the most frequent phone pairs behind true rejects.

    Compare this distribution with false rejects to distinguish borderline
    annotations from systematically strict detection.
    """
    return _top_phone_pairs(judgements, lambda j: j.ground_truth_mispronounced, n)


@dataclass(frozen=True)
class InsertionStats:
    total: int
    utterance_count: int
    per_utterance_rate: float
    vowel_count: int


def summarize_insertions(insertions: list[InsertionRecord], utterance_count: int) -> InsertionStats:
    total = len(insertions)
    vowel = sum(1 for r in insertions if r.is_vowel)
    rate = total / utterance_count if utterance_count else 0.0
    return InsertionStats(total, utterance_count, rate, vowel)


@dataclass(frozen=True)
class MetricsResult:
    confusion: ConfusionCounts
    far: float
    frr: float
    der: float
    der_counts: DerCounts
    insertion_stats: InsertionStats
    top_false_rejects: list[PhonePairCount]
    top_true_rejects: list[PhonePairCount]


def compute_metrics(
    judgements: list[PhonemeJudgement],
    insertions: list[InsertionRecord],
    utterance_count: int,
    top_n: int = 20,
) -> MetricsResult:
    """Aggregate FAR/FRR/DER plus top-N FR/TR pair tables (same N for both)."""
    confusion = compute_confusion(judgements)
    der_value, der_counts = compute_der(judgements)
    return MetricsResult(
        confusion=confusion,
        far=compute_far(confusion),
        frr=compute_frr(confusion),
        der=der_value,
        der_counts=der_counts,
        insertion_stats=summarize_insertions(insertions, utterance_count),
        top_false_rejects=top_false_reject_pairs(judgements, top_n),
        top_true_rejects=top_true_reject_pairs(judgements, top_n),
    )


def to_json_dict(result: MetricsResult, run_metadata: dict) -> dict:
    """Serializable summary for metrics.json.

    Run metadata records the subset, seed, model versions, threshold, and
    commit needed to interpret or reproduce a result.
    """
    return {
        "run_metadata": run_metadata,
        "confusion": asdict(result.confusion),
        "far": result.far,
        "frr": result.frr,
        "der": result.der,
        "der_counts": asdict(result.der_counts),
        "der_adaptation_note": DER_ADAPTATION_NOTE,
        "insertion_stats": asdict(result.insertion_stats),
        "top_false_rejects": [asdict(p) for p in result.top_false_rejects],
        "top_true_rejects": [asdict(p) for p in result.top_true_rejects],
    }
