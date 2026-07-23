"""Ground-truth/system matching and FAR/FRR/DER computation (docs/design_eval.md §3).

Pure functions only; no file or model I/O. build_utterance_judgements is the
"matching" step that combines one utterance's ground truth, this system's
espeak reference, and its detected PhonemeErrors into per-phone judgements;
the compute_* functions turn those judgements into the reported metrics.
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

# Confirmed against the real corpus (2026-07-22, docs/devlog.md): speechocean762
# has no pronounced-phone (or equivalent) annotation of what a mispronounced
# phone was actually replaced with, so DER's adapted definition (design_eval.md
# §3.2 -- compare a detected substitution's content against ground truth) has
# nothing to compare against here; der_counts.eligible is always 0 on this
# corpus. The computation is kept (not removed) for a possible future
# L2-ARCTIC extension where such annotations may exist. Surfaced in both
# metrics.json and the markdown report so the 0.000 is never misread as "the
# detector's diagnoses were all correct".
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
    """Match ground truth against this system's detected errors, one utterance's worth.

    reference_word_spans must be g2p.to_phonemes_by_word(utt.text.lower())
    normalized per word, in the same word order as utt.words (§2.3): the
    caller is responsible for running g2p, since that requires the real
    phonemizer/espeak-ng dependency this module deliberately avoids.
    """
    if len(utt.words) != len(reference_word_spans):
        raise ValueError(
            f"{utt.utt_id}: ground truth has {len(utt.words)} words but the "
            f"system reference has {len(reference_word_spans)}"
        )

    # Insertions never consume a reference position (pipeline.extract_errors
    # attributes them to the *preceding* position), so they can collide with
    # a substitution/deletion's position; only sub/deletion keys are usable
    # for the per-position lookup below.
    flaggable_by_position = {e.position: e for e in errors if e.op in ("substitution", "deletion")}
    insertions = [e for e in errors if e.op == "insertion" and e.actual is not None]

    judgements: list[PhonemeJudgement] = []
    ref_offset = 0
    for word_annotation, (_, ref_phones) in zip(utt.words, reference_word_spans):
        position_map = map_word_positions(word_annotation.phones, ref_phones)
        for local_i, gt_phone in enumerate(word_annotation.phones):
            accuracy = word_annotation.phones_accuracy[local_i]
            mispronounced = is_mispronounced(accuracy, threshold)
            gt_mispron = next(
                (m for m in word_annotation.mispronunciations if m.index == local_i), None
            )
            pronounced_phone = gt_mispron.pronounced_phone if gt_mispron else None

            local_ref_pos = position_map.gt_to_reference[local_i]
            if local_ref_pos is None:
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
    denom = counts.false_accept + counts.true_reject
    return counts.false_accept / denom if denom else 0.0


def compute_frr(counts: ConfusionCounts) -> float:
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
    """(reference_phone, hyp_phone) pairs behind false rejects, most frequent first.

    A false reject is a position where ground truth says the phone was
    pronounced correctly but the system flagged it anyway. High-frequency
    pairs are candidates for manual review: some may be notation variants
    missing from g2p.EQUIVALENCE_CLASSES (docs/design_eval.md known-difficulty
    list); others may be genuine over-detection by the recognizer. This
    function only counts -- classifying a pair is a human judgment call
    against EQUIVALENCE_CLASSES' inclusion criterion, not automated here.
    """
    return _top_phone_pairs(judgements, lambda j: not j.ground_truth_mispronounced, n)


def top_true_reject_pairs(judgements: list[PhonemeJudgement], n: int = 20) -> list[PhonePairCount]:
    """(reference_phone, hyp_phone) pairs behind true rejects, most frequent first.

    A true reject is a position where ground truth says the phone was
    mispronounced and the system correctly flagged it. Comparing this
    distribution against top_false_reject_pairs for the same (reference,
    hyp) pair tests whether a high false-reject count reflects genuine
    rater leniency on borderline cases (the same pair appears often on both
    sides) versus the model being systematically too strict for that pair
    (it appears almost only on the false-reject side).
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

    run_metadata carries execution conditions (subset, seed, model versions,
    threshold, git commit -- see docs/design_eval.md §4) so a result can be
    interpreted and reproduced without cross-referencing the run that
    produced it.
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
