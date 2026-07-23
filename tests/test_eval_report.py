"""Unit tests for evaluation markdown report rendering."""

from pronunciation_coach.evaluation.metrics import (
    ConfusionCounts,
    DER_ADAPTATION_NOTE,
    DerCounts,
    InsertionStats,
    MetricsResult,
    PhonePairCount,
)
from pronunciation_coach.evaluation.report import _pair_rows, render_report


def _result(
    *,
    top_false_rejects: list[PhonePairCount] | None = None,
    top_true_rejects: list[PhonePairCount] | None = None,
) -> MetricsResult:
    return MetricsResult(
        confusion=ConfusionCounts(
            true_accept=10,
            false_reject=2,
            false_accept=1,
            true_reject=3,
            excluded=0,
            total=16,
        ),
        far=0.25,
        frr=0.167,
        der=0.0,
        der_counts=DerCounts(
            eligible=0,
            mismatched=0,
            excluded_ambiguous=1,
            excluded_non_substitution=2,
        ),
        insertion_stats=InsertionStats(
            total=4,
            utterance_count=5,
            per_utterance_rate=0.8,
            vowel_count=1,
        ),
        top_false_rejects=top_false_rejects or [],
        top_true_rejects=top_true_rejects or [],
    )


def test_pair_rows_formats_hyp_phone_and_deleted_branch():
    rows = _pair_rows(
        [
            PhonePairCount(reference_phone="ð", hyp_phone="z", count=3),
            PhonePairCount(reference_phone="t", hyp_phone=None, count=2),
        ]
    )
    assert rows == [
        "| ð | z | 3 |",
        "| t | (deleted) | 2 |",
    ]


def test_pair_rows_empty():
    assert _pair_rows([]) == []


def test_render_report_includes_metrics_and_pair_tables():
    result = _result(
        top_false_rejects=[
            PhonePairCount(reference_phone="ð", hyp_phone="z", count=3),
            PhonePairCount(reference_phone="l", hyp_phone=None, count=1),
        ],
        top_true_rejects=[
            PhonePairCount(reference_phone="θ", hyp_phone="s", count=2),
        ],
    )
    report = render_report(
        result,
        {
            "run_id": "unit",
            "subset_description": "fixture subset",
            "seed": 7,
            "phoneme_recognizer_model": "fixture-model",
            "accuracy_threshold": 0.5,
            "git_commit": "abc1234",
        },
    )

    assert report.startswith("# Detection evaluation report (unit)")
    assert "- Subset: fixture subset" in report
    assert "- Utterances (scored): 5" in report
    assert "- Sampling seed: 7" in report
    assert "- Phoneme recognizer model: fixture-model" in report
    assert "- Mispronunciation threshold (accuracy <): 0.5" in report
    assert "- git commit: abc1234" in report

    assert "- FAR (miss rate): 0.250  [FA=1, TR=3]" in report
    assert "- FRR (false alarm rate): 0.167  [FR=2, TA=10]" in report
    assert "DER is out of scope for this corpus" in report
    assert DER_ADAPTATION_NOTE in report

    assert "## Top false-reject (reference, hyp) pairs" in report
    assert "| ð | z | 3 |" in report
    assert "| l | (deleted) | 1 |" in report

    assert "## Top true-reject (reference, hyp) pairs" in report
    assert "| θ | s | 2 |" in report

    assert "- Insertions: 4 (0.80/utt over 5 utterances)" in report
    assert "- Of which vowel insertions: 1" in report


def test_render_report_uses_placeholders_for_missing_metadata():
    report = render_report(_result(), {})
    assert report.startswith("# Detection evaluation report (?)")
    assert "- Subset: ?" in report
    assert "- Sampling seed: ?" in report
    assert "- Phoneme recognizer model: ?" in report
    assert "- Mispronunciation threshold (accuracy <): ?" in report
    assert "- git commit: ?" in report
