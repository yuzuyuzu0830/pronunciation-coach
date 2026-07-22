import pytest

from pronunciation_coach.evaluation.metrics import (
    ConfusionCounts,
    DER_ADAPTATION_NOTE,
    InsertionRecord,
    PhonemeJudgement,
    build_utterance_judgements,
    compute_confusion,
    compute_der,
    compute_far,
    compute_frr,
    compute_metrics,
    summarize_insertions,
    to_json_dict,
)
from pronunciation_coach.evaluation.so762 import (
    Mispronunciation,
    UtteranceAnnotation,
    WordAnnotation,
)
from pronunciation_coach.types import PhonemeError

# --- build_utterance_judgements ---
#
# Worked example: "this is high" read with a correctly-flagged ð/d error on
# "this" (DH) and a missed error on "high" (HH is mispronounced per ground
# truth but the system's error list has nothing there).
REFERENCE_WORD_SPANS = [
    ("this", ["ð", "ɪ", "s"]),
    ("is", ["ɪ", "z"]),
    ("high", ["h", "aɪ"]),
]


def make_utterance() -> UtteranceAnnotation:
    return UtteranceAnnotation(
        utt_id="0001000001",
        speaker_id="0001",
        text="THIS IS HIGH",
        words=[
            WordAnnotation(
                text="THIS",
                phones=["DH", "IH1", "S"],
                phones_accuracy=[0.2, 2.0, 2.0],
                mispronunciations=[Mispronunciation("DH", 0, "D")],
            ),
            WordAnnotation(
                text="IS",
                phones=["IH1", "Z"],
                phones_accuracy=[2.0, 2.0],
                mispronunciations=[],
            ),
            WordAnnotation(
                text="HIGH",
                phones=["HH", "AY1"],
                phones_accuracy=[0.0, 2.0],
                mispronunciations=[Mispronunciation("HH", 0, "<unk>")],
            ),
        ],
    )


ERRORS = [PhonemeError("substitution", "ð", "d", 0, "this")]


def test_build_utterance_judgements_matches_and_flags():
    judgements, insertions = build_utterance_judgements(make_utterance(), REFERENCE_WORD_SPANS, ERRORS)

    assert insertions == []
    by_phone = {(j.word, j.gt_phone): j for j in judgements}

    dh = by_phone[("THIS", "DH")]
    assert dh.ground_truth_mispronounced is True
    assert dh.system_flagged is True
    assert dh.system_op == "substitution"
    assert dh.system_actual == "d"
    assert dh.pronounced_phone == "D"
    assert dh.excluded is False

    hh = by_phone[("HIGH", "HH")]
    assert hh.ground_truth_mispronounced is True
    assert hh.system_flagged is False  # missed: false accept

    ih1_this = by_phone[("THIS", "IH1")]
    assert ih1_this.ground_truth_mispronounced is False
    assert ih1_this.system_flagged is False  # correctly accepted


def test_build_utterance_judgements_uses_global_reference_positions():
    """A substitution reported at a later word's position must not be picked
    up by an earlier word (offset bookkeeping must be correct)."""
    errors = [PhonemeError("substitution", "aɪ", "a", 6, "high")]  # global position of AY1
    judgements, _ = build_utterance_judgements(make_utterance(), REFERENCE_WORD_SPANS, errors)
    by_phone = {(j.word, j.gt_phone): j for j in judgements}
    assert by_phone[("HIGH", "AY1")].system_flagged is True
    assert by_phone[("THIS", "DH")].system_flagged is False


def test_build_utterance_judgements_raises_on_word_count_mismatch():
    utt = make_utterance()
    with pytest.raises(ValueError, match="words"):
        build_utterance_judgements(utt, REFERENCE_WORD_SPANS[:2], ERRORS)


def test_build_utterance_judgements_marks_unmapped_gt_phone_excluded():
    utt = UtteranceAnnotation(
        utt_id="0003000001",
        speaker_id="0003",
        text="X",
        words=[
            WordAnnotation(
                text="X",
                phones=["T", "IH1"],  # no reference counterpart for T (see test_phone_map)
                phones_accuracy=[2.0, 2.0],
                mispronunciations=[],
            )
        ],
    )
    judgements, _ = build_utterance_judgements(utt, [("x", ["ɪ"])], [])
    t_judgement = next(j for j in judgements if j.gt_phone == "T")
    assert t_judgement.excluded is True
    assert t_judgement.exclusion_reason == "unmapped_position"


def test_build_utterance_judgements_collects_insertions():
    utt = UtteranceAnnotation(
        utt_id="0001000002",
        speaker_id="0001",
        text="HIGH",
        words=[
            WordAnnotation(
                text="HIGH", phones=["HH", "AY1"], phones_accuracy=[2.0, 2.0], mispronunciations=[]
            )
        ],
    )
    errors = [PhonemeError("insertion", None, "ə", 1, "high")]
    _, insertions = build_utterance_judgements(utt, [("high", ["h", "aɪ"])], errors)
    assert insertions == [InsertionRecord("0001000002", "ə", True)]


# --- compute_confusion / compute_far / compute_frr ---


def judgement(mispronounced: bool, flagged: bool, excluded: bool = False) -> PhonemeJudgement:
    return PhonemeJudgement(
        utt_id="u",
        word="w",
        gt_phone="P",
        gt_accuracy=0.0 if mispronounced else 2.0,
        ground_truth_mispronounced=mispronounced,
        system_flagged=flagged,
        system_op="substitution" if flagged else None,
        system_actual="x" if flagged else None,
        pronounced_phone=None,
        excluded=excluded,
        exclusion_reason="unmapped_position" if excluded else None,
    )


def test_compute_confusion_counts_each_quadrant():
    judgements = [
        judgement(mispronounced=False, flagged=False),  # TA
        judgement(mispronounced=False, flagged=False),  # TA
        judgement(mispronounced=False, flagged=True),  # FR
        judgement(mispronounced=True, flagged=False),  # FA
        judgement(mispronounced=True, flagged=True),  # TR
        judgement(mispronounced=True, flagged=True, excluded=True),  # excluded
    ]
    counts = compute_confusion(judgements)
    assert counts == ConfusionCounts(
        true_accept=2, false_reject=1, false_accept=1, true_reject=1, excluded=1, total=6
    )


def test_compute_far_and_frr():
    counts = ConfusionCounts(true_accept=8, false_reject=2, false_accept=1, true_reject=3, excluded=0, total=14)
    assert compute_far(counts) == pytest.approx(1 / 4)
    assert compute_frr(counts) == pytest.approx(2 / 10)


def test_compute_far_and_frr_zero_denominator_is_zero_not_error():
    counts = ConfusionCounts(true_accept=0, false_reject=0, false_accept=0, true_reject=0, excluded=0, total=0)
    assert compute_far(counts) == 0.0
    assert compute_frr(counts) == 0.0


# --- compute_der ---


def der_judgement(
    mispronounced: bool = True,
    flagged: bool = True,
    op: str | None = "substitution",
    actual: str | None = None,
    pronounced_phone: str | None = None,
    excluded: bool = False,
) -> PhonemeJudgement:
    return PhonemeJudgement(
        utt_id="u",
        word="w",
        gt_phone="P",
        gt_accuracy=0.0,
        ground_truth_mispronounced=mispronounced,
        system_flagged=flagged,
        system_op=op,
        system_actual=actual,
        pronounced_phone=pronounced_phone,
        excluded=excluded,
        exclusion_reason=None,
    )


def test_compute_der_correct_diagnosis_is_not_mismatched():
    judgements = [der_judgement(actual="d", pronounced_phone="D")]
    der, counts = compute_der(judgements)
    assert der == 0.0
    assert counts.eligible == 1
    assert counts.mismatched == 0


def test_compute_der_wrong_diagnosis_counts_as_mismatched():
    judgements = [der_judgement(actual="ɛ", pronounced_phone="IY1")]  # expected iː, got ɛ
    der, counts = compute_der(judgements)
    assert der == 1.0
    assert counts.mismatched == 1


def test_compute_der_excludes_ambiguous_pronounced_phone():
    judgements = [
        der_judgement(actual="d", pronounced_phone="<unk>"),
        der_judgement(actual="d", pronounced_phone="R*"),
    ]
    der, counts = compute_der(judgements)
    assert counts.eligible == 0
    assert counts.excluded_ambiguous == 2
    assert der == 0.0  # no eligible cases: defined as 0, not NaN


def test_compute_der_excludes_non_substitution_true_rejects():
    judgements = [der_judgement(op="deletion", pronounced_phone="D")]
    _, counts = compute_der(judgements)
    assert counts.excluded_non_substitution == 1
    assert counts.eligible == 0


def test_compute_der_ignores_false_accepts_and_true_accepts():
    judgements = [
        der_judgement(mispronounced=True, flagged=False),  # false accept: no diagnosis made
        der_judgement(mispronounced=False, flagged=False),  # true accept
    ]
    der, counts = compute_der(judgements)
    assert counts.eligible == 0
    assert der == 0.0


# --- summarize_insertions ---


def test_summarize_insertions_counts_vowels_and_rate():
    insertions = [
        InsertionRecord("u1", "ə", True),
        InsertionRecord("u1", "s", False),
        InsertionRecord("u2", "ɪ", True),
    ]
    stats = summarize_insertions(insertions, utterance_count=2)
    assert stats.total == 3
    assert stats.vowel_count == 2
    assert stats.per_utterance_rate == pytest.approx(1.5)


def test_summarize_insertions_zero_utterances_is_safe():
    stats = summarize_insertions([], utterance_count=0)
    assert stats.per_utterance_rate == 0.0


# --- compute_metrics / to_json_dict integration ---


def test_compute_metrics_and_to_json_dict_shape():
    judgements = [
        judgement(mispronounced=True, flagged=True),
        judgement(mispronounced=False, flagged=False),
    ]
    result = compute_metrics(judgements, insertions=[], utterance_count=1)
    payload = to_json_dict(result, run_metadata={"run_id": "test"})

    assert payload["run_metadata"] == {"run_id": "test"}
    assert payload["far"] == result.far
    assert payload["der_adaptation_note"] == DER_ADAPTATION_NOTE
    assert payload["confusion"]["true_reject"] == 1
    assert "insertion_stats" in payload
