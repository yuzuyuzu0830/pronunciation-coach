import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except (ImportError, OSError, RuntimeError):
    ESPEAK_AVAILABLE = False

from pronunciation_coach.evaluation.metrics import ConfusionCounts
from pronunciation_coach.evaluation.so762 import UtteranceAnnotation
from scripts import evaluate_detection

requires_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng is not installed"
)

SCORES = {
    "0001010011": {
        "text": "THIS IS HIGH",
        "words": [
            {
                "text": "THIS",
                "phones": ["DH", "IH1", "S"],
                "phones-accuracy": [0.2, 2.0, 2.0],
                "mispronunciations": [
                    {"canonical-phone": "DH", "index": 0, "pronounced-phone": "D"}
                ],
            },
            {
                "text": "IS",
                "phones": ["IH1", "Z"],
                "phones-accuracy": [2.0, 2.0],
                "mispronunciations": [],
            },
            {
                "text": "HIGH",
                "phones": ["HH", "AY1"],
                "phones-accuracy": [0.0, 2.0],
                "mispronunciations": [
                    {"canonical-phone": "HH", "index": 0, "pronounced-phone": "<unk>"}
                ],
            },
        ],
    },
    "0002030022": {
        "text": "WATER",
        "words": [
            {
                "text": "WATER",
                "phones": ["W", "AO1", "T", "ER0"],
                "phones-accuracy": [2.0, 2.0, 2.0, 2.0],
                "mispronunciations": [],
            }
        ],
    },
}

HYP_PHONEMES = {
    "0001010011": ["d", "ɪ", "s", "ɪ", "z", "h", "aɪ"],
    "0002030022": ["w", "ɔː", "ɾ", "ɚ"],
}

# DH→D is detected, the HH error is missed, and WATER is a correct control.

SPEAKER_BY_UTT = {"0001010011": "0001", "0002030022": "0002"}

AGE_BY_SPEAKER = {"0001": 10, "0002": 25}


@pytest.mark.parametrize(
    "exc",
    [OSError("git unavailable"), subprocess.CalledProcessError(128, ["git"])],
)
def test_git_commit_short_returns_none_for_expected_git_failures(
    exc: Exception,
) -> None:
    with patch.object(evaluate_detection.subprocess, "run", side_effect=exc):
        assert evaluate_detection._git_commit_short() is None


def test_git_commit_short_does_not_hide_unexpected_errors() -> None:
    with (
        patch.object(
            evaluate_detection.subprocess, "run", side_effect=RuntimeError("bug")
        ),
        pytest.raises(RuntimeError, match="bug"),
    ):
        evaluate_detection._git_commit_short()


def _write_spk2age(tmp_path: Path) -> list[Path]:
    path = tmp_path / "spk2age"
    path.write_text(
        "\n".join(f"{spk} {age}" for spk, age in AGE_BY_SPEAKER.items()) + "\n",
        encoding="utf-8",
    )
    return [path]


def _write_utt2spk(tmp_path: Path) -> list[Path]:
    path = tmp_path / "utt2spk"
    path.write_text(
        "\n".join(f"{utt_id} {spk}" for utt_id, spk in SPEAKER_BY_UTT.items()) + "\n",
        encoding="utf-8",
    )
    return [path]


@requires_espeak
def test_run_stage2_computes_expected_metrics() -> None:
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)
    result, judgements, insertions, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )

    assert skipped == []
    assert result.insertion_stats.utterance_count == 2
    assert result.far == pytest.approx(0.5)
    assert result.frr == pytest.approx(0.0)
    assert result.der == pytest.approx(0.0)
    assert result.der_counts.eligible == 1
    assert result.confusion == ConfusionCounts(
        true_accept=9,
        false_reject=0,
        false_accept=1,
        true_reject=1,
        excluded=0,
        total=11,
    )
    assert insertions == []

    dh = next(j for j in judgements if j.gt_phone == "DH")
    assert dh.system_flagged and dh.system_actual == "d"
    hh = next(j for j in judgements if j.gt_phone == "HH")
    assert not hh.system_flagged


@requires_espeak
def test_cmd_score_writes_expected_output_files(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    hyp_path = tmp_path / "hyp_phonemes.jsonl"
    hyp_path.write_text(
        "\n".join(
            json.dumps({"utt_id": utt_id, "phonemes": phones})
            for utt_id, phones in HYP_PHONEMES.items()
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        utt2spk=_write_utt2spk(tmp_path),
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
        spk2age=None,
        threshold=0.5,
        seed=0,
        run_id="test_run",
        subset_description="fixture",
        model_name="fixture-model",
    )
    evaluate_detection._cmd_score(args)

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["far"] == pytest.approx(0.5)
    assert metrics["run_metadata"]["run_id"] == "test_run"
    assert "der_adaptation_note" in metrics

    judgement_lines = (
        (out_dir / "judgements.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(judgement_lines) == 11

    report = (out_dir / "detection_eval_test_run.md").read_text(encoding="utf-8")
    assert "DER is out of scope for this corpus" in report
    assert not (out_dir / "skipped.txt").exists()


# Age-group breakdown


@requires_espeak
def test_age_group_breakdown_splits_by_child_adult(tmp_path: Path) -> None:
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)
    _, judgements, insertions, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )
    breakdown = evaluate_detection.age_group_breakdown(
        utterances,
        judgements,
        insertions,
        skipped,
        spk2age_paths=_write_spk2age(tmp_path),
    )
    assert breakdown["child"]["n_utterances"] == 1
    assert breakdown["adult"]["n_utterances"] == 1
    assert breakdown["child"]["n_scored"] == 1
    assert breakdown["adult"]["n_scored"] == 1
    assert "n_utterances_unknown_age" not in breakdown


@requires_espeak
def test_age_group_breakdown_counts_unknown_age_speakers(tmp_path: Path) -> None:
    spk2age_path = tmp_path / "spk2age"
    spk2age_path.write_text("0001 10\n", encoding="utf-8")
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)
    _, judgements, insertions, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )
    breakdown = evaluate_detection.age_group_breakdown(
        utterances, judgements, insertions, skipped, spk2age_paths=[spk2age_path]
    )
    assert breakdown["n_utterances_unknown_age"] == 1
    assert breakdown["adult"]["n_utterances"] == 0


def test_age_group_breakdown_reuses_judgements_without_rerunning_stage2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)
    judgements = [
        evaluate_detection.PhonemeJudgement(
            utt_id="0001010011",
            word="THIS",
            gt_phone="DH",
            gt_accuracy=0.2,
            ground_truth_mispronounced=True,
            system_flagged=True,
            system_op="substitution",
            system_actual="d",
            reference_phone="ð",
            pronounced_phone="D",
            excluded=False,
            exclusion_reason=None,
        ),
        evaluate_detection.PhonemeJudgement(
            utt_id="0002030022",
            word="WATER",
            gt_phone="W",
            gt_accuracy=2.0,
            ground_truth_mispronounced=False,
            system_flagged=False,
            system_op=None,
            system_actual=None,
            reference_phone="w",
            pronounced_phone=None,
            excluded=False,
            exclusion_reason=None,
        ),
    ]

    def fail_stage2(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("age_group_breakdown must not call run_stage2")

    def fail_g2p(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("age_group_breakdown must not call g2p")

    monkeypatch.setattr(evaluate_detection, "run_stage2", fail_stage2)
    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", fail_g2p)

    breakdown = evaluate_detection.age_group_breakdown(
        utterances,
        judgements,
        insertions=[],
        skipped=[],
        spk2age_paths=_write_spk2age(tmp_path),
    )
    assert breakdown["child"]["n_scored"] == 1
    assert breakdown["adult"]["n_scored"] == 1
    assert breakdown["child"]["confusion"]["true_reject"] == 1
    assert breakdown["adult"]["confusion"]["true_accept"] == 1


@requires_espeak
def test_cmd_score_with_spk2age_adds_age_breakdown_to_metrics(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    hyp_path = tmp_path / "hyp_phonemes.jsonl"
    hyp_path.write_text(
        "\n".join(
            json.dumps({"utt_id": utt_id, "phonemes": phones})
            for utt_id, phones in HYP_PHONEMES.items()
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        utt2spk=_write_utt2spk(tmp_path),
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
        spk2age=_write_spk2age(tmp_path),
        threshold=0.5,
        seed=0,
        run_id="with_age",
        subset_description="fixture",
        model_name="fixture-model",
    )
    evaluate_detection._cmd_score(args)

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["age_breakdown"]["child"]["n_utterances"] == 1
    assert metrics["age_breakdown"]["adult"]["n_utterances"] == 1


@requires_espeak
def test_cmd_score_reports_skipped_utterances_missing_from_hyp_jsonl(
    tmp_path: Path,
) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    hyp_path = tmp_path / "hyp_phonemes.jsonl"
    hyp_path.write_text(
        json.dumps({"utt_id": "0002030022", "phonemes": HYP_PHONEMES["0002030022"]}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        utt2spk=_write_utt2spk(tmp_path),
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
        spk2age=None,
        threshold=0.5,
        seed=0,
        run_id="partial",
        subset_description="fixture",
        model_name="fixture-model",
    )
    evaluate_detection._cmd_score(args)

    skipped = (out_dir / "skipped.txt").read_text(encoding="utf-8")
    assert "0001010011" in skipped


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("espeak backend failed"),
        OSError("libespeak not found"),
    ],
)
def test_run_stage2_skips_all_when_batched_g2p_raises(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)

    def boom(_texts: list[str]) -> None:
        raise exc

    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", boom)

    result, judgements, insertions, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )

    assert result.insertion_stats.utterance_count == 0
    assert judgements == []
    assert insertions == []
    assert len(skipped) == 2
    assert all("g2p_failed" in s for s in skipped)


def test_run_stage2_skips_utterance_on_per_text_word_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)

    def fake_many(texts: list[str]) -> list[ValueError | list[tuple[str, list[str]]]]:
        assert len(texts) == 2
        return [
            ValueError("word count mismatch"),
            [("water", ["w", "ɔː", "ɾ", "ɚ"])],
        ]

    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", fake_many)

    result, judgements, _, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )

    assert len(skipped) == 1
    assert skipped[0].startswith("0001010011: g2p_word_count_mismatch")
    assert result.insertion_stats.utterance_count == 1
    assert {judgement.utt_id for judgement in judgements} == {"0002030022"}


# Test-split sampling


def test_cmd_sample_restricts_to_utt_ids_filter(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    utt_ids_path = tmp_path / "test_split.txt"
    utt_ids_path.write_text("0002030022\n", encoding="utf-8")
    out_path = tmp_path / "sample_out.txt"

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        utt2spk=_write_utt2spk(tmp_path),
        utt_ids=utt_ids_path,
        n=1,
        seed=0,
        out=out_path,
    )
    evaluate_detection._cmd_sample(args)

    assert out_path.read_text(encoding="utf-8").strip() == "0002030022"


# Recognition and resume behavior


class _FakeRecognizer:
    def __init__(
        self,
        phonemes_by_utt: dict[str, list[str]] | None = None,
        raise_for: frozenset[str] = frozenset(),
    ) -> None:
        self.phonemes_by_utt = phonemes_by_utt or {}
        self.raise_for = raise_for
        self.calls: list[Path] = []

    def recognize(self, path: Path) -> list[str]:
        self.calls.append(path)
        utt_id = path.stem
        if utt_id in self.raise_for:
            raise RuntimeError(f"boom on {utt_id}")
        return self.phonemes_by_utt.get(utt_id, ["x", "y"])


def _make_utterances_with_audio(
    tmp_path: Path,
    utt_ids: list[str],
    missing: frozenset[str] = frozenset(),
) -> tuple[list[UtteranceAnnotation], Path]:
    wave_root = tmp_path / "WAVE"
    utterances: list[UtteranceAnnotation] = []
    for utt_id in utt_ids:
        speaker_id = utt_id[:4]
        utterances.append(
            evaluate_detection.UtteranceAnnotation(
                utt_id=utt_id, speaker_id=speaker_id, text="X", words=[]
            )
        )
        if utt_id in missing:
            continue
        speaker_dir = wave_root / f"SPEAKER{speaker_id}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        (speaker_dir / f"{utt_id}.WAV").write_bytes(b"fake-audio")
    return utterances, wave_root


def test_run_stage1_recognizes_all_and_writes_jsonl(tmp_path: Path) -> None:
    utt_ids = ["0001000001", "0001000002", "0003000001"]
    utterances, wave_root = _make_utterances_with_audio(tmp_path, utt_ids)
    recognizer = _FakeRecognizer({"0001000001": ["a", "b"], "0003000001": ["c"]})
    out_path = tmp_path / "hyp_phonemes.jsonl"

    newly, skipped = evaluate_detection.run_stage1(
        utterances, wave_root, recognizer, out_path
    )

    assert newly == 3
    assert skipped == []
    hyp = evaluate_detection.load_hyp_phonemes(out_path)
    assert hyp == {
        "0001000001": ["a", "b"],
        "0001000002": ["x", "y"],
        "0003000001": ["c"],
    }
    assert len(recognizer.calls) == 3


def test_run_stage1_resumes_and_skips_already_processed(tmp_path: Path) -> None:
    utt_ids = ["0001000001", "0001000002", "0001000003"]
    utterances, wave_root = _make_utterances_with_audio(tmp_path, utt_ids)
    out_path = tmp_path / "hyp_phonemes.jsonl"

    first_recognizer = _FakeRecognizer(raise_for=frozenset({"0001000003"}))
    newly1, skipped1 = evaluate_detection.run_stage1(
        utterances, wave_root, first_recognizer, out_path
    )
    assert newly1 == 2
    assert skipped1 == ["0001000003: recognize_failed: boom on 0001000003"]
    assert set(first_recognizer.calls) == {
        wave_root / "SPEAKER0001" / "0001000001.WAV",
        wave_root / "SPEAKER0001" / "0001000002.WAV",
        wave_root / "SPEAKER0001" / "0001000003.WAV",
    }

    second_recognizer = _FakeRecognizer(
        raise_for=frozenset({"0001000001", "0001000002", "0001000003"})
    )
    newly2, skipped2 = evaluate_detection.run_stage1(
        utterances, wave_root, second_recognizer, out_path
    )

    assert newly2 == 0
    assert skipped2 == ["0001000003: recognize_failed: boom on 0001000003"]
    assert second_recognizer.calls == [wave_root / "SPEAKER0001" / "0001000003.WAV"]

    hyp = evaluate_detection.load_hyp_phonemes(out_path)
    assert set(hyp) == {"0001000001", "0001000002"}


def test_run_stage1_skips_missing_audio_file(tmp_path: Path) -> None:
    utt_ids = ["0001000001", "0001000002"]
    utterances, wave_root = _make_utterances_with_audio(
        tmp_path, utt_ids, missing=frozenset({"0001000002"})
    )
    recognizer = _FakeRecognizer()
    out_path = tmp_path / "hyp_phonemes.jsonl"

    newly, skipped = evaluate_detection.run_stage1(
        utterances, wave_root, recognizer, out_path
    )

    assert newly == 1
    assert len(skipped) == 1
    assert skipped[0].startswith("0001000002: audio_missing:")
    assert recognizer.calls == [wave_root / "SPEAKER0001" / "0001000001.WAV"]


def test_run_stage1_skips_on_recognizer_exception_without_aborting_others(
    tmp_path: Path,
) -> None:
    utt_ids = ["0001000001", "0001000002", "0001000003"]
    utterances, wave_root = _make_utterances_with_audio(tmp_path, utt_ids)
    recognizer = _FakeRecognizer(raise_for=frozenset({"0001000002"}))
    out_path = tmp_path / "hyp_phonemes.jsonl"

    newly, skipped = evaluate_detection.run_stage1(
        utterances, wave_root, recognizer, out_path
    )

    assert newly == 2
    assert skipped == ["0001000002: recognize_failed: boom on 0001000002"]
    hyp = evaluate_detection.load_hyp_phonemes(out_path)
    assert set(hyp) == {"0001000001", "0001000003"}


def test_cmd_score_exits_when_every_utterance_g2p_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    hyp_path = tmp_path / "hyp_phonemes.jsonl"
    hyp_path.write_text(
        "\n".join(
            json.dumps({"utt_id": utt_id, "phonemes": phones})
            for utt_id, phones in HYP_PHONEMES.items()
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"

    def boom(_texts: list[str]) -> None:
        raise RuntimeError("espeak unavailable")

    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", boom)

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        utt2spk=_write_utt2spk(tmp_path),
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
        spk2age=None,
        threshold=0.5,
        seed=0,
        run_id="empty",
        subset_description="fixture",
        model_name="fixture-model",
    )
    with pytest.raises(SystemExit, match="scored 0 utterances"):
        evaluate_detection._cmd_score(args)
