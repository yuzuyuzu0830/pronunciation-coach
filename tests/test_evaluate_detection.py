"""stage2 (scoring) end-to-end test against a small fixture (docs/design_eval.md §8 step 4).

scripts/evaluate_detection.py isn't a package, so it's loaded via importlib
rather than a normal import.
"""

import importlib.util
import json
from pathlib import Path

import pytest

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except Exception:
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng is not installed")

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "evaluate_detection.py"
_spec = importlib.util.spec_from_file_location("evaluate_detection", _SCRIPT_PATH)
evaluate_detection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_detection)

# Ground truth for "THIS IS HIGH" (DH mispronounced as D, HH mispronounced but
# missed by the recognizer) and "WATER" (fully correct), matching real
# espeak-ng output for this text (verified: to_phonemes_by_word("this is
# high") == this/DH,IH1,S is/IH1,Z high/HH,AY1; "water" == W,AO1,T,ER0).
SCORES = {
    "0001010011": {
        "text": "THIS IS HIGH",
        "words": [
            {
                "text": "THIS",
                "phones": ["DH", "IH1", "S"],
                "phones-accuracy": [0.2, 2.0, 2.0],
                "mispronunciations": [{"canonical-phone": "DH", "index": 0, "pronounced-phone": "D"}],
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
                "mispronunciations": [{"canonical-phone": "HH", "index": 0, "pronounced-phone": "<unk>"}],
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

# Recognizer output: "this" read as "dis" (ð->d, the flagged DH error); HIGH's
# HH mispronunciation is NOT caught by the recognizer (hyp says "h", matching
# the reference) -- a realistic false accept. WATER is read perfectly.
HYP_PHONEMES = {
    "0001010011": ["d", "ɪ", "s", "ɪ", "z", "h", "aɪ"],
    "0002030022": ["w", "ɔː", "ɾ", "ɚ"],
}


@requires_espeak
def test_run_stage2_computes_expected_metrics():
    utterances = evaluate_detection.parse_scores(SCORES)
    result, judgements, skipped = evaluate_detection.run_stage2(utterances, HYP_PHONEMES, threshold=0.5)

    assert skipped == []
    assert result.insertion_stats.utterance_count == 2
    # TR=1 (DH correctly flagged), FA=1 (HH missed) -> FAR = 1/2
    assert result.far == pytest.approx(0.5)
    # No false rejects among the 9 true-accept phones -> FRR = 0
    assert result.frr == pytest.approx(0.0)
    # DH's diagnosis (actual "d") matches ground truth's pronounced-phone "D"
    assert result.der == pytest.approx(0.0)
    assert result.der_counts.eligible == 1
    assert result.confusion.total == 11

    dh = next(j for j in judgements if j.gt_phone == "DH")
    assert dh.system_flagged and dh.system_actual == "d"
    hh = next(j for j in judgements if j.gt_phone == "HH")
    assert not hh.system_flagged


@requires_espeak
def test_cmd_score_writes_expected_output_files(tmp_path):
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
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
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

    judgement_lines = (out_dir / "judgements.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(judgement_lines) == 11

    report = (out_dir / "detection_eval_test_run.md").read_text(encoding="utf-8")
    assert "Note on the DER definition" in report
    assert not (out_dir / "skipped.txt").exists()


@requires_espeak
def test_cmd_score_reports_skipped_utterances_missing_from_hyp_jsonl(tmp_path):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    hyp_path = tmp_path / "hyp_phonemes.jsonl"
    # Only include one of the two utterances.
    hyp_path.write_text(
        json.dumps({"utt_id": "0002030022", "phonemes": HYP_PHONEMES["0002030022"]}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
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
def test_run_stage2_skips_all_when_batched_g2p_raises(monkeypatch, exc):
    """Backend-level failure on the batched phonemize skips every pending utt."""
    utterances = evaluate_detection.parse_scores(SCORES)

    def boom(_texts: list[str]):
        raise exc

    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", boom)

    result, judgements, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )

    assert result.insertion_stats.utterance_count == 0
    assert judgements == []
    assert len(skipped) == 2
    assert all("g2p_failed" in s for s in skipped)


def test_run_stage2_skips_utterance_on_per_text_word_count_mismatch(monkeypatch):
    """Word/group mismatches stay per-utterance after the batched phonemize."""
    utterances = evaluate_detection.parse_scores(SCORES)

    def fake_many(texts: list[str]):
        assert len(texts) == 2
        return [
            ValueError("word count mismatch"),
            [("water", ["w", "ɔː", "ɾ", "ɚ"])],
        ]

    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", fake_many)

    result, judgements, skipped = evaluate_detection.run_stage2(
        utterances, HYP_PHONEMES, threshold=0.5
    )

    assert len(skipped) == 1
    assert skipped[0].startswith("0001010011: g2p_word_count_mismatch")
    assert result.insertion_stats.utterance_count == 1
    assert judgements


def test_cmd_score_exits_when_every_utterance_g2p_fails(tmp_path, monkeypatch):
    """All-skip (e.g. espeak missing) must not look like a successful empty run."""
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

    def boom(_texts: list[str]):
        raise RuntimeError("espeak unavailable")

    monkeypatch.setattr(evaluate_detection, "to_phonemes_by_word_many", boom)

    args = evaluate_detection.argparse.Namespace(
        scores_json=scores_path,
        hyp_jsonl=hyp_path,
        utt_ids=None,
        out_dir=out_dir,
        threshold=0.5,
        seed=0,
        run_id="empty",
        subset_description="fixture",
        model_name="fixture-model",
    )
    with pytest.raises(SystemExit, match="scored 0 utterances"):
        evaluate_detection._cmd_score(args)
