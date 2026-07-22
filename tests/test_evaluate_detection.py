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

# The corpus's utt id does not embed its speaker id; scoring always looks it
# up via utt2spk (so762.parse_utt2spk).
SPEAKER_BY_UTT = {"0001010011": "0001", "0002030022": "0002"}


def _write_utt2spk(tmp_path: Path) -> list[Path]:
    path = tmp_path / "utt2spk"
    path.write_text(
        "\n".join(f"{utt_id} {spk}" for utt_id, spk in SPEAKER_BY_UTT.items()) + "\n",
        encoding="utf-8",
    )
    return [path]


@requires_espeak
def test_run_stage2_computes_expected_metrics():
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)
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
        utt2spk=_write_utt2spk(tmp_path),
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
    assert "DER is out of scope for this corpus" in report
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
        utt2spk=_write_utt2spk(tmp_path),
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
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)

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
    utterances = evaluate_detection.parse_scores(SCORES, SPEAKER_BY_UTT)

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


# --- sample: test-split filtering ---


def test_cmd_sample_restricts_to_utt_ids_filter(tmp_path):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SCORES), encoding="utf-8")
    utt_ids_path = tmp_path / "test_split.txt"
    utt_ids_path.write_text("0002030022\n", encoding="utf-8")  # only WATER is "test"
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


# --- stage1 (recognize): resumable JSONL, tested with a fake recognizer ---


class FakeRecognizer:
    def __init__(self, phonemes_by_utt=None, raise_for=frozenset()):
        self.phonemes_by_utt = phonemes_by_utt or {}
        self.raise_for = raise_for
        self.calls: list[Path] = []

    def recognize(self, path: Path) -> list[str]:
        self.calls.append(path)
        utt_id = path.stem
        if utt_id in self.raise_for:
            raise RuntimeError(f"boom on {utt_id}")
        return self.phonemes_by_utt.get(utt_id, ["x", "y"])


def make_utterances_with_audio(tmp_path, utt_ids, missing=frozenset()):
    """Build minimal UtteranceAnnotations and matching dummy WAVE/ files."""
    wave_root = tmp_path / "WAVE"
    utterances = []
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


def test_run_stage1_recognizes_all_and_writes_jsonl(tmp_path):
    utt_ids = ["0001000001", "0001000002", "0003000001"]
    utterances, wave_root = make_utterances_with_audio(tmp_path, utt_ids)
    recognizer = FakeRecognizer({"0001000001": ["a", "b"], "0003000001": ["c"]})
    out_path = tmp_path / "hyp_phonemes.jsonl"

    newly, skipped = evaluate_detection.run_stage1(utterances, wave_root, recognizer, out_path)

    assert newly == 3
    assert skipped == []
    hyp = evaluate_detection.load_hyp_phonemes(out_path)
    assert hyp == {
        "0001000001": ["a", "b"],
        "0001000002": ["x", "y"],
        "0003000001": ["c"],
    }
    assert len(recognizer.calls) == 3


def test_run_stage1_resumes_and_skips_already_processed(tmp_path):
    """Simulates an interrupted run: rerun must not re-recognize completed utts."""
    utt_ids = ["0001000001", "0001000002", "0001000003"]
    utterances, wave_root = make_utterances_with_audio(tmp_path, utt_ids)
    out_path = tmp_path / "hyp_phonemes.jsonl"

    first_recognizer = FakeRecognizer(raise_for={"0001000003"})
    newly1, skipped1 = evaluate_detection.run_stage1(utterances, wave_root, first_recognizer, out_path)
    assert newly1 == 2  # 0001000001, 0001000002 succeed; 0001000003 fails (not written)
    assert skipped1 == ["0001000003: recognize_failed: boom on 0001000003"]
    assert set(first_recognizer.calls) == {
        wave_root / "SPEAKER0001" / "0001000001.WAV",
        wave_root / "SPEAKER0001" / "0001000002.WAV",
        wave_root / "SPEAKER0001" / "0001000003.WAV",
    }

    # "Resume": a fresh recognizer that would raise for ALL utt ids if called --
    # proves the already-written two are skipped, only the failed one retried.
    second_recognizer = FakeRecognizer(raise_for={"0001000001", "0001000002", "0001000003"})
    newly2, skipped2 = evaluate_detection.run_stage1(utterances, wave_root, second_recognizer, out_path)

    assert newly2 == 0
    assert skipped2 == ["0001000003: recognize_failed: boom on 0001000003"]
    # Only the still-missing utt was re-attempted; the two done ones were never touched.
    assert second_recognizer.calls == [wave_root / "SPEAKER0001" / "0001000003.WAV"]

    hyp = evaluate_detection.load_hyp_phonemes(out_path)
    assert set(hyp.keys()) == {"0001000001", "0001000002"}


def test_run_stage1_skips_missing_audio_file(tmp_path):
    utt_ids = ["0001000001", "0001000002"]
    utterances, wave_root = make_utterances_with_audio(tmp_path, utt_ids, missing={"0001000002"})
    recognizer = FakeRecognizer()
    out_path = tmp_path / "hyp_phonemes.jsonl"

    newly, skipped = evaluate_detection.run_stage1(utterances, wave_root, recognizer, out_path)

    assert newly == 1
    assert len(skipped) == 1
    assert skipped[0].startswith("0001000002: audio_missing:")
    assert recognizer.calls == [wave_root / "SPEAKER0001" / "0001000001.WAV"]


def test_run_stage1_skips_on_recognizer_exception_without_aborting_others(tmp_path):
    utt_ids = ["0001000001", "0001000002", "0001000003"]
    utterances, wave_root = make_utterances_with_audio(tmp_path, utt_ids)
    recognizer = FakeRecognizer(raise_for={"0001000002"})
    out_path = tmp_path / "hyp_phonemes.jsonl"

    newly, skipped = evaluate_detection.run_stage1(utterances, wave_root, recognizer, out_path)

    assert newly == 2
    assert skipped == ["0001000002: recognize_failed: boom on 0001000002"]
    hyp = evaluate_detection.load_hyp_phonemes(out_path)
    assert set(hyp.keys()) == {"0001000001", "0001000003"}


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
        utt2spk=_write_utt2spk(tmp_path),
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
