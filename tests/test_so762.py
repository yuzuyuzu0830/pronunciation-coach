import json
from pathlib import Path

import pytest

from pronunciation_coach.evaluation.so762 import (
    Mispronunciation,
    UtteranceAnnotation,
    WordAnnotation,
    audio_path,
    is_mispronounced,
    parse_scores,
    speaker_id_from_utt_id,
    stratified_sample,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "so762_scores_sample.json").read_text(encoding="utf-8")
)


def test_parse_scores_returns_utterances_sorted_by_utt_id():
    utterances = parse_scores(FIXTURE)
    assert [u.utt_id for u in utterances] == ["0001010011", "0002030022"]


def test_parse_scores_extracts_speaker_id():
    utterances = parse_scores(FIXTURE)
    assert utterances[0].speaker_id == "0001"
    assert utterances[1].speaker_id == "0002"


def test_parse_scores_parses_word_and_phone_fields():
    utterances = parse_scores(FIXTURE)
    this_word = utterances[0].words[0]
    assert this_word == WordAnnotation(
        text="THIS",
        phones=["DH", "IH1", "S"],
        phones_accuracy=[2.0, 2.0, 2.0],
        mispronunciations=[],
    )


def test_parse_scores_parses_mispronunciations():
    utterances = parse_scores(FIXTURE)
    high_word = utterances[0].words[2]
    assert high_word.mispronunciations == [
        Mispronunciation(canonical_phone="HH", index=0, pronounced_phone="<unk>")
    ]


def test_parse_scores_raises_on_missing_field():
    malformed = {"0001010011": {"text": "THIS", "words": [{"text": "THIS", "phones": ["DH"]}]}}
    with pytest.raises(ValueError, match="phones-accuracy"):
        parse_scores(malformed)


def test_parse_scores_raises_on_phones_length_mismatch():
    malformed = {
        "0001010011": {
            "text": "THIS",
            "words": [{"text": "THIS", "phones": ["DH", "IH1"], "phones-accuracy": [2.0]}],
        }
    }
    with pytest.raises(ValueError, match="length mismatch"):
        parse_scores(malformed)


@pytest.mark.parametrize(
    "utt_id, expected",
    [
        ("0001010011", "0001"),
        ("0123456789", "0123"),
    ],
)
def test_speaker_id_from_utt_id(utt_id, expected):
    assert speaker_id_from_utt_id(utt_id) == expected


def test_speaker_id_from_utt_id_raises_on_too_short_id():
    with pytest.raises(ValueError):
        speaker_id_from_utt_id("12")


def test_audio_path_builds_speaker_directory_layout():
    utt = UtteranceAnnotation(utt_id="0001010011", speaker_id="0001", text="THIS", words=[])
    assert audio_path(utt, Path("/data/speechocean762/WAVE")) == Path(
        "/data/speechocean762/WAVE/SPEAKER0001/0001010011.WAV"
    )


@pytest.mark.parametrize(
    "accuracy, threshold, expected",
    [
        (2.0, 0.5, False),
        (0.4, 0.5, True),
        (0.5, 0.5, False),  # boundary: exactly at threshold counts as pronounced
        (0.0, 0.5, True),
    ],
)
def test_is_mispronounced(accuracy, threshold, expected):
    assert is_mispronounced(accuracy, threshold) is expected


def _make_utterances(speaker_counts: dict[str, int]) -> list[UtteranceAnnotation]:
    utterances = []
    for speaker, count in speaker_counts.items():
        for i in range(count):
            utterances.append(
                UtteranceAnnotation(
                    utt_id=f"{speaker}{i:06d}", speaker_id=speaker, text="X", words=[]
                )
            )
    return utterances


def test_stratified_sample_returns_requested_count():
    utterances = _make_utterances({"0001": 5, "0002": 5, "0003": 5})
    sample = stratified_sample(utterances, n=6, seed=0)
    assert len(sample) == 6


def test_stratified_sample_is_deterministic_for_same_seed():
    utterances = _make_utterances({"0001": 5, "0002": 5, "0003": 5})
    first = stratified_sample(utterances, n=6, seed=42)
    second = stratified_sample(utterances, n=6, seed=42)
    assert [u.utt_id for u in first] == [u.utt_id for u in second]


def test_stratified_sample_spreads_across_speakers():
    """A subset smaller than the speaker count should still touch every speaker."""
    utterances = _make_utterances({"0001": 10, "0002": 10, "0003": 10})
    sample = stratified_sample(utterances, n=3, seed=0)
    assert {u.speaker_id for u in sample} == {"0001", "0002", "0003"}


def test_stratified_sample_raises_when_n_exceeds_pool():
    utterances = _make_utterances({"0001": 2})
    with pytest.raises(ValueError):
        stratified_sample(utterances, n=3, seed=0)


def test_stratified_sample_raises_on_negative_n():
    utterances = _make_utterances({"0001": 2})
    with pytest.raises(ValueError):
        stratified_sample(utterances, n=-1, seed=0)
