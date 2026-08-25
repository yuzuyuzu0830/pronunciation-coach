"""speechocean762 ground-truth parsing for the detection evaluation .

The parser follows the corpus's Kaldi-recipe layout and raises on malformed
fields instead of guessing (docs/design_eval.md §7).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACCURACY_THRESHOLD_DEFAULT = 0.5  # design_eval.md §1.2: accuracy < 0.5 counts as mispronounced


@dataclass(frozen=True)
class Mispronunciation:
    canonical_phone: str  # ARPAbet, as given by the corpus (may carry a stress digit)
    index: int  # position within the owning word's `phones` list
    pronounced_phone: str  # ARPAbet, or the special markers "R*" / "<unk>"


@dataclass(frozen=True)
class WordAnnotation:
    text: str
    phones: list[str]  # ARPAbet with stress digits (e.g. "EY1")
    phones_accuracy: list[float]  # 0-2 per phone, aligned index-for-index with `phones`
    mispronunciations: list[Mispronunciation]


@dataclass(frozen=True)
class UtteranceAnnotation:
    utt_id: str
    speaker_id: str
    text: str  # as given by the corpus (uppercase); lower() before passing to g2p (§6 issue 7)
    words: list[WordAnnotation]


def is_mispronounced(accuracy: float, threshold: float = ACCURACY_THRESHOLD_DEFAULT) -> bool:
    return accuracy < threshold


def _parse_two_column_file(text: str, file_desc: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            key, value = line.split()
        except ValueError:
            raise ValueError(f"Malformed {file_desc} line (expected 'id value'): {line!r}")
        mapping[key] = value
    return mapping


def parse_utt2spk(text: str) -> dict[str, str]:
    """Parse a Kaldi-style utt2spk file: 'utt_id speaker_id' per line.

    Utterance IDs do not encode their speaker, so this mapping is the
    authoritative source (confirmed against the corpus, 2026-07-22).
    """
    return _parse_two_column_file(text, "utt2spk")


def parse_spk2age(text: str) -> dict[str, int]:
    """Parse a Kaldi-style spk2age file: 'speaker_id age' per line."""
    return {spk: int(age) for spk, age in _parse_two_column_file(text, "spk2age").items()}


# Confirmed against the real corpus (2026-07-22): ages split cleanly into
# 6-15 (25 speakers) and 19-43 (100 speakers) with no speaker aged 16-18, 
# so any cutoff placed in that gap gives the same classification.
CHILD_AGE_MAX = 15


def is_child(age: int) -> bool:
    return age <= CHILD_AGE_MAX


def audio_path(utt: UtteranceAnnotation, wave_root: Path) -> Path:
    return wave_root / f"SPEAKER{utt.speaker_id}" / f"{utt.utt_id}.WAV"


def _parse_word(raw: dict[str, Any]) -> WordAnnotation:
    try:
        phones = raw["phones"]
        accuracy = raw["phones-accuracy"]
    except KeyError as e:
        raise ValueError(f"Word annotation missing field {e}: {raw!r}") from e
    if len(phones) != len(accuracy):
        raise ValueError(
            f"phones/phones-accuracy length mismatch for word {raw.get('text')!r}: "
            f"{len(phones)} vs {len(accuracy)}"
        )
    try:
        mispronunciations = [
            Mispronunciation(
                canonical_phone=m["canonical-phone"],
                index=m["index"],
                pronounced_phone=m["pronounced-phone"],
            )
            for m in raw.get("mispronunciations", [])
        ]
    except KeyError as e:
        raise ValueError(f"Mispronunciation entry missing field {e}: {raw!r}") from e
    return WordAnnotation(
        text=raw.get("text", ""),
        phones=list(phones),
        phones_accuracy=[float(a) for a in accuracy],
        mispronunciations=mispronunciations,
    )


def parse_scores(raw: dict[str, Any], speaker_by_utt: dict[str, str]) -> list[UtteranceAnnotation]:
    """Parse a speechocean762 scores.json dict into UtteranceAnnotations.

    `speaker_by_utt` must cover the relevant train/test mappings because
    scores.json carries no speaker IDs. Utterances are sorted by ID so later
    seeded sampling remains reproducible.
    """
    utterances = []
    for utt_id in sorted(raw):
        entry = raw[utt_id]
        try:
            words = [_parse_word(w) for w in entry["words"]]
            text = entry["text"]
        except KeyError as e:
            raise ValueError(f"Utterance {utt_id!r} missing field {e}") from e
        try:
            speaker_id = speaker_by_utt[utt_id]
        except KeyError:
            raise ValueError(
                f"No speaker mapping for utt id {utt_id!r}; build speaker_by_utt "
                "from {train,test}/utt2spk via parse_utt2spk()"
            )
        utterances.append(
            UtteranceAnnotation(
                utt_id=utt_id,
                speaker_id=speaker_id,
                text=text,
                words=words,
            )
        )
    return utterances


def stratified_sample(
    utterances: list[UtteranceAnnotation], n: int, seed: int
) -> list[UtteranceAnnotation]:
    """Sample deterministically while spreading selections across speakers.

    Each round takes at most one utterance per speaker. Age metadata is not
    used here; callers evaluate child/adult subsets separately.
    """
    if n > len(utterances):
        raise ValueError(f"Requested {n} utterances but only {len(utterances)} available")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    # Build independent queues so one prolific speaker cannot dominate.
    by_speaker: dict[str, list[UtteranceAnnotation]] = {}
    for utt in utterances:
        by_speaker.setdefault(utt.speaker_id, []).append(utt)

    # Shuffle both queue contents and speaker order with the same local RNG.
    rng = random.Random(seed)
    for utts in by_speaker.values():
        rng.shuffle(utts)
    speaker_order = list(by_speaker)
    rng.shuffle(speaker_order)

    # Draw one item from each non-empty queue per round until n is reached.
    queues = {spk: list(utts) for spk, utts in by_speaker.items()}
    selected: list[UtteranceAnnotation] = []
    while len(selected) < n:
        for spk in speaker_order:
            if not queues[spk]:
                continue
            selected.append(queues[spk].pop())
            if len(selected) == n:
                break
    return selected
