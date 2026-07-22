"""speechocean762 ground-truth parsing for the detection evaluation (docs/design_eval.md §1).

The exact on-disk layout of scores.json is confirmed against the real corpus
during data acquisition (docs/design_eval.md §7); this parser targets the
documented Kaldi-recipe format and raises on missing/malformed fields rather
than guessing, so a format mismatch surfaces immediately instead of silently
producing wrong ground truth.
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


def speaker_id_from_utt_id(utt_id: str) -> str:
    """First four digits of the utt id are the speaker id (WAVE/SPEAKERxxxx/...)."""
    if len(utt_id) < 4 or not utt_id[:4].isdigit():
        raise ValueError(f"Cannot derive speaker id from utt id {utt_id!r}")
    return utt_id[:4]


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


def parse_scores(raw: dict[str, Any]) -> list[UtteranceAnnotation]:
    """Parse a speechocean762 scores.json dict into UtteranceAnnotations.

    Utt ids are sorted before parsing (dict order isn't guaranteed stable
    across json libraries/versions) so downstream sampling is reproducible.
    """
    utterances = []
    for utt_id in sorted(raw):
        entry = raw[utt_id]
        try:
            words = [_parse_word(w) for w in entry["words"]]
            text = entry["text"]
        except KeyError as e:
            raise ValueError(f"Utterance {utt_id!r} missing field {e}") from e
        utterances.append(
            UtteranceAnnotation(
                utt_id=utt_id,
                speaker_id=speaker_id_from_utt_id(utt_id),
                text=text,
                words=words,
            )
        )
    return utterances


def stratified_sample(
    utterances: list[UtteranceAnnotation], n: int, seed: int
) -> list[UtteranceAnnotation]:
    """Deterministically sample n utterances, round-robining across speakers.

    Cycles through speakers (in seed-shuffled order), taking one utterance
    per speaker per round, so a small subset still covers most of the
    speaker pool rather than a few speakers' contiguous utterances (§1.3).

    Adult/child balance (also required by §1.3) needs per-speaker metadata
    (spk2age or equivalent) whose exact format is unconfirmed until the real
    corpus is fetched (§7); it is layered on top of this speaker-level
    sampling once that format is known, not implemented here.
    """
    if n > len(utterances):
        raise ValueError(f"Requested {n} utterances but only {len(utterances)} available")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    by_speaker: dict[str, list[UtteranceAnnotation]] = {}
    for utt in utterances:
        by_speaker.setdefault(utt.speaker_id, []).append(utt)

    rng = random.Random(seed)
    for utts in by_speaker.values():
        rng.shuffle(utts)
    speaker_order = list(by_speaker)
    rng.shuffle(speaker_order)

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
