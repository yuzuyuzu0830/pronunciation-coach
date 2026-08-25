"""Fixed target sentences for the pronunciation trial.

Tests verify that each sentence survives the production G2P path and contains
its annotated target phonemes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrialSentence:
    text: str
    target_phonemes: tuple[str, ...]  # empty for a control sentence
    note: str  # for the trial script/moderator


# Map retired logged targets to their current sentence metadata for analysis.
RETIRED_PRESET_TEXTS: dict[str, str] = {
    "My name is Yuki and I live in Tokyo.": "I have some tea in my room.",
}


TRIAL_SENTENCES: tuple[TrialSentence, ...] = (
    TrialSentence(
        text="This is my brother's house.",
        target_phonemes=("ð",),
        note="ð (dh_stopping): this / brother's",
    ),
    TrialSentence(
        text="Thank you for the three thin books.",
        target_phonemes=("θ",),
        note="θ (th_sibilant_substitution): thank / three / thin",
    ),
    TrialSentence(
        text="The red light is very bright.",
        target_phonemes=("l", "ɹ"),
        note="l/ɹ (liquid_confusion): red / light / very / bright",
    ),
    TrialSentence(
        text="I can see the sea from here.",
        target_phonemes=("s", "iː"),
        note="s+i (si_palatalization): see / sea",
    ),
    TrialSentence(
        text="Did you see the sheep near the ship?",
        target_phonemes=("iː", "ɪ"),
        note="vowel length (vowel_length_mismatch): sheep /iː/ vs ship /ɪ/",
    ),
    TrialSentence(
        text="Please put the book on the desk.",
        target_phonemes=("s", "k"),
        note="word-final consonant cluster (induces vowel_epenthesis): desk",
    ),
    TrialSentence(
        text="Although this is difficult, three friends helped me finish it.",
        target_phonemes=("ð", "θ", "ɹ"),
        note="compound sentence: includes ð/θ/ɹ in one sentence (this / three friends)",
    ),
    TrialSentence(
        text="The weather is nice today.",
        target_phonemes=(),
        note="control sentence 1: neutral sentence not targeting specific phonemes",
    ),
    # Keep both controls as neutral baselines; the retired wording remains
    # resolvable through RETIRED_PRESET_TEXTS for earlier trial logs.
    TrialSentence(
        text="I have some tea in my room.",
        target_phonemes=(),
        note="control sentence 2: neutral baseline with incidental /ɹ/ in 'room'",
    ),
)


def resolve_trial_sentence(target_text: str) -> TrialSentence | None:
    """Resolve current or retired target text to its sentence metadata."""
    by_text = {s.text: s for s in TRIAL_SENTENCES}
    if target_text in by_text:
        return by_text[target_text]
    current = RETIRED_PRESET_TEXTS.get(target_text)
    if current is not None:
        return by_text.get(current)
    return None
