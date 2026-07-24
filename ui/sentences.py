"""Trial target-sentence list (docs/design_ui.md §6).

Provisional list -- the final wording is confirmed by the trial plan before
the August session; this list only needs to (a) cover the difficulty
patterns the L1 knowledge base targets and (b) actually phonemize (verified
by tests/test_ui_sentences.py against the real g2p pipeline, the same way
practice_words are verified in pronunciation_coach/knowledge_data).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrialSentence:
    text: str
    target_phonemes: tuple[str, ...]  # empty for a control sentence
    note: str  # for the trial script/moderator


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
    TrialSentence(
        text="My name is Yuki and I live in Tokyo.",
        target_phonemes=(),
        note="control sentence 2: neutral sentence not targeting specific phonemes",
    ),
)
