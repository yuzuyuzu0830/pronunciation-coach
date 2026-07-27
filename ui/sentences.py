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


# Retired preset wordings that still appear in results/trial_logs/trial_log.jsonl
# (e.g. P01). Maps old target_text -> the current TRIAL_SENTENCES text that
# replaced it, so analysis that resolves logged target_text back to a
# TrialSentence (control vs elicitation, target_phonemes, …) still works.
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
    TrialSentence(
        text="I have some tea in my room.",
        target_phonemes=(),
        note=(
            "control sentence 2: neutral sentence not targeting specific "
            "phonemes. Kept as a second control (not repurposed as a regular "
            "elicitation item) -- both control sentences are the neutral "
            "baseline for the L1 knowledge-base analysis, so getting this "
            "one wrong would skew that baseline. No proper nouns (Whisper "
            "mis-transcription risk) and no concentrated l/ɹ, th, or "
            "vowel-length patterns beyond the incidental /ɹ/ in 'room' -- "
            "same tolerance as control sentence 1's incidental /ð/,/ɹ/. "
            "Former wording 'My name is Yuki and I live in Tokyo.' is kept "
            "in RETIRED_PRESET_TEXTS for trial_log.jsonl analysis."
        ),
    ),
)


def resolve_trial_sentence(target_text: str) -> TrialSentence | None:
    """Map a logged or UI target_text to its TrialSentence.

    Recognizes current TRIAL_SENTENCES texts and retired wordings listed in
    RETIRED_PRESET_TEXTS (same role / metadata as the replacement sentence).
    """
    by_text = {s.text: s for s in TRIAL_SENTENCES}
    if target_text in by_text:
        return by_text[target_text]
    current = RETIRED_PRESET_TEXTS.get(target_text)
    if current is not None:
        return by_text.get(current)
    return None

