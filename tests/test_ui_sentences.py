import pytest

from pronunciation_coach.g2p import normalize, to_phonemes_by_word
from ui.sentences import (
    RETIRED_PRESET_TEXTS,
    TRIAL_SENTENCES,
    resolve_trial_sentence,
)

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except Exception:
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng is not installed")


def test_trial_sentences_list_is_non_empty():
    assert len(TRIAL_SENTENCES) >= 5


def test_trial_sentences_have_unique_text():
    texts = [s.text for s in TRIAL_SENTENCES]
    assert len(texts) == len(set(texts))


def test_resolve_trial_sentence_finds_current_preset():
    sentence = TRIAL_SENTENCES[0]
    assert resolve_trial_sentence(sentence.text) is sentence


def test_resolve_trial_sentence_maps_retired_control_2_to_current_control():
    """P01 trial_log.jsonl still has the pre-rename control sentence 2 text.
    Analysis that maps target_text -> TrialSentence must treat it as the
    same control (empty target_phonemes), not as an unknown custom sentence.
    """
    old_text = "My name is Yuki and I live in Tokyo."
    assert old_text in RETIRED_PRESET_TEXTS
    resolved = resolve_trial_sentence(old_text)
    assert resolved is not None
    assert resolved.text == "I have some tea in my room."
    assert resolved.target_phonemes == ()
    assert "control sentence 2" in resolved.note


def test_resolve_trial_sentence_returns_none_for_unknown_text():
    assert resolve_trial_sentence("not a trial sentence") is None


def test_retired_preset_targets_are_not_in_current_list():
    """Retired texts must stay out of the UI dropdown (TRIAL_SENTENCES) while
    still being resolvable for log analysis."""
    current = {s.text for s in TRIAL_SENTENCES}
    for old_text, new_text in RETIRED_PRESET_TEXTS.items():
        assert old_text not in current
        assert new_text in current


@requires_espeak
@pytest.mark.parametrize("sentence", TRIAL_SENTENCES, ids=lambda s: s.text)
def test_trial_sentence_phonemizes_without_error(sentence):
    """Same mechanical check as knowledge_data's practice_words: a sentence
    that can't survive g2p can't be used as a trial target text (it would
    hit the WER-gate g2p path and never reach detection)."""
    word_spans = to_phonemes_by_word(sentence.text)
    assert word_spans, f"{sentence.text!r} produced no words"


@requires_espeak
@pytest.mark.parametrize(
    "sentence", [s for s in TRIAL_SENTENCES if s.target_phonemes], ids=lambda s: s.text
)
def test_trial_sentence_contains_its_target_phonemes(sentence):
    """Each non-control sentence's annotated target phonemes must actually
    appear in its own phonemization (catches an annotation drifting out of
    sync with the sentence text)."""
    phonemes = normalize(
        [p for _, phones in to_phonemes_by_word(sentence.text) for p in phones]
    )
    for target in sentence.target_phonemes:
        assert target in phonemes, (
            f"{target!r} not found in phonemization of {sentence.text!r}: {phonemes}"
        )
