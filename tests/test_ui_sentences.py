from __future__ import annotations

import pytest

from pronunciation_coach.g2p import normalize, to_phonemes_by_word
from ui.sentences import (
    RETIRED_PRESET_TEXTS,
    TRIAL_SENTENCES,
    TrialSentence,
    resolve_trial_sentence,
)

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except (ImportError, OSError):
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng is not installed"
)


def _sentence_id(sentence: TrialSentence) -> str:
    return sentence.text


def test_trial_sentences_include_at_least_five_entries() -> None:
    assert len(TRIAL_SENTENCES) >= 5


def test_trial_sentences_have_unique_text() -> None:
    texts = [sentence.text for sentence in TRIAL_SENTENCES]
    assert len(texts) == len(set(texts))


def test_resolve_trial_sentence_finds_current_preset() -> None:
    sentence = TRIAL_SENTENCES[0]
    assert resolve_trial_sentence(sentence.text) is sentence


def test_resolve_trial_sentence_maps_retired_control_2_to_current_control() -> None:
    old_text = "My name is Yuki and I live in Tokyo."
    assert old_text in RETIRED_PRESET_TEXTS
    resolved = resolve_trial_sentence(old_text)
    assert resolved is not None
    assert resolved.text == "I have some tea in my room."
    assert resolved.target_phonemes == ()
    assert "control sentence 2" in resolved.note


def test_resolve_trial_sentence_returns_none_for_unknown_text() -> None:
    assert resolve_trial_sentence("not a trial sentence") is None


def test_retired_preset_targets_are_not_in_current_list() -> None:
    current = {sentence.text for sentence in TRIAL_SENTENCES}
    for old_text, new_text in RETIRED_PRESET_TEXTS.items():
        assert old_text not in current
        assert new_text in current


@requires_espeak
@pytest.mark.parametrize("sentence", TRIAL_SENTENCES, ids=_sentence_id)
def test_trial_sentence_phonemizes_without_error(sentence: TrialSentence) -> None:
    word_spans = to_phonemes_by_word(sentence.text)
    assert word_spans, f"{sentence.text!r} produced no words"


@requires_espeak
@pytest.mark.parametrize(
    "sentence",
    [sentence for sentence in TRIAL_SENTENCES if sentence.target_phonemes],
    ids=_sentence_id,
)
def test_trial_sentence_contains_its_target_phonemes(sentence: TrialSentence) -> None:
    phonemes = normalize(
        [phone for _, phones in to_phonemes_by_word(sentence.text) for phone in phones]
    )
    for target in sentence.target_phonemes:
        assert target in phonemes, (
            f"{target!r} not found in phonemization of {sentence.text!r}: {phonemes}"
        )
