import pytest

from pronunciation_coach.g2p import normalize, to_phonemes_by_word
from ui.sentences import TRIAL_SENTENCES

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
