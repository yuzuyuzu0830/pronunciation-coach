from pathlib import Path

from pronunciation_coach.pipeline import Pipeline, extract_errors, flag_misread_errors
from pronunciation_coach.types import (
    AlignmentOp,
    CoachingResult,
    DiagnosisReport,
    PhonemeError,
    ReadingMismatch,
)

# --- extract_errors (pure) ---

WORD_SPANS = [("this", ["ð", "ɪ", "s"]), ("high", ["h", "aɪ"])]


def test_extract_errors_skips_matches():
    ops = [AlignmentOp("match", "ð", "ð"), AlignmentOp("match", "ɪ", "ɪ")]
    assert extract_errors(ops, WORD_SPANS) == []


def test_extract_errors_substitution_and_deletion_positions():
    ops = [
        AlignmentOp("substitution", "ð", "d"),
        AlignmentOp("match", "ɪ", "ɪ"),
        AlignmentOp("deletion", "s", None),
        AlignmentOp("match", "h", "h"),
        AlignmentOp("substitution", "aɪ", "a"),
    ]
    assert extract_errors(ops, WORD_SPANS) == [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError("deletion", "s", None, 2, "this"),
        PhonemeError("substitution", "aɪ", "a", 4, "high"),
    ]


def test_extract_errors_insertion_belongs_to_preceding_word():
    """Insertion on a word boundary attributes to the preceding word."""
    ops = [
        AlignmentOp("match", "ð", "ð"),
        AlignmentOp("match", "ɪ", "ɪ"),
        AlignmentOp("match", "s", "s"),
        AlignmentOp("insertion", None, "ɯ"),  # after "this", before "high"
        AlignmentOp("match", "h", "h"),
        AlignmentOp("match", "aɪ", "aɪ"),
    ]
    assert extract_errors(ops, WORD_SPANS) == [
        PhonemeError("insertion", None, "ɯ", 2, "this"),
    ]


def test_extract_errors_insertion_at_start_belongs_to_first_word():
    """A leading insertion has no preceding phone; clamp to the first word."""
    ops = [
        AlignmentOp("insertion", None, "ɯ"),
        AlignmentOp("match", "ð", "ð"),
    ]
    assert extract_errors(ops, WORD_SPANS) == [
        PhonemeError("insertion", None, "ɯ", 0, "this"),
    ]


# --- flag_misread_errors (pure) ---


def test_flag_misread_errors_flags_only_mismatched_words():
    errors = [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError("substitution", "aɪ", "uː", 4, "high"),
    ]
    assert flag_misread_errors(errors, {"high": "buy"}) == [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError(
            "substitution", "aɪ", "uː", 4, "high",
            possibly_misread=True, misread_as="buy",
        ),
    ]


def test_flag_misread_errors_marks_omitted_word_without_read_as():
    errors = [PhonemeError("deletion", "h", None, 3, "high")]
    flagged = flag_misread_errors(errors, {"high": None})
    assert flagged[0].possibly_misread
    assert flagged[0].misread_as is None


def test_flag_misread_errors_without_mismatches_is_identity():
    errors = [PhonemeError("substitution", "ð", "d", 0, "this")]
    assert flag_misread_errors(errors, {}) == errors


# --- Pipeline integration with fakes ---

PHONEMES_BY_WORD = {
    "this": ["ð", "ɪ", "s"],
    "is": ["ɪ", "z"],
    "high": ["h", "aɪ"],
}


def fake_g2p_by_word(text: str) -> list[tuple[str, list[str]]]:
    return [(w, PHONEMES_BY_WORD[w]) for w in text.lower().split()]


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
        self.called = False

    def transcribe(self, audio_path: Path) -> str:
        self.called = True
        return self.text


class FakeRecognizer:
    def __init__(self, phonemes: list[str]) -> None:
        self.phonemes = phonemes
        self.called = False

    def recognize(self, audio_path: Path) -> list[str]:
        self.called = True
        return self.phonemes


class FakeExplainer:
    def __init__(self) -> None:
        self.received: DiagnosisReport | None = None

    def explain(self, report: DiagnosisReport) -> str:
        self.received = report
        return "FAKE EXPLANATION"


AUDIO = Path("dummy.wav")


def make_pipeline(transcriber, recognizer, explainer) -> Pipeline:
    return Pipeline(
        transcriber=transcriber,
        phoneme_recognizer=recognizer,
        explainer=explainer,
        g2p_by_word=fake_g2p_by_word,
    )


def test_pipeline_detects_errors():
    explainer = FakeExplainer()
    pipeline = make_pipeline(
        FakeTranscriber("this"),
        FakeRecognizer(["d", "ɪ"]),  # ð→d substitution, s deleted
        explainer,
    )
    result = pipeline.run(AUDIO, target_text="this")

    assert isinstance(result, CoachingResult)
    assert result.report.errors == [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError("deletion", "s", None, 2, "this"),
    ]
    assert result.explanation == "FAKE EXPLANATION"
    assert result.validation is not None
    assert result.validation.passed
    assert explainer.received is result.report


def test_pipeline_with_perfect_pronunciation_reports_no_errors():
    pipeline = make_pipeline(
        FakeTranscriber("this is high"),
        FakeRecognizer(["ð", "ɪ", "s", "ɪ", "z", "h", "aɪ"]),
        FakeExplainer(),
    )
    result = pipeline.run(AUDIO, target_text="this is high")

    assert isinstance(result, CoachingResult)
    assert result.report.errors == []
    assert result.explanation == "FAKE EXPLANATION"


def test_pipeline_flags_errors_in_misread_words():
    """Learner reads "buy" instead of "high": Whisper hears the other word,
    so the h→b substitution is a reading mistake, not a pronunciation habit.
    The ð→d error in the correctly-read "this" must stay unflagged."""
    explainer = FakeExplainer()
    pipeline = make_pipeline(
        FakeTranscriber("this is buy"),
        FakeRecognizer(["d", "ɪ", "s", "ɪ", "z", "b", "aɪ"]),
        explainer,
    )
    result = pipeline.run(AUDIO, target_text="this is high")

    assert isinstance(result, CoachingResult)
    assert result.validation is not None
    assert result.validation.passed  # WER 1/3 stays under the gate
    assert result.validation.word_mismatches == {"high": "buy"}
    assert result.report.errors == [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError(
            "substitution", "h", "b", 5, "high",
            possibly_misread=True, misread_as="buy",
        ),
    ]


def test_pipeline_gate_failure_skips_phoneme_evaluation():
    recognizer = FakeRecognizer(["ð"])
    explainer = FakeExplainer()
    pipeline = make_pipeline(
        FakeTranscriber("completely different words entirely"),
        recognizer,
        explainer,
    )
    result = pipeline.run(AUDIO, target_text="this is high")

    assert isinstance(result, ReadingMismatch)
    assert not result.validation.passed
    assert result.validation.wer > 0.5
    assert not recognizer.called
    assert explainer.received is None


def test_pipeline_without_target_text_skips_validation():
    pipeline = make_pipeline(
        FakeTranscriber("this"),
        FakeRecognizer(["ð", "ɪ", "s"]),
        FakeExplainer(),
    )
    result = pipeline.run(AUDIO)

    assert isinstance(result, CoachingResult)
    assert result.validation is None
    assert result.report.target_text is None
    assert result.report.errors == []
