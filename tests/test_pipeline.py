from pathlib import Path

from pronunciation_coach.pipeline import (
    Explainer,
    PhonemeRecognizer,
    Pipeline,
    Transcriber,
    extract_errors,
    flag_misread_errors,
)
from pronunciation_coach.types import (
    AlignmentOp,
    CoachingResult,
    Diagnosis,
    DiagnosisReport,
    PhonemeError,
    ReadingMismatch,
)

# Error extraction

WORD_SPANS: list[tuple[str, list[str]]] = [
    ("this", ["ð", "ɪ", "s"]),
    ("high", ["h", "aɪ"]),
]


def test_extract_errors_skips_matches() -> None:
    ops = [AlignmentOp("match", "ð", "ð"), AlignmentOp("match", "ɪ", "ɪ")]
    assert extract_errors(ops, WORD_SPANS) == []


def test_extract_errors_substitution_and_deletion_positions() -> None:
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


def test_extract_errors_insertion_belongs_to_preceding_word() -> None:
    ops = [
        AlignmentOp("match", "ð", "ð"),
        AlignmentOp("match", "ɪ", "ɪ"),
        AlignmentOp("match", "s", "s"),
        AlignmentOp("insertion", None, "ɯ"),
        AlignmentOp("match", "h", "h"),
        AlignmentOp("match", "aɪ", "aɪ"),
    ]
    assert extract_errors(ops, WORD_SPANS) == [
        PhonemeError("insertion", None, "ɯ", 2, "this"),
    ]


def test_extract_errors_insertion_at_start_belongs_to_first_word() -> None:
    ops = [
        AlignmentOp("insertion", None, "ɯ"),
        AlignmentOp("match", "ð", "ð"),
    ]
    assert extract_errors(ops, WORD_SPANS) == [
        PhonemeError("insertion", None, "ɯ", 0, "this"),
    ]


# Misread flags


def test_flag_misread_errors_flags_only_mismatched_words() -> None:
    errors = [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError("substitution", "aɪ", "uː", 4, "high"),
    ]
    assert flag_misread_errors(errors, {"high": "buy"}) == [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError(
            "substitution",
            "aɪ",
            "uː",
            4,
            "high",
            possibly_misread=True,
            misread_as="buy",
        ),
    ]


def test_flag_misread_errors_marks_omitted_word_without_read_as() -> None:
    errors = [PhonemeError("deletion", "h", None, 3, "high")]
    flagged = flag_misread_errors(errors, {"high": None})
    assert flagged[0].possibly_misread
    assert flagged[0].misread_as is None


def test_flag_misread_errors_without_mismatches_is_identity() -> None:
    errors = [PhonemeError("substitution", "ð", "d", 0, "this")]
    assert flag_misread_errors(errors, {}) == errors


def test_flag_misread_errors_normalizes_capitalized_word() -> None:
    errors = [PhonemeError("substitution", "ð", "d", 0, "This")]
    flagged = flag_misread_errors(errors, {"this": "dis"})
    assert flagged[0].possibly_misread
    assert flagged[0].misread_as == "dis"


def test_flag_misread_errors_matches_digit_form_word_to_number_word_key() -> None:
    errors = [PhonemeError("substitution", "θ", "t", 0, "3")]
    flagged = flag_misread_errors(errors, {"three": "free"})
    assert flagged[0].possibly_misread
    assert flagged[0].misread_as == "free"


# Pipeline integration

PHONEMES_BY_WORD: dict[str, list[str]] = {
    "this": ["ð", "ɪ", "s"],
    "is": ["ɪ", "z"],
    "high": ["h", "aɪ"],
}


def _fake_g2p_by_word(text: str) -> list[tuple[str, list[str]]]:
    return [(w, PHONEMES_BY_WORD[w.lower()]) for w in text.split()]


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, _audio_path: Path) -> str:
        return self.text


class _FakeRecognizer:
    def __init__(self, phonemes: list[str]) -> None:
        self.phonemes = phonemes
        self.called = False

    def recognize(self, _audio_path: Path) -> list[str]:
        self.called = True
        return self.phonemes


class _FakeExplainer:
    def __init__(self) -> None:
        self.received: DiagnosisReport | None = None

    def explain(self, report: DiagnosisReport) -> str:
        self.received = report
        return "FAKE EXPLANATION"


AUDIO: Path = Path("dummy.wav")


def _make_pipeline(
    transcriber: Transcriber,
    recognizer: PhonemeRecognizer,
    explainer: Explainer,
) -> Pipeline:
    return Pipeline(
        transcriber=transcriber,
        phoneme_recognizer=recognizer,
        explainer=explainer,
        g2p_by_word=_fake_g2p_by_word,
    )


def test_pipeline_detects_errors() -> None:
    explainer = _FakeExplainer()
    pipeline = _make_pipeline(
        _FakeTranscriber("this"),
        _FakeRecognizer(["d", "ɪ"]),
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


def test_pipeline_with_perfect_pronunciation_reports_no_errors() -> None:
    pipeline = _make_pipeline(
        _FakeTranscriber("this is high"),
        _FakeRecognizer(["ð", "ɪ", "s", "ɪ", "z", "h", "aɪ"]),
        _FakeExplainer(),
    )
    result = pipeline.run(AUDIO, target_text="this is high")

    assert isinstance(result, CoachingResult)
    assert result.report.errors == []
    assert result.explanation == "FAKE EXPLANATION"


def test_pipeline_flags_errors_in_misread_words() -> None:
    explainer = _FakeExplainer()
    pipeline = _make_pipeline(
        _FakeTranscriber("this is buy"),
        _FakeRecognizer(["d", "ɪ", "s", "ɪ", "z", "b", "aɪ"]),
        explainer,
    )
    result = pipeline.run(AUDIO, target_text="this is high")

    assert isinstance(result, CoachingResult)
    assert result.validation is not None
    assert result.validation.passed
    assert result.validation.word_mismatches == {"high": "buy"}
    assert result.report.errors == [
        PhonemeError("substitution", "ð", "d", 0, "this"),
        PhonemeError(
            "substitution",
            "h",
            "b",
            5,
            "high",
            possibly_misread=True,
            misread_as="buy",
        ),
    ]


def test_pipeline_flags_misread_on_capitalized_target_word() -> None:
    pipeline = _make_pipeline(
        _FakeTranscriber("dis is high"),
        _FakeRecognizer(["d", "ɪ", "s", "ɪ", "z", "h", "aɪ"]),
        _FakeExplainer(),
    )
    result = pipeline.run(AUDIO, target_text="This is high")

    assert isinstance(result, CoachingResult)
    assert result.validation is not None
    assert result.validation.word_mismatches == {"this": "dis"}
    assert result.report.errors == [
        PhonemeError(
            "substitution",
            "ð",
            "d",
            0,
            "This",
            possibly_misread=True,
            misread_as="dis",
        ),
    ]


def test_pipeline_gate_failure_skips_phoneme_evaluation() -> None:
    recognizer = _FakeRecognizer(["ð"])
    explainer = _FakeExplainer()
    pipeline = _make_pipeline(
        _FakeTranscriber("completely different words entirely"),
        recognizer,
        explainer,
    )
    result = pipeline.run(AUDIO, target_text="this is high")

    assert isinstance(result, ReadingMismatch)
    assert not result.validation.passed
    assert result.validation.wer > 0.5
    assert not recognizer.called
    assert explainer.received is None


# Diagnose and explain


def test_diagnose_gate_failure_returns_reading_mismatch() -> None:
    recognizer = _FakeRecognizer(["ð"])
    explainer = _FakeExplainer()
    pipeline = _make_pipeline(
        _FakeTranscriber("completely different words entirely"),
        recognizer,
        explainer,
    )
    diagnosis = pipeline.diagnose(AUDIO, target_text="this is high")

    assert isinstance(diagnosis, ReadingMismatch)
    assert not diagnosis.validation.passed
    assert not recognizer.called
    assert explainer.received is None


def test_diagnose_report_matches_run_and_can_be_explained() -> None:
    explainer_via_run = _FakeExplainer()
    pipeline_run = _make_pipeline(
        _FakeTranscriber("this"),
        _FakeRecognizer(["d", "ɪ"]),
        explainer_via_run,
    )
    run_result = pipeline_run.run(AUDIO, target_text="this")
    assert isinstance(run_result, CoachingResult)

    explainer_via_diagnose = _FakeExplainer()
    pipeline_diagnose = _make_pipeline(
        _FakeTranscriber("this"),
        _FakeRecognizer(["d", "ɪ"]),
        explainer_via_diagnose,
    )
    diagnosis = pipeline_diagnose.diagnose(AUDIO, target_text="this")
    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.report == run_result.report
    assert diagnosis.validation == run_result.validation

    explanation = explainer_via_diagnose.explain(diagnosis.report)
    assert explanation == run_result.explanation
    assert explainer_via_diagnose.received is diagnosis.report


def test_explain_composes_with_diagnose_to_match_run() -> None:
    explainer = _FakeExplainer()
    pipeline = _make_pipeline(
        _FakeTranscriber("this"), _FakeRecognizer(["d", "ɪ"]), explainer
    )

    diagnosis = pipeline.diagnose(AUDIO, target_text="this")
    assert isinstance(diagnosis, Diagnosis)
    coaching = pipeline.explain(diagnosis)

    assert isinstance(coaching, CoachingResult)
    assert coaching.report == diagnosis.report
    assert coaching.validation == diagnosis.validation
    assert coaching.explanation == "FAKE EXPLANATION"
    assert explainer.received is diagnosis.report


def test_diagnose_without_target_text_has_no_validation() -> None:
    pipeline = _make_pipeline(
        _FakeTranscriber("this"),
        _FakeRecognizer(["ð", "ɪ", "s"]),
        _FakeExplainer(),
    )
    diagnosis = pipeline.diagnose(AUDIO)

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.validation is None
    assert diagnosis.report.target_text is None
    assert diagnosis.report.errors == []


def test_pipeline_without_target_text_skips_validation() -> None:
    pipeline = _make_pipeline(
        _FakeTranscriber("this"),
        _FakeRecognizer(["ð", "ɪ", "s"]),
        _FakeExplainer(),
    )
    result = pipeline.run(AUDIO)

    assert isinstance(result, CoachingResult)
    assert result.validation is None
    assert result.report.target_text is None
    assert result.report.errors == []
