from pronunciation_coach.types import (
    Diagnosis,
    DiagnosisReport,
    PhonemeError,
    ReadingMismatch,
    ReadingValidation,
)
from ui.interface import (
    _errors_to_dataframe,
    _log_status_markdown,
    _transcript_markdown,
    resolve_target,
)
from ui.runner import TrialState


# --- resolve_target ---


def test_resolve_target_custom_text_wins_over_preset():
    assert resolve_target("preset sentence", "  my custom text  ") == ("my custom text", "custom")


def test_resolve_target_falls_back_to_preset_when_custom_empty():
    assert resolve_target("preset sentence", "") == ("preset sentence", "preset")
    assert resolve_target("preset sentence", "   ") == ("preset sentence", "preset")


def test_resolve_target_none_when_both_empty():
    assert resolve_target(None, "") == (None, "preset")
    assert resolve_target("", "") == (None, "preset")


# --- _errors_to_dataframe ---


def make_diagnosis(errors: list[PhonemeError], validation=None) -> Diagnosis:
    report = DiagnosisReport(
        transcript="this is high",
        target_text="this is high",
        reference_phonemes=["ð", "ɪ", "s"],
        hypothesis_phonemes=["d", "ɪ", "s"],
        errors=errors,
        learner_l1="Japanese",
    )
    return Diagnosis(report=report, validation=validation)


def test_errors_to_dataframe_empty_for_no_diagnosis():
    assert _errors_to_dataframe(None) == []


def test_errors_to_dataframe_empty_for_no_errors():
    assert _errors_to_dataframe(make_diagnosis([])) == []


def test_errors_to_dataframe_renders_substitution_row():
    diagnosis = make_diagnosis([PhonemeError("substitution", "ð", "d", 0, "this")])
    rows = _errors_to_dataframe(diagnosis)
    assert rows == [[1, "this", "substitution", "ð", "d", ""]]


def test_errors_to_dataframe_marks_misread_with_replacement():
    diagnosis = make_diagnosis(
        [
            PhonemeError(
                "substitution", "h", "b", 5, "high", possibly_misread=True, misread_as="buy"
            )
        ]
    )
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][5] == '→ "buy"?'


def test_errors_to_dataframe_marks_misread_without_replacement_as_skipped():
    diagnosis = make_diagnosis(
        [PhonemeError("deletion", "h", None, 3, "high", possibly_misread=True, misread_as=None)]
    )
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][5] == "(skipped?)"


def test_errors_to_dataframe_uses_dash_for_missing_phones():
    diagnosis = make_diagnosis([PhonemeError("deletion", "h", None, 3, "high")])
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][3] == "h"
    assert rows[0][4] == "-"


# --- _transcript_markdown ---


def make_validation(passed: bool, wer: float = 0.0) -> ReadingValidation:
    return ReadingValidation(
        target_words=["this", "is", "high"],
        transcript_words=["this", "is", "high"],
        wer=wer,
        threshold=0.5,
        passed=passed,
    )


def test_transcript_markdown_empty_for_running_state():
    assert _transcript_markdown(TrialState(stage="running", status_message="Transcribing…")) == ""


def test_transcript_markdown_shows_gate_pass_with_validation():
    diagnosis = make_diagnosis([], validation=make_validation(passed=True, wer=0.1))
    state = TrialState(stage="detected", status_message="", diagnosis=diagnosis)
    md = _transcript_markdown(state)
    assert "this is high" in md
    assert "passed" in md
    assert "0.10" in md


def test_transcript_markdown_shows_no_gate_note_without_target_text():
    diagnosis = make_diagnosis([], validation=None)
    state = TrialState(stage="detected", status_message="", diagnosis=diagnosis)
    md = _transcript_markdown(state)
    assert "no target sentence" in md


def test_transcript_markdown_shows_reading_mismatch_details():
    validation = ReadingValidation(
        target_words=["this", "is", "high"],
        transcript_words=["completely", "different"],
        wer=1.0,
        threshold=0.5,
        passed=False,
    )
    state = TrialState(
        stage="reading_mismatch",
        status_message="",
        reading_mismatch=ReadingMismatch(validation),
    )
    md = _transcript_markdown(state)
    assert "failed" in md
    assert "this is high" in md
    assert "completely different" in md


# --- _log_status_markdown ---


def test_log_status_markdown_empty_while_running_or_detected():
    assert _log_status_markdown(TrialState(stage="running", status_message="")) == ""
    assert _log_status_markdown(TrialState(stage="detected", status_message="")) == ""


def test_log_status_markdown_shows_success():
    state = TrialState(stage="done", status_message="", logged=True)
    assert "Record saved" in _log_status_markdown(state)


def test_log_status_markdown_shows_failure_with_reason():
    state = TrialState(stage="done", status_message="", logged=False, log_error="disk full")
    md = _log_status_markdown(state)
    assert "Failed to save record" in md
    assert "disk full" in md
