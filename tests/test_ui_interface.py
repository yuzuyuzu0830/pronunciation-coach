import pytest

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
import ui.interface as interface_module
from ui.runner import TrialState

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except Exception:
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng is not installed")


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


def make_diagnosis(errors: list[PhonemeError], validation=None, target_text="this is high") -> Diagnosis:
    report = DiagnosisReport(
        transcript=target_text,
        target_text=target_text,
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


@requires_espeak
def test_errors_to_dataframe_highlights_substitution_in_word():
    """target_text="this is high"; "this" -> [ð, ɪ, s] (real g2p), so the ð
    at local index 0 should highlight "th" (grapheme for ð) in bold/color."""
    diagnosis = make_diagnosis([PhonemeError("substitution", "ð", "d", 0, "this")])
    rows = _errors_to_dataframe(diagnosis)
    word_html, op, expected, actual, misread = rows[0][1:]
    assert word_html == '<span style="color: crimson; font-weight: bold;">th</span>is'
    assert op == "substitution"
    assert expected == "/ð/"
    assert actual == "/d/"
    assert misread == ""


@requires_espeak
def test_errors_to_dataframe_highlights_capitalized_target_matching_diagnose_casing():
    """_report_word_spans must g2p the cased target (same as diagnose()), not
    lower() it first -- otherwise case-sensitive espeak outputs can shift
    offsets and break highlights. Trial sentences start with capitals."""
    diagnosis = make_diagnosis(
        [PhonemeError("substitution", "ð", "d", 0, "This")],
        target_text="This is high",
    )
    rows = _errors_to_dataframe(diagnosis)
    word_html = rows[0][1]
    assert word_html == '<span style="color: crimson; font-weight: bold;">Th</span>is'


@requires_espeak
def test_errors_to_dataframe_highlights_deletion_with_underline_and_missing_actual():
    """"desk" -> [d, ɛ, s, k] (real g2p); deleting the final k should
    underline the "k" and show "(missing)" rather than a phoneme symbol.
    ("high" is deliberately avoided here: its silent "gh" makes the "h"
    grapheme ambiguous -- see test_locate_grapheme_returns_none_for_irregular_spelling.)"""
    diagnosis = make_diagnosis(
        [PhonemeError("deletion", "k", None, 3, "desk")], target_text="desk"
    )
    rows = _errors_to_dataframe(diagnosis)
    word_html, op, expected, actual, _ = rows[0][1:]
    assert "text-decoration: underline" in word_html
    assert ">k<" in word_html
    assert expected == "/k/"
    assert actual == "(missing)"


def test_errors_to_dataframe_insertion_appends_suffix_without_highlighting_a_letter():
    """An insertion isn't tied to any letter in the target spelling, so the
    word itself stays plain and the extra sound is appended after it."""
    diagnosis = make_diagnosis([PhonemeError("insertion", None, "ə", 2, "this")])
    rows = _errors_to_dataframe(diagnosis)
    word_html, op, expected, actual, _ = rows[0][1:]
    assert word_html == 'this <span style="color: crimson;">+/ə/</span>'
    assert expected == "-"
    assert actual == "/ə/"


def test_errors_to_dataframe_falls_back_to_plain_word_when_g2p_unavailable(monkeypatch):
    """Word-span recomputation is best-effort (docs/design_ui.md §11): if
    g2p can't run, the table must still render, just without a highlight."""

    def boom(text):
        raise RuntimeError("espeak backend unavailable")

    monkeypatch.setattr(interface_module, "to_phonemes_by_word", boom)
    diagnosis = make_diagnosis([PhonemeError("substitution", "ð", "d", 0, "this")])
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][1] == "this"  # plain, no <span>


def test_errors_to_dataframe_falls_back_to_plain_word_when_position_unlocatable():
    """position=99 doesn't fall inside any recomputed word span."""
    diagnosis = make_diagnosis([PhonemeError("substitution", "ð", "d", 99, "this")])
    rows = _errors_to_dataframe(diagnosis)
    # No <span> means locate_grapheme was never even reached for this row.
    assert "<span" not in rows[0][1]


def test_errors_to_dataframe_marks_misread_with_replacement():
    """Softer than a confident '→ "buy"?': the transcript itself may be the
    thing that's wrong (Whisper mis-transcription observed in the P01
    trial), not necessarily what the learner said."""
    diagnosis = make_diagnosis(
        [
            PhonemeError(
                "substitution", "h", "b", 5, "high", possibly_misread=True, misread_as="buy"
            )
        ]
    )
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][5] == '≠ transcript ("buy"?)'


def test_errors_to_dataframe_marks_misread_without_replacement_as_skipped():
    diagnosis = make_diagnosis(
        [PhonemeError("deletion", "h", None, 3, "high", possibly_misread=True, misread_as=None)]
    )
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][5] == "≠ transcript (missing?)"


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
