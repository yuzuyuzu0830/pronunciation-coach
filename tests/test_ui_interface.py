from __future__ import annotations

from typing import Never

import pytest

import ui.interface as interface_module
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

try:
    from phonemizer.backend import EspeakBackend

    ESPEAK_AVAILABLE = EspeakBackend.is_available()
except (ImportError, OSError):
    ESPEAK_AVAILABLE = False

requires_espeak = pytest.mark.skipif(
    not ESPEAK_AVAILABLE, reason="espeak-ng is not installed"
)


def test_resolve_target_custom_text_wins_over_preset() -> None:
    assert resolve_target("preset sentence", "  my custom text  ") == (
        "my custom text",
        "custom",
    )


def test_resolve_target_falls_back_to_preset_when_custom_empty() -> None:
    assert resolve_target("preset sentence", "") == ("preset sentence", "preset")
    assert resolve_target("preset sentence", "   ") == ("preset sentence", "preset")


def test_resolve_target_none_when_both_empty() -> None:
    assert resolve_target(None, "") == (None, "preset")
    assert resolve_target("", "") == (None, "preset")


def _make_diagnosis(
    errors: list[PhonemeError],
    validation: ReadingValidation | None = None,
    target_text: str = "this is high",
) -> Diagnosis:
    report = DiagnosisReport(
        transcript=target_text,
        target_text=target_text,
        reference_phonemes=["ð", "ɪ", "s"],
        hypothesis_phonemes=["d", "ɪ", "s"],
        errors=errors,
        learner_l1="Japanese",
    )
    return Diagnosis(report=report, validation=validation)


def test_errors_to_dataframe_empty_for_no_diagnosis() -> None:
    assert _errors_to_dataframe(None) == []


def test_errors_to_dataframe_empty_for_no_errors() -> None:
    assert _errors_to_dataframe(_make_diagnosis([])) == []


@requires_espeak
def test_errors_to_dataframe_highlights_substitution_in_word() -> None:
    diagnosis = _make_diagnosis([PhonemeError("substitution", "ð", "d", 0, "this")])
    rows = _errors_to_dataframe(diagnosis)
    word_html, op, expected, actual, misread = rows[0][1:]
    assert word_html == '<span style="color: crimson; font-weight: bold;">th</span>is'
    assert op == "substitution"
    assert expected == "/ð/"
    assert actual == "/d/"
    assert misread == ""


@requires_espeak
def test_errors_to_dataframe_preserves_case_when_highlighting() -> None:
    diagnosis = _make_diagnosis(
        [PhonemeError("substitution", "ð", "d", 0, "This")],
        target_text="This is high",
    )
    rows = _errors_to_dataframe(diagnosis)
    word_html = rows[0][1]
    assert word_html == '<span style="color: crimson; font-weight: bold;">Th</span>is'


@requires_espeak
def test_errors_to_dataframe_formats_deletion() -> None:
    diagnosis = _make_diagnosis(
        [PhonemeError("deletion", "k", None, 3, "desk")], target_text="desk"
    )
    rows = _errors_to_dataframe(diagnosis)
    word_html = rows[0][1]
    assert "text-decoration: underline" in word_html
    assert ">k<" in word_html
    assert rows[0][2:] == ["deletion", "/k/", "(missing)", ""]


def test_errors_to_dataframe_formats_insertion() -> None:
    diagnosis = _make_diagnosis([PhonemeError("insertion", None, "ə", 2, "this")])
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][1:] == [
        'this <span style="color: crimson;">+/ə/</span>',
        "insertion",
        "-",
        "/ə/",
        "",
    ]


def test_errors_to_dataframe_falls_back_to_plain_word_when_g2p_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_g2p(_text: str) -> Never:
        raise RuntimeError("espeak backend unavailable")

    monkeypatch.setattr(interface_module, "to_phonemes_by_word", fail_g2p)
    diagnosis = _make_diagnosis([PhonemeError("substitution", "ð", "d", 0, "this")])
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][1] == "this"


def test_errors_to_dataframe_propagates_unexpected_g2p_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_g2p(_text: str) -> Never:
        raise TypeError("unexpected implementation error")

    monkeypatch.setattr(interface_module, "to_phonemes_by_word", fail_g2p)
    diagnosis = _make_diagnosis([PhonemeError("substitution", "ð", "d", 0, "this")])

    with pytest.raises(TypeError, match="unexpected implementation error"):
        _errors_to_dataframe(diagnosis)


def test_errors_to_dataframe_uses_plain_word_for_unmapped_position() -> None:
    diagnosis = _make_diagnosis([PhonemeError("substitution", "ð", "d", 99, "this")])
    rows = _errors_to_dataframe(diagnosis)
    assert "<span" not in rows[0][1]


def test_errors_to_dataframe_marks_misread_with_replacement() -> None:
    diagnosis = _make_diagnosis(
        [
            PhonemeError(
                "substitution",
                "h",
                "b",
                5,
                "high",
                possibly_misread=True,
                misread_as="buy",
            )
        ]
    )
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][5] == '≠ transcript ("buy"?)'


def test_errors_to_dataframe_marks_misread_without_replacement_as_skipped() -> None:
    diagnosis = _make_diagnosis(
        [
            PhonemeError(
                "deletion", "h", None, 3, "high", possibly_misread=True, misread_as=None
            )
        ]
    )
    rows = _errors_to_dataframe(diagnosis)
    assert rows[0][5] == "≠ transcript (missing?)"


def _make_validation(passed: bool, wer: float = 0.0) -> ReadingValidation:
    return ReadingValidation(
        target_words=["this", "is", "high"],
        transcript_words=["this", "is", "high"],
        wer=wer,
        threshold=0.5,
        passed=passed,
    )


def test_transcript_markdown_empty_for_running_state() -> None:
    assert (
        _transcript_markdown(
            TrialState(stage="running", status_message="Transcribing…")
        )
        == ""
    )


def test_transcript_markdown_shows_gate_pass_with_validation() -> None:
    diagnosis = _make_diagnosis([], validation=_make_validation(passed=True, wer=0.1))
    state = TrialState(stage="detected", status_message="", diagnosis=diagnosis)
    md = _transcript_markdown(state)
    assert "this is high" in md
    assert "passed" in md
    assert "0.10" in md


def test_transcript_markdown_shows_no_gate_note_without_target_text() -> None:
    diagnosis = _make_diagnosis([], validation=None)
    state = TrialState(stage="detected", status_message="", diagnosis=diagnosis)
    md = _transcript_markdown(state)
    assert "no target sentence" in md


def test_transcript_markdown_shows_reading_mismatch_details() -> None:
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


def test_log_status_markdown_empty_while_running_or_detected() -> None:
    assert _log_status_markdown(TrialState(stage="running", status_message="")) == ""
    assert _log_status_markdown(TrialState(stage="detected", status_message="")) == ""


def test_log_status_markdown_shows_success() -> None:
    state = TrialState(stage="done", status_message="", logged=True)
    assert "Record saved" in _log_status_markdown(state)


def test_log_status_markdown_shows_failure_with_reason() -> None:
    state = TrialState(
        stage="done", status_message="", logged=False, log_error="disk full"
    )
    md = _log_status_markdown(state)
    assert "Failed to save record" in md
    assert "disk full" in md
