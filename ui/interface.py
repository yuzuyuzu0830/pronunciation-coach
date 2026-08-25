"""Gradio layout, result formatting, and event wiring.

Model-backed Blocks wiring is covered by the manual E2E check in
docs/design_ui.md; the formatting helpers are unit-tested directly.
"""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from pronunciation_coach.g2p import normalize, to_phonemes_by_word
from pronunciation_coach.types import Diagnosis, DiagnosisReport, PhonemeError
from ui import config
from ui.models import AppModels
from ui.phoneme_hints import locate_grapheme
from ui.runner import TrialState, run_trial
from ui.sentences import TRIAL_SENTENCES
from ui.trial_logging import TargetSource

RESULT_HEADERS = ["#", "Word", "Type", "expected", "actual", "Misread?"]
RESULT_DATATYPES = ["str", "html", "str", "str", "str", "str"]  # Word column renders HTML

# Keep crimson error highlights distinct from the theme's primary colour.
THEME = gr.themes.Default(primary_hue=gr.themes.colors.blue)

# Gradio uses an editing cursor for Dataframe cells even when interactive=False.
# Header buttons retain their normal cursor.
CSS = """
.readonly-table td,
.readonly-table td .cell-wrap,
.readonly-table td .cell-wrap span {
    cursor: default !important;
}
"""


def resolve_target(sentence_choice: str | None, custom_text: str) -> tuple[str | None, TargetSource]:
    """Prefer free text over a preset; return no target for free practice."""
    custom_text = (custom_text or "").strip()
    if custom_text:
        return custom_text, "custom"
    if sentence_choice:
        return sentence_choice, "preset"
    return None, "preset"


def _report_word_spans(report: DiagnosisReport) -> list[tuple[list[str], int]] | None:
    """Rebuild word phoneme offsets for optional grapheme highlighting."""
    base_text = report.target_text if report.target_text is not None else report.transcript
    try:
        # Match Pipeline.diagnose(): espeak can treat casing as pronunciation.
        raw_spans = to_phonemes_by_word(base_text)
    except (OSError, RuntimeError, ValueError):
        return None
    spans: list[tuple[list[str], int]] = []
    offset = 0
    for _, phones in raw_spans:
        phones = normalize(phones)
        spans.append((phones, offset))
        offset += len(phones)
    return spans


def _word_phonemes_at(
    word_spans: list[tuple[list[str], int]] | None, position: int
) -> tuple[list[str], int] | None:
    """Find the word and local index for a reference-phoneme position."""
    if word_spans is None:
        return None
    for phones, offset in word_spans:
        if offset <= position < offset + len(phones):
            return phones, position - offset
    return None


def _highlight_span(word: str, span: tuple[int, int], style: str) -> str:
    start, end = span
    before, target, after = word[:start], word[start:end], word[end:]
    css = (
        "text-decoration: underline; text-decoration-color: crimson; "
        "text-decoration-thickness: 2px;"
        if style == "deletion"
        else "color: crimson; font-weight: bold;"
    )
    return (
        html.escape(before)
        + f'<span style="{css}">{html.escape(target)}</span>'
        + html.escape(after)
    )


def _word_cell_html(error: PhonemeError, word_spans: list[tuple[list[str], int]] | None) -> str:
    """Format a word with its erroneous grapheme or inserted phone marked."""
    word = error.word or "-"
    if error.op == "insertion":
        escaped = html.escape(word)
        if error.actual is None:
            return escaped
        return escaped + f' <span style="color: crimson;">+/{html.escape(error.actual)}/</span>'

    span = None
    local = _word_phonemes_at(word_spans, error.position)
    if local is not None and error.expected is not None:
        phones, local_index = local
        span = locate_grapheme(word, error.expected, local_index, phones)
    if span is None:
        return html.escape(word)
    style = "deletion" if error.op == "deletion" else "substitution"
    return _highlight_span(word, span, style)


def _actual_cell(error: PhonemeError) -> str:
    if error.op == "deletion":
        return "(missing)"
    return f"/{error.actual}/" if error.actual else "-"


def _errors_to_dataframe(diagnosis: Diagnosis | None) -> list[list[str | int]]:
    """Format detected errors as rows for the Gradio results table."""
    if diagnosis is None:
        return []
    word_spans = _report_word_spans(diagnosis.report)
    rows: list[list[str | int]] = []
    for i, error in enumerate(diagnosis.report.errors, start=1):
        if error.possibly_misread:
            # Whisper may be wrong, so avoid claiming what the learner said.
            misread = (
                f'≠ transcript ("{error.misread_as}"?)'
                if error.misread_as
                else "≠ transcript (missing?)"
            )
        else:
            misread = ""
        rows.append(
            [
                i,
                _word_cell_html(error, word_spans),
                error.op,
                f"/{error.expected}/" if error.expected else "-",
                _actual_cell(error),
                misread,
            ]
        )
    return rows


def _transcript_markdown(state: TrialState) -> str:
    """Summarize the transcript and reading-validation result."""
    if state.stage == "reading_mismatch" and state.reading_mismatch is not None:
        v = state.reading_mismatch.validation
        return (
            f'**Transcript:** "{" ".join(v.transcript_words)}"  \n'
            f"**WER gate:** failed (WER={v.wer:.2f}, threshold={v.threshold:.2f})\n\n"
            f"Does not match the target sentence. Please read again.\n\n"
            f"- Target: {' '.join(v.target_words)}\n"
            f"- Transcript: {' '.join(v.transcript_words)}"
        )
    if state.diagnosis is not None:
        report = state.diagnosis.report
        v = state.diagnosis.validation
        gate = f"**WER gate:** passed (WER={v.wer:.2f})" if v is not None else "(no target sentence)"
        return f'**Transcript:** "{report.transcript}"  \n{gate}'
    return ""


def _log_status_markdown(state: TrialState) -> str:
    if state.stage in ("running", "detected"):
        return ""
    if state.logged:
        return "✅ Record saved"
    if state.log_error:
        return f"⚠️ Failed to save record (results are shown above): {state.log_error}"
    return ""


def build_app(models: AppModels) -> gr.Blocks:
    """Build the single-session Gradio trial interface."""
    session_trial_count = {"n": 0}

    def on_run(participant_id, l1, sentence_choice, custom_text, audio_path):
        target_text, target_source = resolve_target(sentence_choice, custom_text)
        session_trial_count["n"] += 1
        count = session_trial_count["n"]

        for state in run_trial(
            pipeline=models.pipeline,
            audio_path=Path(audio_path) if audio_path else None,
            participant_id=participant_id or "",
            learner_l1=l1,
            target_text=target_text,
            target_source=target_source,
            app_session_id=models.app_session_id,
            trial_logs_dir=models.trial_logs_dir,
            config=models.run_config,
        ):
            status = f"{state.status_message} (trial {count} this session)"
            yield (
                status,
                _transcript_markdown(state),
                _errors_to_dataframe(state.diagnosis),
                state.explanation or "",
                _log_status_markdown(state),
            )

    def on_clear():
        return None, "", "", [], "", ""

    with gr.Blocks(title="Pronunciation Coach — User Trial", theme=THEME, css=CSS) as demo:
        gr.Markdown("# Pronunciation Coach — User Trial")

        with gr.Row():
            participant_id = gr.Textbox(
                label="Participant ID",
                placeholder="e.g. P01 (do not enter a real name)",
            )
            l1 = gr.Dropdown(
                choices=list(config.L1_CHOICES),
                value=config.DEFAULT_L1,
                label="L1",
            )

        sentence_choice = gr.Dropdown(
            choices=[s.text for s in TRIAL_SENTENCES],
            label="Target sentence",
        )
        custom_text = gr.Textbox(
            label="Free text (overrides the target sentence when filled)",
            placeholder="",
        )

        # Participants re-record mistakes instead of editing the waveform.
        audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Recording (use ✕ to re-record)",
            editable=False,
        )

        with gr.Row():
            run_button = gr.Button("▶ Run", variant="primary")
            clear_button = gr.Button("Clear results")

        status = gr.Markdown()
        transcript_display = gr.Markdown()
        results_table = gr.Dataframe(
            headers=RESULT_HEADERS,
            datatype=RESULT_DATATYPES,
            label="Detected errors",
            interactive=False,
            elem_classes=["readonly-table"],
        )
        explanation = gr.Markdown(label="Explanation")
        log_status = gr.Markdown()

        run_button.click(
            fn=on_run,
            inputs=[participant_id, l1, sentence_choice, custom_text, audio],
            outputs=[status, transcript_display, results_table, explanation, log_status],
        )
        clear_button.click(
            fn=on_clear,
            inputs=[],
            outputs=[audio, status, transcript_display, results_table, explanation, log_status],
        )

    demo.queue(default_concurrency_limit=1)
    return demo
