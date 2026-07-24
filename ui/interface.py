"""Gradio layout and event wiring (docs/design_ui.md §1).

The pure helpers (resolve_target, _errors_to_dataframe, _transcript_markdown,
_log_status_markdown) are unit-tested directly; build_app()'s gr.Blocks
wiring itself is exercised by the manual E2E check (docs/design_ui.md §9
step 6), the same way this project smoke-tests other model-backed glue code
instead of unit-testing it.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from pronunciation_coach.types import Diagnosis
from ui import config
from ui.models import AppModels
from ui.runner import TrialState, run_trial
from ui.sentences import TRIAL_SENTENCES
from ui.trial_logging import TargetSource

RESULT_HEADERS = ["#", "Word", "Type", "expected", "actual", "Misread?"]


def resolve_target(sentence_choice: str | None, custom_text: str) -> tuple[str | None, TargetSource]:
    """Free-text input wins over the preset dropdown when both are filled
    (docs/design_ui.md §1). Neither filled means no target text (free
    practice, no WER gate)."""
    custom_text = (custom_text or "").strip()
    if custom_text:
        return custom_text, "custom"
    if sentence_choice:
        return sentence_choice, "preset"
    return None, "preset"


def _errors_to_dataframe(diagnosis: Diagnosis | None) -> list[list]:
    if diagnosis is None:
        return []
    rows = []
    for i, error in enumerate(diagnosis.report.errors, start=1):
        if error.possibly_misread:
            misread = f'→ "{error.misread_as}"?' if error.misread_as else "(skipped?)"
        else:
            misread = ""
        rows.append(
            [i, error.word or "-", error.op, error.expected or "-", error.actual or "-", misread]
        )
    return rows


def _transcript_markdown(state: TrialState) -> str:
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

    with gr.Blocks(title="Pronunciation Coach — User Trial") as demo:
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

        audio = gr.Audio(sources=["microphone"], type="filepath", label="Recording")

        with gr.Row():
            run_button = gr.Button("▶ Run", variant="primary")
            clear_button = gr.Button("Clear results")

        status = gr.Markdown()
        transcript_display = gr.Markdown()
        results_table = gr.Dataframe(headers=RESULT_HEADERS, label="Detected errors")
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
