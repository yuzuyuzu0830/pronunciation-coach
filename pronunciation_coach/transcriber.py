"""Speech-to-text transcription (Whisper)."""

from __future__ import annotations

from pathlib import Path

import whisper


class WhisperTranscriber:
    """Transcriber implementation backed by openai-whisper.

    The model is loaded once at construction so multiple files can be
    processed without reloading.
    """

    def __init__(self, model_size: str = "base") -> None:
        self._model = whisper.load_model(model_size)

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        result = self._model.transcribe(str(audio_path), language="en")
        text = str(result["text"]).strip()
        if not text:
            raise ValueError(
                f"Whisper produced an empty transcript for {audio_path}. "
                "Check that the recording contains speech."
            )
        return text
