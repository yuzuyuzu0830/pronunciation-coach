"""Phoneme recognition from audio (wav2vec2, espeak IPA output)."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

DEFAULT_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# wav2vec2 expects 16 kHz mono input.
TARGET_SAMPLE_RATE = 16000


def default_device() -> torch.device:
    """Prefer Apple MPS when available; otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_audio(audio_path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
    return waveform.squeeze(0)


def _load_processor(model_name: str) -> Wav2Vec2Processor:
    """Build the processor from an explicit tokenizer + feature extractor.

    Wav2Vec2Processor.from_pretrained(model_name) can fail to auto-resolve
    the phoneme tokenizer class for some espeak-phoneme CTC repos (a known
    issue observed when comparing candidate models); constructing each part
    explicitly and composing them sidesteps that regardless of which repo
    triggers it.
    """
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


class Wav2Vec2PhonemeRecognizer:
    """PhonemeRecognizer implementation backed by a wav2vec2 CTC model.

    The model is loaded once at construction so multiple files can be
    processed without reloading. Greedy argmax decoding is enough for the
    prototype (docs/design.md §1).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: torch.device | None = None,
    ) -> None:
        self._device = device if device is not None else default_device()
        self._processor = _load_processor(model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self._device).eval()

    def recognize(self, audio_path: Path) -> list[str]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = _load_audio(audio_path)
        inputs = self._processor(
            waveform.numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(inputs.input_values.to(self._device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = self._processor.batch_decode(predicted_ids)[0].split()

        if not phonemes:
            raise ValueError(
                f"Phoneme recognition produced an empty sequence for {audio_path}. "
                "Check that the recording contains speech."
            )
        return phonemes
