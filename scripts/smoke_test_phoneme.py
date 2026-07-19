import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEFAULT_AUDIO_PATH = Path("data/samples/sample.m4a")
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# wav2vec2 expects 16 kHz mono input.
TARGET_SAMPLE_RATE = 16000


def get_device() -> torch.device:
    """Prefer Apple MPS when available; otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_audio(audio_path: Path) -> torch.Tensor:
    """Load audio as a mono 16 kHz waveform tensor.

    Stereo inputs are averaged to mono. Resampling uses torchaudio so the
    script stays independent of ffmpeg CLI tools when torchcodec is installed.
    """
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
    return waveform.squeeze(0)


def recognize_phonemes(audio_path: Path, device: torch.device) -> str:
    """Run greedy CTC decoding and return the predicted phoneme string.

    Greedy argmax is enough for a smoke test; production MDD may use
    beam search or forced alignment against a reference transcript.
    """
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device).eval()

    waveform = load_audio(audio_path)
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH, type=Path)
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    phonemes = recognize_phonemes(args.audio_path, device)
    print(f"[device={device}] {args.audio_path}")
    print(phonemes)


if __name__ == "__main__":
    main()
