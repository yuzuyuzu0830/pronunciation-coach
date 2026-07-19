# pronunciation-coach

English pronunciation coaching system that orchestrates Whisper (ASR),
wav2vec2/WavLM (phoneme-level detection), and a local LLM (explanations).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

System dependency for phonemizer / espeak-based phoneme models:

```bash
brew install espeak-ng
```

## Smoke tests

```bash
# Whisper transcription (tiny or base)
python scripts/smoke_test_whisper.py data/samples/sample.m4a

# wav2vec2 phoneme recognition
python scripts/smoke_test_phoneme.py data/samples/sample.m4a
```

## Tests

```bash
pytest tests/
```
