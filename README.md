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

Local LLM explanations use [Ollama](https://ollama.com/) (`http://localhost:11434`).
A one-off `ollama serve` in a terminal dies when that session ends or the Mac
reboots, so the LLM smoke test will fail with a connection error until you
start Ollama again.

```bash
brew install ollama
# Prefer this during development so the server survives reboots:
brew services start ollama
# Or start only for the current session (must re-run after reboot):
# ollama serve

ollama pull llama3.2:3b
```
## Smoke tests

```bash
# Whisper transcription (tiny or base)
python scripts/smoke_test_whisper.py data/samples/sample.m4a

# wav2vec2 phoneme recognition
python scripts/smoke_test_phoneme.py data/samples/sample.m4a

# Local LLM explanation via Ollama (detection is mocked in the prompt)
python scripts/smoke_test_llm.py
```

## Tests

```bash
pytest tests/
```
