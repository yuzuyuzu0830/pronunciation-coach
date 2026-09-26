# Pronunciation Coach

An English pronunciation coaching system. It combines:

- Whisper for speech transcription
- a wav2vec2 CTC model for phoneme recognition and mispronunciation detection
- espeak-ng for reference pronunciation generation
- a local Ollama model for explanations of detected errors

The acoustic pipeline detects pronunciation errors. The LLM receives the
structured detection result and explains it; it does not perform detection.

## Requirements

- Python 3.10 or later
- Homebrew
- Enough disk space to download the Whisper, wav2vec2, and Ollama models

The first model-backed run downloads the selected Whisper and wav2vec2 models.

## Setup

Install the required system packages:

```bash
brew install espeak-ng ffmpeg ollama
```

Create a virtual environment and install the Python packages:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Ollama and download the model used by the trial UI:

```bash
# Keep Ollama running across terminal sessions and reboots.
brew services start ollama

# Alternatively, run it only in the current terminal.
# ollama serve

ollama pull llama3.1:8b
```

The explanation-comparison script uses `llama3.2:3b` by default. Pull it only
when running that experiment:

```bash
ollama pull llama3.2:3b
```

## Trial UI

Run the application from the repository root:

```bash
source venv/bin/activate
python app.py
```

The application checks espeak-ng, the configured models, and Ollama before
opening the local Gradio UI. The fixed trial configuration is defined in
`ui/config.py`.

In the UI:

1. Enter an anonymous participant ID. Do not enter a real name.
2. Select a target sentence or enter custom text.
3. Record the sentence with the microphone.
4. Select **Run** to transcribe, validate, diagnose, and explain the recording.

Trial metadata and copies of submitted recordings are saved under
`results/trial_logs/`. Treat this directory as participant data and do not
commit it.

## Command-line pipeline

Run the complete pipeline on an audio file:

```bash
python scripts/run_pipeline.py path/to/recording.wav \
  --target-text "This is the sentence to read."
```

`--target-text` is recommended because it enables the reading-mismatch check
before pronunciation diagnosis. Use `--help` to see the optional L1 and
Whisper model arguments.

## Smoke tests

The repository does not include sample recordings. Supply a local audio file
when testing Whisper or the phoneme recognizer:

```bash
# Whisper transcription
python scripts/smoke_test_whisper.py path/to/recording.wav

# wav2vec2 phoneme recognition
python scripts/smoke_test_phoneme.py path/to/recording.wav

# Ollama explanation from a fixed, structured detection result
python scripts/smoke_test_llm.py
```

These commands use real models and external services. They are separate from
the automated test suite.

## Evaluation data

Detection evaluation uses the SpeechOcean762 corpus. Download it from
[OpenSLR SLR101](https://www.openslr.org/101/) and extract it as follows:

```bash
mkdir -p data/.download
curl -C - --retry 10 --retry-delay 5 --retry-all-errors \
  -o data/.download/speechocean762.tar.gz \
  https://openslr.trmal.net/resources/101/speechocean762.tar.gz
tar -xzf data/.download/speechocean762.tar.gz -C data/
```

The expected corpus root is `data/speechocean762/`. Corpus files and generated
evaluation results are intentionally excluded from Git.

The evaluation command has three independent stages:

```bash
# Select a reproducible subset from the test split.
python scripts/evaluate_detection.py sample \
  --scores-json data/speechocean762/resource/scores.json \
  --utt2spk data/speechocean762/test/utt2spk \
  --n 300 \
  --seed 0 \
  --out data/speechocean762_subset/utt_ids.txt

# Run phoneme recognition. The JSONL output is resumable.
python scripts/evaluate_detection.py recognize \
  --scores-json data/speechocean762/resource/scores.json \
  --utt2spk data/speechocean762/test/utt2spk \
  --utt-ids data/speechocean762_subset/utt_ids.txt \
  --wave-root data/speechocean762/WAVE \
  --out results/so762_eval/hyp_phonemes.jsonl

# Compute FAR, FRR, and the available DER statistics.
python scripts/evaluate_detection.py score \
  --scores-json data/speechocean762/resource/scores.json \
  --utt2spk data/speechocean762/test/utt2spk \
  --utt-ids data/speechocean762_subset/utt_ids.txt \
  --hyp-jsonl results/so762_eval/hyp_phonemes.jsonl \
  --spk2age data/speechocean762/test/spk2age \
  --out-dir results/so762_eval \
  --seed 0 \
  --run-id so762 \
  --subset-description "300 test utterances"
```

SpeechOcean762 provides pronunciation accuracy scores but not the learner's
actual substituted phoneme. FAR and FRR can be computed, while substitution
identity DER may therefore be unavailable in the generated report.

## Experiment and analysis scripts

Additional scripts operate on trial logs or evaluation outputs:

```bash
# Compare Whisper sizes on saved trial recordings.
python scripts/compare_whisper_models.py --participant P01

# Compare Ollama models and prompt versions using fixed fixtures.
python scripts/compare_explanations.py

# Preview a saved session before replaying it with the current pipeline.
python scripts/rerun_session.py --session SESSION_ID_PREFIX --dry-run
```

Comparison reports are written under `docs/experiments/`. Trial replays and
evaluation artifacts are written under `results/`.

## Tests

Run the automated test suite from the repository root:

```bash
source venv/bin/activate
pytest tests/
```

The automated tests use fakes and mocks where appropriate. Run the smoke tests
separately when verifying local models and services.

## Project structure

```text
pronunciation_coach/
  transcriber.py          Whisper transcription
  phoneme_recognizer.py   wav2vec2 phoneme recognition
  g2p.py                  reference phoneme generation
  aligner.py              reference/hypothesis alignment
  pipeline.py             pipeline orchestration and error extraction
  knowledge.py            L1-specific and phoneme fallback knowledge
  explainer.py            Ollama explanation generation
  evaluation/             SpeechOcean762 parsing and metrics
ui/                       Gradio trial interface and logging
scripts/                  CLI, evaluation, replay, and experiment tools
tests/                    automated test suite
```
