import argparse
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

# Prompt embeds a known detection result so the LLM explains only;
# it must not be asked to discover errors from audio or raw transcripts.
DEFAULT_PROMPT = (
    'You are a pronunciation coach. A Japanese learner of English pronounced '
    '"this" as /d ɪ s/ instead of /ð ɪ s/. Briefly explain the error and how '
    "to fix it, in simple language."
)


def generate(prompt: str, model: str) -> str:
    """Call Ollama once and return the generated explanation text."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if "response" not in payload:
        raise KeyError(
            f"Ollama response missing 'response' field: {sorted(payload)}"
        )
    return payload["response"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Ollama explanation generation for MDD feedback."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        text = generate(args.prompt, args.model)
    except requests.exceptions.ConnectionError:
        print(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Start the server (e.g. `ollama serve`) and pull the model first.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"Ollama HTTP error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[model={args.model}]")
    print(text)


if __name__ == "__main__":
    main()
