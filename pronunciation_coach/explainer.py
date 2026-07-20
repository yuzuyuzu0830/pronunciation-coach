"""Explanation generation from structured error reports (Ollama).

The LLM explains only; detection stays with the acoustic side. The prompt
therefore embeds already-detected errors and forbids re-judging them
(docs/design.md §7).
"""

from __future__ import annotations

import requests

from pronunciation_coach.types import DiagnosisReport, PhonemeError

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_BASE_URL = "http://localhost:11434"

_ROLE_INSTRUCTION_V1 = """\
You are a pronunciation coach for English learners.
A separate acoustic system has already detected the pronunciation errors \
listed below. Your job is ONLY to explain them:
- Do not re-judge, add, or remove errors.
- Do not invent phoneme symbols; use only the symbols given below.
- Explain in simple, plain English without specialised phonetic jargon."""

# v2: hardened against three issues seen in practice (symbol invention,
# structural collapse, and dubious L1 generalizations). v1 is kept for comparison.
_ROLE_INSTRUCTION_V2 = """\
You are a pronunciation coach for English learners.
A separate acoustic system has already detected the pronunciation errors \
listed below. Your job is ONLY to explain them:
- Do not re-judge, add, or remove errors.
- Do not write any new phonetic or IPA symbols: quote only the symbols that \
appear in the error list below. Refer to a reading mistake using ordinary \
word spelling only, never symbols.
- Keep the exact order of the error list: one numbered item per error, and \
start each item with the error's number from the list. Do not merge, split, \
or repeat items.
- Do not make generalised claims about the learner's L1 phonology (such as \
"Japanese speakers tend to ...") unless you are certain they are true. Focus \
on describing what happened and giving a practice method.
- Explain in simple, plain English without specialised phonetic jargon."""

_ROLE_INSTRUCTIONS = {"v1": _ROLE_INSTRUCTION_V1, "v2": _ROLE_INSTRUCTION_V2}
PROMPT_VERSIONS = tuple(_ROLE_INSTRUCTIONS)
DEFAULT_PROMPT_VERSION = "v1"


def _format_error(index: int, error: PhonemeError) -> str:
    location = f'in word "{error.word}"' if error.word else f"at position {error.position}"
    if error.op == "substitution":
        detail = f"expected /{error.expected}/ but heard /{error.actual}/"
    elif error.op == "deletion":
        detail = f"expected /{error.expected}/ but it was missing"
    else:
        detail = f"extra /{error.actual}/ was added"
    return f"{index}. {error.op} {location}: {detail}"


def _format_misread_notice(word: str | None, read_as: str | None) -> str:
    if read_as is None:
        return f'- The word "{word}" seems to have been skipped.'
    return f'- The word "{word}" may have been read as "{read_as}".'


def build_prompt(
    report: DiagnosisReport, version: str = DEFAULT_PROMPT_VERSION
) -> str:
    if version not in _ROLE_INSTRUCTIONS:
        raise ValueError(
            f"Unknown prompt version {version!r}; available: {PROMPT_VERSIONS}"
        )
    lines = [_ROLE_INSTRUCTIONS[version], ""]
    lines.append(f"Learner's first language (L1): {report.learner_l1}")
    if report.target_text is not None:
        lines.append(f'Target sentence: "{report.target_text}"')
    lines.append(f'What the learner said (transcript): "{report.transcript}"')
    lines.append("")

    if not report.errors:
        lines.append(
            "No pronunciation errors were detected. Briefly praise the "
            "learner and encourage them to keep practicing."
        )
        return "\n".join(lines)

    pronunciation_errors = [e for e in report.errors if not e.possibly_misread]
    # One notice per misread word, not per phoneme error inside it.
    misread_notices = dict.fromkeys(
        (e.word, e.misread_as) for e in report.errors if e.possibly_misread
    )

    if pronunciation_errors:
        lines.append("Detected pronunciation errors:")
        lines.extend(
            _format_error(i, error)
            for i, error in enumerate(pronunciation_errors, start=1)
        )
        lines.append("")
    if misread_notices:
        lines.append("Possible reading mistakes (a different word was read):")
        lines.extend(
            _format_misread_notice(word, read_as)
            for word, read_as in misread_notices
        )
        lines.append("")

    if pronunciation_errors:
        lines.append(
            "For each pronunciation error, explain what happened and give one "
            "practical tip to fix it, considering difficulties typical for "
            "the learner's L1."
        )
    if misread_notices:
        lines.append(
            "For each possible reading mistake, do not explain individual "
            "sounds — these are not pronunciation habits. Instead, point out "
            "which word was likely read (or that it was skipped), and ask the "
            "learner to check the target word and read it again."
        )
    return "\n".join(lines)


class OllamaExplainer:
    """Explainer implementation backed by a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        if prompt_version not in _ROLE_INSTRUCTIONS:
            raise ValueError(
                f"Unknown prompt version {prompt_version!r}; "
                f"available: {PROMPT_VERSIONS}"
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._prompt_version = prompt_version

    def explain(self, report: DiagnosisReport) -> str:
        prompt = build_prompt(report, version=self._prompt_version)
        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Start the server with `ollama serve` (or `brew services "
                f"start ollama`) and pull the model with `ollama pull {self._model}`."
            ) from exc

        payload = response.json()
        if "response" not in payload:
            raise KeyError(
                f"Ollama response missing 'response' field: {sorted(payload)}"
            )
        return payload["response"]
