"""Explanation generation from structured error reports (Ollama).

The LLM explains only; detection stays with the acoustic side. The prompt
therefore embeds already-detected errors and forbids re-judging them
(docs/design.md §7).
"""

from __future__ import annotations

import requests

from pronunciation_coach.knowledge import match_knowledge
from pronunciation_coach.types import DiagnosisReport, KnowledgeRecord, PhonemeError

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
- For the numbered error list only: keep its exact order, write one numbered \
item per error, and start each item with the error's number from the list. \
Do not merge, split, or repeat items.
- If a "Possible reading mistakes" section is given, always address it in a \
separate final section after the numbered items; never drop it.
- Do not make generalised claims about the learner's L1 phonology (such as \
"Japanese speakers tend to ...") unless you are certain they are true. Focus \
on describing what happened and giving a practice method.
- Explain in simple, plain English without specialised phonetic jargon."""

# v3: structured-knowledge injection (docs/design_3c.md §5). Detection- and
# structure-related v2 rules still apply; the role is narrowed from "explain"
# to "rephrase pre-written material", since build_prompt now supplies a cause
# and tip for the errors it can classify (§0: fixes 3b's fabricated symbols
# and dubious L1 generalizations by not leaving that content to the model).
_ROLE_INSTRUCTION_V3 = """\
You are a pronunciation coach for English learners.
A separate acoustic system has already detected the pronunciation errors \
listed below. For each error, you are given a pre-written cause and practice \
tip when available. Your job is only to rephrase the given material into \
natural, encouraging coaching language:
- Do not re-judge, add, or remove errors.
- Do not invent facts, phoneme symbols, causes, or practice words beyond \
what is given below.
- If no cause or tip is given for an error, state only that the error \
occurred and do not speculate about why.
- Do not write any new phonetic or IPA symbols: quote only the symbols that \
appear in the error list below. Refer to a reading mistake using ordinary \
word spelling only, never symbols.
- For the numbered error list only: keep its exact order, write one numbered \
item per error, and start each item with the error's number from the list. \
Do not merge, split, or repeat items.
- If a "Possible reading mistakes" section is given, always address it in a \
separate final section after the numbered items; never drop it.
- Explain in simple, plain English without specialised phonetic jargon."""

_ROLE_INSTRUCTIONS = {
    "v1": _ROLE_INSTRUCTION_V1,
    "v2": _ROLE_INSTRUCTION_V2,
    "v3": _ROLE_INSTRUCTION_V3,
}
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


def _tier_sort_key(record: KnowledgeRecord | None) -> int:
    if record is None:
        return 2
    return 0 if record.tier == "l1_specific" else 1


def rank_errors_by_tier(
    errors: list[PhonemeError], reference_phonemes: list[str], l1: str
) -> list[tuple[PhonemeError, KnowledgeRecord | None]]:
    """Pair each error with its matched knowledge record and sort by tier.

    Tier order: l1_specific > phoneme_fallback > no match (docs/design_3c.md
    §4). The sort is stable, so errors within the same tier keep their
    original relative order — §4 does not define a secondary key. Callers
    should pass only non-misread errors; misread words are handled
    separately and never go through knowledge matching.
    """
    pairs = [
        (error, match_knowledge(error, reference_phonemes, l1)) for error in errors
    ]
    return sorted(pairs, key=lambda pair: _tier_sort_key(pair[1]))


def _format_knowledge_block(record: KnowledgeRecord) -> list[str]:
    """Inline knowledge directly under its error (docs/design_3c.md §5): no
    separate section for the model to (mis)associate with the wrong item.
    Fallback-tier records have cause=None (no L1-transfer claim, §3), so the
    Cause line is omitted rather than printed as empty/None.
    """
    block = [f"   Known phenomenon: {record.id}"]
    if record.cause is not None:
        block.append(f"   Cause: {record.cause}")
    block.append(f"   Tip: {record.articulation_tip}")
    words = ", ".join(
        f"{pw.word} (/{pw.target_phoneme}/)" for pw in record.practice_words
    )
    block.append(f"   Practice words: {words}")
    return block


def _append_v3_error_section(
    lines: list[str],
    report: DiagnosisReport,
    pronunciation_errors: list[PhonemeError],
    full_explanation_limit: int,
) -> None:
    """v3 tiering: sort by tier, inline knowledge for the top N, facts-only
    for the rest — every error is still listed, only the level of detail
    differs (docs/design_3c.md §4).
    """
    if not pronunciation_errors:
        return
    ranked = rank_errors_by_tier(
        pronunciation_errors, report.reference_phonemes, report.learner_l1
    )
    total = len(ranked)
    explained = min(full_explanation_limit, total)
    lines.append(f"Detected {total} errors, {explained} explained in detail below.")
    lines.append("")
    # Literal count guards against the model inventing extra items when
    # total == 1 (regression observed in the 2026-07-20 comparison run, §5).
    lines.append(
        f"There are exactly {total} numbered items below (this holds even "
        f"when {total} == 1). Output exactly {total} numbered items — do "
        "not add an extra item such as 'no other errors were found', even "
        "if there is only one."
    )
    lines.append("")
    lines.append("Detected pronunciation errors:")
    for i, (error, record) in enumerate(ranked, start=1):
        lines.append(_format_error(i, error))
        if i <= full_explanation_limit and record is not None:
            lines.extend(_format_knowledge_block(record))
    lines.append("")


def build_prompt(
    report: DiagnosisReport,
    version: str = DEFAULT_PROMPT_VERSION,
    full_explanation_limit: int = 3,
) -> str:
    if version not in _ROLE_INSTRUCTIONS:
        raise ValueError(
            f"Unknown prompt version {version!r}; available: {PROMPT_VERSIONS}"
        )
    # Single source of truth for the v3-vs-legacy split, checked once here
    # instead of comparing `version` against "v3" independently at each spot
    # below. Two independent comparisons previously had to be kept in sync by
    # hand (one `== "v3"`, one `!= "v3"`); a future version added to only one
    # of them would silently mix its error-section format with the wrong
    # trailing instruction text instead of failing loudly.
    uses_structured_knowledge = version == "v3"

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

    if uses_structured_knowledge:
        _append_v3_error_section(lines, report, pronunciation_errors, full_explanation_limit)
    elif pronunciation_errors:
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

    if not uses_structured_knowledge and pronunciation_errors:
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
        full_explanation_limit: int = 3,
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
        self._full_explanation_limit = full_explanation_limit

    def explain(self, report: DiagnosisReport) -> str:
        prompt = build_prompt(
            report,
            version=self._prompt_version,
            full_explanation_limit=self._full_explanation_limit,
        )
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
