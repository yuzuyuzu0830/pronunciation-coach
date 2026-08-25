"""Explanation generation from structured error reports (Ollama).

The LLM explains only; detection stays with the acoustic side. 
The prompt therefore embeds already-detected errors and forbids re-judging them.
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

# v2 prevents observed symbol invention, structural collapse, and unsupported
# L1 generalizations. v1 remains available for comparison.
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

# v3 limits the LLM to rephrasing matched knowledge.
# Unmatched errors are rendered deterministically because the model invented
# tips when asked to describe facts-only items in the 2026-07-22 run.
_ROLE_INSTRUCTION_V3 = """\
You are a pronunciation coach for English learners.
A separate acoustic system has already detected the pronunciation errors \
listed below. For each error, you are given a pre-written cause and practice \
tip when available. Your job is only to rephrase the given material into \
natural, encouraging coaching language:
- Do not re-judge, add, or remove errors.
- Do not invent facts, phoneme symbols, causes, or practice words beyond \
what is given below.
- If the given cause mentions uncertainty or alternative explanations, you \
must preserve them in your rephrasing — do not present an uncertain cause \
as certain.
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
    """Match errors to knowledge and sort them by explanation priority.

    The stable order is L1-specific, phoneme fallback, then unmatched.
    Misread words are handled separately.
    """
    pairs = [
        (error, match_knowledge(error, reference_phonemes, l1)) for error in errors
    ]
    return sorted(pairs, key=lambda pair: _tier_sort_key(pair[1]))


def _format_knowledge_block(record: KnowledgeRecord) -> list[str]:
    """Place knowledge under its error to preserve the association.

    Fallback records omit L1 causes and internal phenomenon ids after the
    model exposed an id verbatim in the 2026-07-22 run.
    """
    block = []
    if record.tier == "l1_specific":
        block.append(f"   Known phenomenon: {record.phenomenon}")
    if record.cause is not None:
        block.append(f"   Cause: {record.cause}")
    block.append(f"   Tip: {record.articulation_tip}")
    words = ", ".join(
        f"{pw.word} (/{pw.target_phoneme}/)" for pw in record.practice_words
    )
    block.append(f"   Practice words: {words}")
    return block


def format_facts_only_error(error: PhonemeError) -> str:
    """Deterministic learner-facing sentence for a facts-only error.

    These errors bypass the LLM because it invented unsupported tips for them
    in the 2026-07-22 comparison run.
    """
    location = (
        f'In the word "{error.word}"'
        if error.word
        else f"At position {error.position}"
    )
    if error.op == "substitution":
        return f"{location}, expected /{error.expected}/ but heard /{error.actual}/."
    if error.op == "deletion":
        return f"{location}, expected /{error.expected}/ but it was missing."
    return f"{location}, an extra /{error.actual}/ was added."


FACTS_ONLY_HEADING = "Other detected differences:"

# At M=0 the block is the whole output, so "Other" has nothing to contrast
# with; only that call site overrides the heading.
ZERO_M_HEADING = "Detected differences:"

# M=0 does not prove correct pronunciation, so this deterministic preamble
# states only what the system found and avoids unsupported praise.
ZERO_M_PREAMBLE = (
    "Differences were detected, and none of them matched a known "
    "pronunciation pattern.\n"
    "There is no detailed explanation for them, so the list below is "
    "reference information."
)


def render_facts_only_section(
    errors: list[PhonemeError], *, heading: str = FACTS_ONLY_HEADING
) -> str:
    """Render facts-only errors, omitting the heading for empty input."""
    if not errors:
        return ""
    lines = [heading]
    lines.extend(f"- {format_facts_only_error(error)}" for error in errors)
    return "\n".join(lines)


def _require_non_negative_limit(full_explanation_limit: int) -> None:
    """Reject negatives: Python would treat them as reverse slice indices."""
    if full_explanation_limit < 0:
        raise ValueError(
            f"full_explanation_limit must be >= 0, got {full_explanation_limit}"
        )


def _split_full_and_facts_only(
    pronunciation_errors: list[PhonemeError],
    reference_phonemes: list[str],
    l1: str,
    full_explanation_limit: int,
) -> tuple[list[tuple[PhonemeError, KnowledgeRecord]], list[PhonemeError]]:
    """Split errors into LLM-bound and deterministic groups.

    Only the top N errors with matched knowledge reach the LLM.
    Unmatched and lower-ranked errors remain facts-only.
    """
    _require_non_negative_limit(full_explanation_limit)
    ranked = rank_errors_by_tier(pronunciation_errors, reference_phonemes, l1)
    matched = sum(1 for _, record in ranked if record is not None)
    full_count = min(full_explanation_limit, matched)
    return ranked[:full_count], [error for error, _ in ranked[full_count:]]


def _facts_only_errors(
    report: DiagnosisReport, full_explanation_limit: int
) -> list[PhonemeError]:
    """Return errors that must be rendered outside the LLM response."""
    pronunciation_errors = [e for e in report.errors if not e.possibly_misread]
    if not pronunciation_errors:
        return []
    _, facts_only = _split_full_and_facts_only(
        pronunciation_errors,
        report.reference_phonemes,
        report.learner_l1,
        full_explanation_limit,
    )
    return facts_only


def _append_v3_error_section(
    lines: list[str],
    report: DiagnosisReport,
    pronunciation_errors: list[PhonemeError],
    full_explanation_limit: int,
) -> None:
    """Add only the top-N matched errors to a v3 prompt.

    Facts-only errors are rendered after the LLM output.
    """
    if not pronunciation_errors:
        return
    full_pairs, _ = _split_full_and_facts_only(
        pronunciation_errors,
        report.reference_phonemes,
        report.learner_l1,
        full_explanation_limit,
    )
    total = len(pronunciation_errors)
    explained = len(full_pairs)
    lines.append(f"Detected {total} errors, {explained} explained in detail below.")
    lines.append("")
    if not full_pairs:
        # M=0: the LLM sees no error details, so forbid reconstructing them
        # from the transcript; the factual list is appended deterministically.
        lines.append(
            "No errors are listed for detailed explanation. Do not describe "
            "or guess any specific sounds or errors; briefly encourage the "
            "learner to keep practicing. A factual list of the detected "
            "differences will be shown to the learner separately."
        )
        lines.append("")
        return
    # Literal count guards against the model inventing extra items when
    # explained == 1 (regression observed in the 2026-07-20 comparison run).
    lines.append(
        f"There are exactly {explained} numbered items below (this holds even "
        f"when {explained} == 1). Output exactly {explained} numbered items — do "
        "not add an extra item such as 'no other errors were found', even "
        "if there is only one."
    )
    if explained < total:
        lines.append(
            "Any remaining differences beyond these items are shown to the "
            "learner separately; do not mention, guess, or explain them."
        )
    lines.append("")
    lines.append("Detected pronunciation errors:")
    for i, (error, record) in enumerate(full_pairs, start=1):
        lines.append(_format_error(i, error))
        lines.extend(_format_knowledge_block(record))
    lines.append("")


def _is_zero_m(
    uses_structured_knowledge: bool,
    pronunciation_errors: list[PhonemeError],
    report: DiagnosisReport,
    full_explanation_limit: int,
) -> bool:
    """Check whether v3 has errors but none selected for LLM explanation.

    This repeats a pure knowledge lookup also used when building the prompt.
    """
    if not uses_structured_knowledge or not pronunciation_errors:
        return False
    full_pairs, _ = _split_full_and_facts_only(
        pronunciation_errors,
        report.reference_phonemes,
        report.learner_l1,
        full_explanation_limit,
    )
    return not full_pairs


def _should_skip_llm(
    report: DiagnosisReport,
    version: str,
    full_explanation_limit: int,
) -> bool:
    """True when the prompt would leave the LLM nothing to say.

    At M=0, the 2026-08-23 run produced unsupported numbered items and
    reading-mistake sections. The deterministic facts-only block did not, so
    it replaces the LLM call. Reports with misreads still need word-level
    coaching and remain eligible for the LLM.
    """
    _require_non_negative_limit(full_explanation_limit)
    if any(error.possibly_misread for error in report.errors):
        return False
    return _is_zero_m(version == "v3", report.errors, report, full_explanation_limit)


def build_prompt(
    report: DiagnosisReport,
    version: str = DEFAULT_PROMPT_VERSION,
    full_explanation_limit: int = 3,
) -> str:
    """Build a versioned prompt for a diagnosis report.

    In v3, only errors with matched knowledge are included for explanation.
    """
    _require_non_negative_limit(full_explanation_limit)
    if version not in _ROLE_INSTRUCTIONS:
        raise ValueError(
            f"Unknown prompt version {version!r}; available: {PROMPT_VERSIONS}"
        )
    # Keep the v3-versus-legacy decision consistent across all prompt sections.
    uses_structured_knowledge = version == "v3"

    pronunciation_errors = [e for e in report.errors if not e.possibly_misread]
    # One notice per misread word, not per phoneme error inside it.
    misread_notices = dict.fromkeys(
        (e.word, e.misread_as) for e in report.errors if e.possibly_misread
    )
    # Withhold the transcript and misread section at M=0; either can prompt
    # the model to reconstruct details absent from the structured error list.
    # The UI still displays misread facts directly.
    zero_m = _is_zero_m(uses_structured_knowledge, pronunciation_errors, report, full_explanation_limit)

    lines = [_ROLE_INSTRUCTIONS[version], ""]
    lines.append(f"Learner's first language (L1): {report.learner_l1}")
    if report.target_text is not None:
        lines.append(f'Target sentence: "{report.target_text}"')
    if not zero_m:
        lines.append(f'What the learner said (transcript): "{report.transcript}"')
    lines.append("")

    if not report.errors:
        lines.append(
            "No pronunciation errors were detected. Briefly praise the "
            "learner and encourage them to keep practicing."
        )
        return "\n".join(lines)

    if uses_structured_knowledge:
        _append_v3_error_section(lines, report, pronunciation_errors, full_explanation_limit)
    elif pronunciation_errors:
        lines.append("Detected pronunciation errors:")
        lines.extend(
            _format_error(i, error)
            for i, error in enumerate(pronunciation_errors, start=1)
        )
        lines.append("")
    if misread_notices and not zero_m:
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
    if misread_notices and not zero_m:
        lines.append(
            "For each possible reading mistake, do not explain individual "
            "sounds — these are not pronunciation habits. Instead, point out "
            "which word was likely read (or that it was skipped), and ask the "
            "learner to check the target word and read it again. Note that "
            "the transcript itself may be a mis-transcription rather than a "
            "genuine reading mistake, so phrase this as a possibility, not "
            "a certainty."
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
        _require_non_negative_limit(full_explanation_limit)
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._prompt_version = prompt_version
        self._full_explanation_limit = full_explanation_limit

    def explain(self, report: DiagnosisReport) -> str:
        """Generate coaching text, bypassing the LLM for facts-only v3 output."""
        if _should_skip_llm(report, self._prompt_version, self._full_explanation_limit):
            # M=0 always has at least one facts-only error here.
            return ZERO_M_PREAMBLE + "\n\n" + render_facts_only_section(
                _facts_only_errors(report, self._full_explanation_limit),
                heading=ZERO_M_HEADING,
            )
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
        text = payload["response"]
        if self._prompt_version == "v3":
            section = render_facts_only_section(
                _facts_only_errors(report, self._full_explanation_limit)
            )
            if section:
                text = text.rstrip() + "\n\n" + section
        return text
