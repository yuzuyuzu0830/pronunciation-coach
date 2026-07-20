"""Compare explanation quality across model × prompt-version combinations.

Runs every combination against fixed DiagnosisReport fixtures
(tests/fixtures/*.json) and writes the explanations, together with the input
conditions, to docs/experiments/explanations_<date>.md for side-by-side
review. Requires a running Ollama server with the requested models pulled.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pronunciation_coach.explainer import (
    DEFAULT_MODEL,
    PROMPT_VERSIONS,
    OllamaExplainer,
)
from pronunciation_coach.types import DiagnosisReport, PhonemeError

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_FIXTURES = ["e2e_feature_future", "simple_dh_d"]
OUTPUT_DIR = REPO_ROOT / "docs" / "experiments"


def load_report(path: Path) -> DiagnosisReport:
    data = json.loads(path.read_text())
    data.pop("_comment", None)
    errors = [PhonemeError(**e) for e in data.pop("errors")]
    return DiagnosisReport(errors=errors, **data)


def format_conditions(name: str, report: DiagnosisReport) -> list[str]:
    lines = [
        f"## Fixture: {name}",
        "",
        f'- target_text: `{report.target_text}`',
        f'- transcript: `{report.transcript}`',
        f"- learner_l1: {report.learner_l1}",
        f"- reference: `{' '.join(report.reference_phonemes)}`",
        f"- hypothesis: `{' '.join(report.hypothesis_phonemes)}`",
        "",
        "| # | op | expected | actual | word | possibly_misread | misread_as |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(report.errors, start=1):
        lines.append(
            f"| {i} | {e.op} | {e.expected or ''} | {e.actual or ''} "
            f"| {e.word or ''} | {e.possibly_misread} | {e.misread_as or ''} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare explanations across models and prompt versions"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=[DEFAULT_MODEL],
        help="Ollama model names (multiple allowed)",
    )
    parser.add_argument(
        "--prompt-version",
        nargs="+",
        choices=PROMPT_VERSIONS,
        default=list(PROMPT_VERSIONS),
        help="prompt versions to compare (default: all)",
    )
    parser.add_argument(
        "--fixture",
        nargs="+",
        default=DEFAULT_FIXTURES,
        help="fixture names under tests/fixtures/ (without .json)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="suffix for the output filename, to avoid overwriting "
        "an earlier run on the same day",
    )
    args = parser.parse_args()

    today = datetime.date.today().isoformat()
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUTPUT_DIR / f"explanations_{today}{suffix}.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Comparison experiment for explanation generation ({today})",
        "",
        f"- Model: {', '.join(args.model)}",
        f"- Prompt: {', '.join(args.prompt_version)}",
        "- Generation parameters: Ollama defaults (temperature is fixed so the text varies on re-runs)",
        "",
    ]

    for fixture_name in args.fixture:
        fixture_path = FIXTURES_DIR / f"{fixture_name}.json"
        report = load_report(fixture_path)
        lines.extend(format_conditions(fixture_name, report))

        for model in args.model:
            for version in args.prompt_version:
                print(f"[{fixture_name}] {model} × {version} ...", flush=True)
                explainer = OllamaExplainer(model=model, prompt_version=version)
                started = time.perf_counter()
                explanation = explainer.explain(report)
                elapsed = time.perf_counter() - started
                lines.extend(
                    [
                        f"### {model} × {version}  ({elapsed:.1f}s)",
                        "",
                        "```",
                        explanation.strip(),
                        "```",
                        "",
                    ]
                )

    out_path.write_text("\n".join(lines))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
