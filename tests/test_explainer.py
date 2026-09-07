import importlib.util
import re
from pathlib import Path

import pytest

from pronunciation_coach.explainer import (
    NO_ERRORS_MESSAGE,
    ZERO_M_HEADING,
    ZERO_M_PREAMBLE,
    OllamaExplainer,
    _should_skip_llm,
    build_prompt,
    format_facts_only_error,
    rank_errors_by_tier,
    render_facts_only_section,
)
from pronunciation_coach.types import DiagnosisReport, PhonemeError


_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "compare_explanations.py"
_SPEC = importlib.util.spec_from_file_location("compare_explanations_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_SCRIPT_PATH}")
compare_explanations = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_explanations)


def make_report(errors: list[PhonemeError]) -> DiagnosisReport:
    return DiagnosisReport(
        transcript="dis is high",
        target_text="this is high",
        reference_phonemes=["ð", "ɪ", "s", "ɪ", "z", "h", "aɪ"],
        hypothesis_phonemes=["d", "ɪ", "ɪ", "z", "h", "aɪ", "ɯ"],
        errors=errors,
        learner_l1="Japanese",
    )


SAMPLE_ERRORS = [
    PhonemeError("substitution", "ð", "d", 0, "this"),
    PhonemeError("deletion", "s", None, 2, "this"),
    PhonemeError("insertion", None, "ɯ", 6, "high"),
]


def test_prompt_contains_every_error_field():
    """All op types, phonemes, and word attributions must appear."""
    prompt = build_prompt(make_report(SAMPLE_ERRORS))
    assert "substitution" in prompt
    assert "deletion" in prompt
    assert "insertion" in prompt
    assert "/ð/" in prompt
    assert "/d/" in prompt
    assert "/s/" in prompt
    assert "/ɯ/" in prompt
    assert '"this"' in prompt
    assert '"high"' in prompt


def test_prompt_contains_l1_and_texts():
    prompt = build_prompt(make_report(SAMPLE_ERRORS))
    assert "Japanese" in prompt
    assert "this is high" in prompt  # target text
    assert "dis is high" in prompt  # transcript


def test_prompt_restricts_llm_to_explanation_only():
    """The role instruction must forbid detection and symbol invention."""
    prompt = build_prompt(make_report(SAMPLE_ERRORS))
    assert "pronunciation coach" in prompt
    assert "Do not re-judge, add, or remove errors" in prompt
    assert "Do not invent phoneme symbols" in prompt


def test_prompt_without_errors_asks_for_praise():
    prompt = build_prompt(make_report([]))
    assert "No pronunciation errors were detected" in prompt
    assert "substitution" not in prompt
    assert "Japanese" in prompt


# --- possibly-misread errors: word-level notice instead of phoneme detail ---

MISREAD_ERRORS = [
    PhonemeError(
        "substitution", "h", "b", 5, "high",
        possibly_misread=True, misread_as="buy",
    ),
]


def test_prompt_turns_flagged_errors_into_word_level_notice():
    prompt = build_prompt(make_report(MISREAD_ERRORS))
    assert '"high"' in prompt
    assert '"buy"' in prompt
    assert "read as" in prompt
    # No phoneme-level detail for reading mistakes:
    assert "/h/" not in prompt
    assert "/b/" not in prompt


def test_prompt_flagged_omitted_word_reports_skip():
    errors = [
        PhonemeError("deletion", "h", None, 5, "high", possibly_misread=True)
    ]
    prompt = build_prompt(make_report(errors))
    assert '"high"' in prompt
    assert "skipped" in prompt
    assert "/h/" not in prompt


def test_prompt_mixed_errors_keeps_phoneme_detail_for_unflagged():
    errors = [PhonemeError("substitution", "ð", "d", 0, "this")] + MISREAD_ERRORS
    prompt = build_prompt(make_report(errors))
    assert "/ð/" in prompt
    assert "/d/" in prompt
    assert '"buy"' in prompt
    assert "/b/" not in prompt


def test_prompt_dedupes_flagged_errors_of_the_same_word():
    errors = [
        PhonemeError(
            "substitution", "h", "b", 5, "high",
            possibly_misread=True, misread_as="buy",
        ),
        PhonemeError(
            "substitution", "aɪ", "i", 6, "high",
            possibly_misread=True, misread_as="buy",
        ),
    ]
    prompt = build_prompt(make_report(errors))
    assert prompt.count('"buy"') == 1


def test_prompt_instructs_word_level_guidance_for_misreads():
    prompt = build_prompt(make_report(MISREAD_ERRORS))
    assert "do not explain individual sounds" in prompt.lower()
    assert "read it again" in prompt


def test_prompt_hedges_misread_notice_as_possible_transcription_error():
    """A misread flag can itself be a Whisper mis-transcription, not a real
    reading mistake (observed in the P01 trial: 'weather' transcribed as
    'river'). The instruction must not tell the model to treat it as fact."""
    prompt = build_prompt(make_report(MISREAD_ERRORS))
    assert "transcription" in prompt.lower()
    assert "possibility" in prompt.lower() or "may " in prompt.lower()


# --- prompt v2: strengthened instructions, selectable per version ---


def test_v1_is_the_default_version():
    report = make_report(SAMPLE_ERRORS)
    assert build_prompt(report) == build_prompt(report, version="v1")


def test_unknown_prompt_version_raises_value_error():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        build_prompt(make_report(SAMPLE_ERRORS), version="v99")


def test_negative_full_explanation_limit_raises_in_build_prompt():
    """Negatives become reverse slice indices in ranked[:limit]; reject early."""
    with pytest.raises(ValueError, match="full_explanation_limit must be >= 0"):
        build_prompt(
            make_report(SAMPLE_ERRORS),
            version="v3",
            full_explanation_limit=-1,
        )


def test_negative_full_explanation_limit_raises_in_explainer_init():
    with pytest.raises(ValueError, match="full_explanation_limit must be >= 0"):
        OllamaExplainer(full_explanation_limit=-1)


def test_v2_forbids_writing_new_symbols():
    prompt = build_prompt(make_report(SAMPLE_ERRORS), version="v2")
    assert "Do not write any new phonetic or IPA symbols" in prompt
    assert "quote only the symbols that appear in the error list" in prompt
    assert "ordinary word spelling" in prompt


def test_v2_forces_one_numbered_item_per_error():
    prompt = build_prompt(make_report(SAMPLE_ERRORS), version="v2")
    assert "one numbered item per error" in prompt
    assert "start each item with the error's number" in prompt
    assert "Do not merge, split, or repeat items" in prompt


def test_v2_restrains_l1_generalisations():
    prompt = build_prompt(make_report(SAMPLE_ERRORS), version="v2")
    assert "generalised claims" in prompt
    assert "unless you are certain" in prompt


def test_v2_structure_rule_is_scoped_to_the_numbered_list():
    """The one-item-per-error rule must not suppress the misread section
    (regression observed in the 2026-07-20 comparison run)."""
    prompt = build_prompt(make_report(SAMPLE_ERRORS), version="v2")
    assert "For the numbered error list only" in prompt
    assert 'If a "Possible reading mistakes" section is given' in prompt
    assert "separate final section" in prompt


def test_v2_keeps_error_list_and_coach_role():
    prompt = build_prompt(make_report(SAMPLE_ERRORS), version="v2")
    assert "pronunciation coach" in prompt
    assert "/ð/" in prompt
    assert '"this"' in prompt
    assert "Do not re-judge, add, or remove errors" in prompt


# --- rank_errors_by_tier: tier priority sort + top-N selection (design_3c.md §4) ---


def test_rank_errors_by_tier_orders_l1_then_fallback_then_none():
    l1_error = PhonemeError("substitution", "ð", "d", 0, "this")  # dh_stopping
    fallback_error = PhonemeError("substitution", "v", "b", 1, "van")  # phoneme fallback only
    none_error = PhonemeError("substitution", "x", "y", 2, "word")  # no match at all

    ranked = rank_errors_by_tier(
        [none_error, fallback_error, l1_error],
        reference_phonemes=["ð", "v", "x"],
        l1="Japanese",
    )

    assert [error for error, _ in ranked] == [l1_error, fallback_error, none_error]
    assert ranked[0][1].tier == "l1_specific"
    assert ranked[1][1].tier == "phoneme_fallback"
    assert ranked[2][1] is None


def test_rank_errors_by_tier_is_stable_within_the_same_tier():
    first = PhonemeError("substitution", "v", "b", 0, "van")  # fallback tier
    second = PhonemeError("substitution", "f", "b", 1, "fan")  # fallback tier

    ranked = rank_errors_by_tier(
        [first, second], reference_phonemes=["v", "f"], l1="Japanese"
    )

    assert [error for error, _ in ranked] == [first, second]


def test_rank_errors_by_tier_top_n_slice_keeps_only_the_highest_tiers():
    l1_error = PhonemeError("substitution", "ð", "d", 0, "this")
    fallback_error = PhonemeError("substitution", "v", "b", 1, "van")
    none_error = PhonemeError("substitution", "x", "y", 2, "word")

    ranked = rank_errors_by_tier(
        [none_error, l1_error, fallback_error],
        reference_phonemes=["ð", "v", "x"],
        l1="Japanese",
    )
    full_explanation, facts_only = ranked[:2], ranked[2:]

    assert [error for error, _ in full_explanation] == [l1_error, fallback_error]
    assert [error for error, _ in facts_only] == [none_error]


# --- prompt v3: structured-knowledge injection (design_3c.md §5) ---


def make_v3_report(errors: list[PhonemeError]) -> DiagnosisReport:
    return DiagnosisReport(
        transcript="dis is high",
        target_text="this is high",
        reference_phonemes=["x", "v", "ð", "ɪ", "s"],
        hypothesis_phonemes=["y", "b", "d", "ɪ", "s"],
        errors=errors,
        learner_l1="Japanese",
    )


V3_ERRORS = [
    PhonemeError("substitution", "x", "y", 0, "wordx"),  # no match at all
    PhonemeError("substitution", "v", "b", 1, "van"),  # phoneme_fallback tier
    PhonemeError("substitution", "ð", "d", 2, "this"),  # l1_specific tier (dh_stopping)
]


def test_v3_role_instruction_limits_llm_to_rephrasing():
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert "Your job is only to rephrase the given" in prompt
    assert "do not speculate about why" in prompt


def test_v3_header_reports_total_and_explained_counts():
    prompt = build_prompt(
        make_v3_report(V3_ERRORS), version="v3", full_explanation_limit=2
    )
    assert "Detected 3 errors, 2 explained in detail below." in prompt


def test_v3_item_count_literal_matches_explained_count():
    """The literal count covers only the items actually in the prompt: the
    unmatched "wordx" error is facts-only and no longer sent to the LLM."""
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert "There are exactly 2 numbered items below" in prompt
    assert "Output exactly 2 numbered items" in prompt


def test_v3_item_count_literal_handles_a_single_error_without_extra_items():
    """Regression test for the 2026-07-20 comparison run's fabricated
    'item 2: no errors' behavior when there is exactly one real error."""
    single = [PhonemeError("substitution", "ð", "d", 0, "this")]
    prompt = build_prompt(make_v3_report(single), version="v3")
    assert "There are exactly 1 numbered items below" in prompt
    assert "Output exactly 1 numbered items" in prompt


def test_v3_orders_items_by_tier_not_by_detection_order():
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert prompt.index('"this"') < prompt.index('"van"')


def test_v3_facts_only_items_are_not_sent_in_the_prompt():
    """Unmatched errors have nothing to rephrase; leaving them in the prompt
    made the model invent tips for them (2026-07-22 comparison run)."""
    prompt = build_prompt(
        make_v3_report(V3_ERRORS), version="v3", full_explanation_limit=3
    )
    assert '"wordx"' not in prompt
    assert "/x/" not in prompt
    assert "/y/" not in prompt


def test_v3_matched_item_beyond_the_limit_is_facts_only_too():
    prompt = build_prompt(
        make_v3_report(V3_ERRORS), version="v3", full_explanation_limit=1
    )
    # Tier-sorted rank 1 is "this" (l1_specific); "van" (fallback, rank 2)
    # falls outside the limit and must not be sent to the LLM at all.
    assert '"this"' in prompt
    assert '"van"' not in prompt


def test_v3_known_phenomenon_line_uses_human_readable_name_not_id():
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert "Known phenomenon: ð→d substitution (dental stopping)" in prompt
    assert "dh_stopping" not in prompt


def test_v3_fallback_tier_item_has_tip_but_no_cause_or_phenomenon_line():
    """Fallback records make no L1 claim and their machine id must not leak
    into learner-facing text ("fallback_v" was quoted verbatim in the
    2026-07-22 run), so the block is Tip + Practice words only."""
    prompt = build_prompt(
        make_v3_report(V3_ERRORS), version="v3", full_explanation_limit=3
    )
    lines = prompt.splitlines()
    van_index = next(i for i, line in enumerate(lines) if '"van"' in line)
    block = []
    for line in lines[van_index + 1 :]:
        stripped = line.strip()
        if stripped == "" or stripped[:1].isdigit():
            break
        block.append(stripped)
    assert any(line.startswith("Tip:") for line in block)
    assert not any(line.startswith("Cause:") for line in block)
    assert not any(line.startswith("Known phenomenon") for line in block)


def test_v3_role_instruction_preserves_uncertainty_in_causes():
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert "you must preserve them in your rephrasing" in prompt
    assert "do not present an uncertain cause as certain" in prompt


def test_v3_prompt_with_only_facts_only_errors_forbids_describing_them():
    """When no error has matched knowledge (M=0), the LLM sees no error
    details at all and must not try to reconstruct them."""
    only_unmatched = [PhonemeError("substitution", "x", "y", 0, "wordx")]
    prompt = build_prompt(make_v3_report(only_unmatched), version="v3")
    assert "Detected 1 errors, 0 explained in detail below." in prompt
    assert "There are exactly" not in prompt
    assert '"wordx"' not in prompt
    assert "Do not describe or guess" in prompt


def test_v3_prompt_at_zero_m_drops_transcript_line():
    """Regression test for the fabrication observed in the
    2026-07-22 v3 llm-bypass confirmation run: with the transcript still in
    the prompt, the model reconstructed a word-level paraphrase from it
    ('zoo' said as 'sue') even though M=0 forbids describing specific
    errors. Dropping the transcript line removes that raw material."""
    only_unmatched = [PhonemeError("substitution", "x", "y", 0, "wordx")]
    report = make_v3_report(only_unmatched)
    prompt = build_prompt(report, version="v3")
    assert report.transcript not in prompt
    # The target sentence is not raw material for reconstructing the
    # learner's actual errors, so it's still shown.
    assert report.target_text in prompt


def test_v3_prompt_at_zero_m_drops_misread_section_and_instruction_entirely():
    """Same run also saw an empty 'Possible reading mistakes' heading
    fabricated even though no such section was given -- the generic role
    instruction ("if given, always address it") was apparently enough to
    make the model invent one. At M=0, the misread section (data +
    instruction) is dropped altogether rather than just its instruction:
    the learner still sees it via the UI's Misread? column
    (ui/interface.py), so no information is lost."""
    errors = [PhonemeError("substitution", "x", "y", 0, "wordx")] + MISREAD_ERRORS
    prompt = build_prompt(make_v3_report(errors), version="v3")
    assert "Detected" in prompt and "0 explained in detail below." in prompt
    # The dynamic section (header + this report's notices) is gone. The
    # fixed role-instruction template still mentions the section generically
    # ("if given, always address it") -- that shared string is untouched by
    # this fix and is exercised by test_v2_structure_rule_is_scoped_to_the_numbered_list.
    assert "Possible reading mistakes (a different word was read):" not in prompt
    assert '"buy"' not in prompt
    assert "check the target word and read it again" not in prompt


def test_v3_prompt_keeps_transcript_and_misread_section_when_m_is_nonzero():
    """Only the M=0 case is affected -- when at least one error is fully
    explained, transcript and misread handling stay exactly as before."""
    errors = V3_ERRORS + MISREAD_ERRORS
    prompt = build_prompt(make_v3_report(errors), version="v3")
    assert make_v3_report(errors).transcript in prompt
    assert "Possible reading mistakes" in prompt
    assert '"buy"' in prompt


# --- facts-only template: deterministic, LLM-free rendering ---


def test_facts_only_template_covers_all_op_types():
    sub = PhonemeError("substitution", "ð", "d", 0, "this")
    dele = PhonemeError("deletion", "s", None, 2, "this")
    ins = PhonemeError("insertion", None, "ɯ", 6, "high")
    assert (
        format_facts_only_error(sub)
        == 'In the word "this", expected /ð/ but heard /d/.'
    )
    assert (
        format_facts_only_error(dele)
        == 'In the word "this", expected /s/ but it was missing.'
    )
    assert (
        format_facts_only_error(ins)
        == 'In the word "high", an extra /ɯ/ was added.'
    )


def test_facts_only_template_falls_back_to_position_without_word():
    err = PhonemeError("substitution", "ð", "d", 3, None)
    assert format_facts_only_error(err) == "At position 3, expected /ð/ but heard /d/."


def test_render_facts_only_section_lists_each_error_as_a_bullet():
    errors = [
        PhonemeError("substitution", "x", "y", 0, "wordx"),
        PhonemeError("insertion", None, "ɯ", 6, "high"),
    ]
    section = render_facts_only_section(errors)
    assert section.splitlines()[0] == "Other detected differences:"
    assert '- In the word "wordx", expected /x/ but heard /y/.' in section
    assert '- In the word "high", an extra /ɯ/ was added.' in section


def test_render_facts_only_section_is_empty_without_errors():
    assert render_facts_only_section([]) == ""


# --- explain(): deterministic facts-only section appended after LLM output ---


class _FakeOllamaResponse:
    def __init__(self, text: str = "LLM TEXT") -> None:
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._text}


def _fake_ollama(monkeypatch, captured, response="LLM TEXT"):
    """Install a fake Ollama endpoint and count the calls made to it."""
    captured["calls"] = 0

    def fake_post(url, json=None, timeout=None):
        captured["calls"] += 1
        captured["prompt"] = json["prompt"]
        return _FakeOllamaResponse(response)

    monkeypatch.setattr(
        "pronunciation_coach.explainer.requests.post", fake_post
    )


def test_v3_explain_appends_facts_only_section_after_llm_output(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(V3_ERRORS))
    assert out.startswith("LLM TEXT")
    assert "Other detected differences:" in out
    assert '- In the word "wordx", expected /x/ but heard /y/.' in out
    assert '"wordx"' not in captured["prompt"]


def test_v3_explain_appends_nothing_when_every_error_is_fully_explained(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    matched_only = [PhonemeError("substitution", "ð", "d", 2, "this")]
    out = explainer.explain(make_v3_report(matched_only))
    assert out == "LLM TEXT"


def test_v2_explain_never_appends_facts_only_section(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v2")
    out = explainer.explain(make_v3_report(V3_ERRORS))
    assert out == "LLM TEXT"


def test_v3_misread_section_keeps_v2_word_level_notice_format():
    errors = [PhonemeError("substitution", "ð", "d", 0, "this")] + MISREAD_ERRORS
    prompt = build_prompt(make_report(errors), version="v3")
    assert '"buy"' in prompt
    assert "read as" in prompt
    assert "do not explain individual sounds" in prompt.lower()
    assert "/b/" not in prompt


# --- explain(): the LLM is skipped entirely at M=0 with no misread ---
#
# The 2026-08-23 E2 (N=10) run showed that at M=0 the call was still made
# purely for an encouragement sentence, and the surviving role-instruction
# template leaked numbered items (9/10 trials), an invented bracketed
# transcription (1/10) and empty reading-mistake sections (7/10) into the
# learner-facing output. With nothing to rephrase and no misread to report,
# there is nothing for the LLM to do, so it is not called at all.


ZERO_M_ERRORS = [PhonemeError("substitution", "x", "y", 0, "wordx")]

# Shape of the leakage actually observed in the 2026-08-23 run: a phantom
# numbered item and a reading-mistake section, neither backed by the input.
_LEAKY_RESPONSE = (
    "Great effort!\n\n"
    "1. In the word \"zoo\", you said [a zo].\n\n"
    "Possible reading mistakes:\n"
)


def test_v3_explain_does_not_call_the_llm_at_zero_m_without_misreads(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert captured["calls"] == 0


def test_v3_explain_at_zero_m_returns_the_preamble_and_the_facts_only_section(
    monkeypatch,
):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    expected = ZERO_M_PREAMBLE + "\n\n" + render_facts_only_section(
        ZERO_M_ERRORS, heading=ZERO_M_HEADING
    )
    assert out == expected
    assert "LLM TEXT" not in out


def test_v3_explain_at_zero_m_keeps_the_deterministic_facts_only_block(monkeypatch):
    """The deterministic block is unchanged by the skip: it is still the
    same rendering that used to be appended after the LLM output."""
    captured = {}
    _fake_ollama(monkeypatch, captured, response=_LEAKY_RESPONSE)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert ZERO_M_HEADING in out
    assert '- In the word "wordx", expected /x/ but heard /y/.' in out


def test_v3_explain_at_zero_m_output_has_no_numbered_items_or_misread_section(
    monkeypatch,
):
    captured = {}
    _fake_ollama(monkeypatch, captured, response=_LEAKY_RESPONSE)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert not re.search(r"(?m)^\s*\d+[.)]\s", out)
    assert "reading mistake" not in out.lower()


def test_v3_explain_calls_the_llm_at_zero_m_when_a_misread_is_present(monkeypatch):
    """A misread still needs word-level coaching, so the call stays."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS + MISREAD_ERRORS))
    assert captured["calls"] == 1
    assert out.startswith("LLM TEXT")


def test_v3_explain_still_calls_the_llm_when_m_is_nonzero(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    explainer.explain(make_v3_report(V3_ERRORS))
    assert captured["calls"] == 1


def test_v3_explain_does_not_call_the_llm_when_no_error_was_detected(monkeypatch):
    """No errors was excluded because it is not M=0, has no facts-only
    material, and the prompt asks only for praise."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report([]))
    assert captured["calls"] == 0
    assert out == NO_ERRORS_MESSAGE


def test_should_skip_llm_is_true_with_no_errors_and_no_misread():
    assert _should_skip_llm(make_v3_report([]), "v3", 3)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_should_skip_llm_is_false_with_no_errors_on_v1_and_v2(version):
    assert not _should_skip_llm(make_v3_report([]), version, 3)


def test_v3_explain_with_no_errors_returns_the_no_errors_message(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    report = make_v3_report([])

    outputs = [explainer.explain(report) for _ in range(5)]

    assert outputs == [NO_ERRORS_MESSAGE] * 5
    assert captured["calls"] == 0


def test_v3_explain_with_no_errors_ignores_a_leaky_llm_response(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured, response=_LEAKY_RESPONSE)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report([]))

    assert not re.search(r"(?m)^\s*\d+[.)]\s", out)
    assert "reading mistake" not in out.lower()
    assert not re.search(r"\[[^]]+\]", out)
    assert captured["calls"] == 0


def test_no_errors_message_reports_detection_without_claiming_correctness():
    lowered = NO_ERRORS_MESSAGE.lower()
    for banned in (
        "perfect",
        "correct",
        "accurate",
        "flawless",
        "no mistakes",
        "native",
        "accent",
    ):
        assert banned not in lowered
    assert "detected" in lowered
    assert NO_ERRORS_MESSAGE.isascii()
    assert "!" not in NO_ERRORS_MESSAGE


def test_no_errors_output_does_not_use_the_zero_m_preamble(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report([]))

    assert ZERO_M_PREAMBLE not in out
    assert ZERO_M_HEADING not in out


def test_v3_explain_calls_the_llm_when_only_misreads_are_detected(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    report = make_v3_report(MISREAD_ERRORS)

    out = explainer.explain(report)

    assert captured["calls"] == 1
    assert "Possible reading mistakes" in captured["prompt"]
    assert report.transcript in captured["prompt"]
    assert out == "LLM TEXT"


def test_no_errors_fixture_bypasses_the_llm(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    fixture_path = compare_explanations.FIXTURES_DIR / "e2e_no_errors.json"
    report = compare_explanations.load_report(fixture_path)
    explainer = OllamaExplainer(prompt_version="v3")

    out = explainer.explain(report)

    assert captured["calls"] == 0
    assert out == NO_ERRORS_MESSAGE


def test_v2_explain_still_calls_the_llm_at_zero_m(monkeypatch):
    """The skip is v3-only: v2 has no knowledge tiering, so M=0 does not
    exist for it and its measured behaviour must not move."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v2")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert captured["calls"] == 1
    assert out == "LLM TEXT"


def test_v1_explain_still_calls_the_llm_at_zero_m(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v1")
    explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert captured["calls"] == 1


def test_v3_explain_skips_the_llm_when_the_limit_leaves_nothing_to_explain(
    monkeypatch,
):
    """full_explanation_limit=0 leaves zero items to explain, which is the
    same condition build_prompt already treats as M=0, so the skip applies
    there too."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3", full_explanation_limit=0)
    out = explainer.explain(make_v3_report(V3_ERRORS))
    assert captured["calls"] == 0
    assert out.startswith(ZERO_M_PREAMBLE)
    assert ZERO_M_HEADING in out
    assert out.count("\n- ") == len(V3_ERRORS)


# --- explain(): deterministic preamble on the M=0 skip path ---
#
# Skipping the LLM (2026-08-23) removed the encouragement sentence that used
# to open the M=0 output, leaving the learner with a bare heading and a list
# of phoneme differences. The replacement is a fixed template, not generated
# text: detection precision is 0.069, so an absence of explainable errors is
# not evidence of correct pronunciation and must not be phrased as praise.


def test_zero_m_preamble_states_the_facts_without_praising(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert out.startswith(ZERO_M_PREAMBLE)
    assert "Differences were detected" in out
    assert "none of them matched a known pronunciation pattern" in out
    assert "reference information" in out


def test_zero_m_preamble_avoids_evaluative_language():
    """Guards the wording requirement itself, not just its current text."""
    lowered = ZERO_M_PREAMBLE.lower()
    for banned in (
        "well done",
        "good job",
        "great",
        "nice",
        "correct",
        "accurate",
        "improv",
        "progress",
        "keep practicing",
        "keep practising",
    ):
        assert banned not in lowered
    assert "!" not in ZERO_M_PREAMBLE
    assert ZERO_M_PREAMBLE.isascii()


def test_zero_m_output_is_byte_identical_across_calls(monkeypatch):
    """No LLM, so the whole output must be reproducible byte for byte."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    report = make_v3_report(ZERO_M_ERRORS)
    outs = [explainer.explain(report) for _ in range(5)]
    assert len(set(outs)) == 1
    assert captured["calls"] == 0


def test_zero_m_output_layout_is_preamble_blank_line_then_the_list(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert out.split("\n\n") == [
        ZERO_M_PREAMBLE,
        render_facts_only_section(ZERO_M_ERRORS, heading=ZERO_M_HEADING),
    ]


def test_zero_m_heading_drops_the_other_prefix(monkeypatch):
    """At M=0 the block is the only content, so "Other" has nothing to
    contrast with."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(ZERO_M_ERRORS))
    assert ZERO_M_HEADING == "Detected differences:"
    assert "Other detected differences:" not in out


def test_nonzero_m_output_has_no_preamble_and_keeps_the_other_heading(monkeypatch):
    """M>0 is untouched: LLM text first, then the original heading."""
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v3")
    out = explainer.explain(make_v3_report(V3_ERRORS))
    assert captured["calls"] == 1
    assert ZERO_M_PREAMBLE not in out
    assert "Differences were detected" not in out
    assert out == "LLM TEXT\n\nOther detected differences:\n" + (
        '- In the word "wordx", expected /x/ but heard /y/.'
    )


def test_nonzero_m_v2_output_has_no_preamble(monkeypatch):
    captured = {}
    _fake_ollama(monkeypatch, captured)
    explainer = OllamaExplainer(prompt_version="v2")
    out = explainer.explain(make_v3_report(V3_ERRORS))
    assert out == "LLM TEXT"


def test_render_facts_only_section_keeps_its_original_heading_by_default():
    """The M>0 call site passes no heading, so its output cannot move."""
    section = render_facts_only_section(ZERO_M_ERRORS)
    assert section.startswith("Other detected differences:")


def test_render_facts_only_section_heading_override_only_changes_the_heading():
    default = render_facts_only_section(ZERO_M_ERRORS)
    overridden = render_facts_only_section(ZERO_M_ERRORS, heading="Detected differences:")
    assert default.split("\n")[1:] == overridden.split("\n")[1:]
    assert overridden.split("\n")[0] == "Detected differences:"


def test_render_facts_only_section_heading_override_still_renders_nothing_when_empty():
    assert render_facts_only_section([], heading="Detected differences:") == ""
