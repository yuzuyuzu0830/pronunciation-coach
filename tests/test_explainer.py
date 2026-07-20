import pytest

from pronunciation_coach.explainer import build_prompt
from pronunciation_coach.types import DiagnosisReport, PhonemeError


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


# --- prompt v2: strengthened instructions, selectable per version ---


def test_v1_is_the_default_version():
    report = make_report(SAMPLE_ERRORS)
    assert build_prompt(report) == build_prompt(report, version="v1")


def test_unknown_prompt_version_raises_value_error():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        build_prompt(make_report(SAMPLE_ERRORS), version="v99")


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
