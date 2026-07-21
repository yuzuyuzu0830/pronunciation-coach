import pytest

from pronunciation_coach.explainer import build_prompt, rank_errors_by_tier
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


def test_v3_item_count_literal_matches_total_errors():
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert "There are exactly 3 numbered items below" in prompt
    assert "Output exactly 3 numbered items" in prompt


def test_v3_item_count_literal_handles_a_single_error_without_extra_items():
    """Regression test for the 2026-07-20 comparison run's fabricated
    'item 2: no errors' behavior when there is exactly one real error."""
    single = [PhonemeError("substitution", "ð", "d", 0, "this")]
    prompt = build_prompt(make_v3_report(single), version="v3")
    assert "There are exactly 1 numbered items below" in prompt
    assert "Output exactly 1 numbered items" in prompt


def test_v3_orders_items_by_tier_not_by_detection_order():
    prompt = build_prompt(make_v3_report(V3_ERRORS), version="v3")
    assert prompt.index('"this"') < prompt.index('"van"') < prompt.index('"wordx"')


def test_v3_embeds_knowledge_only_for_items_within_the_limit():
    prompt = build_prompt(
        make_v3_report(V3_ERRORS), version="v3", full_explanation_limit=1
    )
    # Tier-sorted rank 1 is "this" (l1_specific); "van" (fallback, rank 2)
    # must not get a knowledge block when the limit is 1.
    assert "Known phenomenon: dh_stopping" in prompt
    assert "Known phenomenon: fallback_v" not in prompt


def test_v3_unmatched_item_never_gets_a_knowledge_block():
    prompt = build_prompt(
        make_v3_report(V3_ERRORS), version="v3", full_explanation_limit=3
    )
    lines = prompt.splitlines()
    wordx_index = next(i for i, line in enumerate(lines) if '"wordx"' in line)
    trailing = lines[wordx_index + 1 :]
    assert not any(line.strip().startswith("Known phenomenon") for line in trailing)


def test_v3_fallback_tier_item_has_tip_but_no_cause_line():
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
    assert any(line.startswith("Known phenomenon: fallback_v") for line in block)
    assert any(line.startswith("Tip:") for line in block)
    assert not any(line.startswith("Cause:") for line in block)


def test_v3_misread_section_keeps_v2_word_level_notice_format():
    errors = [PhonemeError("substitution", "ð", "d", 0, "this")] + MISREAD_ERRORS
    prompt = build_prompt(make_report(errors), version="v3")
    assert '"buy"' in prompt
    assert "read as" in prompt
    assert "do not explain individual sounds" in prompt.lower()
    assert "/b/" not in prompt
