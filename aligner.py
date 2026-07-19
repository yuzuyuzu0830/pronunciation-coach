"""Phoneme sequence alignment for mispronunciation detection.

Compares a canonical (reference) phoneme sequence with a recognised
(hypothesis) sequence and returns edit operations. Detection stays in this
module; LLM explanation must not perform alignment or error detection.
"""

from __future__ import annotations


def align_phonemes(
    reference: list[str], hypothesis: list[str]
) -> list[tuple[str, str | None, str | None]]:
    """Align reference and hypothesis phonemes with edit operations.

    Returns a list of where op is one of:
    match, substitution, deletion, insertion. Deletion uses hyp_phone=None;
    insertion uses ref_phone=None.
    """
    raise NotImplementedError
