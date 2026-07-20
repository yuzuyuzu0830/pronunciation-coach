"""Phoneme sequence alignment via edit-distance dynamic programming.

Pure alignment only: converting non-match operations into PhonemeError is the
pipeline's responsibility (docs/design.md §2, §5).
"""

from __future__ import annotations

from typing import Callable

from pronunciation_coach.types import AlignmentOp


def unit_substitution_cost(ref_phone: str, hyp_phone: str) -> float:
    return 0.0 if ref_phone == hyp_phone else 1.0


def align_phonemes(
    reference: list[str],
    hypothesis: list[str],
    substitution_cost: Callable[[str, str], float] = unit_substitution_cost,
    gap_cost: float = 1.0,
) -> list[AlignmentOp]:
    """Align reference and hypothesis phonemes with edit operations.

    Needleman–Wunsch-style DP. Ties resolve substitution > deletion >
    insertion during backtrace, so the result is deterministic. The
    substitution cost function is injectable to allow a feature-weighted
    cost later without touching this module.
    """
    n, m = len(reference), len(hypothesis)

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * gap_cost
    for j in range(1, m + 1):
        cost[0][j] = j * gap_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + substitution_cost(reference[i - 1], hypothesis[j - 1]),
                cost[i - 1][j] + gap_cost,
                cost[i][j - 1] + gap_cost,
            )

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and cost[i][j]
            == cost[i - 1][j - 1] + substitution_cost(reference[i - 1], hypothesis[j - 1])
        ):
            op = "match" if reference[i - 1] == hypothesis[j - 1] else "substitution"
            ops.append(AlignmentOp(op, reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + gap_cost:
            ops.append(AlignmentOp("deletion", reference[i - 1], None))
            i -= 1
        else:
            ops.append(AlignmentOp("insertion", None, hypothesis[j - 1]))
            j -= 1

    ops.reverse()
    return ops
