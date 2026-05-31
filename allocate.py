"""Spend compute where it helps: more samples for harder questions.

The headline of Snell et al. 2024 isn't "sample more" — it's "sample more
*optimally*." A flat budget wastes compute: easy questions are already solved at
N=1, while hard ones stay wrong even at N=64. Compute-optimal allocation reads
each question's difficulty and shifts the budget toward the hard ones.

We can't see the gold answer at inference time, so difficulty is estimated from
the model's own behavior: send a small **probe** batch and measure how much the
candidates *disagree*. If a cheap probe already agrees unanimously, the question
is easy — stop. If the probe scatters across many answers, it's hard — spend the
rest of the budget there.

All of this is pure arithmetic over probe results, so the allocation policy is
unit-tested offline. The disagreement signal (normalized entropy of the answer
distribution) is the same quantity that makes Self-Consistency work, reused here
as a difficulty meter.
"""

from __future__ import annotations

import math

from .aggregate import Candidate, tally_votes


def agreement(candidates: list[Candidate]) -> float:
    """Fraction of valid candidates that landed on the plurality answer, in ``[0, 1]``.

    1.0 means unanimous (easy); near 0 means everyone disagreed (hard). Returns
    0.0 when nothing produced an answer.
    """
    votes = tally_votes(candidates)
    total = sum(votes.values())
    if total == 0:
        return 0.0
    return max(votes.values()) / total


def disagreement(candidates: list[Candidate]) -> float:
    """Normalized Shannon entropy of the answer distribution, in ``[0, 1]``.

    0.0 when all candidates agree (one answer), rising toward 1.0 as the votes
    spread evenly across many distinct answers. This is the difficulty signal:
    high entropy = the model is unsure = worth spending more compute on.
    """
    votes = tally_votes(candidates)
    total = sum(votes.values())
    if total <= 1 or len(votes) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in votes.values())
    return entropy / math.log(len(votes))


def optimal_n(
    probe: list[Candidate],
    *,
    n_min: int = 1,
    n_max: int = 16,
    confident_at: float = 1.0,
) -> int:
    """How many *total* samples this question deserves, from a probe batch.

    Linear in the probe's disagreement: unanimous probe -> ``n_min`` (don't waste
    compute on a solved question); maximally split probe -> ``n_max``. ``confident_at``
    is the agreement level already treated as "solved" (default 1.0 = unanimous).

    The result is the *target total* including the probe, never below the probe
    size already spent and never below ``n_min``.
    """
    if n_max < n_min:
        raise ValueError(f"n_max ({n_max}) must be >= n_min ({n_min})")
    a = agreement(probe)
    if a >= confident_at:
        target = n_min
    else:
        # Map agreement in [0, confident_at) -> budget in (n_max, n_min].
        frac = a / confident_at if confident_at > 0 else 0.0
        target = round(n_max - frac * (n_max - n_min))
    return max(n_min, len(probe), target)


def allocate(
    probes: dict[str, list[Candidate]],
    *,
    n_min: int = 1,
    n_max: int = 16,
    confident_at: float = 1.0,
) -> dict[str, int]:
    """Target total sample count per question id, from each question's probe.

    A convenience wrapper over :func:`optimal_n` for a batch of questions — the
    shape a benchmark or scheduler consumes.
    """
    return {
        qid: optimal_n(
            probe, n_min=n_min, n_max=n_max, confident_at=confident_at
        )
        for qid, probe in probes.items()
    }
