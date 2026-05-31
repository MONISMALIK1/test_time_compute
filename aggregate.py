"""Turn N scored candidates into one answer — three ways the paper compares.

This is the core of test-time compute. You've spent the compute to sample many
solutions; now you have to *pick one*. Snell et al. (and the verifier line going
back to Cobbe et al. 2021) compare exactly these aggregation rules:

  - **majority_vote**  — count how many candidates reached each answer, take the
    plurality. This is Self-Consistency (Wang et al., 2022): a verifier-free
    baseline that just trusts the crowd.
  - **best_score**     — Best-of-N: ask the verifier to score each candidate and
    take the answer of the single highest-scoring one. Strong when the verifier
    is good, fragile when it's noisy (one confident wrong score wins).
  - **weighted_majority** — the usual winner in the paper: sum the verifier's
    scores per answer and take the highest total. Combines the crowd's wisdom
    with the verifier's discrimination, so a lone high score can't override a
    well-supported consensus.

Everything here is a pure function over a list of :class:`Candidate`. No network,
no model — the LLM already ran upstream. That's what makes the selection logic
fully unit-testable, and it's where the interesting behavior lives.

Ties are broken deterministically (highest tally, then highest summed score, then
the answer that appeared earliest) so the same candidates always yield the same
pick — important for reproducible benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    """One sampled solution: its full text, the answer parsed from it, its score.

    ``answer`` is the normalized string from :func:`extract.extract_answer` (or
    ``None`` if the candidate produced no number). ``score`` is the verifier's
    estimate that this candidate is correct, in ``[0, 1]``; the verifier-free
    strategies ignore it.
    """

    text: str
    answer: str | None
    score: float = 1.0


def _valid(candidates: list[Candidate]) -> list[Candidate]:
    """Candidates that actually produced an answer (drop the ``None``s)."""
    return [c for c in candidates if c.answer is not None]


def tally_votes(candidates: list[Candidate]) -> dict[str, int]:
    """How many candidates reached each answer."""
    counts: dict[str, int] = {}
    for c in _valid(candidates):
        counts[c.answer] = counts.get(c.answer, 0) + 1
    return counts


def tally_weight(candidates: list[Candidate]) -> dict[str, float]:
    """Summed verifier score per answer (the weighted-vote tally)."""
    weights: dict[str, float] = {}
    for c in _valid(candidates):
        weights[c.answer] = weights.get(c.answer, 0.0) + c.score
    return weights


def _first_index(candidates: list[Candidate]) -> dict[str, int]:
    """Position each answer first appeared — the final, stable tie-breaker."""
    order: dict[str, int] = {}
    for i, c in enumerate(_valid(candidates)):
        order.setdefault(c.answer, i)
    return order


def majority_vote(candidates: list[Candidate]) -> str | None:
    """Self-Consistency: the answer the most candidates agree on.

    Ties broken by summed score, then earliest appearance.
    """
    votes = tally_votes(candidates)
    if not votes:
        return None
    weights = tally_weight(candidates)
    order = _first_index(candidates)
    return max(votes, key=lambda a: (votes[a], weights[a], -order[a]))


def best_score(candidates: list[Candidate]) -> str | None:
    """Best-of-N: the answer of the single highest-scoring candidate.

    Ties on score broken by earliest appearance among valid candidates.
    """
    valid = _valid(candidates)
    if not valid:
        return None
    best = max(range(len(valid)), key=lambda i: (valid[i].score, -i))
    return valid[best].answer


def weighted_majority(candidates: list[Candidate]) -> str | None:
    """Verifier-weighted vote: the answer with the highest summed score.

    The paper's strongest general strategy. Ties broken by raw vote count, then
    earliest appearance.
    """
    weights = tally_weight(candidates)
    if not weights:
        return None
    votes = tally_votes(candidates)
    order = _first_index(candidates)
    return max(weights, key=lambda a: (weights[a], votes[a], -order[a]))


# Strategy name -> function, for the CLI and benchmark.
STRATEGIES = {
    "majority": majority_vote,
    "best": best_score,
    "weighted": weighted_majority,
}


def aggregate(candidates: list[Candidate], strategy: str) -> str | None:
    """Dispatch to a named strategy from :data:`STRATEGIES`."""
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}"
        ) from None
    return fn(candidates)
