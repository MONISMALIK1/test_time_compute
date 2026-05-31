"""The test-time compute loop: sample many, verify, then choose.

Reference: Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally Can Be
More Effective than Scaling Model Parameters", https://arxiv.org/abs/2408.03314
(and the outcome-verifier line from Cobbe et al. 2021, https://arxiv.org/abs/2110.14168).

    candidates = sample(question, n)      # spend compute: many reasoning paths
    score each candidate (verifier)       # how likely is each one correct?
    answer = aggregate(candidates, ...)   # majority / best-of-N / weighted vote

``solve`` runs that at a fixed budget. ``solve_adaptive`` adds the paper's
headline move — *compute-optimal* allocation: probe cheaply, measure how much the
candidates disagree, and only spend the rest of the budget on questions that need
it. ``solve_baseline`` is the matched n=1 greedy comparison the benchmark plots
the curve against.

Sampling and verification are the only parts that touch the network; everything
that *decides* (extract, aggregate, allocate) is pure, so the whole control flow
is unit-tested with a mocked sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aggregate import Candidate, aggregate
from .allocate import agreement, disagreement, optimal_n
from .extract import extract_answer
from .llm import DEFAULT_MODEL, sample
from .prompts import COT_PROMPT
from .verifier import MajorityVerifier, Verifier


@dataclass
class TTCResult:
    answer: str | None                                # the selected final answer
    candidates: list[Candidate] = field(default_factory=list)  # every sample, scored
    strategy: str = "majority"                        # how the answer was chosen
    n: int = 0                                        # samples actually drawn
    used_compute: bool = True                         # False for the n=1 baseline

    @property
    def agreement(self) -> float:
        """Fraction of candidates that landed on the plurality answer."""
        return agreement(self.candidates)

    @property
    def disagreement(self) -> float:
        """Difficulty signal: normalized entropy of the answer spread."""
        return disagreement(self.candidates)


def _candidates(question: str, n: int, model: str | None, temperature: float) -> list[Candidate]:
    """Sample ``n`` CoT completions and parse each into a scored-able candidate."""
    prompt = COT_PROMPT.format(question=question)
    texts = sample(prompt, n=n, model=model, temperature=temperature)
    return [Candidate(text=t, answer=extract_answer(t)) for t in texts]


def solve(
    question: str,
    n: int = 8,
    strategy: str = "weighted",
    verifier: Verifier | None = None,
    model: str | None = None,
    temperature: float = 0.8,
) -> TTCResult:
    """Sample ``n`` candidates, score them, and aggregate by ``strategy``.

    Default verifier is :class:`MajorityVerifier` (no network) — pair it with
    ``strategy="majority"`` for the Self-Consistency baseline, or pass an
    :class:`LLMVerifier` with ``"weighted"``/``"best"`` to use real scores.
    """
    verifier = verifier or MajorityVerifier()
    candidates = _candidates(question, n, model, temperature)
    verifier.score(question, candidates)
    answer = aggregate(candidates, strategy)
    return TTCResult(answer=answer, candidates=candidates, strategy=strategy, n=len(candidates))


def solve_adaptive(
    question: str,
    probe_n: int = 4,
    n_max: int = 16,
    strategy: str = "weighted",
    verifier: Verifier | None = None,
    model: str | None = None,
    temperature: float = 0.8,
    confident_at: float = 1.0,
) -> TTCResult:
    """Compute-optimal solve: probe, gauge difficulty, top up only if needed.

    Draw a ``probe_n`` batch, let :func:`allocate.optimal_n` set a target total
    from the probe's agreement, sample the shortfall (if any), then verify and
    aggregate the combined pool. Easy questions cost ``probe_n``; hard ones grow
    toward ``n_max``.
    """
    verifier = verifier or MajorityVerifier()
    candidates = _candidates(question, probe_n, model, temperature)

    target = optimal_n(candidates, n_min=probe_n, n_max=n_max, confident_at=confident_at)
    if target > len(candidates):
        candidates += _candidates(question, target - len(candidates), model, temperature)

    verifier.score(question, candidates)
    answer = aggregate(candidates, strategy)
    return TTCResult(answer=answer, candidates=candidates, strategy=strategy, n=len(candidates))


def solve_baseline(
    question: str,
    model: str | None = None,
) -> TTCResult:
    """The matched comparison: a single greedy (temperature 0) sample."""
    candidates = _candidates(question, n=1, model=model, temperature=0.0)
    answer = candidates[0].answer if candidates else None
    return TTCResult(
        answer=answer, candidates=candidates, strategy="baseline",
        n=len(candidates), used_compute=False,
    )


__all__ = [
    "TTCResult",
    "solve",
    "solve_adaptive",
    "solve_baseline",
    "DEFAULT_MODEL",
]
