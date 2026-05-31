"""Scoring candidates — the half of test-time compute that does the *choosing*.

A verifier looks at a candidate solution and estimates the chance it's correct.
That estimate is what ``aggregate.best_score`` and ``aggregate.weighted_majority``
rank on. Two implementations, deliberately swappable behind one interface so the
benchmark can isolate exactly what the verifier buys you:

  - **MajorityVerifier** — gives every candidate the same score (1.0). With it,
    weighted-majority collapses to plain majority vote: this is the verifier-free
    Self-Consistency baseline, and it needs no network, so it's the default and
    the one the offline tests exercise.

  - **LLMVerifier** — LLM-as-judge: prompt the model to grade each candidate 0..1
    (Cobbe et al. 2021's outcome verifier, prompted rather than trained). This is
    where the real signal comes from, and the only part that calls out.

Faithfulness note: a *trained* verifier / process-reward model needs token-level
signals OpenRouter's chat API doesn't expose, so the strongest version of the
paper isn't reproducible here. The interface is, and a prompted judge is a real,
working stand-in — see the README's honest-caveat section.
"""

from __future__ import annotations

from .aggregate import Candidate
from .prompts import VERIFIER_PROMPT, parse_score


class Verifier:
    """Assigns a correctness score in ``[0, 1]`` to each candidate, in place.

    Subclasses implement :meth:`score`. The contract: return the same candidate
    objects (mutated with their ``score``), in the same order.
    """

    def score(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        raise NotImplementedError


class MajorityVerifier(Verifier):
    """The trivial verifier: every candidate scores 1.0.

    Reduces weighted strategies to vote-counting, giving the Self-Consistency
    baseline. No network — this is the default and what the tests rely on.
    """

    def score(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            c.score = 1.0
        return candidates


class LLMVerifier(Verifier):
    """LLM-as-judge: ask the model how likely each candidate is correct.

    ``chat_fn`` defaults to :func:`llm.chat` but is injectable, so tests drive a
    fake judge with no network. Candidates with no parsed answer are scored 0 —
    a solution that never reached a number can't be the right one to pick.
    """

    def __init__(self, chat_fn=None, model: str | None = None, temperature: float = 0.0):
        if chat_fn is None:
            from .llm import chat as chat_fn  # lazy: importing must not need a key
        self._chat = chat_fn
        self._model = model
        self._temperature = temperature

    def score(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            if c.answer is None:
                c.score = 0.0
                continue
            prompt = VERIFIER_PROMPT.format(question=question, solution=c.text)
            reply = self._chat(prompt, model=self._model, temperature=self._temperature)
            c.score = parse_score(reply)
        return candidates


# Verifier name -> zero-arg factory, for the CLI.
VERIFIERS = {
    "majority": MajorityVerifier,
    "llm": LLMVerifier,
}


def make_verifier(name: str, **kwargs) -> Verifier:
    """Build a verifier by name from :data:`VERIFIERS`."""
    try:
        cls = VERIFIERS[name]
    except KeyError:
        raise ValueError(
            f"unknown verifier {name!r}; choose from {sorted(VERIFIERS)}"
        ) from None
    return cls(**kwargs) if kwargs else cls()
