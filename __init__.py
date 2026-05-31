"""Test-Time Compute: scaling inference instead of parameters (Snell et al., 2024).

Don't answer hard questions in one shot. Spend compute at *inference* time: sample
many reasoning paths, score them with a verifier, and choose — majority vote,
best-of-N, or verifier-weighted vote. Then go one better and spend that compute
*optimally*, sampling more only on the questions the model finds hard.

This is the capstone of the series: Self-Consistency (majority vote), Tree of
Thoughts (search), and Reflexion (revise from a verdict) are all special cases of
"use more inference to get a better answer." Here they sit behind one interface.

Everything that *decides* — answer extraction, the aggregation strategies, the
compute-optimal allocator — is pure stdlib and tested offline; only sampling and
the LLM-as-judge verifier touch the network.

Public API:
    solve(question, n, strategy, verifier)   # fixed-budget: sample N -> verify -> choose
    solve_adaptive(question, probe_n, n_max)  # compute-optimal: spend where it's hard
    solve_baseline(question)                  # matched n=1 greedy comparison
    Candidate / aggregate / STRATEGIES        # the selection strategies
    Verifier / MajorityVerifier / LLMVerifier # scoring (no-network and LLM-judge)
    optimal_n / agreement / disagreement      # the allocation policy
    extract_answer / grade                    # deterministic answer parsing
    SAMPLE_PROBLEMS                           # bundled math eval set
"""

from .aggregate import (
    STRATEGIES,
    Candidate,
    aggregate,
    best_score,
    majority_vote,
    weighted_majority,
)
from .allocate import agreement, allocate, disagreement, optimal_n
from .core import TTCResult, solve, solve_adaptive, solve_baseline
from .dataset import SAMPLE_PROBLEMS, Problem
from .extract import extract_answer, grade, normalize
from .llm import DEFAULT_MODEL, chat, sample
from .prompts import COT_PROMPT, VERIFIER_PROMPT, parse_score
from .verifier import (
    VERIFIERS,
    LLMVerifier,
    MajorityVerifier,
    Verifier,
    make_verifier,
)

__all__ = [
    "STRATEGIES",
    "VERIFIERS",
    "COT_PROMPT",
    "Candidate",
    "DEFAULT_MODEL",
    "LLMVerifier",
    "MajorityVerifier",
    "Problem",
    "SAMPLE_PROBLEMS",
    "TTCResult",
    "VERIFIER_PROMPT",
    "Verifier",
    "aggregate",
    "agreement",
    "allocate",
    "best_score",
    "chat",
    "disagreement",
    "extract_answer",
    "grade",
    "majority_vote",
    "make_verifier",
    "normalize",
    "optimal_n",
    "parse_score",
    "sample",
    "solve",
    "solve_adaptive",
    "solve_baseline",
    "weighted_majority",
]
