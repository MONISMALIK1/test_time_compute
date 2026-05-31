"""The two prompts test-time compute needs — and how to read the verifier's reply.

A *generator* prompt asks the model to reason step by step and end with a
``#### <answer>`` line, so :func:`extract.extract_answer` can read the result
cleanly. We sample this one many times at non-zero temperature to get diverse
candidates.

A *verifier* prompt is the other half of the system: hand the model a question
and one candidate solution and ask, "is this correct?", scored 0 to 1. This is
LLM-as-judge — an outcome verifier in the sense of Cobbe et al. 2021, just
prompted instead of trained. ``parse_score`` turns its reply back into a float,
deterministically, so that parsing is unit-tested without the network.
"""

from __future__ import annotations

import re

COT_PROMPT = """\
Solve the problem step by step. Show your reasoning, then on the final line write
the answer in the exact form:
#### <number>

Problem: {question}
Solution:"""

VERIFIER_PROMPT = """\
You are grading a candidate solution to a math problem. Decide how likely it is
to be correct. Consider the reasoning, not just the final number.

Problem: {question}

Candidate solution:
{solution}

Reply with a single number between 0 and 1 — your probability that the candidate's
final answer is correct — in the exact form:
SCORE: <0..1>"""

# "SCORE: 0.8" wins; otherwise fall back to the first number we see. The leading
# sign is captured so an out-of-range "-0.2" clamps to 0 rather than parsing 0.2.
_SCORE_RE = re.compile(r"SCORE:\s*(-?(?:[01](?:\.\d+)?|0?\.\d+))", re.IGNORECASE)
_FLOAT_RE = re.compile(r"-?(?:\d?\.\d+|\d+)")


def parse_score(text: str, default: float = 0.5) -> float:
    """Read the verifier's ``SCORE: x`` reply as a float clamped to ``[0, 1]``.

    Falls back to the first parseable number if the marker is missing, and to
    ``default`` if the reply has no number at all — a judge that won't commit is
    treated as maximally uncertain, not as a hard zero.
    """
    if not text:
        return default
    m = _SCORE_RE.search(text)
    raw = m.group(1) if m else None
    if raw is None:
        m2 = _FLOAT_RE.search(text)
        if not m2:
            return default
        raw = m2.group(0)
    try:
        val = float(raw)
    except ValueError:
        return default
    return max(0.0, min(1.0, val))
