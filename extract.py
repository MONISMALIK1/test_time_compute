"""Pull the final numeric answer out of a chain of thought, and grade it.

Test-time compute is about *choosing among* candidate solutions, and you can't
choose until you can read each candidate's answer off its reasoning. That parsing
is the deterministic floor of this whole repo: given a model's text, extract the
final number, normalize it, and decide whether it matches the gold answer — all
without a single network call. Every selection strategy in ``aggregate.py`` is
built on top of these functions, so they're tested hard.

Grading is intentionally simple and numeric: strip ``$``/``,``/whitespace, parse
as a number, compare with a small tolerance. Math word-problem answers are
numbers, so this is enough — and being deterministic, it never disagrees with
itself the way an LLM judge might.
"""

from __future__ import annotations

import re

# A signed integer or decimal, optionally with thousands separators: 1,024  -3.5  42
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# GSM8K marks its gold answer with "#### 42"; models prompted in that style copy
# it, so an explicit marker wins over "just grab the last number".
_HASH_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")

# "The answer is 42", "answer: 42", "= 42" — common natural-language endings.
_PHRASE_RE = re.compile(
    r"(?:answer\s*(?:is|:)?\s*|=\s*)\$?(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def normalize(answer: str) -> str:
    """Canonicalize a numeric answer string: drop ``$``/``,``/spaces, trim ``.0``.

    ``"$1,024.00"`` and ``" 1024 "`` both normalize to ``"1024"`` so that string
    equality on the result is a meaningful comparison.
    """
    s = answer.strip().lstrip("$").replace(",", "").replace(" ", "")
    try:
        f = float(s)
    except ValueError:
        return s
    # Render integers without a trailing ".0" so 7 and 7.0 compare equal as text.
    if f == int(f):
        return str(int(f))
    return repr(f)


def extract_answer(text: str) -> str | None:
    """Read the final numeric answer out of a reasoning chain.

    Resolution order, most explicit first:
      1. a ``#### <n>`` marker (GSM8K style),
      2. an "answer is <n>" / "= <n>" phrase (last occurrence),
      3. otherwise the last number anywhere in the text.

    Returns the normalized answer string, or ``None`` if there's no number at all.
    """
    if not text:
        return None

    m = _HASH_RE.search(text)
    if m:
        return normalize(m.group(1))

    phrases = _PHRASE_RE.findall(text)
    if phrases:
        return normalize(phrases[-1])

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return normalize(numbers[-1])

    return None


def grade(predicted: str | None, gold: str, tol: float = 1e-6) -> bool:
    """True if ``predicted`` matches ``gold`` numerically (within ``tol``).

    Both sides are normalized first. Non-numeric values fall back to a normalized
    string comparison so the function never raises on junk input.
    """
    if predicted is None:
        return False
    p, g = normalize(predicted), normalize(gold)
    try:
        return abs(float(p) - float(g)) <= tol
    except ValueError:
        return p == g
