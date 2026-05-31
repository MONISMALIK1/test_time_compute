"""OpenRouter LLM wrapper — one ``chat`` call, plus threaded N-sampling.

Same proven, dependency-free pattern as the rest of this series: HTTP over stdlib
``urllib``, ``OPENROUTER_API_KEY`` read only from the environment and never logged
or written to disk.

Test-time compute lives or dies on sampling *many* candidates, so ``sample()``
fans N completions out across a thread pool (``concurrent.futures``, stdlib) —
the calls are I/O-bound, so threads are the right tool and we stay zero-dependency.
Set ``temperature`` above 0 there; identical greedy samples would defeat the
entire point of voting.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = os.environ.get("TTC_MODEL", "openai/gpt-oss-120b:free")
_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RETRIES = 5
_RETRY_BASE = 2.0
_MAX_WORKERS = 8

# Strip control characters that break json.loads (e.g. NUL bytes).
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CTRL.sub("", text)


def chat(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 768,
    stop: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Single chat completion via OpenRouter. Returns the assistant text."""
    if not _API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Export it before running: export OPENROUTER_API_KEY=sk-or-..."
        )

    payload: dict = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MONISMALIK1/test_time_compute",
        "X-Title": "Test-Time Compute: verifier-guided sampling",
    }

    for attempt in range(_MAX_RETRIES):
        req = urllib.request.Request(_API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = _clean(resp.read().decode())
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After", _RETRY_BASE ** attempt))
                time.sleep(wait)
                continue
            raise

        # Inline error in a 200 response (some free-tier models do this).
        if "error" in data and "choices" not in data:
            raise RuntimeError(f"API error: {data['error']}")

        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        return text.strip()

    raise RuntimeError("Exceeded max retries calling OpenRouter API")


def sample(
    prompt: str,
    n: int,
    model: str | None = None,
    temperature: float = 0.8,
    max_tokens: int = 768,
    timeout: int = 120,
) -> list[str]:
    """Draw ``n`` independent completions for ``prompt``, concurrently.

    Temperature defaults to 0.8 — diversity is the whole point of sampling many.
    Results preserve call order. Used to generate the candidate pool the
    verifier and aggregation strategies then choose among.
    """
    if n <= 0:
        return []

    def _one(_: int) -> str:
        return chat(prompt, model=model, temperature=temperature,
                    max_tokens=max_tokens, timeout=timeout)

    if n == 1:
        return [_one(0)]

    with ThreadPoolExecutor(max_workers=min(n, _MAX_WORKERS)) as pool:
        return list(pool.map(_one, range(n)))
