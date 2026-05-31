"""CLI for test-time compute.

Usage:
    # Solve one problem by sampling 8 candidates and taking a weighted vote
    python -m test_time_compute "A train travels 60 mph for 3 hours. How far?"

    # Pick the selection strategy: majority | best | weighted
    python -m test_time_compute "..." --n 16 --strategy majority

    # Use the LLM-as-judge verifier (otherwise every candidate scores 1.0)
    python -m test_time_compute "..." --verifier llm --strategy weighted

    # Compute-optimal: probe cheaply, then spend more only if the probe disagrees
    python -m test_time_compute "..." --adaptive --probe-n 4 --n-max 16

    # Baseline: a single greedy sample (the n=1 comparison)
    python -m test_time_compute "..." --baseline

    # See every sampled candidate, its parsed answer, and its score
    python -m test_time_compute "..." --show-candidates

    # Benchmark the accuracy curve on the bundled math set
    python -m test_time_compute --bench --n 8 --verifier llm --strategy weighted
"""

from __future__ import annotations

import argparse
import sys
import time

from .aggregate import STRATEGIES
from .core import solve, solve_adaptive, solve_baseline
from .dataset import SAMPLE_PROBLEMS
from .extract import grade
from .llm import DEFAULT_MODEL
from .verifier import VERIFIERS, make_verifier


def _print_candidates(res) -> None:
    print("--- candidates ---")
    for i, c in enumerate(res.candidates, 1):
        head = c.text.replace("\n", " ")
        if len(head) > 80:
            head = head[:77] + "..."
        print(f"[{i:2d}] answer={str(c.answer):<8} score={c.score:.2f}  {head}")
    print(f"agreement={res.agreement:.2f}  disagreement={res.disagreement:.2f}")
    print("------------------")


def _solve_one(args, model: str):
    verifier = make_verifier(args.verifier)
    if args.adaptive:
        return solve_adaptive(
            args.question, probe_n=args.probe_n, n_max=args.n_max,
            strategy=args.strategy, verifier=verifier, model=model,
            temperature=args.temperature,
        )
    return solve(
        args.question, n=args.n, strategy=args.strategy,
        verifier=verifier, model=model, temperature=args.temperature,
    )


def _bench(args, model: str) -> int:
    verifier_name = args.verifier
    problems = SAMPLE_PROBLEMS[: args.num] if args.num else SAMPLE_PROBLEMS

    base_correct = 0
    ttc_correct = 0
    total_samples = 0
    total_secs = 0.0

    for i, prob in enumerate(problems, 1):
        t0 = time.monotonic()
        base = solve_baseline(prob.question, model=model)
        if args.adaptive:
            ttc = solve_adaptive(prob.question, probe_n=args.probe_n, n_max=args.n_max,
                                 strategy=args.strategy, verifier=make_verifier(verifier_name),
                                 model=model, temperature=args.temperature)
        else:
            ttc = solve(prob.question, n=args.n, strategy=args.strategy,
                        verifier=make_verifier(verifier_name), model=model,
                        temperature=args.temperature)
        secs = time.monotonic() - t0
        total_secs += secs
        total_samples += ttc.n

        base_ok = grade(base.answer, prob.answer)
        ttc_ok = grade(ttc.answer, prob.answer)
        base_correct += int(base_ok)
        ttc_correct += int(ttc_ok)

        print(
            f"[{i:2d}/{len(problems)}] TTC {'OK' if ttc_ok else '--'} "
            f"(n={ttc.n:2d} agree={ttc.agreement:.2f} ans={ttc.answer}) "
            f"| base {'OK' if base_ok else '--'} | {secs:4.1f}s  gold={prob.answer}",
            flush=True,
        )

    n = len(problems)
    print("\n" + "=" * 72)
    mode = (f"adaptive probe={args.probe_n} n_max={args.n_max}"
            if args.adaptive else f"n={args.n}")
    print(f"Math eval — {n} problems, model={model}")
    print(f"  strategy={args.strategy}  verifier={verifier_name}  {mode}")
    print("=" * 72)
    print(f"  Test-time compute: {ttc_correct}/{n} = {ttc_correct / n * 100:5.1f}%")
    print(f"  Baseline (n=1):    {base_correct}/{n} = {base_correct / n * 100:5.1f}%")
    print(f"  TTC advantage:     {ttc_correct - base_correct:+d} problems")
    print(f"  Avg samples/problem: {total_samples / n:.1f}   avg time: {total_secs / n:.1f}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="test_time_compute",
        description="Test-Time Compute (Snell et al., 2024): sample many, "
                    "verify, and choose — instead of scaling parameters.",
    )
    p.add_argument("question", nargs="?", help="The problem to solve.")
    p.add_argument("--n", type=int, default=8, help="Candidates to sample (default: 8).")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="weighted",
                   help="How to pick the answer (default: weighted).")
    p.add_argument("--verifier", choices=sorted(VERIFIERS), default="majority",
                   help="Scorer: 'majority' (no network) or 'llm' (default: majority).")
    p.add_argument("--model", default=None, help=f"Model slug (default: {DEFAULT_MODEL}).")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for candidate diversity (default: 0.8).")
    p.add_argument("--baseline", action="store_true",
                   help="Single greedy sample — the n=1 comparison.")
    p.add_argument("--show-candidates", action="store_true",
                   help="Print every sampled candidate with its answer and score.")

    p.add_argument("--adaptive", action="store_true",
                   help="Compute-optimal: probe, then top up only if candidates disagree.")
    p.add_argument("--probe-n", type=int, default=4, help="Adaptive: probe batch size.")
    p.add_argument("--n-max", type=int, default=16, help="Adaptive: max total samples.")

    p.add_argument("--bench", action="store_true",
                   help="Benchmark TTC vs the n=1 baseline on the bundled math set.")
    p.add_argument("--num", type=int, default=0, help="Bench: limit number of problems.")
    args = p.parse_args()

    model = args.model or DEFAULT_MODEL

    if args.bench:
        return _bench(args, model)

    if not args.question:
        p.error("provide a problem to solve, or use --bench")

    print(f"\nProblem: {args.question}", file=sys.stderr)
    print(f"Model: {model}\n", file=sys.stderr, flush=True)

    if args.baseline:
        res = solve_baseline(args.question, model=model)
        if args.show_candidates:
            _print_candidates(res)
        print("=" * 60)
        print(f"Answer (baseline, n=1): {res.answer}")
        return 0

    res = _solve_one(args, model)
    if args.show_candidates:
        _print_candidates(res)
    print("=" * 60)
    print(f"Answer ({res.strategy}, n={res.n}): {res.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
