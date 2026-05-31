# Test-Time Compute — sample many, verify, choose

[![tests](https://github.com/MONISMALIK1/test_time_compute/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/test_time_compute/actions/workflows/test.yml)

A from-scratch, dependency-free implementation of **scaling test-time compute**
(Snell et al., 2024 — [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)),
with the outcome-verifier idea from Cobbe et al., 2021
([arXiv:2110.14168](https://arxiv.org/abs/2110.14168)).

For years the only knob that made models smarter was *bigger* — more parameters,
more training. Test-time compute is the other knob: keep the model fixed and
spend more effort **at inference**. Sample many reasoning paths, score them with
a verifier, and choose. A small model that's allowed to think longer can beat a
much larger model answering in one shot — and you only pay the thinking tax on
the questions that are actually hard.

```
question ──► sample N solutions ──► verifier scores each ──► aggregate ──► answer
                  │                        │                     │
          (diverse reasoning,      (how likely correct?)   (majority / best-of-N /
           temperature > 0)                                  weighted vote)
```

## The capstone of the series

The earlier repos are all special cases of "use more inference to get a better
answer" — this one puts them behind a single interface:

| Earlier repo | What it is, in this frame |
| --- | --- |
| **Self-Consistency** | majority vote over N samples — the verifier-free baseline |
| **Tree of Thoughts** | search over partial solutions with a self-scored heuristic |
| **Reflexion** | use a verdict to *revise*, not just re-sample |

Test-time compute is the unifying idea: **a candidate generator + a verifier +
an aggregation rule**, plus a policy for *how much* to spend per question.

## The three ways to choose (and why it matters)

Once you've sampled N candidates, you have to pick one. The paper compares:

- **`majority`** — count answers, take the plurality. Trusts the crowd; needs no
  verifier. This is Self-Consistency.
- **`best`** — Best-of-N: take the single highest-scoring candidate. Strong with
  a good verifier, fragile with a noisy one (one overconfident wrong score wins).
- **`weighted`** — sum verifier scores per answer, take the highest total. The
  usual winner: a good verifier can rescue a *wrong* majority, while a lone bad
  score can't override a well-supported consensus.

```text
Candidates (answer, verifier score):
    42 (0.1)   42 (0.1)   42 (0.1)   7 (0.9)   7 (0.9)

  majority  -> 42   (three votes beat two — but it's wrong)
  best      -> 7    (highest single score)
  weighted  -> 7    (0.3 for 42 vs 1.8 for 7 — the verifier rescues it)
```

That `weighted` flip is the whole point, and it's exercised directly in the tests
(`test_good_verifier_flips_wrong_majority`).

## Compute-optimal allocation

A flat budget is wasteful: easy questions are solved at N=1, hard ones stay wrong
at N=64. So `--adaptive` does what the paper's headline result prescribes —
**probe cheaply, then spend more only where it's needed**:

1. sample a small probe batch,
2. measure how much the candidates *disagree* (normalized entropy of the answer
   spread — the same signal that makes voting work),
3. a unanimous probe stops early; a scattered one tops up toward `--n-max`.

The disagreement → budget policy is pure arithmetic (`allocate.py`) and fully
unit-tested.

## What's deterministic (and tested offline)

Everything that *decides* runs without a network call, so the logic is unit-tested
with the LLM mocked — only sampling and the LLM-judge touch the wire:

| Module | Role | Network? |
| --- | --- | --- |
| `extract.py` | parse + normalize + grade the final answer | no |
| `aggregate.py` | the three selection strategies | no |
| `allocate.py` | compute-optimal budget from disagreement | no |
| `verifier.py` | `MajorityVerifier` (no-op) / `LLMVerifier` (judge) | judge only |
| `core.py` | the solve loop wiring it together | sampling only |

56 tests, no API key required:

```bash
make test          # or: cd .. && python -m unittest discover -s test_time_compute/tests -t .
```

## Usage

Set your key (read only from the environment, never written to disk):

```bash
export OPENROUTER_API_KEY=sk-or-...
```

```bash
# Solve by sampling 8 candidates and taking a weighted vote
python -m test_time_compute "A train travels 60 mph for 3 hours. How far?"

# Pick the strategy and budget
python -m test_time_compute "..." --n 16 --strategy majority

# Use the LLM-as-judge verifier (otherwise every candidate scores 1.0)
python -m test_time_compute "..." --verifier llm --strategy weighted

# Compute-optimal: probe 4, top up toward 16 only if the probe disagrees
python -m test_time_compute "..." --adaptive --probe-n 4 --n-max 16

# Baseline: a single greedy sample (the n=1 comparison)
python -m test_time_compute "..." --baseline

# See every candidate, its parsed answer, and its score
python -m test_time_compute "..." --show-candidates

# Benchmark the accuracy curve on the bundled math set
python -m test_time_compute --bench --n 8 --verifier llm --strategy weighted
```

The model defaults to a free OpenRouter slug; override with `--model` or the
`TTC_MODEL` environment variable.

## ⚠️ Honest caveat: scaffolding, not a trained verifier

The strongest forms of test-time compute — speculative decoding, or a **trained
process-reward model** that scores each reasoning *step* — need token-level
logits, which OpenRouter's chat API doesn't expose. So the RL-trained verifier
from the paper isn't faithfully reproducible here.

What this repo *is*, and tests honestly: the **search + verifier scaffolding** —
best-of-N, weighted majority, verifier-best, an LLM-as-judge outcome verifier,
and compute-optimal allocation. That's the reproducible core of the idea, and the
selection logic is identical regardless of where the scores come from. Swap in a
better verifier and everything downstream still holds.

## Layout

```
extract.py     answer parsing + numeric grading        (pure)
aggregate.py   majority / best-of-N / weighted vote     (pure)
allocate.py    disagreement -> compute-optimal budget   (pure)
verifier.py    MajorityVerifier / LLMVerifier            (judge calls out)
prompts.py     CoT generator + verifier prompts          (parse is pure)
dataset.py     12 bundled math problems with answers
llm.py         OpenRouter wrapper + threaded N-sampling
core.py        solve / solve_adaptive / solve_baseline
__main__.py    CLI
tests/         5 offline suites, LLM always mocked
```

## License

MIT — see [LICENSE](LICENSE).
