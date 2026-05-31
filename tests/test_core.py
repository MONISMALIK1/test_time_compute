"""Tests for the solve loop — the sampler is mocked, so this is fully offline.

These prove the control flow the paper describes actually happens: the question
reaches the generator prompt, candidates are scored and aggregated by the chosen
strategy, a good verifier can flip a wrong majority, the baseline is a single
greedy sample, and adaptive allocation only tops up when the probe disagrees.
"""

import unittest
from unittest.mock import patch

from test_time_compute import core
from test_time_compute.core import solve, solve_adaptive, solve_baseline
from test_time_compute.verifier import LLMVerifier, MajorityVerifier


def fixed_sampler(texts):
    """A core.sample stand-in that returns a fixed list, ignoring n."""
    def _s(prompt, n, model=None, temperature=0.8):
        return list(texts)
    return _s


class SolveTests(unittest.TestCase):
    def test_majority_picks_plurality(self):
        texts = ["Reason... #### 42", "#### 42", "#### 7"]
        with patch.object(core, "sample", fixed_sampler(texts)):
            res = solve("q", n=3, strategy="majority", verifier=MajorityVerifier())
        self.assertEqual(res.answer, "42")
        self.assertEqual(res.n, 3)
        self.assertEqual([c.answer for c in res.candidates], ["42", "42", "7"])

    def test_question_reaches_the_generator_prompt(self):
        captured = {}

        def spy(prompt, n, model=None, temperature=0.8):
            captured["prompt"] = prompt
            captured["n"] = n
            return ["#### 4"] * n

        with patch.object(core, "sample", spy):
            solve("When was Helix founded?", n=5)
        self.assertIn("When was Helix founded?", captured["prompt"])
        self.assertEqual(captured["n"], 5)

    def test_good_verifier_flips_wrong_majority(self):
        # Majority says 42 (wrong); the judge scores the '7' solutions high.
        texts = ["#### 42", "#### 42", "#### 42", "sol seven #### 7", "sol seven #### 7"]

        def judge(prompt, **kwargs):
            return "SCORE: 0.9" if "seven" in prompt else "SCORE: 0.1"

        with patch.object(core, "sample", fixed_sampler(texts)):
            res = solve("q", n=5, strategy="weighted", verifier=LLMVerifier(chat_fn=judge))
        self.assertEqual(res.answer, "7")

    def test_agreement_property(self):
        texts = ["#### 7", "#### 7", "#### 7", "#### 3"]
        with patch.object(core, "sample", fixed_sampler(texts)):
            res = solve("q", n=4, strategy="majority")
        self.assertEqual(res.answer, "7")
        self.assertAlmostEqual(res.agreement, 0.75)


class BaselineTests(unittest.TestCase):
    def test_single_greedy_sample(self):
        captured = {}

        def spy(prompt, n, model=None, temperature=0.8):
            captured["n"] = n
            captured["temperature"] = temperature
            return ["#### 180"]

        with patch.object(core, "sample", spy):
            res = solve_baseline("how far?")
        self.assertEqual(captured["n"], 1)
        self.assertEqual(captured["temperature"], 0.0)  # greedy, not sampled
        self.assertEqual(res.answer, "180")
        self.assertFalse(res.used_compute)


class AdaptiveTests(unittest.TestCase):
    def test_easy_question_stops_at_probe(self):
        calls = []

        def sampler(prompt, n, model=None, temperature=0.8):
            calls.append(n)
            return ["#### 7"] * n  # unanimous probe -> "easy"

        with patch.object(core, "sample", sampler):
            res = solve_adaptive("q", probe_n=4, n_max=16, strategy="majority")
        self.assertEqual(calls, [4])      # probed once, no top-up
        self.assertEqual(res.n, 4)
        self.assertEqual(res.answer, "7")

    def test_hard_question_tops_up(self):
        batches = iter([
            ["#### 1", "#### 2", "#### 3", "#### 4"],  # split probe -> "hard"
            ["#### 5"] * 9,                            # the top-up
        ])
        calls = []

        def sampler(prompt, n, model=None, temperature=0.8):
            calls.append(n)
            return next(batches)

        with patch.object(core, "sample", sampler):
            res = solve_adaptive("q", probe_n=4, n_max=16, strategy="majority")
        self.assertEqual(calls[0], 4)     # probe
        self.assertEqual(len(calls), 2)   # topped up
        self.assertEqual(res.n, 13)       # 4 probe + 9 top-up


if __name__ == "__main__":
    unittest.main()
