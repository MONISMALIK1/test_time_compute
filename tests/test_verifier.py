"""Tests for the verifiers and the deterministic score parser.

The LLM judge is driven by a fake ``chat_fn`` so the whole thing runs offline:
we prove the judge's reply is parsed into a usable score, that no-answer
candidates are zeroed, and that the trivial verifier reproduces Self-Consistency.
"""

import unittest

from test_time_compute.aggregate import Candidate
from test_time_compute.prompts import parse_score
from test_time_compute.verifier import (
    LLMVerifier,
    MajorityVerifier,
    make_verifier,
)


class ParseScoreTests(unittest.TestCase):
    def test_explicit_marker(self):
        self.assertAlmostEqual(parse_score("Looks right. SCORE: 0.8"), 0.8)

    def test_marker_case_insensitive(self):
        self.assertAlmostEqual(parse_score("score: 1"), 1.0)

    def test_clamped(self):
        self.assertEqual(parse_score("SCORE: 1.5"), 1.0)
        self.assertEqual(parse_score("SCORE: -0.2"), 0.0)

    def test_falls_back_to_first_number(self):
        self.assertAlmostEqual(parse_score("I'd say 0.3 likely"), 0.3)

    def test_no_number_uses_default(self):
        self.assertEqual(parse_score("no idea", default=0.5), 0.5)
        self.assertEqual(parse_score(""), 0.5)


class MajorityVerifierTests(unittest.TestCase):
    def test_scores_everything_one(self):
        c = [Candidate("a", "42", 0.0), Candidate("b", "7", 0.0)]
        MajorityVerifier().score("q", c)
        self.assertEqual([x.score for x in c], [1.0, 1.0])


class LLMVerifierTests(unittest.TestCase):
    def test_uses_chat_reply_as_score(self):
        replies = iter(["SCORE: 0.9", "SCORE: 0.2"])

        def fake_chat(prompt, **kwargs):
            return next(replies)

        c = [Candidate("good sol", "42"), Candidate("bad sol", "7")]
        LLMVerifier(chat_fn=fake_chat).score("q", c)
        self.assertAlmostEqual(c[0].score, 0.9)
        self.assertAlmostEqual(c[1].score, 0.2)

    def test_question_and_solution_reach_the_judge(self):
        captured = {}

        def fake_chat(prompt, **kwargs):
            captured["prompt"] = prompt
            return "SCORE: 0.5"

        LLMVerifier(chat_fn=fake_chat).score("What is 2+2?", [Candidate("the sol text", "4")])
        self.assertIn("What is 2+2?", captured["prompt"])
        self.assertIn("the sol text", captured["prompt"])

    def test_no_answer_candidate_scored_zero_without_calling(self):
        calls = []

        def fake_chat(prompt, **kwargs):
            calls.append(prompt)
            return "SCORE: 1.0"

        c = [Candidate("rambling, no number", None)]
        LLMVerifier(chat_fn=fake_chat).score("q", c)
        self.assertEqual(c[0].score, 0.0)
        self.assertEqual(calls, [])  # didn't waste a call on an answerless candidate


class FactoryTests(unittest.TestCase):
    def test_make_verifier_by_name(self):
        self.assertIsInstance(make_verifier("majority"), MajorityVerifier)
        self.assertIsInstance(make_verifier("llm"), LLMVerifier)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            make_verifier("nope")


if __name__ == "__main__":
    unittest.main()
