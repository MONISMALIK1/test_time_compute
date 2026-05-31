"""Tests for the three selection strategies — the core of test-time compute.

The point of these is to prove the strategies actually differ where the paper
says they should: majority follows the crowd, best-of-N follows the top score,
and weighted combines the two (and can be rescued by a good verifier from a wrong
majority).
"""

import unittest

from test_time_compute.aggregate import (
    Candidate,
    aggregate,
    best_score,
    majority_vote,
    tally_votes,
    tally_weight,
    weighted_majority,
)


def cands(pairs):
    """pairs: list of (answer, score) -> list[Candidate]."""
    return [Candidate(text=f"sol {a}", answer=a, score=s) for a, s in pairs]


class TallyTests(unittest.TestCase):
    def test_votes_and_weights(self):
        c = cands([("42", 0.2), ("42", 0.3), ("7", 0.9)])
        self.assertEqual(tally_votes(c), {"42": 2, "7": 1})
        self.assertAlmostEqual(tally_weight(c)["42"], 0.5)
        self.assertAlmostEqual(tally_weight(c)["7"], 0.9)

    def test_none_answers_are_dropped(self):
        c = [Candidate("junk", None, 1.0), Candidate("ok", "5", 1.0)]
        self.assertEqual(tally_votes(c), {"5": 1})


class MajorityTests(unittest.TestCase):
    def test_plurality_wins(self):
        c = cands([("42", 0.1), ("42", 0.1), ("7", 0.99)])
        self.assertEqual(majority_vote(c), "42")  # ignores scores entirely

    def test_empty_is_none(self):
        self.assertIsNone(majority_vote([]))
        self.assertIsNone(majority_vote([Candidate("x", None)]))


class BestScoreTests(unittest.TestCase):
    def test_takes_highest_scoring_candidate(self):
        c = cands([("42", 0.1), ("42", 0.2), ("7", 0.95)])
        self.assertEqual(best_score(c), "7")  # one strong score beats the crowd

    def test_score_tie_breaks_to_earliest(self):
        c = cands([("3", 0.8), ("9", 0.8)])
        self.assertEqual(best_score(c), "3")


class WeightedMajorityTests(unittest.TestCase):
    def test_good_verifier_overrides_wrong_majority(self):
        # Three weak wrong votes vs two strong right ones: weighted picks right.
        c = cands([("42", 0.1), ("42", 0.1), ("42", 0.1), ("7", 0.9), ("7", 0.9)])
        self.assertEqual(majority_vote(c), "42")      # crowd is wrong
        self.assertEqual(weighted_majority(c), "7")   # verifier rescues it

    def test_reduces_to_majority_when_scores_equal(self):
        c = cands([("42", 1.0), ("42", 1.0), ("7", 1.0)])
        self.assertEqual(weighted_majority(c), majority_vote(c))


class DispatchTests(unittest.TestCase):
    def test_aggregate_dispatches_by_name(self):
        # Clear plurality for 42, but 7 carries the single highest score, so the
        # two strategies must disagree — proving dispatch routes to each.
        c = cands([("42", 0.1), ("42", 0.1), ("7", 0.95)])
        self.assertEqual(aggregate(c, "majority"), "42")
        self.assertEqual(aggregate(c, "best"), "7")

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            aggregate(cands([("1", 1.0)]), "nonsense")


if __name__ == "__main__":
    unittest.main()
