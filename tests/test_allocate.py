"""Tests for compute-optimal allocation — spend more on harder questions.

These pin down the policy: a unanimous probe is "easy" and gets the floor budget;
a split probe is "hard" and gets more, scaling monotonically with disagreement
and never exceeding n_max.
"""

import unittest

from test_time_compute.aggregate import Candidate
from test_time_compute.allocate import agreement, allocate, disagreement, optimal_n


def cands(answers):
    return [Candidate(text=f"sol {a}", answer=a) for a in answers]


class AgreementTests(unittest.TestCase):
    def test_unanimous(self):
        self.assertEqual(agreement(cands(["7", "7", "7"])), 1.0)

    def test_split(self):
        self.assertAlmostEqual(agreement(cands(["7", "7", "3", "9"])), 0.5)

    def test_empty(self):
        self.assertEqual(agreement([]), 0.0)


class DisagreementTests(unittest.TestCase):
    def test_unanimous_is_zero(self):
        self.assertEqual(disagreement(cands(["7", "7", "7"])), 0.0)

    def test_max_spread_is_one(self):
        # Four distinct answers, evenly split -> normalized entropy 1.0.
        self.assertAlmostEqual(disagreement(cands(["1", "2", "3", "4"])), 1.0)

    def test_partial_between(self):
        d = disagreement(cands(["7", "7", "7", "3"]))
        self.assertTrue(0.0 < d < 1.0)

    def test_single_candidate(self):
        self.assertEqual(disagreement(cands(["7"])), 0.0)


class OptimalNTests(unittest.TestCase):
    def test_unanimous_probe_gets_floor(self):
        probe = cands(["7", "7", "7", "7"])
        self.assertEqual(optimal_n(probe, n_min=4, n_max=16), 4)

    def test_fully_split_probe_gets_ceiling(self):
        probe = cands(["1", "2", "3", "4"])  # agreement 0.25 -> high budget
        self.assertEqual(optimal_n(probe, n_min=1, n_max=16), 12)

    def test_more_disagreement_means_more_budget(self):
        easy = cands(["7", "7", "7", "3"])     # agreement 0.75
        hard = cands(["7", "7", "3", "9"])     # agreement 0.50
        self.assertGreater(
            optimal_n(hard, n_min=1, n_max=16),
            optimal_n(easy, n_min=1, n_max=16),
        )

    def test_never_below_probe_already_spent(self):
        probe = cands(["7", "7", "7", "7", "7", "7"])  # 6 spent, but "easy"
        self.assertGreaterEqual(optimal_n(probe, n_min=1, n_max=16), 6)

    def test_bad_bounds_raise(self):
        with self.assertRaises(ValueError):
            optimal_n(cands(["1"]), n_min=10, n_max=4)


class AllocateBatchTests(unittest.TestCase):
    def test_per_question_targets(self):
        probes = {
            "easy": cands(["7", "7", "7", "7"]),
            "hard": cands(["1", "2", "3", "4"]),
        }
        out = allocate(probes, n_min=4, n_max=16)
        self.assertEqual(out["easy"], 4)
        self.assertGreater(out["hard"], out["easy"])


if __name__ == "__main__":
    unittest.main()
