"""Sanity checks on the bundled eval set.

Cheap guards that catch a typo'd problem before it silently skews a benchmark:
ids are unique, every gold answer is numeric, and each gold answer survives a
round-trip through the same extraction/grading the benchmark uses.
"""

import unittest

from test_time_compute.dataset import SAMPLE_PROBLEMS
from test_time_compute.extract import grade, normalize


class DatasetTests(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [p.id for p in SAMPLE_PROBLEMS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_questions_and_answers_present(self):
        for p in SAMPLE_PROBLEMS:
            self.assertTrue(p.question.strip(), f"{p.id} has an empty question")
            self.assertTrue(p.answer.strip(), f"{p.id} has an empty answer")

    def test_gold_answers_are_numeric(self):
        for p in SAMPLE_PROBLEMS:
            # normalize() leaves non-numeric strings untouched, so a numeric gold
            # answer must parse as a float after normalization.
            float(normalize(p.answer))  # raises if a gold answer isn't a number

    def test_gold_answer_round_trips_through_grading(self):
        # A model that emits exactly the gold answer must be graded correct — this
        # is what the benchmark relies on, so prove it can't silently fail.
        for p in SAMPLE_PROBLEMS:
            self.assertTrue(grade(p.answer, p.answer), f"{p.id} fails self-grade")


if __name__ == "__main__":
    unittest.main()
