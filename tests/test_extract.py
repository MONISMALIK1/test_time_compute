"""Tests for answer extraction, normalization, and grading.

This is the deterministic floor every selection strategy stands on, so the
parsing is exercised across the messy ways a model ends a reasoning chain.
"""

import unittest

from test_time_compute.extract import extract_answer, grade, normalize


class NormalizeTests(unittest.TestCase):
    def test_strips_currency_commas_and_spaces(self):
        self.assertEqual(normalize("$1,024.00"), "1024")
        self.assertEqual(normalize(" 1024 "), "1024")

    def test_integers_and_floats_canonicalize(self):
        self.assertEqual(normalize("7.0"), "7")
        self.assertEqual(normalize("7"), "7")
        self.assertEqual(normalize("3.5"), "3.5")

    def test_non_numeric_passes_through(self):
        self.assertEqual(normalize("none"), "none")


class ExtractTests(unittest.TestCase):
    def test_hash_marker_wins(self):
        text = "Lots of reasoning, maybe 12 here and 99 there.\n#### 42"
        self.assertEqual(extract_answer(text), "42")

    def test_answer_phrase(self):
        self.assertEqual(extract_answer("So the answer is 180 miles."), "180")
        self.assertEqual(extract_answer("Therefore = 15"), "15")

    def test_falls_back_to_last_number(self):
        self.assertEqual(extract_answer("Step 1: 5 apples. Step 2: 3 more. Total 8."), "8")

    def test_handles_commas_and_dollars(self):
        self.assertEqual(extract_answer("The total cost is $1,250."), "1250")

    def test_negative_number(self):
        self.assertEqual(extract_answer("#### -7"), "-7")

    def test_no_number_returns_none(self):
        self.assertIsNone(extract_answer("I have no idea."))
        self.assertIsNone(extract_answer(""))

    def test_hash_beats_a_later_stray_number(self):
        # The gold marker should win even if prose continues after it.
        self.assertEqual(extract_answer("#### 4\n(checking: 4 boxes, looks right, 6 each)"), "4")


class GradeTests(unittest.TestCase):
    def test_numeric_equality_ignores_formatting(self):
        self.assertTrue(grade("1,024", "1024"))
        self.assertTrue(grade("15.0", "15"))
        self.assertTrue(grade("$180", "180"))

    def test_wrong_answer_fails(self):
        self.assertFalse(grade("41", "42"))

    def test_none_prediction_fails(self):
        self.assertFalse(grade(None, "42"))

    def test_tolerance(self):
        self.assertTrue(grade("0.3333333", "0.3333334", tol=1e-3))
        self.assertFalse(grade("0.3", "0.4", tol=1e-3))

    def test_non_numeric_falls_back_to_string(self):
        self.assertTrue(grade("none", "none"))
        self.assertFalse(grade("none", "42"))


if __name__ == "__main__":
    unittest.main()
