"""Tests for the sampling layer — chat is patched, so this runs offline.

The contract that matters for voting: N-sampling tolerates a few failed
completions (a smaller vote is fine) but raises if every one fails (nothing to
choose among).
"""

import unittest
from unittest.mock import patch

from test_time_compute import llm


class LocalEndpointTests(unittest.TestCase):
    def test_detects_local_hosts(self):
        self.assertTrue(llm._is_local("http://localhost:11434/v1/chat/completions"))
        self.assertTrue(llm._is_local("http://127.0.0.1:1234/v1/chat/completions"))

    def test_remote_is_not_local(self):
        self.assertFalse(llm._is_local("https://openrouter.ai/api/v1/chat/completions"))


class SampleTests(unittest.TestCase):
    def test_zero_returns_empty(self):
        self.assertEqual(llm.sample("p", 0), [])

    def test_collects_all_successes(self):
        with patch.object(llm, "chat", return_value="ok"):
            out = llm.sample("p", 5)
        self.assertEqual(out, ["ok"] * 5)

    def test_one_failure_is_dropped_not_fatal(self):
        calls = {"i": 0}

        def flaky(prompt, **kwargs):
            calls["i"] += 1
            if calls["i"] == 2:          # the 2nd completion blows up
                raise RuntimeError("boom")
            return "ok"

        with patch.object(llm, "chat", side_effect=flaky):
            out = llm.sample("p", 4)
        self.assertEqual(out, ["ok"] * 3)   # 4 requested, 1 dropped, 3 survive

    def test_all_failures_raise(self):
        with patch.object(llm, "chat", side_effect=RuntimeError("nope")):
            with self.assertRaises(RuntimeError):
                llm.sample("p", 3)

    def test_single_sample_failure_propagates(self):
        # n == 1 has nothing to be resilient about — let the error surface.
        with patch.object(llm, "chat", side_effect=RuntimeError("nope")):
            with self.assertRaises(RuntimeError):
                llm.sample("p", 1)


if __name__ == "__main__":
    unittest.main()
