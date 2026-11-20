import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentspec.llm.mock import MockTokenModel
from agentspec.llm.speculative import (
    residual_distribution,
    speculative_decode,
)


def _naive_greedy(model, prompt_ids, n):
    """Reference: plain autoregressive greedy decoding with the target."""
    ids = list(prompt_ids)
    for _ in range(n):
        probs = model.next_token_probs(ids)
        ids.append(max(range(len(probs)), key=probs.__getitem__))
    return ids[len(prompt_ids):]


class TestSpeculativeDecode(unittest.TestCase):
    def test_greedy_output_matches_target_autoregressive(self):
        """Core correctness: draft == target + greedy => identical output."""
        target = MockTokenModel(vocab_size=64, shared_seed=7)
        prompt = [0, 1, 2]
        result = speculative_decode(
            target, target, prompt, max_new_tokens=32, gamma=4, temperature=0.0
        )
        expected = _naive_greedy(target, prompt, 32)
        self.assertEqual(result.token_ids, expected)

    def test_greedy_accepts_everything_when_draft_equals_target(self):
        target = MockTokenModel(vocab_size=32, shared_seed=3)
        result = speculative_decode(
            target, target, [5], max_new_tokens=20, gamma=4, temperature=0.0
        )
        self.assertAlmostEqual(result.stats.acceptance_rate, 1.0)

    def test_output_length_never_exceeds_budget(self):
        target = MockTokenModel(vocab_size=32, shared_seed=1)
        draft = MockTokenModel(vocab_size=32, shared_seed=1, noise_seed=99, noise_scale=2.0)
        for budget in (1, 3, 7, 16):
            result = speculative_decode(
                target, draft, [0], max_new_tokens=budget, gamma=4, seed=42
            )
            self.assertLessEqual(len(result.token_ids), budget)

    def test_sampling_is_deterministic_with_seed(self):
        target = MockTokenModel(vocab_size=64, shared_seed=11)
        draft = MockTokenModel(vocab_size=64, shared_seed=11, noise_seed=5, noise_scale=1.0)
        a = speculative_decode(target, draft, [0], max_new_tokens=24, temperature=0.8, seed=123)
        b = speculative_decode(target, draft, [0], max_new_tokens=24, temperature=0.8, seed=123)
        self.assertEqual(a.token_ids, b.token_ids)

    def test_correlated_draft_gets_high_acceptance(self):
        target = MockTokenModel(vocab_size=64, shared_seed=21)
        draft = MockTokenModel(vocab_size=64, shared_seed=21, noise_seed=8, noise_scale=0.3)
        result = speculative_decode(
            target, draft, [0], max_new_tokens=64, gamma=4, temperature=0.8, seed=7
        )
        self.assertGreater(result.stats.acceptance_rate, 0.5)

    def test_vocab_mismatch_raises(self):
        target = MockTokenModel(vocab_size=64)
        draft = MockTokenModel(vocab_size=32)
        with self.assertRaises(ValueError):
            speculative_decode(target, draft, [0])

    def test_eos_stops_generation(self):
        target = MockTokenModel(vocab_size=16, shared_seed=2)
        result = speculative_decode(
            target, target, [0], max_new_tokens=64, gamma=4,
            temperature=0.0, eos_token_id=target.next_token_probs([0]).index(
                max(target.next_token_probs([0]))
            ),
        )
        self.assertLess(len(result.token_ids), 64)

    def test_residual_distribution(self):
        p = [0.5, 0.3, 0.2]
        q = [0.4, 0.3, 0.3]
        r = residual_distribution(p, q)
        self.assertAlmostEqual(sum(r), 1.0)
        self.assertGreater(r[0], 0.9)  # all residual mass is on index 0
        # p == q -> falls back to p
        self.assertEqual(residual_distribution(p, p), p)


if __name__ == "__main__":
    unittest.main()
