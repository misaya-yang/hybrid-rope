import unittest
import torch

from experiments.pm_keep.balanced_queries import type_weights, balanced_plan
from experiments.pm_keep.ops import make_sampling_plan


class BalancedQueriesTests(unittest.TestCase):
    def test_repeating_background_does_not_increase_its_total_probability(self):
        for count in [1, 100]:
            ids = torch.tensor([99] + [7] * count + [8, 9, 9])
            weights = type_weights(ids, 1)
            for token in [7, 8, 9]:
                self.assertAlmostEqual(float(weights[ids[1:] == token].sum()), 1/3)

    def test_keeps_positions_and_reproducible_prefix_only_samples(self):
        ids = torch.tensor([99, 7, 7, 7, 8, 9, 9])
        plan = make_sampling_plan(7, 4, samples_per_head=32, horizon=128, sink_tokens=1)
        weights = type_weights(ids, 1)
        changed = balanced_plan(plan, weights, 1)
        torch.testing.assert_close(changed.future_positions, plan.future_positions)
        torch.testing.assert_close(changed.query_indices, balanced_plan(plan, weights, 1).query_indices)
        self.assertTrue(bool(((changed.query_indices >= 1) & (changed.query_indices < 7)).all()))


if __name__ == "__main__":
    unittest.main()
