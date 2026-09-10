import unittest

import torch

from .causal_probe import paired_swap_sets


class SetInterventionTests(unittest.TestCase):
    def test_same_budget_same_removals_common_protection_no_mutation(self):
        c = torch.tensor([[0, 1, 4, 5, 14, 15], [0, 1, 2, 3, 14, 15]])
        p = torch.tensor([[0, 1, 6, 7, 14, 15], [0, 1, 4, 5, 14, 15]])
        before = c.clone()
        actual, sham, receipt = paired_swap_sets(c, p, 16)
        self.assertTrue(torch.equal(actual, p))
        self.assertTrue(torch.equal(c, before))
        self.assertEqual(sham.shape, c.shape)
        for h in range(2):
            self.assertEqual(set(c[h].tolist()) - set(actual[h].tolist()), set(c[h].tolist()) - set(sham[h].tolist()))
            self.assertTrue({0, 1, 14, 15}.issubset(sham[h].tolist()))
            self.assertEqual(receipt[h]["swaps"], 2)

    def test_identical_sets_explicitly_have_no_intervention(self):
        c = torch.tensor([[0, 1, 4, 5]])
        actual, sham, receipt = paired_swap_sets(c, c, 8)
        self.assertTrue(torch.equal(actual, c) and torch.equal(sham, c))
        self.assertEqual(receipt[0]["swaps"], 0)

    def test_fixed_seed_and_limited_replacements(self):
        c, p = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[0, 1, 4, 5]])
        a, s, r = paired_swap_sets(c, p, 16, max_swaps=1)
        a2, s2, _ = paired_swap_sets(c, p, 16, max_swaps=1)
        self.assertTrue(torch.equal(a, a2) and torch.equal(s, s2))
        self.assertEqual(r[0]["swaps"], 1)


if __name__ == "__main__":
    unittest.main()
