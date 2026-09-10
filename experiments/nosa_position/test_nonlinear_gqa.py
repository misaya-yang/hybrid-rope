import unittest

import torch

from experiments.nosa_position.nonlinear_gqa import improve_set, improve_sets_batched, NonlinearGQASelector
from experiments.nosa_position.exact_probe import ExactBlockSelector
from experiments.nosa_position.runtime import AttentionSettings, SelectionContext, mandatory_blocks


class NonlinearGQATests(unittest.TestCase):
    def test_complementary_heads_trade_additive_mass_for_lower_risk(self):
        # Additive top1 is block 1, but it starves head 2.
        p = torch.tensor([[.05, .75, .20, 0], [.05, .01, .55, .39]], dtype=torch.float64)
        self.assertEqual(int(p[:, 1:].sum(0).argmax()) + 1, 1)
        chosen, risk = improve_set(p, torch.tensor([0, 1]), torch.tensor([True, False, False, False]), torch.ones(4, dtype=torch.bool))
        self.assertEqual(chosen.tolist(), [0, 2])
        self.assertGreater(risk[0], risk[-1])
        self.assertLessEqual(len(risk), 3)

    def test_single_head_top_mass_null_and_full_set_null(self):
        p = torch.tensor([[.1, .5, .3, .1]], dtype=torch.float64)
        fixed = torch.tensor([True, False, False, False])
        for ids in ([0, 1, 2], [0, 1, 2, 3]):
            chosen, risk = improve_set(p, torch.tensor(ids), fixed, torch.ones(4, dtype=torch.bool))
            self.assertEqual(chosen.tolist(), ids)
            self.assertEqual(len(risk), 1)

    def test_fixed_filler_and_future_block_cannot_be_swapped(self):
        p = torch.tensor([[.05, .85, .1, 0], [.05, .01, .94, 0]])
        chosen, risk = improve_set(p, torch.tensor([0, 1]), torch.tensor([True, True, False, False]), torch.tensor([True, True, True, False]))
        self.assertEqual(chosen.tolist(), [0, 1])
        self.assertEqual(len(risk), 1)

    def test_live_selector_preserves_quota_anchors_and_causality(self):
        torch.manual_seed(102)
        settings = AttentionSettings(topk=4, select_blocks=1, local_blocks=1, attention_query_chunk_size=2)
        ctx = SelectionContext(torch.randn(4, 4, 8), torch.randn(2, 449, 8),
                               torch.randn(2, 449, 8), torch.randn(2, 449),
                               torch.tensor([0, 127, 320, 448]), 0, settings, torch.ones(4), 1.)
        b0 = ExactBlockSelector("exact_mass")(ctx)
        e02 = NonlinearGQASelector("e02_nonlinear")
        chosen = e02(ctx)
        torch.testing.assert_close((chosen >= 0).sum(-1), (b0 >= 0).sum(-1))
        anchors = mandatory_blocks(ctx, 8)
        for h in range(2):
            for t in range(4):
                valid = chosen[h, t][chosen[h, t] >= 0]
                self.assertEqual(valid.numel(), valid.unique().numel())
                self.assertTrue((valid <= ctx.query_positions[t] // 64).all())
                self.assertTrue(torch.isin(torch.where(anchors[t])[0], valid).all())
        self.assertGreaterEqual(e02.metrics["e02_risk_reduction"], 0.)


if __name__ == "__main__":
    unittest.main()
