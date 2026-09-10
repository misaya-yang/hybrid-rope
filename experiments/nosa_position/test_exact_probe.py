import unittest
import torch

from experiments.nosa_position.exact_probe import ExactBlockSelector
from experiments.nosa_position.selector_controls import BlockSummarySelector
from experiments.nosa_position.runtime import AttentionSettings, SelectionContext


class ExactBlockMassTests(unittest.TestCase):
    def test_matches_explicit_block_reference_with_causality_and_partial_block(self):
        torch.manual_seed(71)
        settings = AttentionSettings(topk=4, select_blocks=1, local_blocks=1,
                                     attention_query_chunk_size=2)
        length, dim = 385, 8
        ctx = SelectionContext(torch.randn(4, 5, dim), torch.randn(2, length, dim),
                               torch.randn(2, length, dim), torch.randn(2, length),
                               torch.tensor([0, 63, 64, 191, 384]), 0, settings,
                               torch.ones(dim//2), 1.0)
        reference, fast = BlockSummarySelector("exact_mass"), ExactBlockSelector("exact_mass")
        torch.testing.assert_close(fast.logmass(ctx), reference.logmass(ctx), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(fast(ctx), reference(ctx), atol=0, rtol=0)
        # Future keys must not change any earlier row's block mass.
        before = fast.logmass(ctx)[:, :, :4].clone()
        ctx.k[:, 192:] += 100
        torch.testing.assert_close(fast.logmass(ctx)[:, :, :4], before, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
