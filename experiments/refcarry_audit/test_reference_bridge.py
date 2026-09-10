import unittest

import torch

from experiments.refcarry_audit.reference_bridge import ReferenceBridge


class BridgeChecks(unittest.TestCase):
    def test_two_heads_causal_handoff_and_zero_native(self):
        torch.manual_seed(1729)
        bridge = ReferenceBridge(5, 2, 3, torch.tensor([.5, .03], dtype=torch.float64))
        writer_pos = torch.tensor([[2, 7, 19], [2, 8, 19]])
        writer_prob = torch.tensor([[.2, .5, .3], [.2, .6, .2]], dtype=torch.float64)
        positions = torch.tensor([[1, 4, 8, 19], [2, 5, 9, 19], [0, 6, 12, 19]])
        logits = torch.randn(3, 4, dtype=torch.float64)
        values = torch.randn(3, 4, 7, dtype=torch.float64)
        hidden = torch.randn(5, dtype=torch.float64)
        with self.assertRaises(ValueError):
            bridge.read(hidden, logits, values, positions, 19)
        bridge.write(writer_pos, writer_prob, 19)
        out, probs, retained = bridge.read(hidden, logits, values, positions, 19)
        self.assertTrue(torch.equal(probs, logits.softmax(-1)))
        torch.testing.assert_close(out, torch.einsum('hs,hsd->hd', probs, values))
        # Uniform head mixture: p7=.25, p8=.30, p19=.25, p2 discarded
        # by each writer's top two; top two ties are resolved by source address.
        torch.testing.assert_close(retained, torch.full((3,), .55, dtype=torch.float64))
        out.square().sum().backward()
        self.assertGreater(sum(b.projection.weight.grad.norm().item() for b in bridge.biases), 0.)
        with self.assertRaises(ValueError):
            bridge.read(hidden, logits, values, positions, 20)
        bridge.clear()
        with self.assertRaises(ValueError):
            bridge.read(hidden, logits, values, positions, 19)

    def test_positive_bias_stays_on_existing_support(self):
        bridge = ReferenceBridge(1, 1, 1, torch.tensor([1.], dtype=torch.float64))
        bridge.write(torch.tensor([[0, 2]]), torch.tensor([[.4, .6]], dtype=torch.float64), 4)
        with torch.no_grad():
            bridge.biases[0].projection.weight.fill_(2.)
        out, prob, _ = bridge.read(torch.ones(1, dtype=torch.float64),
            torch.tensor([[0., -torch.inf, 0.]], dtype=torch.float64),
            torch.eye(3, dtype=torch.float64)[None], torch.tensor([[0, 1, 4]]), 4)
        self.assertEqual(prob[0, 1].item(), 0.)
        self.assertAlmostEqual(prob.sum().item(), 1.)
        torch.testing.assert_close(out, prob)


if __name__ == '__main__':
    unittest.main()
