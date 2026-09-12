#!/usr/bin/env python3
"""CPU checks for exact next-token and tail128 accounting."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from .evaluate import lm_loss_rows


class TinyBackbone(torch.nn.Module):
    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 7).float())


class TinyModel(torch.nn.Module):
    def __init__(self, vocab=7):
        super().__init__()
        self.lm_head = torch.nn.Linear(vocab, vocab, bias=False)
        self.lm_head.weight.data.copy_(torch.eye(vocab))
        self.model = TinyBackbone()


class EvaluateTests(unittest.TestCase):
    def test_whole_targets_are_shifted_once_and_tail_is_last_128(self):
        model = TinyModel()
        ids = (torch.arange(131) % 7).unsqueeze(0)
        result = lm_loss_rows(model, ids, tail=128, chunk_size=17)
        logits = torch.nn.functional.one_hot(ids[:, :-1], 7).float()
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")[0]
        self.assertEqual(result["whole_target_count"], 130)
        self.assertEqual(result["tail128_target_count"], 128)
        self.assertAlmostEqual(result["whole_loss_sum"], float(losses.sum()), places=4)
        self.assertAlmostEqual(result["tail128_loss_sum"], float(losses[-128:].sum()), places=4)


if __name__ == "__main__":
    unittest.main()
