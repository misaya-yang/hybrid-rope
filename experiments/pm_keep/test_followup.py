"""Scientific invariants for the prepared value-objective ablation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .adapter import PrefillSession
from .run_followup import ValueAwareSession, value_weights


class ValueObjectiveTests(unittest.TestCase):
    def test_raw_norm_matches_explicit_gqa_value_magnitudes(self):
        v = torch.tensor([[[[3., 4.], [0., 2.]], [[5., 12.], [8., 15.]]]])
        torch.testing.assert_close(value_weights(v, "raw_norm"), torch.tensor([[5., 2.], [13., 17.]]))

    def test_centered_weights_are_value_translation_invariant(self):
        generator = torch.Generator().manual_seed(9)
        v = torch.randn(1, 2, 7, 4, generator=generator)
        shift = torch.randn(1, 2, 1, 4, generator=generator)
        torch.testing.assert_close(value_weights(v, "centered_norm"), value_weights(v + shift, "centered_norm"))

    def test_uniform_value_norm_preserves_score_ranking(self):
        v = torch.tensor([[[[3., 4.], [-3., 4.], [0., 5.]]]])
        score = torch.tensor([[.1, .7, .2]])
        self.assertTrue(torch.equal(score.argsort(), (score * value_weights(v, "raw_norm")).argsort()))

    def test_all_three_arms_share_weighting_and_cached_call_not_squared(self):
        session = object.__new__(ValueAwareSession)
        session.scores, session.score_metrics = {}, {}
        session.timings = {"score_seconds": {}}
        session.device = torch.device("cpu")
        session.config = SimpleNamespace(value_objective="raw_norm")
        session.cache = SimpleNamespace(layers=[SimpleNamespace(values=torch.tensor([[[[3., 4.], [0., 2.]]]]))])
        def parent(s, arm):
            if arm not in s.scores:
                s.scores[arm] = [torch.tensor([[.2, .8]])]
                s.score_metrics[arm] = [{}]
                s.timings["score_seconds"][arm] = 0.
            return s.scores[arm]
        with patch.object(PrefillSession, "score", parent):
            for arm in ("P", "C", "U"):
                torch.testing.assert_close(session.score(arm)[0], torch.tensor([[1., 1.6]]))
                torch.testing.assert_close(session.score(arm)[0], torch.tensor([[1., 1.6]]))

    def test_pruning_error_identity_and_mass_counterexample(self):
        a = torch.tensor([.6, .3, .1], dtype=torch.float64)
        v = torch.tensor([[0.], [0.], [100.]], dtype=torch.float64)
        full = (a[:, None] * v).sum(0)
        errors = []
        for keep in (torch.tensor([0, 1]), torch.tensor([0, 2])):
            kept = (a[keep, None] * v[keep]).sum(0) / a[keep].sum()
            mask = torch.ones(3, dtype=torch.bool)
            mask[keep] = False
            rhs = (a[mask, None] * (full - v[mask])).sum(0) / a[keep].sum()
            torch.testing.assert_close(kept - full, rhs)
            errors.append((kept - full).norm())
        self.assertGreater(errors[0], errors[1])

    def test_value_only_never_calls_query_position_scorer(self):
        session = object.__new__(ValueAwareSession)
        session.scores, session.score_metrics = {}, {}
        session.timings = {"score_seconds": {}}
        session.device = torch.device("cpu")
        session.config = SimpleNamespace(value_objective="value_only")
        session.cache = SimpleNamespace(layers=[SimpleNamespace(values=torch.tensor([[[[3., 4.], [0., 2.]]]]))])
        with patch.object(PrefillSession, "score", side_effect=AssertionError("query scorer invoked")):
            torch.testing.assert_close(session.score("P")[0], torch.tensor([[5., 2.]]))


if __name__ == "__main__":
    unittest.main()
