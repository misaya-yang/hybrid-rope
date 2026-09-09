"""Checks for the three follow-up interventions, not their scientific quality."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from .operator import OperatorFactors, Shape, native_response
from .progressive import fit_progressive
from .run import save_factors
from .study import FitConfig, capture, digest, load_record, squared_metrics, write_json
from .test_operator import tiny_model


class FollowupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_balancing_undoes_scaling_when_content_rank_is_full(self):
        shape = Shape(4, 2, 8, 28, 4)
        k, v = torch.randn(80, 16) * 15, torch.randn(80, 16)
        factors = OperatorFactors.from_freqfold(k, v, shape, balance_kv=True)
        self.assertGreater(factors.initialization_balance['key_divisor'], 5)
        q, positions = torch.randn(80, 4, 8), torch.zeros(80, dtype=torch.long)
        teacher = native_response(q, k, v, positions, positions, shape)
        student = factors.response(q, k, v, positions, positions)
        torch.testing.assert_close(student[0], teacher[0], atol=1e-4, rtol=2e-5)
        torch.testing.assert_close(student[2], teacher[2], atol=2e-5, rtol=2e-5)

    def test_attention_kl_ignores_row_shift_and_has_finite_gradient(self):
        valid = torch.ones(3, 5, dtype=torch.bool).tril(diagonal=2)
        scores = torch.randn(2, 3, 5)
        output, value = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
        shifted = (scores + torch.randn(2, 3, 1) * 5).requires_grad_()
        metrics = squared_metrics((scores, output, value, valid), (shifted, output, value, valid))
        self.assertLess(abs(float(metrics['attention_kl'].detach())), 1e-6)
        self.assertGreater(float(metrics['score_mse'].detach()), 0.1)
        metrics['attention_kl'].backward()
        self.assertTrue(torch.isfinite(shifted.grad).all())

    def test_progressive_uses_changed_student_states_and_native_targets(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            model = tiny_model()
            rows = [dict(id=f'calibration_{i:04d}', source_id=f'book_{i}', split='calibration',
                         input_ids=torch.randint(3, 64, (17,)).tolist()) for i in range(2)]
            manifest = capture(model, rows, root / 'capture', queries=4)
            shape = Shape(4, 2, 8, 12, 4, 10000)
            initial = root / 'initial'
            initial.mkdir()
            for layer in range(2):
                records = [load_record(root / 'capture' / f'layer_{layer:03d}' / f'record_{i:05d}.pt') for i in range(2)]
                factors = OperatorFactors.from_freqfold(torch.cat([r['k'] for r in records]),
                                                       torch.cat([r['v'] for r in records]), shape)
                save_factors(initial / f'layer_{layer:03d}.pt', factors)
            write_json(initial / 'manifest.json', dict(status='complete', checkpoint_role='unoptimized_initialization',
                       specification=dict(capture_sha256=digest(root / 'capture/manifest.json'),
                                          shape=factors.metadata()['shape'], fold=None)))
            result = fit_progressive(model, rows, root / 'capture', initial, root / 'fit',
                                     FitConfig(steps=3, score_weight=0, output_weight=1), None, {'device':'cpu'})
            self.assertEqual(result['completed_layers'], [0, 1])
            old = load_record(root / 'capture/layer_001/record_00000.pt')
            new = load_record(root / 'fit/student_inputs/layer_001/record_00000.pt')
            self.assertGreater(float((old['k']-new['k']).abs().max()), 1e-5)
            receipt = json.loads((root / 'fit/layer_001/result.json').read_text())
            self.assertEqual(receipt['optimizer_updates'], 3)
            self.assertNotEqual(receipt['data_position_schedule_sha256'], receipt['teacher_data_position_schedule_sha256'])


if __name__ == '__main__':
    unittest.main()
