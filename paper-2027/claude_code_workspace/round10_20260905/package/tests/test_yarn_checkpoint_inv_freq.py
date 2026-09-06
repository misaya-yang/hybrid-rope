import unittest
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.core_text_phases import eval_extended_3seeds
from scripts.core_text_phases import yarn_finetune_eval


class _Rope(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq.clone())


class _Attn(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()
        self.rope = _Rope(inv_freq)


class _Block(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()
        self.attn = _Attn(inv_freq)


class _Model(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(inv_freq)])

    def get_rope(self):
        return self.blocks[0].attn.rope


class CheckpointInvFreqTests(unittest.TestCase):
    def test_requires_checkpoint_inv_freq_key(self):
        state = {"blocks.0.attn.rope.inv_freq": torch.ones(4)}
        eval_extended_3seeds.require_checkpoint_inv_freq(state, "dummy.pt")
        yarn_finetune_eval.require_checkpoint_inv_freq(state, "dummy.pt")

        with self.assertRaises(KeyError):
            eval_extended_3seeds.require_checkpoint_inv_freq({"weight": torch.ones(1)}, "dummy.pt")
        with self.assertRaises(KeyError):
            yarn_finetune_eval.require_checkpoint_inv_freq({"weight": torch.ones(1)}, "dummy.pt")

    def test_checkpoint_inv_freq_returns_loaded_clone(self):
        constructed = torch.tensor([1.0, 0.5, 0.25, 0.125])
        loaded = torch.tensor([0.9, 0.45, 0.225, 0.1125])
        model = _Model(constructed)
        model.blocks[0].attn.rope.inv_freq.copy_(loaded)

        with redirect_stdout(StringIO()):
            base_eval = eval_extended_3seeds.checkpoint_inv_freq(model, "eval")
            base_ft = yarn_finetune_eval.checkpoint_inv_freq(model, "ft")

        torch.testing.assert_close(base_eval, loaded)
        torch.testing.assert_close(base_ft, loaded)

        model.blocks[0].attn.rope.inv_freq.add_(1.0)
        torch.testing.assert_close(base_eval, loaded)
        torch.testing.assert_close(base_ft, loaded)

    def test_yarn_scaling_depends_on_loaded_not_constructed_freq(self):
        constructed = torch.tensor([1.0, 0.5, 0.25, 0.125])
        loaded = torch.tensor([0.9, 0.45, 0.225, 0.1125])
        model = _Model(constructed)
        model.blocks[0].attn.rope.inv_freq.copy_(loaded)

        with redirect_stdout(StringIO()):
            base = eval_extended_3seeds.checkpoint_inv_freq(model, "eval")
        yarn_from_loaded = eval_extended_3seeds.yarn_inv_freq(base, 2.0, train_seq=16)
        yarn_from_constructed = eval_extended_3seeds.yarn_inv_freq(
            constructed, 2.0, train_seq=16
        )

        self.assertFalse(torch.allclose(yarn_from_loaded, yarn_from_constructed))

    def test_resolves_current_and_legacy_mla_run_dirs(self):
        with TemporaryDirectory() as td:
            work_dir = Path(td)
            current = work_dir / "350m_mla_tau1.414_seed42" / "model.pt"
            current.parent.mkdir()
            current.write_text("placeholder", encoding="utf-8")

            resolved, candidates = eval_extended_3seeds.resolve_checkpoint_path(
                work_dir, 1.414, 42, "model.pt"
            )
            self.assertEqual(resolved, current)
            self.assertIn(current, candidates)

        with TemporaryDirectory() as td:
            work_dir = Path(td)
            legacy = work_dir / "350m_tau1.41_seed42" / "model.pt"
            legacy.parent.mkdir()
            legacy.write_text("placeholder", encoding="utf-8")

            resolved, candidates = yarn_finetune_eval.resolve_checkpoint_path(
                work_dir, 1.414, 42, "model.pt"
            )
            self.assertEqual(resolved, legacy)
            self.assertIn(legacy, candidates)


if __name__ == "__main__":
    unittest.main()
