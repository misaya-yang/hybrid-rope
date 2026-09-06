import tempfile
import unittest
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.core_text_phases.audit_training_artifacts import build_report


class TrainingArtifactAuditTest(unittest.TestCase):
    def test_token_math_and_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            torch.save(
                torch.arange(10 * 8, dtype=torch.int64).view(10, 8),
                work / "train_fineweb-edu_80_8.pt",
            )
            torch.save(torch.arange(16, dtype=torch.int64), work / "val_fineweb-edu_16.pt")

            run = work / "350m_mla_tau1.414_seed42"
            run.mkdir()
            (run / "results.json").write_text(
                '{"tau": 1.414, "seed": 42, "attn_type": "mla", '
                '"d_rope": 32, "kv_lora_rank": 256, "ppl": {"16": 12.3}}\n',
                encoding="utf-8",
            )
            (run / "model.pt").write_bytes(b"placeholder")
            (run / "inv_freq.npy").write_bytes(b"placeholder")

            report = build_report(
                Namespace(
                    work_dir=str(work),
                    dataset="fineweb-edu",
                    seq_len=8,
                    batch_size=4,
                    train_tokens=80,
                    val_tokens=16,
                )
            )

            self.assertEqual(report["token_math"]["optimizer_steps_from_cache"], 2)
            self.assertEqual(report["token_math"]["used_tokens_by_train_loop"], 64)
            self.assertEqual(report["token_math"]["dropped_tokens_from_batch_floor"], 16)
            self.assertEqual(report["run_artifacts"][0]["run_id"], "350m_mla_tau1.414_seed42")
            self.assertEqual(report["run_artifacts"][0]["ppl_lengths"], ["16"])


if __name__ == "__main__":
    unittest.main()
