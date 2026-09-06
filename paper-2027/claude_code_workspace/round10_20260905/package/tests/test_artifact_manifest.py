import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.core_text_phases import make_artifact_manifest
from scripts.lib.rope.schedules import evq_cosh_inv_freq


def _args(entries, **overrides):
    defaults = dict(
        entry=entries,
        output="-",
        note=["unit-test"],
        include_paths=False,
        recursive=False,
        glob=None,
        inspect_tensors=True,
        rope_audit=True,
        base=500000.0,
        tau=1.5,
        rope_dim=64,
        d_rope=None,
        d_head=64,
        d_eff=64,
        tolerance=1e-6,
        estimate_tau_max=4.0,
        estimate_tau_steps=80,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_hashes_and_audits_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "evq_seed42_run"
            run_dir.mkdir()
            inv = evq_cosh_inv_freq(head_dim=64, tau=1.5, base=500000.0)
            np.save(run_dir / "inv_freq.npy", inv.numpy())
            torch.save(torch.arange(8), run_dir / "train_tokens.pt")
            (run_dir / "results.json").write_text('{"ppl": 12.3}\n', encoding="utf-8")

            manifest = make_artifact_manifest.build_manifest(
                _args([f"evq_1b_seed42={run_dir}"])
            )
            dumped = json.dumps(manifest, ensure_ascii=False)

            self.assertNotIn(str(run_dir), dumped)
            self.assertEqual(manifest["path_policy"], "sanitized_path_hints_only")

            entry = manifest["entries"][0]
            self.assertEqual(entry["label"], "evq_1b_seed42")
            self.assertEqual(entry["path_hint"], run_dir.name)

            files = {item["path_hint"]: item for item in entry["files"]}
            self.assertIn("inv_freq.npy", files)
            self.assertIn("train_tokens.pt", files)
            self.assertIn("sha256", files["inv_freq.npy"])
            self.assertEqual(files["inv_freq.npy"]["tensor"]["shape"], [32])
            self.assertEqual(files["train_tokens.pt"]["tensor"]["shape"], [8])

            audit = entry["rope_audit"]
            self.assertEqual(audit["classification"], "matches_requested_evq")
            self.assertEqual(audit["source"]["kind"], "npy")
            self.assertIn("sha256", audit["actual_inv_freq"])

    def test_missing_rope_audit_error_is_sanitized_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "missing_inv_freq"
            run_dir.mkdir()

            manifest = make_artifact_manifest.build_manifest(
                _args([f"empty_run={run_dir}"])
            )
            dumped = json.dumps(manifest, ensure_ascii=False)

            self.assertNotIn(str(run_dir), dumped)
            audit = manifest["entries"][0]["rope_audit"]
            self.assertFalse(audit["available"])
            self.assertEqual(audit["error_type"], "FileNotFoundError")
            self.assertEqual(
                audit["error"],
                "RoPE audit did not find an inv_freq/model artifact in this entry.",
            )


if __name__ == "__main__":
    unittest.main()
