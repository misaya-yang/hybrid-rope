#!/usr/bin/env python3
"""Offline tests for real_dape_compare (no FineWeb, GPU optional)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO))


class IdentityTests(unittest.TestCase):
    def test_method_identity_map(self):
        from run_dape_compare import METHOD_IDENTITY, ALL_METHODS

        self.assertIn("free_inv_freq", ALL_METHODS)
        self.assertIn("dape_kerple_mlp", ALL_METHODS)
        self.assertNotIn("DAPE", ALL_METHODS)
        free = METHOD_IDENTITY["free_inv_freq"]
        self.assertEqual(free["identity"], "learnable_inv_freq_32")
        self.assertIn("zheng2024_dape", free["not"])
        dape = METHOD_IDENTITY["dape_kerple_mlp"]
        self.assertEqual(dape["identity"], "zheng_inspired_kerple_plus_attn_mlp")
        self.assertIn("free_inv_freq_32", dape["not"])

    def test_evq_schedule_finite(self):
        import torch
        from run_dape_compare import evq_cosh_inv_freq, midpoint_geometric_inv_freq

        a = evq_cosh_inv_freq(64, tau=5.0, base=500_000.0)
        b = midpoint_geometric_inv_freq(64, 500_000.0)
        self.assertEqual(tuple(a.shape), (32,))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.isfinite(b).all())
        # tau=0 should match midpoint geometric
        z = evq_cosh_inv_freq(64, tau=0.0, base=500_000.0)
        self.assertTrue(torch.allclose(z, b, rtol=1e-5, atol=1e-6))


class CliTests(unittest.TestCase):
    def test_dry_run(self):
        cmd = [
            sys.executable,
            str(PKG / "run_dape_compare.py"),
            "--dry-run",
            "--protocol",
            "p1",
            "--seeds",
            "42",
            "--methods",
            "geo,evq,free_inv_freq,kerple,dape_kerple_mlp",
        ]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("p1_pe_dominant_l128_geo_seed42", r.stdout)
        self.assertIn("dape_kerple_mlp", r.stdout)
        self.assertIn("free_inv_freq", r.stdout)

    def test_reject_name_dape(self):
        cmd = [
            sys.executable,
            str(PKG / "run_dape_compare.py"),
            "--smoke",
            "--methods",
            "dape",
        ]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    def test_smoke_one_method(self):
        work = PKG / "work" / "test_smoke_geo"
        if work.exists():
            import shutil

            shutil.rmtree(work)
        cmd = [
            sys.executable,
            str(PKG / "run_dape_compare.py"),
            "--smoke",
            "--seeds",
            "42",
            "--methods",
            "geo",
            "--work",
            str(work),
        ]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        summary = json.loads((work / "aggregate" / "summary.json").read_text())
        self.assertEqual(summary["n_results"], 1)
        result = json.loads(
            (work / "runs" / "smoke_geo_seed42" / "result.json").read_text()
        )
        self.assertEqual(result["method_id"], "geo")
        self.assertIn("ppl", result)
        self.assertTrue(result["ppl"])


class FindingsTests(unittest.TestCase):
    def test_findings_states_no_win_on_real_dape(self):
        text = (PKG / "FINDINGS.md").read_text()
        self.assertIn("EVQ 没有赢", text)
        self.assertIn("free", text.lower())
        self.assertIn("55.9", text)
        self.assertIn("56.8", text)


if __name__ == "__main__":
    unittest.main()
