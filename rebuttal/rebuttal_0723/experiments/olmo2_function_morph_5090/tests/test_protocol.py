from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..authorization import AuthorizationError, require_gpu_authorization
from ..protocol import METHOD_ID, MORPH_GRID, ROPE_BASE, ROTARY_PAIRS, protocol_manifest
from ..targets import (
    anchored_evq_phi,
    build_target_manifest,
    log_morph,
    matched_exponential_control,
)


class TargetTests(unittest.TestCase):
    def _collection(self, root: Path) -> Path:
        path = root / "r0.npz"
        native = 1.0 / (
            ROPE_BASE ** (np.arange(ROTARY_PAIRS, dtype=np.float64) / ROTARY_PAIRS)
        )
        distance = np.arange(4096, dtype=np.float64)
        profile = np.exp(-distance / 128.0)
        profile[0] = 0.0
        mass = np.broadcast_to(profile, (16, 16, 4096)).copy()
        metadata = {
            "backend": "hf_olmo2",
            "base": ROPE_BASE,
            "head_dim": 128,
            "heads": 16,
            "layers": 16,
            "length": 4096,
            "model_revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
            "tokens_sha256": "test",
            "training_or_parameter_updates": False,
        }
        np.savez(
            path,
            inv_freq=native,
            mass=mass,
            metadata=json.dumps(metadata),
        )
        return path

    def test_control_matches_rms_and_bends_opposite(self) -> None:
        uniform = np.linspace(0.0, 1.0, ROTARY_PAIRS)
        phase = uniform**1.7
        control, _ = matched_exponential_control(phase)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(phase - uniform)))),
            float(np.sqrt(np.mean(np.square(control - uniform)))),
            places=12,
        )
        self.assertLess(
            float(np.mean(phase - uniform) * np.mean(control - uniform)), 0.0
        )
        self.assertEqual(control[0], 0.0)
        self.assertEqual(control[-1], 1.0)

    def test_anchored_evq_and_morph_endpoints(self) -> None:
        phi = anchored_evq_phi()
        self.assertEqual(phi[0], 0.0)
        self.assertEqual(phi[-1], 1.0)
        self.assertTrue(np.all(np.diff(phi) > 0.0))
        native = 1.0 / (
            ROPE_BASE ** (np.arange(ROTARY_PAIRS, dtype=np.float64) / ROTARY_PAIRS)
        )
        target = np.exp(
            np.log(native[0]) + phi * (np.log(native[-1]) - np.log(native[0]))
        )
        self.assertTrue(np.array_equal(log_morph(native, target, 0.0), native))
        self.assertTrue(np.allclose(log_morph(native, target, 1.0), target))

    def test_manifest_freezes_all_candidate_morphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_target_manifest(self._collection(Path(directory)))
        self.assertEqual(manifest["status"], "FINITE_TARGETS_FROZEN")
        self.assertEqual(manifest["protocol"]["method_id"], METHOD_ID)
        self.assertEqual(
            sorted(manifest["candidates"]),
            sorted(protocol_manifest()["candidate_order"]),
        )
        for rows in manifest["morph_tables"].values():
            self.assertEqual([row["t"] for row in rows], list(MORPH_GRID))


class AuthorizationTests(unittest.TestCase):
    def test_gpu_gate_fails_closed(self) -> None:
        with self.assertRaises(AuthorizationError):
            require_gpu_authorization(cli_authorize=False, environment={})
        with self.assertRaises(AuthorizationError):
            require_gpu_authorization(cli_authorize=True, environment={})
        require_gpu_authorization(
            cli_authorize=True,
            environment={"OLMO_FUNCTION_MORPH_GPU_AUTHORIZED": "1"},
        )


if __name__ == "__main__":
    unittest.main()
