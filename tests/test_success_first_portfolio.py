"""Contract tests for the success-first zero-training tournament code.

CPU only: parameterisation identities, table construction, the frozen selection
rule, the confirmation verdict, the portfolio freezer on a synthetic R0
collection, and the evaluator's no-GPU contract mode.  No model, no CUDA.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lib.rope.schedules import geometric_inv_freq  # noqa: E402
from scripts.lib.rope.knot_allocation import (  # noqa: E402
    Z5KnotRotaryEmbedding,
    init_gap_logits_from_table,
)
from scripts.lib.rope.hat_projection import (  # noqa: E402
    hat_basis,
    phase_chord_direction,
    project_direction_onto_basis,
    table_from_coefficients,
)
from scripts.eval.zero_training_selection import (  # noqa: E402
    apply_selection_rule,
    confirmation_verdict,
    is_feasible,
)
from scripts.eval.eval_zero_training_tournament import _install  # noqa: E402
from scripts.eval.eval_zero_training_tournament import (  # noqa: E402
    checkpoint_identity,
    merge_standalone_native_prefix,
    validate_checkpoint_identity,
)
from scripts.eval.develop_zero_training_family import (  # noqa: E402
    install_z5_from_native,
)
from scripts.data.build_success_first_splits import (  # noqa: E402
    load_token_prefix_exclusions,
    token_prefix_sha256,
)


def native_table(pairs: int = 64) -> np.ndarray:
    return geometric_inv_freq(head_dim=2 * pairs, base=500_000.0).numpy().astype(np.float64)


class KnotAllocationTests(unittest.TestCase):
    def test_native_initialisation_is_bitwise(self):
        native = native_table()
        knot = Z5KnotRotaryEmbedding(native)
        realized = knot.realized_inv_freq().detach().cpu().numpy()
        self.assertTrue(np.allclose(realized, native.astype(np.float32), atol=0, rtol=0))

    def test_strictly_decreasing_and_endpoints(self):
        native = native_table()
        knot = Z5KnotRotaryEmbedding(native)
        knot.set_gap_logits_(np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.3], dtype=np.float32))
        realized = knot.realized_inv_freq().detach().cpu().numpy()
        self.assertTrue(np.all(realized[:-1] > realized[1:]))
        self.assertEqual(realized[0], native[0].astype(np.float32))

    def test_support_factor_stretches_slow_endpoint(self):
        native = native_table()
        knot = Z5KnotRotaryEmbedding(native, support_factor=4.0)
        realized = knot.realized_inv_freq().detach().cpu().numpy()
        expected_slow = native[-1].astype(np.float32) / np.float32(4.0)
        self.assertEqual(realized[0], native[0].astype(np.float32))
        self.assertAlmostEqual(float(realized[-1]), float(expected_slow), places=6)
        self.assertTrue(np.all(realized[:-1] > realized[1:]))

    def test_gradient_flows_to_gap_logits(self):
        import torch

        native = native_table()
        knot = Z5KnotRotaryEmbedding(native)
        knot.set_gap_logits_(np.array([0.3, -0.2, 0.1, 0.0, 0.2, -0.1], dtype=np.float32))
        loss = knot.realized_inv_freq().sum()
        loss.backward()
        self.assertIsNotNone(knot.gap_logits.grad)
        self.assertTrue(bool(np.isfinite(knot.gap_logits.grad.numpy()).all()))

    def test_cuda_coordinates_follow_trainable_parameter_device(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        native = torch.as_tensor(native_table(), dtype=torch.float32, device="cuda")
        knot = Z5KnotRotaryEmbedding(native)
        coordinates = knot.normalized_coordinates()
        realized = knot.realized_inv_freq()
        self.assertEqual(coordinates.device, knot.gap_logits.device)
        self.assertEqual(realized.device, knot.gap_logits.device)
        realized.sum().backward()
        self.assertIsNotNone(knot.gap_logits.grad)

    def test_init_from_table_recovers_native(self):
        native = native_table()
        logits = init_gap_logits_from_table(native, native)
        knot = Z5KnotRotaryEmbedding(native)
        knot.set_gap_logits_(logits.numpy())
        realized = knot.realized_inv_freq().detach().cpu().numpy()
        self.assertTrue(np.allclose(realized, native.astype(np.float32), atol=1e-4))


class HatBasisTests(unittest.TestCase):
    def test_basis_vanishes_at_endpoints(self):
        basis = hat_basis(64)
        self.assertEqual(basis.shape, (64, 5))
        self.assertTrue(np.allclose(basis[0].numpy(), 0.0))
        self.assertTrue(np.allclose(basis[-1].numpy(), 0.0))

    def test_projection_and_table_identity(self):
        native = native_table()
        target = np.exp(np.log(native) + 0.05 * np.sin(np.linspace(0, np.pi, 64)))
        target[0], target[-1] = native[0], native[-1]
        direction = phase_chord_direction(native, target)
        basis = hat_basis(64)
        coefficients = project_direction_onto_basis(direction, basis)
        table = table_from_coefficients(native, coefficients.numpy())
        self.assertEqual(table[0], native[0].astype(np.float32))
        self.assertEqual(table[-1], native[-1].astype(np.float32))
        self.assertTrue(np.all(table[:-1] > table[1:]))
        target_delta = np.log(target) - np.log(native)
        realised_delta = np.log(table.astype(np.float64)) - np.log(native)
        cosine = np.dot(target_delta, realised_delta) / (
            np.linalg.norm(target_delta) * np.linalg.norm(realised_delta)
        )
        self.assertGreater(cosine, 0.99)


class DevelopmentIsolationTests(unittest.TestCase):
    def test_each_z5_install_starts_from_the_same_native_table(self):
        import torch

        class DummyModel(torch.nn.Module):
            def __init__(self, native):
                super().__init__()
                self.model = torch.nn.Module()
                rotary = torch.nn.Module()
                rotary.register_buffer("inv_freq", torch.from_numpy(native.astype(np.float32)))
                rotary.attention_scaling = 1.0
                self.model.rotary_emb = rotary

        native = native_table()
        model = DummyModel(native)
        native_rotary = model.model.rotary_emb
        first, _ = install_z5_from_native(model, native_rotary, support_factor=1.0)
        first.set_gap_logits_(torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]))
        self.assertFalse(torch.equal(first.inv_freq.cpu(), native_rotary.inv_freq.cpu()))

        second, _ = install_z5_from_native(model, native_rotary, support_factor=1.25)
        self.assertTrue(torch.equal(second.original_inv_freq.cpu(), native_rotary.inv_freq.cpu()))


class FirewallTests(unittest.TestCase):
    def test_r0_token_prefix_tensor_is_an_exclusion_owner(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r0_tokens.pt"
            rows = torch.arange(3 * 8, dtype=torch.int64).reshape(3, 8)
            torch.save(rows, path)
            exclusions, receipts = load_token_prefix_exclusions([path], prefix_length=4)
            self.assertEqual(len(exclusions), 3)
            self.assertEqual(receipts[0]["rows"], 3)
            self.assertIn(token_prefix_sha256(rows[1, :4].numpy()), exclusions)


class SelectionRuleTests(unittest.TestCase):
    def _candidate(
        self, name, prefix, dense, tail, dof=1, long_range=False, chord=0.1,
        support=1.0, yarn_prefix=None, yarn_dense=None,
    ):
        candidate = {
            "name": name,
            "native_prefix_delta": prefix,
            "long_dense_delta": dense,
            "far_tail_nll": tail,
            "calibrated_dof": dof,
            "used_long_range_in_construction": long_range,
            "chord_displacement_rms": chord,
            "support_movement": np.log(support),
        }
        if yarn_prefix is not None:
            candidate["yarn_native_prefix_delta"] = yarn_prefix
        if yarn_dense is not None:
            candidate["yarn_long_dense_delta"] = yarn_dense
        return candidate

    def test_infeasible_candidates_are_rejected(self):
        rows = [
            self._candidate("a", 0.05, 0.0, 1.0),
            self._candidate("b", 0.0, 0.02, 0.9),
        ]
        outcome = apply_selection_rule(rows)
        self.assertIsNone(outcome["winner"])
        self.assertEqual(outcome["ledger"]["feasible"], 0)

    def test_lowest_far_tail_wins_among_feasible(self):
        rows = [
            self._candidate("a", 0.001, 0.001, 1.5),
            self._candidate("b", 0.0, 0.0, 1.2),
        ]
        outcome = apply_selection_rule(rows)
        self.assertEqual(outcome["winner"]["name"], "b")

    def test_tiebreak_prefers_fewer_dof(self):
        rows = [
            self._candidate("hi", 0.0, 0.0, 1.0, dof=5),
            self._candidate("lo", 0.0, 0.0, 1.0, dof=1),
        ]
        outcome = apply_selection_rule(rows)
        self.assertEqual(outcome["winner"]["name"], "lo")

    def test_anchored_mode_admits_candidate_absolute_mode_rejects(self):
        row = self._candidate(
            "cheaper_than_yarn", 0.05, 0.02, 1.0, yarn_prefix=0.09, yarn_dense=0.04
        )
        self.assertFalse(is_feasible(row))
        self.assertTrue(is_feasible(row, mode="ANCHORED"))
        outcome = apply_selection_rule([row], mode="ANCHORED")
        self.assertEqual(outcome["winner"]["name"], "cheaper_than_yarn")
        self.assertEqual(outcome["ledger"]["feasibility_mode"], "ANCHORED")

    def test_anchored_mode_still_rejects_costlier_than_yarn(self):
        row = self._candidate(
            "costlier_than_yarn", 0.12, 0.02, 1.0, yarn_prefix=0.09, yarn_dense=0.04
        )
        self.assertFalse(is_feasible(row, mode="ANCHORED"))

    def test_anchored_mode_requires_measured_yarn_reference(self):
        row = self._candidate("no_reference", 0.05, 0.02, 1.0)
        with self.assertRaises(ValueError):
            is_feasible(row, mode="ANCHORED")

    def test_anchored_mode_leaves_absolute_pass_unchanged(self):
        row = self._candidate("clean", 0.0, 0.0, 1.0, yarn_prefix=-1.0, yarn_dense=-1.0)
        self.assertTrue(is_feasible(row))
        self.assertTrue(is_feasible(row, mode="ANCHORED"))

    def test_confirmation_verdicts(self):
        improving = {"native_prefix": {"mean_delta": -0.01, "ci_low": -0.02, "ci_high": -0.001},
                     "far_tail": {"mean_delta": -0.05, "ci_low": -0.08, "ci_high": -0.02},
                     "long_dense": {"mean_delta": 0.001, "ci_low": -0.002, "ci_high": 0.004}}
        self.assertEqual(confirmation_verdict(improving), "JOINT_IMPROVEMENT")
        mechanism = {
            "native_prefix": {"mean_delta": 0.02, "ci_low": 0.01, "ci_high": 0.03},
            "far_tail": {"mean_delta": -0.05, "ci_low": -0.08, "ci_high": -0.02},
            "long_dense": {"mean_delta": 0.001, "ci_low": 0.0, "ci_high": 0.002},
        }
        self.assertEqual(confirmation_verdict(mechanism), "MECHANISM_ONLY")
        fail = {
            "native_prefix": {"mean_delta": 0.0, "ci_low": -0.01, "ci_high": 0.01},
            "far_tail": {"mean_delta": 0.05, "ci_low": 0.02, "ci_high": 0.08},
            "long_dense": {"mean_delta": 0.0, "ci_low": -0.01, "ci_high": 0.01},
        }
        self.assertEqual(confirmation_verdict(fail), "FAIL")

    def test_yarn_anchored_verdict_needs_anchored_intervals(self):
        anchored = {
            "native_prefix": {"mean_delta": 0.02, "ci_low": 0.01, "ci_high": 0.03},
            "far_tail": {"mean_delta": -0.05, "ci_low": -0.08, "ci_high": -0.02},
            "long_dense": {"mean_delta": 0.001, "ci_low": 0.0, "ci_high": 0.002},
            "native_prefix_vs_yarn": {"mean_delta": -0.04, "ci_low": -0.07, "ci_high": -0.01},
            "long_dense_vs_yarn": {"mean_delta": -0.02, "ci_low": -0.04, "ci_high": -0.005},
        }
        self.assertEqual(confirmation_verdict(anchored), "YARN_ANCHORED_PARETO")
        without_reference = {k: v for k, v in anchored.items() if not k.endswith("_vs_yarn")}
        self.assertEqual(confirmation_verdict(without_reference), "MECHANISM_ONLY")
        worse_than_yarn = dict(anchored)
        worse_than_yarn["native_prefix_vs_yarn"] = {
            "mean_delta": 0.03, "ci_low": 0.01, "ci_high": 0.05,
        }
        self.assertEqual(confirmation_verdict(worse_than_yarn), "MECHANISM_ONLY")

    def test_strict_tier_wins_over_anchored_tier(self):
        strict_and_anchored = {
            "native_prefix": {"mean_delta": -0.01, "ci_low": -0.02, "ci_high": -0.001},
            "far_tail": {"mean_delta": -0.05, "ci_low": -0.08, "ci_high": -0.02},
            "long_dense": {"mean_delta": 0.001, "ci_low": -0.002, "ci_high": 0.004},
            "native_prefix_vs_yarn": {"mean_delta": -0.04, "ci_low": -0.07, "ci_high": -0.01},
            "long_dense_vs_yarn": {"mean_delta": -0.02, "ci_low": -0.04, "ci_high": -0.005},
        }
        self.assertEqual(confirmation_verdict(strict_and_anchored), "JOINT_IMPROVEMENT")


class FreezerTests(unittest.TestCase):
    def _synthetic_r0(self, path: Path) -> None:
        native = native_table()
        rng = np.random.default_rng(0)
        mass = rng.random((16, 16, 4096)) + 0.01
        metadata = json.dumps({"model_revision": "test", "tokens_sha256": "0" * 64})
        np.savez(path, inv_freq=native, mass=mass, metadata=np.array(metadata))

    def test_freezer_materialises_portfolio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            r0 = tmp_path / "r0.npz"
            self._synthetic_r0(r0)
            native = native_table()
            learned = native * np.exp(0.01 * np.linspace(0, 1, 64))
            learned[0], learned[-1] = native[0], native[-1]
            learned_path = tmp_path / "learned.npy"
            np.save(learned_path, learned.astype("<f4"))
            budgeted_path = tmp_path / "budgeted.npy"
            np.save(budgeted_path, learned.astype("<f4"))
            out = tmp_path / "portfolio"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/analysis/freeze_success_first_portfolio.py"),
                 "--r0-collection", str(r0), "--learned-table", str(learned_path),
                 "--budgeted-table", str(budgeted_path), "--output", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "portfolio_manifest.json").read_text())
            f1_tables = manifest["families"]["F1_PC_MORPH"]["tables"]
            self.assertEqual(len(f1_tables), 7)
            for entry in f1_tables:
                table = np.load(entry["path"], allow_pickle=False)
                self.assertTrue(np.all(table[:-1] > table[1:]), entry["name"])
            self.assertTrue(any(e["is_bitwise_native"] for e in f1_tables))
            target = manifest["phase_chord_target"]
            self.assertTrue(target["fast_endpoint_fixed"])
            self.assertTrue(target["slow_endpoint_fixed"])
            self.assertEqual(len(manifest["families"]["F2_PC_RETENTION_PROJECT"]["coefficients"]), 5)


class EvaluatorContractTests(unittest.TestCase):
    def test_standalone_1x_loss_owns_native_prefix(self):
        summary = {
            "native_prefix": {"numerator": 99.0, "denominator": 3, "nll": 33.0},
            "long_dense": {"numerator": 10.0, "denominator": 7, "nll": 10.0 / 7.0},
            "far_tail": {"numerator": 5.0, "denominator": 2, "nll": 2.5},
        }
        merged = merge_standalone_native_prefix(summary, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(merged["native_prefix"]["nll"], 2.0)
        self.assertEqual(merged["long_dense"], summary["long_dense"])
        self.assertEqual(merged["far_tail"], summary["far_tail"])

    def test_checkpoint_identity_detects_weight_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text("{}\n")
            (checkpoint / "model.safetensors").write_bytes(b"weights-v1")
            expected = checkpoint_identity(checkpoint)
            validate_checkpoint_identity(checkpoint, expected)
            (checkpoint / "model.safetensors").write_bytes(b"weights-v2")
            with self.assertRaises(RuntimeError):
                validate_checkpoint_identity(checkpoint, expected)

    def test_table_install_switches_frequency_and_attention_scaling(self):
        import torch

        class RotaryModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.ones(4))
                self.attention_scaling = 1.0

        model = RotaryModel()
        table = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
        _install(model, table, 1.138629436111989)
        self.assertTrue(torch.equal(model.inv_freq, torch.from_numpy(table)))
        self.assertEqual(model.attention_scaling, 1.138629436111989)

    def test_contract_mode_validates_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            native = native_table().astype("<f4")
            table_path = tmp_path / "native.npy"
            np.save(table_path, native)
            import hashlib

            table_hash = hashlib.sha256(np.ascontiguousarray(native).tobytes()).hexdigest()
            manifest = {
                "candidates": [{
                    "name": "Native", "family": "F1_PC_MORPH", "path": str(table_path),
                    "table_sha256_float32": table_hash, "pair_count": 64,
                    "is_bitwise_native": True, "calibrated_dof": 0,
                    "used_long_range_in_construction": False,
                    "chord_displacement_rms": 0.0, "support_factor": 1.0,
                }],
            }
            manifest_path = tmp_path / "candidates.json"
            manifest_path.write_text(json.dumps(manifest))
            rows_path = tmp_path / "rows.jsonl"
            rows = []
            for multiplier in (1, 4):
                rows.append(json.dumps({
                    "multiplier": multiplier, "source_row": 0,
                    "input_ids": [1] * (4096 * multiplier),
                    "source_text_sha256": "a" * 64,
                }))
            rows_path.write_text("\n".join(rows) + "\n")
            out = tmp_path / "contract"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/eval/eval_zero_training_tournament.py"),
                 "--contract", "--candidates", str(manifest_path),
                 "--rows", str(rows_path), "--output", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((out / "contract_receipt.json").read_text())
            self.assertEqual(receipt["status"], "ZERO_TRAINING_TOURNAMENT_CONTRACT_OK")
            self.assertFalse(receipt["cuda_initialised"])


if __name__ == "__main__":
    unittest.main()
