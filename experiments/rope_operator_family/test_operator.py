"""CPU correctness tests. Tiny/random models are fixtures, not research results."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import CompactAttention, cache_bytes, chunked_attention
from .operator import OperatorFactors, Shape, freqfold_rotation, native_response, rotate, split_to_pairs
from .prepare import prepare_text
from .run import parser
from .study import FitConfig, diagnose, fit_layer, generator_diagnostics, responses, squared_metrics


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_exact_full_native_gqa_representation(self):
        factors = OperatorFactors.identity(4, 2, 8, 10000)
        q, k, v = torch.randn(11, 4, 8), torch.randn(11, 16), torch.randn(11, 16)
        positions = torch.arange(11) + 37
        teacher = native_response(q, k, v, positions, positions, factors.shape)
        student = factors.response(q, k, v, positions, positions)
        torch.testing.assert_close(student[0], teacher[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(student[1], teacher[1], atol=2e-6, rtol=2e-6)

    def test_freqfold_orthogonality_and_same_frequency_identity(self):
        shape = Shape(4, 2, 8, 12, 8, 10000)
        keys = torch.randn(40, 16)
        j = freqfold_rotation(keys, shape, 1)
        torch.testing.assert_close(j @ j.T, torch.eye(16), atol=1e-6, rtol=1e-6)
        frequencies = 10000 ** (-torch.arange(0, 8, 2).float() / 8)
        positions = torch.arange(40)
        # Native rotation transformed into the same-frequency PCA basis.
        native_interleaved = rotate(split_to_pairs(keys.reshape(40, 2, 8)), positions, frequencies)
        native_split = native_interleaved.reshape(40, 2, 4, 2).transpose(-1, -2).flatten(-2).flatten(-2)
        expected = native_split @ j.T
        actual = rotate((keys @ j.T).reshape(40, 2, 8), positions, frequencies).flatten(-2)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_static_phase_intervention_keeps_causal_order(self):
        factors = OperatorFactors.identity(4, 2, 8)
        record = dict(q=torch.randn(3, 4, 8), k=torch.randn(7, 16), v=torch.randn(7, 16),
                      query_positions=torch.tensor([1, 3, 6]), key_positions=torch.arange(7))
        teacher, student = responses(factors, record, 0)
        self.assertEqual(teacher[3].sum(-1).tolist(), [2, 4, 7])
        torch.testing.assert_close(student[1], teacher[1], atol=1e-6, rtol=1e-6)

    def test_changing_only_frequencies_preserves_static_content(self):
        k, v = torch.randn(20, 16), torch.randn(20, 16)
        factors = OperatorFactors.from_freqfold(k, v, Shape(4, 2, 8, 12, 4))
        record = dict(q=torch.randn(6, 4, 8), k=k, v=v,
                      query_positions=torch.arange(14, 20), key_positions=torch.arange(20))
        before = responses(factors, record, 0)[1]
        with torch.no_grad():
            factors.phase.add_(torch.randn_like(factors.phase) * 100)
        after = responses(factors, record, 0)[1]
        torch.testing.assert_close(after[0], before[0], atol=0, rtol=0)
        torch.testing.assert_close(after[1], before[1], atol=0, rtol=0)

    def test_full_representation_has_zero_generator_residual(self):
        metrics = generator_diagnostics(OperatorFactors.identity(4, 2, 8, 10000))
        self.assertLess(metrics["generator_intertwining_residual_frobenius"], 1e-7)
        self.assertLess(metrics["subspace_leakage_frobenius"], 1e-6)

    def test_position_response_metric_removes_static_score_offset(self):
        k, v = torch.randn(24, 16), torch.randn(24, 16)
        factors = OperatorFactors.from_freqfold(k, v, Shape(4, 2, 8, 12, 4))
        record = dict(q=torch.randn(8, 4, 8), k=k, v=v, id="fixture", source_id="fixture",
                      query_positions=torch.arange(16, 24), key_positions=torch.arange(24))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.pt"
            torch.save(record, path)
            before = diagnose(factors, [path], 4)["means"]
            with torch.no_grad():
                factors.P.add_(torch.randn_like(factors.P) * 0.5)
            after = diagnose(factors, [path], 4)["means"]
        self.assertAlmostEqual(after["position_response_mse"], before["position_response_mse"], delta=1e-5)
        self.assertGreater(abs(after["static_score_mse"] - before["static_score_mse"]), 0.01)

    def test_chunked_matches_sdpa_with_cache_offset_and_padding(self):
        q, k, v = torch.randn(2, 4, 5, 12), torch.randn(2, 1, 13, 12), torch.randn(2, 1, 13, 7)
        mask = torch.arange(13)[None, :] <= (8 + torch.arange(5))[:, None]
        mask = mask[None, None].expand(2, 1, -1, -1).clone()
        mask[0, :, :, :2] = False
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.2, enable_gqa=True)
        actual = chunked_attention(q, k, v, mask, 0.2, query_chunk=2, key_chunk=4)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_all_masked_rows_are_finite(self):
        q, k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 5)
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        actual = chunked_attention(q, k, v, mask, 0.5, key_chunk=1)
        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_fitting_reduces_real_score_loss(self):
        k, v, q = torch.randn(32, 16), torch.randn(32, 16), torch.randn(12, 4, 8)
        record = dict(q=q, k=k, v=v, query_positions=torch.arange(20, 32), key_positions=torch.arange(32), id="fixture", source_id="fixture", split="calibration")
        factors = OperatorFactors.from_freqfold(k, v, Shape(4, 2, 8, 12, 4, 10000))
        before = float(squared_metrics(*responses(factors, record))["relative_score_mse"].detach())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.pt"
            torch.save(record, path)
            with contextlib.redirect_stdout(io.StringIO()):
                fit_layer(factors, [path], FitConfig(steps=60, learning_rate=0.01, max_position_scale=1,
                                                   learn_frequency=False, value_weight=0.2), Path(directory) / "fit")
        after = float(squared_metrics(*responses(factors, record))["relative_score_mse"].detach())
        self.assertLess(after, before * 0.7)


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
                        max_position_embeddings=256, rope_theta=10000,
                        eos_token_id=2, bos_token_id=1, pad_token_id=0)
    config._attn_implementation = "sdpa"
    return Qwen2ForCausalLM(config).eval()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_real_hf_forward_cached_decode_and_storage(self):
        model = tiny_model()
        ids = torch.randint(3, 64, (1, 17))
        with torch.no_grad():
            expected = model(ids).logits
            for index, layer in enumerate(model.model.layers):
                layer.self_attn = CompactAttention(layer.self_attn, OperatorFactors.identity(4, 2, 8, 10000), index)
            actual = model(ids).logits
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            state = model(ids[:, :9], use_cache=True)
            self.assertEqual(cache_bytes(state.past_key_values), 2 * 9 * 32 * 4)
            state = model(ids[:, 9:14], past_key_values=state.past_key_values, use_cache=True)
            torch.testing.assert_close(state.logits, expected[:, 9:14], atol=1e-6, rtol=1e-5)
            state = model(ids[:, 14:], past_key_values=state.past_key_values, use_cache=True)
            torch.testing.assert_close(state.logits, expected[:, 14:], atol=1e-6, rtol=1e-5)
            self.assertEqual(cache_bytes(state.past_key_values), 2 * 17 * 32 * 4)

    def test_affine_fusion_matches_factor_response(self):
        model = tiny_model()
        original = model.model.layers[0].self_attn
        hidden = torch.randn(1, 19, 32)
        q = original.q_proj(hidden)[0].reshape(19, 4, 8)
        k, v = original.k_proj(hidden)[0], original.v_proj(hidden)[0]
        factors = OperatorFactors.from_freqfold(k.detach(), v.detach(), Shape(4, 2, 8, 12, 4, 10000))
        positions = torch.arange(19)
        reference = factors.response(q, k, v, positions, positions)[1].transpose(0, 1).flatten(-2)
        reference = original.o_proj(reference)
        for backend in ("sdpa", "chunked"):
            compact = CompactAttention(original, factors, 0, backend=backend)
            actual = compact(hidden, position_ids=positions[None])[0][0]
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    def test_bfloat16_model_path(self):
        model = tiny_model().to(torch.bfloat16)
        ids = torch.randint(3, 64, (1, 11))
        with torch.no_grad():
            expected = model(ids).logits
            for index, layer in enumerate(model.model.layers):
                layer.self_attn = CompactAttention(layer.self_attn, OperatorFactors.identity(4, 2, 8, 10000), index)
            actual = model(ids).logits
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.04)

    def test_padded_hf_model(self):
        model = tiny_model()
        ids = torch.randint(3, 64, (2, 13))
        mask = torch.ones_like(ids)
        mask[0, :3] = 0
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.no_grad():
            expected = model(ids, attention_mask=mask, position_ids=positions).logits
            for index, layer in enumerate(model.model.layers):
                layer.self_attn = CompactAttention(layer.self_attn, OperatorFactors.identity(4, 2, 8, 10000), index)
            actual = model(ids, attention_mask=mask, position_ids=positions).logits
            torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=1e-6, rtol=1e-5)

    def test_complete_single_method_cli_pipeline(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, data, captured, factors = (root / name for name in ("model", "data", "capture", "factors"))
            tiny_model().save_pretrained(model_path)
            vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, **{f"word{i}": i + 4 for i in range(60)}}
            tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
            tokenizer.save_pretrained(model_path)
            source = root / "texts.jsonl"
            source.write_text("\n".join(json.dumps({"source_id": f"doc{i}", "text": " ".join(f"word{(j + i) % 60}" for j in range(100))}) for i in range(9)))
            def invoke(arguments):
                with contextlib.redirect_stdout(io.StringIO()):
                    args = parser().parse_args([str(arg) for arg in arguments])
                    args.function(args)
            invoke(["prepare", "--model", model_path, "--source", source, "--out", data,
                    "--calibration-documents", 3, "--validation-documents", 2,
                    "--calibration-length", 24, "--evaluation-length", 48])
            invoke(["capture", "--model", model_path, "--data", data, "--out", captured,
                    "--queries", 8, "--device", "cpu", "--dtype", "bfloat16"])
            invoke(["fit", "--capture", captured, "--out", factors, "--content-rank", 12,
                    "--rotary-dim", 4, "--steps", 3, "--max-position-scale", 2, "--device", "cpu"])
            initial = root / "initialization"
            invoke(["fit", "--capture", captured, "--out", initial, "--content-rank", 12,
                    "--rotary-dim", 4, "--initialize-only", "--device", "cpu"])
            initial_manifest = json.loads((initial / "manifest.json").read_text())
            self.assertEqual(initial_manifest["checkpoint_role"], "unoptimized_initialization")
            self.assertEqual(initial_manifest["specification"]["fit"]["steps"], 0)
            self.assertFalse((initial / "layer_000" / "fit.jsonl").exists())
            # Finished layers can be reused; this must not run a second trajectory.
            invoke(["fit", "--capture", captured, "--out", factors, "--content-rank", 12,
                    "--rotary-dim", 4, "--steps", 3, "--max-position-scale", 2, "--device", "cpu"])
            invoke(["diagnose", "--capture", captured, "--factors", factors, "--layer", 0,
                    "--position-scale", 2, "--out", root / "diagnostic.json", "--device", "cpu"])
            invoke(["evaluate", "--model", model_path, "--data", data, "--factors", factors,
                    "--out", root / "nll.json", "--target-tokens", 8, "--device", "cpu", "--dtype", "bfloat16"])
            prompt = root / "prompt.txt"
            prompt.write_text("word1 word2 word3")
            invoke(["generate", "--model", model_path, "--factors", factors, "--prompt", prompt,
                    "--expected-answer", "word7", "--out", root / "answer.json", "--max-new-tokens", 4,
                    "--device", "cpu", "--dtype", "bfloat16"])
            invoke(["profile", "--model", model_path, "--factors", factors, "--length", 8,
                    "--decode-tokens", 2, "--repeats", 1, "--out", root / "profile.json", "--device", "cpu", "--dtype", "bfloat16"])
            invoke(["report", "--factors", factors, "--diagnostic", root / "diagnostic.json",
                    "--evaluation", root / "nll.json", "--generation", root / "answer.json",
                    "--profile", root / "profile.json", "--out", root / "result.md"])
            self.assertEqual(json.loads((factors / "manifest.json").read_text())["status"], "complete")
            self.assertEqual(len(json.loads((root / "nll.json").read_text())["rows"]), 2)
            self.assertEqual(json.loads((root / "profile.json").read_text())["samples"][0]["cached_bytes_at_prefill"], 2 * 8 * 16 * 2)
            self.assertIn("output", json.loads((root / "answer.json").read_text()))
            self.assertGreaterEqual(json.loads((root / "answer.json").read_text())["answer_nll"], 0)
            self.assertIn("完整模型 NLL", (root / "result.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
