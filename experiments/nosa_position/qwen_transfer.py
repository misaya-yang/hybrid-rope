"""Conditional Qwen3.5 transfer: alter only the six full-attention readers.

The official Transformers model owns all 18 Gated DeltaNet layers, Q/K/V
projections, Q/K normalization, partial RoPE, dynamic cache and output gates.
This module receives already-rotated cached Q/K/V through HF's attention
interface. No NOSA weights, CIS term, or NOSA QK/CIS quota is transplanted.

``native_full`` leaves official SDPA active. Converted modes use one declared
query-only budget: initial/local blocks plus highest query score until topk
blocks are selected. They are converted references, not native Qwen sparse
attention or production kernel speed claims. PC2 uses the actual rotary width
and a diagonal covariance on the remaining NoPE channels.

Requires PyTorch >=2.8, Transformers 5.15.x with Qwen3.5, safetensors and the
checkpoint's tokenizer dependencies. The inspected server environment is
/root/autodl-tmp/reference_position_20260909/code_env/bin/python. CPU operator
tests do not need Transformers; optional HF integration tests need that runtime.

Prepare only (no model execution):
  python -m experiments.nosa_position.qwen_transfer --model MODEL --data RAW.jsonl --output OUT
Explicit conditional run, initially native_full only:
  python -m experiments.nosa_position.qwen_transfer --model MODEL --data RAW.jsonl --output OUT --run --device cuda --selectors native_full pc2

RAW rows require ``messages`` or unrendered ``prompt_text``. NOSA ``prompt`` or
``prompt_ids`` alone are rejected. Retokenization always uses Qwen's own chat
template with thinking disabled; input is never truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor

from .runtime import AttentionSettings, SelectionContext, native_scores, selected_causal_attention
from .selector_controls import BlockSummarySelector


BACKEND_LABEL = "qwen35_full_layers_converted_query_only_reference_v1"
INTERFACE_NAME = "qwen35_position_reference_v1"
MODES = ("native_full", "compressed_mean", "weighted_mean", "pc2", "cobs_rank1",
         "cobs_rank2", "split2", "split4", "quest", "exact_mass")


def query_only_topk(context: SelectionContext, scores: Tensor) -> Tensor:
    """Fixed total support budget, with all initial/local anchors protected."""
    s = context.settings
    blocks = math.ceil(context.k.shape[1] / s.block_size)
    shape = (context.k.shape[0], context.q.shape[1], blocks)
    if scores.shape != shape:
        raise ValueError(f"query block scores must have shape {shape}")
    ids = torch.arange(blocks, device=context.q.device)
    qb = context.query_positions[:, None] // s.block_size
    visible = ids <= qb
    mandatory = ((ids < s.init_blocks) | ((ids <= qb) & (qb <= ids + s.local_blocks))) & visible
    if bool((mandatory.sum(-1) > s.topk).any()):
        raise ValueError("topk is smaller than the required initial/local support")
    ranked = scores.masked_fill(mandatory, torch.inf).masked_fill(~visible, -torch.inf)
    selected = torch.argsort(ranked, descending=True, stable=True, dim=-1)[..., :min(s.topk, blocks)]
    selected = selected.sort(-1).values
    return selected.masked_fill(selected > qb[None], -1)


def _validate_mask(mask, queries, length, positions):
    if mask is None:
        return
    if not isinstance(mask, Tensor):
        raise ValueError("only unpadded causal tensor masks are supported")
    if mask.ndim == 2:
        if tuple(mask.shape) != (1, length) or not bool((mask == 1).all()):
            raise ValueError("padding or nonstandard attention masks are outside this reference")
        return
    if mask.ndim != 4 or tuple(mask.shape) != (1, 1, queries, length):
        raise ValueError("unexpected attention mask shape")
    visible = torch.arange(length, device=positions.device)[None] <= positions[:, None]
    if mask.dtype == torch.bool:
        correct = torch.equal(mask[0, 0], visible)
    else:
        correct = bool(((mask[0, 0] == 0) == visible).all()) and bool((mask[0, 0][~visible] < -1e4).all())
    if not correct:
        raise ValueError("custom/padded attention masks cannot be discarded")


class QwenTransferAttention:
    """HF-compatible attention callable, shared across full layers only."""

    def __init__(self, mode="pc2", *, settings=None, rotary_dim=64, trace_callback=None):
        if mode not in MODES or mode == "native_full":
            raise ValueError("native_full uses the unmodified official SDPA interface")
        self.mode = mode
        self.settings = settings or AttentionSettings()
        self.rotary_dim = rotary_dim
        self.trace_callback = trace_callback
        self.selector = None if mode == "compressed_mean" else BlockSummarySelector(mode, rotary_dim=rotary_dim)
        self.full_layer_calls = 0

    def reset(self):
        self.full_layer_calls = 0
        if self.selector is not None:
            self.selector.reset()

    @torch.no_grad()
    def __call__(self, module, query, key, value, attention_mask=None, *, scaling, dropout=0.0,
                 position_ids=None, **kwargs):
        if dropout != 0.0 or getattr(module, "training", False):
            raise ValueError("Qwen transfer is inference-only and requires model.eval()")
        if kwargs.get("output_attentions", False):
            raise ValueError("attention matrices are not materialized by this reference")
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape or query.shape[0] != 1 or key.shape[0] != 1:
            raise ValueError("expected batch-one Q/K/V with equal key/value head dimensions")
        _, qh, queries, dim = query.shape
        _, kvh, length, _ = key.shape
        if qh % kvh or length < queries or key.shape[-1] != dim:
            raise ValueError("invalid cached GQA geometry")
        if self.rotary_dim > dim or self.rotary_dim < 0 or self.rotary_dim % 2:
            raise ValueError("rotary width does not match Qwen attention geometry")
        positions = torch.arange(length - queries, length, device=query.device)
        if position_ids is not None:
            if tuple(position_ids.shape) != (1, queries) or not torch.equal(position_ids[0], positions):
                raise ValueError("only contiguous text positions and a complete DynamicCache prefix are supported")
        _validate_mask(attention_mask, queries, length, positions)
        # The shared reference divides by sqrt(D). This conversion preserves
        # the exact scaling supplied by HF; it does not rerotate or mutate Q/K.
        factor = float(scaling) * math.sqrt(dim)
        q = query[0] if factor == 1.0 else query[0] * factor
        context = SelectionContext(q, key[0], value[0], key.new_zeros(kvh, length), positions,
                                   module.layer_idx, self.settings)
        blocks = math.ceil(length / self.settings.block_size)
        if blocks <= self.settings.topk:
            if self.selector is not None and int(positions[0]) == 0:
                self.selector.cache.pop(module.layer_idx, None)
            ids = torch.arange(blocks, device=query.device).expand(kvh, queries, -1)
            selected = ids.masked_fill(ids > positions[None, :, None] // self.settings.block_size, -1)
        else:
            if self.selector is None:
                scores, _, _ = native_scores(context)
            else:
                scores = self.selector.logmass(context).softmax(-1).sum(1)
            selected = query_only_topk(context, scores)
        if self.trace_callback is not None:
            self.trace_callback(context, selected)
        output = selected_causal_attention(context, selected)
        self.full_layer_calls += 1
        # Official Qwen3_5Attention applies sigmoid(output_gate) and o_proj.
        return output.transpose(0, 1).unsqueeze(0).contiguous(), None


def _dispatch(module, *args, **kwargs):
    controller = getattr(module, "_qwen_position_controller", None)
    if controller is None:
        raise RuntimeError("custom Qwen attention was called outside a configured full-attention module")
    return controller(module, *args, **kwargs)


def _unmaterialized_causal_mask(*, batch_size, kv_offset=0, attention_mask=None, use_vmap=False, **kwargs):
    """Avoid an NxN HF mask; the attention callable enforces absolute causality."""
    if batch_size != 1 or kv_offset != 0 or use_vmap:
        raise ValueError("only unpadded, unpacked, complete-prefix batch-one attention is supported")
    if attention_mask is not None and (attention_mask.ndim != 2 or not bool((attention_mask == 1).all())):
        raise ValueError("Qwen transfer does not support padded masks")
    return None


def configure_qwen_transfer(model, mode="native_full", *, settings=None, trace_callback=None):
    """Install/replace one registered HF function; never replace model layers.

    Repeated calls replace the per-module controller, so old selector caches
    can be released. The global HF registry stores only a stateless dispatch
    function. ``native_full`` restores official SDPA for the text model.
    """
    if mode not in MODES:
        raise ValueError(mode)
    modules = [(name, module) for name, module in model.named_modules()
               if module.__class__.__name__ == "Qwen3_5Attention"]
    gdn = [module for module in model.modules() if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"]
    if not modules:
        raise ValueError("no official Qwen3_5Attention modules found; no fallback architecture is allowed")
    cfg = modules[0][1].config
    expected = [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"]
    actual = [module.layer_idx for _, module in modules]
    if actual != expected or len(gdn) != cfg.layer_types.count("linear_attention"):
        raise ValueError("full-attention/GDN module counts do not match the official configuration")
    rd = int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0))
    controller = None
    if mode != "native_full":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.register(INTERFACE_NAME, _dispatch)
        ALL_MASK_ATTENTION_FUNCTIONS.register(INTERFACE_NAME, _unmaterialized_causal_mask)
        controller = QwenTransferAttention(mode, settings=settings, rotary_dim=rd, trace_callback=trace_callback)
    for _, module in modules:
        module.config._attn_implementation = "sdpa" if controller is None else INTERFACE_NAME
        if hasattr(module, "_qwen_position_controller"):
            delattr(module, "_qwen_position_controller")
        if controller is not None:
            module._qwen_position_controller = controller
    report = {"backend": "qwen35_official_sdpa" if mode == "native_full" else BACKEND_LABEL,
              "mode": mode, "full_attention_indices": actual, "unchanged_gdn_layers": len(gdn),
              "rotary_dim": rd, "head_dim": cfg.head_dim, "cis": "absent; zero in reference context",
              "support_rule": "native full" if mode == "native_full" else "initial/local anchors plus query-only topk",
              "pc2_nope_covariance": "diagonal", "native_sparse_qwen_claim": False}
    return controller, report


def load_qwen_model(model_dir, *, device="cpu", dtype=torch.float32):
    """Load the official entire conditional-generation checkpoint locally."""
    from transformers import Qwen3_5ForConditionalGeneration
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, dtype=dtype,
        attn_implementation="sdpa", output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Qwen checkpoint loading was not exact: {key}: {info[key][:8]}")
    return model.to(device).eval()


def retokenize_row(row, tokenizer):
    messages = row.get("messages")
    if messages is None:
        if not isinstance(row.get("prompt_text"), str):
            raise ValueError("Qwen rows require raw messages or prompt_text; NOSA prompt/prompt_ids are not reusable")
        messages = [{"role": "user", "content": row["prompt_text"]}]
    if not isinstance(messages, list) or not messages or any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("only nonempty text-only chat messages are supported")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    max_new = row.get("max_new_tokens", 32)
    if max_new < 1 or not ids:
        raise ValueError("prompt and generation budget must be nonempty")
    if row.get("length_cap") is not None and len(ids) + max_new > row["length_cap"]:
        raise ValueError(f"Qwen-retokenized row exceeds its declared length cap: {row.get('row_id')}")
    return {**row, "prompt_ids": ids, "max_new_tokens": max_new, "tokenizer_model": "Qwen3.5",
            "qwen_rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "template_scope": "Qwen checkpoint chat template; enable_thinking=False"}


@torch.inference_mode()
def greedy_generate_qwen(model, input_ids, *, max_new_tokens=32, eos_ids=(), chunk_size=128):
    """Official hybrid cache throughout chunked prefill and one-token decode."""
    if chunk_size < 1 or max_new_tokens < 1 or tuple(input_ids.shape[:1]) != (1,) or input_ids.shape[1] < 1:
        raise ValueError("nonempty batch-one input and positive budgets are required")
    output = None
    for start in range(0, input_ids.shape[1], chunk_size):
        output = model(input_ids=input_ids[:, start:start+chunk_size], use_cache=True,
                       past_key_values=None if output is None else output.past_key_values, logits_to_keep=1)
    generated = []
    for step in range(max_new_tokens):
        if not bool(torch.isfinite(output.logits).all()):
            raise FloatingPointError("nonfinite Qwen generation logits")
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        output = model(input_ids=torch.tensor([[token]], device=input_ids.device), use_cache=True,
                       past_key_values=output.past_key_values, logits_to_keep=1)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selectors", nargs="+", choices=MODES, default=["native_full"])
    parser.add_argument("--run", action="store_true", help="explicitly execute the conditional transfer")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    raw = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    rows = [retokenize_row(row, tokenizer) for row in raw[:args.max_rows]]
    if not rows or args.max_rows < 1:
        raise ValueError("no raw transfer rows selected")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "generations.jsonl").exists():
        raise ValueError("transfer output already contains generations; use a new directory")
    (output / "prepared_rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    report = {"status": "PREPARED_ONLY", "rows": len(rows), "model": str(Path(args.model).resolve()),
              "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("qwen_transfer.py", "runtime.py", "selector_controls.py")},
              "raw_data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "model_config_sha256": hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest(),
              "selectors": args.selectors, "scientific_transfer_verified": False}
    (output / "ready.json").write_text(json.dumps(report, indent=2)+"\n")
    if not args.run:
        print(json.dumps(report), flush=True)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    model = load_qwen_model(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    cfg = model.config.text_config
    eos = getattr(cfg, "eos_token_id", tokenizer.eos_token_id)
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    with (output / "generations.jsonl").open("w") as stream:
        for row in rows:
            ids = torch.tensor([row["prompt_ids"]], device=args.device)
            for mode in args.selectors:
                controller, architecture = configure_qwen_transfer(model, mode, settings=AttentionSettings(topk=args.topk))
                if args.device == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                generated = greedy_generate_qwen(model, ids, max_new_tokens=row["max_new_tokens"],
                                                 eos_ids=eos_ids, chunk_size=args.chunk_size)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                ended = bool(generated and generated[-1] in eos_ids)
                text = tokenizer.decode(generated[:-1] if ended else generated, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                result = {"row_id": row["row_id"], "selector": mode, "architecture": architecture,
                          "generated_token_ids": generated, "output_text": text, "ended_with_eos": ended,
                          "total_seconds": time.perf_counter()-begin, "input_tokens": ids.numel()}
                if "expected" in row:
                    result["exact_string"] = text == row["expected"]
                    result["exact_plus_eos"] = ended and text == row["expected"]
                stream.write(json.dumps(result, ensure_ascii=False)+"\n")
                stream.flush()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    report["status"] = "CONDITIONAL_RUN_COMPLETE"
    (output / "ready.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
