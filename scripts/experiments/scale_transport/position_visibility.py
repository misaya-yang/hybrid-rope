"""Fixed-position visibility interventions; preparation, not a frequency method.

O: original prompt; L: original prefill, retain only selected prompt KV at readout;
P: selected tokens with original positions; C: same tokens at contiguous positions.
All four use the ORIGINAL prompt for logits-processor history. C is therefore a
controlled diagnostic, not an ordinary shortened-prompt benchmark.

The pure layout checks can run locally. The Qwen/Flash runtime must be qualified
on the work machine before interpreting any intervention; no automatic launch.
"""
from __future__ import annotations

import math
import time


def check_decoder(parameters):
    """Qualify the saved greedy contract supported by this narrow replay."""
    required = {"do_sample": False, "num_beams": 1, "min_length": 0,
                "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
                "encoder_repetition_penalty": 1.0, "remove_invalid_values": False}
    if any(parameters.get(k) != v for k, v in required.items()):
        raise ValueError("saved decoder is outside the supported greedy contract")
    inactive = ("min_new_tokens", "bad_words_ids", "forced_bos_token_id",
                "forced_eos_token_id", "exponential_decay_length_penalty",
                "suppress_tokens", "begin_suppress_tokens", "sequence_bias",
                "token_healing", "guidance_scale", "watermarking_config",
                "stop_strings", "constraints", "force_words_ids", "use_mtp")
    if any(parameters.get(k) not in (None, False) for k in inactive):
        raise ValueError("additional logits processors/stopping rules require explicit support")
    if parameters.get("renormalize_logits") not in (None, False):
        raise ValueError("logit normalization differs from the frozen reference")


def layout(prompt_ids, keep_positions, mode):
    if mode not in {"O", "L", "P", "C"}:
        raise ValueError("mode must be O, L, P or C")
    ids, keep = list(prompt_ids), list(keep_positions)
    if len(ids) < 2 or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("at least two nonnegative integer token IDs required")
    if (len(keep) < 2 or any(type(x) is not int for x in keep)
            or keep != sorted(set(keep)) or keep[0] < 0
            or keep[-1] != len(ids) - 1):
        raise ValueError("ordered unique retained positions must include the last prompt query")
    visible = list(range(len(ids))) if mode in {"O", "L"} else keep
    positions = list(range(len(visible))) if mode == "C" else visible
    return {
        "mode": mode,
        "prefill_ids": [ids[i] for i in visible[:-1]],
        "prefill_positions": positions[:-1],
        "query_id": ids[-1],
        "query_position": positions[-1],
        "cache_keep_positions": keep[:-1] if mode == "L" else None,
        "processor_prompt_ids": ids,
    }


def _retain_dynamic_prompt_cache(cache, keep):
    """Filter already-rotated KV without changing any retained value or angle."""
    import torch

    if not getattr(cache, "layers", None) or getattr(cache, "offloading", False):
        raise ValueError("layer-based DynamicCache required")
    for layer in cache.layers:
        # Sliding/static/compressed caches have extra indexing state. This
        # intervention is scoped to the frozen Qwen2.5 full-attention runtime.
        if type(layer).__name__ != "DynamicLayer":
            raise ValueError("only full-attention DynamicLayer is qualified here")
        index = layer.keys.new_tensor(keep, dtype=torch.long)
        layer.keys = layer.keys.index_select(-2, index)
        layer.values = layer.values.index_select(-2, index)
    if cache.get_seq_length() != len(keep):
        raise ValueError("filtered cache retained stale sequence-length state")
    return cache


def replay(model, prompt_ids, keep_positions, mode, *, max_new_tokens,
           eos_token_id, repetition_penalty, absolute_deadline_unix):
    """Greedy Qwen replay with unchanged original-prompt repetition penalty.

    Caller freezes model/table/gain/token/mask identities, checks the resolved
    decoder has no other active logits processors, and qualifies O against the
    existing generation. Gold content is never inserted into a prefix.
    """
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import RepetitionPenaltyLogitsProcessor

    view = layout(prompt_ids, keep_positions, mode)
    if (model.training or model.config.model_type != "qwen2" or model.device.type != "cuda"
            or model.config._attn_implementation != "sdpa"
            or getattr(model.config, "use_sliding_window", False)
            or any(kind != "full_attention" for kind in (getattr(model.config, "layer_types", None) or []))
            or model.model.rotary_emb.rope_type != "default"):
        raise ValueError("eval-mode Qwen2.5 CUDA/Flash-SDPA full attention with static RoPE required")
    if (type(max_new_tokens) is not int or max_new_tokens < 1
            or type(eos_token_id) is not int
            or not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
        raise ValueError("finite positive decoding parameters required")
    if not math.isfinite(absolute_deadline_unix) or absolute_deadline_unix <= time.time():
        raise ValueError("finite future deadline required; outer supervisor owns the hard timeout")
    device = model.device
    tensor = lambda x: torch.tensor([x], dtype=torch.long, device=device)
    processor = RepetitionPenaltyLogitsProcessor(repetition_penalty)
    generated, scores = [], []

    def forward(ids, positions, cache=None):
        if time.time() >= absolute_deadline_unix:
            raise TimeoutError("visibility replay deadline")
        past = 0 if cache is None else cache.get_seq_length()
        # Physical cache indices are contiguous; RoPE positions are supplied
        # independently and retain the original gaps in L/P.
        return model(input_ids=tensor(ids), position_ids=tensor(positions),
                     attention_mask=torch.ones((1, past + len(ids)), dtype=torch.long, device=device),
                     past_key_values=cache, use_cache=True, logits_to_keep=1)

    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        prefill = forward(view["prefill_ids"], view["prefill_positions"])
        cache = prefill.past_key_values
        del prefill
        if mode == "L":
            cache = _retain_dynamic_prompt_cache(cache, view["cache_keep_positions"])
        current = view["query_id"]
        for step in range(max_new_tokens):
            # Recompute the LAST PROMPT QUERY before predicting the first
            # answer token; filtering only later decode steps would miss it.
            result = forward([current], [view["query_position"] + step], cache)
            cache = result.past_key_values
            raw = result.logits[:, -1].float()
            processed = processor(tensor(view["processor_prompt_ids"] + generated), raw)
            if not torch.isfinite(processed).all():
                raise FloatingPointError("nonfinite processed logits")
            scores.append(processed[0].detach().cpu().numpy().copy())
            current = int(processed.argmax(dim=-1).item())
            generated.append(current)
            del result
            if current == eos_token_id:
                break
    return {
        "generated_ids": generated,
        "processed_scores": np.stack(scores),
        "mode": mode,
        "prefill_tokens": len(view["prefill_ids"]),
        "first_query_position": view["query_position"],
        "processor_prompt_tokens": len(prompt_ids),
        "selected_prompt_tokens": len(keep_positions),
        "visible_prompt_tokens": len(prompt_ids) if mode == "O" else len(keep_positions),
        "eos": bool(generated and generated[-1] == eos_token_id),
        "scope": "ORACLE_VISIBILITY_DIAGNOSTIC_NOT_A_DEPLOYMENT_METHOD",
    }
