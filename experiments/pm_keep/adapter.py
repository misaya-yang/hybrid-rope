"""Question-blind, once-per-prefix compaction for the actual HF Qwen2 model.

Only original post-RoPE K/V are gathered. The model still owns projection, RoPE,
GQA, attention, and generation logits. Physical cache length and logical token
position are deliberately separate after compaction. Batch size one without
padding is the supported experiment contract, rather than an implicit batch
approximation. No model weights or forward methods are replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, Iterable, List, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM

from .ops import NativeRope, make_sampling_plan, pm_keep_scores, select_fixed_budget


@dataclass(frozen=True)
class AdapterConfig:
    samples_per_head: int = 256
    horizon: int = 512
    seed: int = 20260909
    query_policy: str = "uniform_prefix"
    recent_query_window: int = 512
    sink_tokens: int = 4
    recent_tokens: int = 256
    keep_fraction: float = 0.25
    query_chunk_size: int = 32
    key_chunk_size: int = 2048


@dataclass
class LayerPrefixData:
    """Ephemeral author-scorer input; hidden_states must not be retained."""

    attention_module: torch.nn.Module
    hidden_states: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    position_embeddings: tuple
    query_samples: torch.Tensor
    sample_indices: torch.Tensor
    future_positions: torch.Tensor
    prefix_length: int
    layer_idx: int


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_nbytes(cache) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.layers
        for tensor in (layer.keys, layer.values)
        if tensor is not None
    )


def _ids(value, device, *, allow_empty=False):
    result = torch.as_tensor(value, dtype=torch.long, device=device)
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if result.ndim != 2 or result.shape[0] != 1:
        raise ValueError("PM-Keep adapter supports one unpadded token sequence")
    if not allow_empty and not result.shape[1]:
        raise ValueError("token sequence must be nonempty")
    return result


def _score_tensor(result, heads, length, device):
    if hasattr(result, "scores"):
        result = result.scores
    if not isinstance(result, torch.Tensor):
        raise TypeError("prefix scorer must return a tensor or a result with .scores")
    if result.shape == (1, heads, length):
        result = result[0]
    if result.shape != (heads, length):
        raise ValueError(f"scorer shape {tuple(result.shape)} != {(heads, length)}")
    if not torch.isfinite(result).all():
        raise ValueError("prefix scorer returned non-finite scores")
    return result.detach().to(device=device, dtype=torch.float32)


class PrefillSession:
    """One full prefill shared by independently cloned inference branches.

    ``prefill`` accepts only prefix IDs, never a future question or label. An
    external scorer is called transiently after its attention layer has completed
    the full prefix; it cannot change any cache tensor. C/P/U use exactly the
    sampled native Q projection outputs captured during that same prefill.
    """

    def __init__(self, model, prefix_ids, config: Optional[AdapterConfig] = None):
        self.model = model
        self.config = config or AdapterConfig()
        self.device = next(model.parameters()).device
        self.prefix_ids = _ids(prefix_ids, self.device).clone()
        self.prefix_length = self.prefix_ids.shape[1]
        if self.config.query_policy not in {"uniform_prefix", "recent_prefix"}:
            raise ValueError("query_policy must be uniform_prefix or recent_prefix")
        if self.config.recent_query_window <= 0:
            raise ValueError("recent_query_window must be positive")
        self._validate_model()
        self.total_budget = math.floor(self.prefix_length * self.config.keep_fraction)
        if not 0 < self.total_budget <= self.prefix_length:
            raise ValueError("keep_fraction must yield a budget between 1 and T")
        protected = len(set(range(min(self.config.sink_tokens, self.prefix_length))) |
                        set(range(max(0, self.prefix_length - self.config.recent_tokens),
                                  self.prefix_length)))
        if self.total_budget < protected:
            raise ValueError("protected sink/recent entries exceed the total KV budget")
        policy_kwargs = ({"query_start": max(self.config.sink_tokens,
                            self.prefix_length - self.config.recent_query_window)}
                         if self.config.query_policy == "recent_prefix" else {})
        self.plans = [make_sampling_plan(
            self.prefix_length, model.config.num_attention_heads,
            samples_per_head=self.config.samples_per_head,
            horizon=self.config.horizon, seed=self.config.seed,
            layer_idx=i, sink_tokens=self.config.sink_tokens,
            **policy_kwargs,
        ) for i in range(model.config.num_hidden_layers)]
        rotary = model.model.rotary_emb
        self.rope = NativeRope(rotary.inv_freq.detach().clone(), layout="split_half",
                               amplitude=float(rotary.attention_scaling))
        self.query_samples = [None] * len(self.plans)
        self.scores: Dict[str, List[torch.Tensor]] = {}
        self.score_metrics = {}
        self.cache = None
        self.last_logits = None
        self.timings = {"query_capture_seconds": 0.0, "score_seconds": {}}

    def _validate_model(self):
        cfg = self.model.config
        if not isinstance(self.model, Qwen2ForCausalLM) or cfg.model_type != "qwen2":
            raise ValueError("this adapter is verified only for native HF Qwen2ForCausalLM")
        if self.model.training:
            raise ValueError("frozen inference requires model.eval()")
        if cfg.num_attention_heads % cfg.num_key_value_heads:
            raise ValueError("Q heads must form intact native GQA groups")
        if any(kind != "full_attention" for kind in cfg.layer_types):
            raise ValueError("sliding/recurrent layers require a separate verified adapter")
        if self.prefix_length + self.config.horizon > cfg.max_position_embeddings:
            raise ValueError("prefix plus score horizon exceeds the native position window")
        rotary = self.model.model.rotary_emb
        if rotary.rope_type != "default" or float(rotary.attention_scaling) != 1.0:
            raise ValueError("dynamic/scaled/non-unit RoPE is outside this native adapter")
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if not isinstance(attn, Qwen2Attention) or hasattr(attn, "q_norm"):
                raise ValueError("unsupported Q projection / normalization implementation")
            if rotary.inv_freq.numel() * 2 != attn.head_dim:
                raise ValueError("this Qwen2 path requires its native full rotary dimension")
            if not math.isclose(attn.scaling, attn.head_dim ** -0.5):
                raise ValueError("attention scale differs from native Qwen2")

    @torch.inference_mode()
    def prefill(self, external_scorers: Optional[Dict[str, Callable]] = None):
        if self.cache is not None:
            raise RuntimeError("a PrefillSession may prefill only once")
        external_scorers = external_scorers or {}
        if any(arm in {"F", "P", "C", "U"} for arm in external_scorers):
            raise ValueError("external scorers cannot replace native or matched-control arms")
        self.scores.update({arm: [None] * len(self.plans) for arm in external_scorers})
        handles = []
        rotary_bindings = []
        self.timings["score_seconds"].update({arm: 0.0 for arm in external_scorers})

        def query_hook(index):
            def capture(module, args, output):
                # Author scorers may re-use q_proj on these same hidden states.
                # Capture only the native forward's first projection call.
                if self.query_samples[index] is not None:
                    return
                _sync(self.device)
                start = time.perf_counter()
                heads = self.model.config.num_attention_heads
                q = output.view(1, self.prefix_length, heads, -1).transpose(1, 2)
                indices = self.plans[index].query_indices.to(self.device)
                gather = indices[None, :, :, None].expand(1, heads, -1, q.shape[-1])
                self.query_samples[index] = torch.gather(q, 2, gather).detach()
                _sync(self.device)
                self.timings["query_capture_seconds"] += time.perf_counter() - start
            return capture

        def layer_hook(index):
            def score(module, args, kwargs, output):
                cache = kwargs["past_key_values"]
                layer = cache.layers[index]
                plan = self.plans[index]
                data = LayerPrefixData(
                    attention_module=module, hidden_states=kwargs["hidden_states"],
                    keys=layer.keys, values=layer.values,
                    position_embeddings=kwargs["position_embeddings"],
                    query_samples=self.query_samples[index], sample_indices=plan.query_indices,
                    future_positions=plan.future_positions, prefix_length=self.prefix_length,
                    layer_idx=index,
                )
                shape = tuple(layer.keys.shape)
                for arm, scorer in external_scorers.items():
                    _sync(self.device)
                    start = time.perf_counter()
                    result = scorer(data)
                    self.scores[arm][index] = _score_tensor(
                        result, self.model.config.num_key_value_heads,
                        self.prefix_length, self.device,
                    )
                    if tuple(layer.keys.shape) != shape or tuple(layer.values.shape) != shape:
                        raise RuntimeError("external scorer compacted cache during full prefill")
                    _sync(self.device)
                    self.timings["score_seconds"][arm] += time.perf_counter() - start
            return score

        for index, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn
            handles.append(attention.q_proj.register_forward_hook(query_hook(index)))
            if external_scorers:
                # KVPress authors expect the model's shared native rotary module
                # on each attention module. Restore this compatibility binding.
                existed = hasattr(attention, "rotary_emb")
                rotary_bindings.append((attention, existed, getattr(attention, "rotary_emb", None)))
                attention.rotary_emb = self.model.model.rotary_emb
                handles.append(attention.register_forward_hook(layer_hook(index), with_kwargs=True))
        _sync(self.device)
        start = time.perf_counter()
        try:
            positions = torch.arange(self.prefix_length, device=self.device)
            output = self.model(
                self.prefix_ids, position_ids=positions[None], cache_position=positions,
                use_cache=True, logits_to_keep=1,
            )
            self.cache = output.past_key_values
            self.last_logits = output.logits[:, -1].detach().clone()
            _sync(self.device)
            self.timings["prefill_total_seconds"] = time.perf_counter() - start
            self.timings["native_prefill_residual_seconds"] = (
                self.timings["prefill_total_seconds"] - self.timings["query_capture_seconds"]
                - sum(self.timings["score_seconds"].values())
            )
            if any(q is None for q in self.query_samples):
                raise RuntimeError("not all native layers supplied prefix Q samples")
            if any(layer.get_seq_length() != self.prefix_length for layer in self.cache.layers):
                raise RuntimeError("full prefix cache was changed during prefill")
        finally:
            for handle in handles:
                handle.remove()
            for attention, existed, previous in rotary_bindings:
                if existed:
                    attention.rotary_emb = previous
                else:
                    delattr(attention, "rotary_emb")
        return self

    @torch.inference_mode()
    def score(self, arm: str):
        if self.cache is None:
            raise RuntimeError("prefill must precede scoring")
        if arm in self.scores:
            return self.scores[arm]
        method = {"P": "pm", "C": "collapse", "U": "unit"}.get(arm)
        if method is None:
            raise ValueError(f"unknown or uncaptured scoring arm {arm}")
        scores, metrics = [], []
        _sync(self.device)
        start = time.perf_counter()
        for index, layer in enumerate(self.cache.layers):
            result = pm_keep_scores(
                self.query_samples[index][0], layer.keys[0], self.rope,
                arm=method, future_start=self.prefix_length, horizon=self.config.horizon,
                future_positions=self.plans[index].future_positions,
                attention_scale=self.model.model.layers[index].self_attn.scaling,
                query_chunk_size=self.config.query_chunk_size,
                key_chunk_size=self.config.key_chunk_size,
            )
            scores.append(_score_tensor(result, layer.keys.shape[1], self.prefix_length, self.device))
            metrics.append(getattr(result, "metrics", {}))
        _sync(self.device)
        self.timings["score_seconds"][arm] = time.perf_counter() - start
        self.scores[arm], self.score_metrics[arm] = scores, metrics
        return scores

    @torch.inference_mode()
    def keep_indices(self, arm: str):
        if self.cache is None:
            raise RuntimeError("prefill must precede selection")
        if arm == "F":
            return [torch.arange(self.prefix_length, device=self.device).expand(
                layer.keys.shape[1], -1).clone() for layer in self.cache.layers]
        return [select_fixed_budget(
            score, self.total_budget, sink_tokens=self.config.sink_tokens,
            recent_tokens=self.config.recent_tokens,
        ) for score in self.score(arm)]

    @torch.inference_mode()
    def branch(self, arm: str, indices: Optional[Iterable[torch.Tensor]] = None):
        """Gather a separate cache; later branches never inherit generated K/V."""
        if self.cache is None:
            raise RuntimeError("prefill must precede cache branching")
        keep = self.keep_indices(arm) if indices is None else list(indices)
        if len(keep) != len(self.cache.layers):
            raise ValueError("one keep-set per layer is required")
        normalized = []
        budget = None
        _sync(self.device)
        start = time.perf_counter()
        cache = DynamicCache(config=self.model.config)
        for index, (layer, selected) in enumerate(zip(self.cache.layers, keep)):
            selected = torch.as_tensor(selected, device=self.device, dtype=torch.long)
            if selected.ndim != 2 or selected.shape[0] != layer.keys.shape[1]:
                raise ValueError("keep indices must be [native_KV_heads, fixed_budget]")
            if not selected.shape[1] or (selected < 0).any() or (selected >= self.prefix_length).any():
                raise ValueError("keep indices must reference valid original prefix positions")
            if selected.shape[1] > 1 and not (selected[:, 1:] > selected[:, :-1]).all():
                raise ValueError("keep indices must be sorted and unique within every KV head")
            if budget is not None and selected.shape[1] != budget:
                raise ValueError("all layers must have the same physical KV budget")
            budget = selected.shape[1]
            gather = selected[None, :, :, None].expand(1, -1, -1, layer.keys.shape[-1])
            keys = torch.gather(layer.keys, 2, gather)
            values = torch.gather(layer.values, 2, gather)
            cache.update(keys, values, index)
            normalized.append(selected.detach().clone())
        _sync(self.device)
        gather_seconds = time.perf_counter() - start
        return DecodeState(
            self.model, cache, self.prefix_length, self.last_logits.clone(), normalized,
            arm=arm, gather_seconds=gather_seconds,
        )

    def memory_receipt(self):
        return {
            "full_prefix_kv_bytes": cache_nbytes(self.cache) if self.cache is not None else 0,
            "query_samples_bytes": sum(q.numel() * q.element_size() for q in self.query_samples
                                       if q is not None),
            "sampling_plan_host_bytes": sum(
                p.query_indices.numel() * p.query_indices.element_size()
                + p.future_positions.numel() * p.future_positions.element_size()
                for p in self.plans),
            "score_tensor_bytes": sum(s.numel() * s.element_size()
                                      for layers in self.scores.values() for s in layers if s is not None),
            "full_prefix_remains_resident_for_paired_branch_reuse": self.cache is not None,
        }


class DecodeState:
    """Append-only inference from a selected cache at original logical positions."""

    def __init__(self, model, cache, logical_position, last_logits, keep_indices,
                 *, arm, gather_seconds):
        self.model, self.cache = model, cache
        self.device = next(model.parameters()).device
        self.prefix_length = logical_position
        self.logical_position = logical_position
        self.last_logits = last_logits
        self.keep_indices = keep_indices
        self.arm = arm
        self.initial_kv_bytes = cache_nbytes(cache)
        self.initial_index_bytes = sum(t.numel() * t.element_size() for t in keep_indices)
        self.logical_positions = []
        self.physical_cache_lengths = []
        self.timings = {"gather_seconds": gather_seconds, "question_seconds": 0.0,
                        "decode_seconds": 0.0}
        self._generated = False

    @torch.inference_mode()
    def step(self, token: int):
        if self.logical_position >= self.model.config.max_position_embeddings:
            raise ValueError("generation would exceed the native context window")
        position = torch.tensor([self.logical_position], device=self.device, dtype=torch.long)
        # One new token can attend to every physical retained slot and to itself.
        # A rectangular causal mask based on re-numbered cache slots is neither
        # needed nor valid for a multi-token suffix. Therefore ingestion is Q=1.
        output = self.model(
            torch.tensor([[int(token)]], device=self.device, dtype=torch.long),
            past_key_values=self.cache, position_ids=position[None], cache_position=position,
            attention_mask={"full_attention": None}, use_cache=True, logits_to_keep=1,
        )
        self.cache = output.past_key_values
        self.last_logits = output.logits[:, -1].detach()
        self.logical_positions.append(self.logical_position)
        self.logical_position += 1
        lengths = [layer.get_seq_length() for layer in self.cache.layers]
        if len(set(lengths)) != 1:
            raise RuntimeError("appended K/V lengths disagree across native layers")
        self.physical_cache_lengths.append(lengths[0])
        return self.last_logits

    @torch.inference_mode()
    def consume(self, suffix_ids):
        if self._generated:
            raise RuntimeError("cannot add a new question after generation")
        suffix = _ids(suffix_ids, self.device, allow_empty=True)
        if self.logical_position + suffix.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("question would exceed the native context window")
        _sync(self.device)
        start = time.perf_counter()
        # Avoid one GPU-to-host synchronization for every suffix token.
        for token in suffix[0].tolist():
            self.step(token)
        _sync(self.device)
        self.timings["question_seconds"] += time.perf_counter() - start
        return self

    @torch.inference_mode()
    def generate(self, max_new_tokens: int, eos_token_ids: Iterable[int]):
        if self._generated:
            raise RuntimeError("generate may be called only once per independent branch")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.logical_position + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("prompt and full answer budget exceed the native context window")
        eos = {int(e) for e in eos_token_ids}
        self._generated = True
        generated = []
        _sync(self.device)
        start = time.perf_counter()
        for index in range(max_new_tokens):
            token = int(self.last_logits[0].argmax())
            generated.append(token)
            if token in eos or index + 1 == max_new_tokens:
                break
            self.step(token)
        _sync(self.device)
        self.timings["decode_seconds"] = time.perf_counter() - start
        return {
            "generated_ids": generated, "ended_eos": bool(generated and generated[-1] in eos),
            "logical_positions": list(self.logical_positions),
            "physical_cache_lengths": list(self.physical_cache_lengths),
            "logical_next_position": self.logical_position,
            "initial_prefix_kv_bytes": self.initial_kv_bytes,
            "retained_indices_bytes": self.initial_index_bytes,
            "final_kv_bytes": cache_nbytes(self.cache),
            "timings": dict(self.timings),
            "last_sampled_token_not_ingested": True,
        }
