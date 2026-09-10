"""Checkpoint-compatible, batch-one NOSA inference reference (no custom CUDA).

This is a source-derived *reference*, not a measured parity claim for the NOSA
CUDA kernels. It retains all pretrained Llama weights, RoPE, and the learned
key bias ``A * softplus(delta(V))``. The published two-attention ratio equals
attention with that additive bias. Selection follows the inspected 32/16 mean
compression, group-summed softmax, five-window max pooling and QK/CIS union.

Reference choices: explicit absolute-position causality for chunked prefill;
only completed compression windows can influence a query; deterministic
low-index tie breaking; native CUDA reduced-precision arithmetic is not copied.
All results must retain ``BACKEND_LABEL`` until kernel parity is measured.

Main API: ``NosaReferenceForCausalLM.from_pretrained(local_dir, ...)``,
``model.prefill(ids, chunk_size=128)``, then ``model(next_id,
past_key_values=output.past_key_values)``. An external selector is a callable
``SelectionContext -> int tensor [kv_heads, query_length, selected_blocks]``;
block IDs are global, unique per row, with -1 padding. No labels or answers are
available to selectors. CPU tests use the same complete model with tiny config.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BACKEND_LABEL = "nosa_source_reference_causal_v1"


@dataclass(frozen=True)
class AttentionSettings:
    kernel_size: int = 32
    kernel_stride: int = 16
    block_size: int = 64
    init_blocks: int = 1
    local_blocks: int = 16  # Official rule includes the current block as well.
    select_blocks: int = 16
    topk: int = 64
    dense: bool = False
    attention_query_chunk_size: int = 16

    def __post_init__(self):
        if self.kernel_size != 2 * self.kernel_stride:
            raise ValueError("reference pooling requires kernel_size = 2 * stride")
        if self.kernel_stride < 1 or self.block_size < 1 or self.block_size % self.kernel_stride:
            raise ValueError("block_size must be a positive multiple of stride")
        if min(self.init_blocks, self.local_blocks, self.select_blocks) < 0 or self.topk < 1:
            raise ValueError("invalid selection budget")
        if self.attention_query_chunk_size < 1:
            raise ValueError("attention query chunk must be positive")


@dataclass
class SelectionContext:
    q: Tensor  # [query_heads, queries, head_dim], after pretrained RoPE
    k: Tensor  # [kv_heads, prefix_length, head_dim], after pretrained RoPE
    v: Tensor  # [kv_heads, prefix_length, head_dim]
    cis: Tensor  # [kv_heads, prefix_length]
    query_positions: Tensor  # [queries], absolute, consecutive
    layer_idx: int
    settings: AttentionSettings
    rope_inv_freq: Optional[Tensor] = None
    rope_attention_factor: float = 1.0


@dataclass
class LayerCache:
    k: Tensor
    v: Tensor
    cis: Tensor


@dataclass
class NosaOutput:
    logits: Tensor
    past_key_values: Optional[list[LayerCache]]


def _safe_softmax(scores: Tensor, valid: Tensor) -> Tensor:
    masked = scores.float().masked_fill(~valid, -torch.inf)
    any_valid = valid.any(-1, keepdim=True)
    # All-masked rows precede the first complete window; their score is zero.
    return torch.softmax(torch.where(any_valid, masked, torch.zeros_like(masked)), -1) * valid


def _max_pool_windows(scores: Tensor, block_count: int, stride_ratio: int) -> Tensor:
    """Official max-pool: block j takes windows [r*j-1, ..., r*j+r-1]."""
    shape = (*scores.shape[:-1], block_count)
    if scores.shape[-1] == 0:
        return scores.new_full(shape, -torch.inf)
    idx = (torch.arange(block_count, device=scores.device)[:, None] * stride_ratio
           + torch.arange(-1, stride_ratio, device=scores.device)[None])
    valid = (idx >= 0) & (idx < scores.shape[-1])
    return scores[..., idx.clamp(0, scores.shape[-1] - 1)].masked_fill(~valid, -torch.inf).amax(-1)


def mandatory_blocks(context: SelectionContext, block_count: Optional[int] = None) -> Tensor:
    s = context.settings
    count = block_count or math.ceil(context.k.shape[1] / s.block_size)
    ids = torch.arange(count, device=context.q.device)
    qb = context.query_positions[:, None] // s.block_size
    return ((ids < s.init_blocks) | ((ids <= qb) & (qb <= ids + s.local_blocks))) & (ids <= qb)


@torch.no_grad()
def _compute_cis_scores(context: SelectionContext) -> Tensor:
    """CIS branch only, reusable without recomputing native QK selection."""
    s, cis, pos = context.settings, context.cis, context.query_positions
    blocks = math.ceil(context.k.shape[1] / s.block_size)
    if cis.shape[1] < s.kernel_size:
        scores = cis.new_full((cis.shape[0], pos.numel(), blocks), -torch.inf, dtype=torch.float32)
    else:
        pooled = cis.unfold(1, s.kernel_size, s.kernel_stride).mean(-1)
        endpoints = torch.arange(pooled.shape[-1], device=cis.device) * s.kernel_stride + s.kernel_size - 1
        complete = endpoints[None] <= pos[:, None]
        per_query = pooled.float()[:, None].expand(-1, pos.numel(), -1).masked_fill(~complete, -torch.inf)
        scores = _max_pool_windows(per_query, blocks, s.block_size // s.kernel_stride)
    visible = torch.arange(blocks, device=cis.device)[None] <= pos[:, None] // s.block_size
    return scores.masked_fill(mandatory_blocks(context, blocks), torch.inf).masked_fill(~visible, -torch.inf)


cis_scores = _compute_cis_scores


@torch.no_grad()
def native_scores(context: SelectionContext) -> tuple[Tensor, Tensor, Tensor]:
    """Source-derived QK and CIS block scores plus [queries, blocks] anchors.

    Stage1 independently softmaxes each duplicated query head over completed
    compressed windows, then sums within its KV group (hence factor two).
    CIS is mean-pooled over the same windows then max-pooled to blocks.
    """
    q, k, cis, pos, s = context.q, context.k, context.cis, context.query_positions, context.settings
    kvh, length, dim = k.shape
    qh, queries, _ = q.shape
    if qh % kvh:
        raise ValueError("query heads must divide into contiguous KV groups")
    blocks = math.ceil(length / s.block_size)
    mandatory = mandatory_blocks(context, blocks)
    if length < s.kernel_size:
        blank = q.new_full((kvh, queries, blocks), -torch.inf, dtype=torch.float32)
        return blank.masked_fill(mandatory, torch.inf), blank.masked_fill(mandatory, torch.inf), mandatory
    ck = k.unfold(1, s.kernel_size, s.kernel_stride).mean(-1)
    endpoints = torch.arange(ck.shape[1], device=q.device) * s.kernel_stride + s.kernel_size - 1
    complete = endpoints[None, :] <= pos[:, None]
    qgroup = q.reshape(kvh, qh // kvh, queries, dim)
    logits = torch.einsum("hgqd,hmd->hgqm", qgroup.float(), ck.float()) / math.sqrt(dim)
    score = _safe_softmax(logits, complete[None, None]).sum(1) * 2
    ratio = s.block_size // s.kernel_stride
    qk_blocks = _max_pool_windows(score, blocks, ratio)
    visible = torch.arange(blocks, device=q.device)[None] <= pos[:, None] // s.block_size
    qk_blocks = qk_blocks.masked_fill(mandatory, torch.inf).masked_fill(~visible, -torch.inf)
    return qk_blocks, cis_scores(context), mandatory


def _stable_topk(scores: Tensor, count: int) -> Tensor:
    # A specified tie policy makes chunk/full equivalence testable. CUDA topk
    # tie ordering is not specified by the original implementation.
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]


@torch.no_grad()
def select_with_scores(context: SelectionContext, qk_scores: Tensor,
                       cis_scores: Optional[Tensor] = None) -> Tensor:
    """Apply the same QK/CIS quota and anchors to external QK block scores.

    ``cis_scores=None`` computes only the native CIS branch. QK score scale is
    arbitrary; only its ranking matters before the official union operation.
    """
    s = context.settings
    blocks = math.ceil(context.k.shape[1] / s.block_size)
    budget = min(s.topk, blocks)
    if blocks <= s.topk:
        selected = torch.arange(blocks, device=context.q.device).expand(context.k.shape[0], context.q.shape[1], -1)
    else:
        shape = (context.k.shape[0], context.q.shape[1], blocks)
        if qk_scores.shape != shape:
            raise ValueError(f"QK block scores must have shape {shape}")
        visible = torch.arange(blocks, device=context.q.device)[None] <= context.query_positions[:, None] // s.block_size
        qk = qk_scores.masked_fill(mandatory_blocks(context, blocks), torch.inf).masked_fill(~visible, -torch.inf)
        cis = _compute_cis_scores(context) if cis_scores is None else cis_scores
        if cis.shape != shape:
            raise ValueError(f"CIS block scores must have shape {shape}")
        cis = cis.masked_fill(mandatory_blocks(context, blocks), torch.inf).masked_fill(~visible, -torch.inf)
        qk_budget = min(s.init_blocks + s.local_blocks + s.select_blocks, blocks)
        chosen_qk = _stable_topk(qk, qk_budget)
        cis = cis.scatter(-1, chosen_qk, torch.inf)
        selected = _stable_topk(cis, budget).sort(-1).values
    future = selected > context.query_positions[None, :, None] // s.block_size
    return selected.masked_fill(future, -1).to(torch.long)


@torch.no_grad()
def native_select(context: SelectionContext) -> Tensor:
    if math.ceil(context.k.shape[1] / context.settings.block_size) <= context.settings.topk:
        # Scores are unused while every block fits into the support budget.
        return select_with_scores(context, context.q.new_empty(0))
    qk, cis, _ = native_scores(context)
    return select_with_scores(context, qk, cis)


def selected_causal_attention(context: SelectionContext, selected: Tensor) -> Tensor:
    """Gather only chosen blocks, preserving per-token CIS and GQA mapping.

    Treat query heads within a KV group as SDPA's query dimension; this avoids
    physically repeating each selected K/V eight times for the 2B checkpoint.
    Output is [query_heads, queries, head_dim].
    """
    q, k, v, cis, pos, s = context.q, context.k, context.v, context.cis, context.query_positions, context.settings
    kvh, length, dim = k.shape
    qh, queries, _ = q.shape
    if selected.ndim != 3 or tuple(selected.shape[:2]) != (kvh, queries):
        raise ValueError("selector must return [kv_heads, queries, selected_blocks]")
    if selected.dtype not in (torch.int32, torch.int64) or selected.device != q.device:
        raise ValueError("selector IDs must be integer tensors on the query device")
    if selected.shape[-1] == 0:
        raise ValueError("selector returned no blocks")
    # Reject duplicate *valid* blocks: duplicates change the softmax measure.
    ordered = selected.sort(-1).values
    if bool(((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any()):
        raise ValueError("selector returned duplicate blocks")
    if bool(((selected < -1) | (selected >= math.ceil(length / s.block_size))).any()):
        raise ValueError("selector returned an out-of-range block")
    heads = torch.arange(kvh, device=q.device)[:, None, None]
    group = qh // kvh
    query = q.reshape(kvh, group, queries, dim).permute(0, 2, 1, 3)
    outputs = []
    # Bound gather copies independently of the model/MLP prefill chunk size.
    # Flatten to 4D SDPA inputs so CUDA fused backends remain eligible.
    for begin in range(0, queries, s.attention_query_chunk_size):
        end = min(queries, begin + s.attention_query_chunk_size)
        picked = selected[:, begin:end]
        ids = picked[..., None] * s.block_size + torch.arange(s.block_size, device=q.device)
        valid = (picked[..., None] >= 0) & (ids < length) & (ids <= pos[None, begin:end, None, None])
        ids, valid = ids.clamp(0, length - 1).flatten(-2), valid.flatten(-2)
        if not bool(valid.any(-1).all()):
            raise ValueError("selector left a query with no causal keys")
        gathered_k, gathered_v = k[heads, ids], v[heads, ids]
        bias = cis[heads, ids].masked_fill(~valid, -torch.inf)
        batch = kvh * (end - begin)
        output = F.scaled_dot_product_attention(query[:, begin:end].reshape(batch, 1, group, dim),
            gathered_k.reshape(batch, 1, -1, dim), gathered_v.reshape(batch, 1, -1, dim),
            attn_mask=bias.reshape(batch, 1, 1, -1), dropout_p=0.0)
        outputs.append(output.reshape(kvh, end - begin, group, dim))
    return torch.cat(outputs, 1).permute(0, 2, 1, 3).reshape(qh, queries, dim)


def dense_causal_attention(context: SelectionContext) -> Tensor:
    """Same final operator over all keys; chunked callers avoid NxN memory."""
    q, k, v, cis = context.q, context.k, context.v, context.cis
    kvh, length, dim = k.shape
    group, queries = q.shape[0] // kvh, q.shape[1]
    query = q.reshape(kvh, group * queries, dim)
    valid = torch.arange(length, device=q.device)[None] <= context.query_positions[:, None]
    bias = cis[:, None, :].expand(-1, queries, -1).masked_fill(~valid, -torch.inf)
    bias = bias[:, None].expand(-1, group, -1, -1).reshape(kvh, group * queries, length)
    output = F.scaled_dot_product_attention(query[:, None], k[:, None], v[:, None],
                                           attn_mask=bias[:, None], dropout_p=0.0)
    return output.reshape(q.shape)


def _config(value):
    raw = dict(value) if isinstance(value, dict) else dict(vars(value))
    defaults = dict(attention_bias=False, mlp_bias=False, hidden_act="silu", rms_norm_eps=1e-6,
                    rope_theta=10000.0, rope_scaling=None, pad_token_id=None,
                    tie_word_embeddings=False, partial_rotary_factor=1.0)
    defaults.update(raw)
    defaults.setdefault("head_dim", defaults["hidden_size"] // defaults["num_attention_heads"])
    cfg = SimpleNamespace(**defaults)
    if cfg.hidden_act != "silu" or cfg.partial_rotary_factor != 1.0:
        raise ValueError("only the downloaded checkpoint's SiLU/full-RoPE architecture is supported")
    if cfg.num_attention_heads % cfg.num_key_value_heads or cfg.head_dim % 2:
        raise ValueError("invalid GQA or RoPE dimensions")
    return cfg


def rope_parameters(config, device):
    dim, base = config.head_dim, config.rope_theta
    factors, amplitude = 1.0, 1.0
    if config.rope_scaling:
        scaling = config.rope_scaling
        if scaling.get("rope_type", scaling.get("type")) != "longrope":
            raise ValueError("only default RoPE or the checkpoint's fixed LongRoPE is supported")
        if scaling["short_factor"] != scaling["long_factor"]:
            raise ValueError("variable LongRoPE frequencies need an explicit cache rerotation implementation")
        if len(scaling["short_factor"]) != dim // 2 or "attention_factor" not in scaling:
            raise ValueError("invalid or ambiguous LongRoPE configuration")
        factors = torch.tensor(scaling["short_factor"], device=device, dtype=torch.float32)
        amplitude = float(scaling["attention_factor"])
    frequency = 1.0 / (factors * base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return frequency, amplitude


def apply_rope(x: Tensor, positions: Tensor, inv_freq: Tensor, amplitude: float = 1.0) -> Tensor:
    # Source forces fp32 phase formation regardless of model dtype.
    angles = positions.float()[:, None] * inv_freq.float()[None]
    phases = torch.cat((angles, angles), -1)
    cos, sin = (phases.cos() * amplitude).to(x.dtype), (phases.sin() * amplitude).to(x.dtype)
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), -1) * sin


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        result = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * result.to(x.dtype)


class NosaAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.config, self.layer_idx = cfg, layer_idx
        h, kh, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, h * d, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kh * d, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kh * d, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(h * d, cfg.hidden_size, bias=cfg.attention_bias)
        self.A = nn.Parameter(torch.zeros(kh))
        self.delta = nn.Linear(kh * d, kh, bias=cfg.attention_bias)

    def forward(self, x, positions, inv_freq, amplitude, cache, selector, settings, trace_callback):
        cfg = self.config
        q = self.q_proj(x)[0].reshape(-1, cfg.num_attention_heads, cfg.head_dim).transpose(0, 1)
        k = self.k_proj(x)[0].reshape(-1, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 1)
        flat_v = self.v_proj(x)
        v = flat_v[0].reshape(-1, cfg.num_key_value_heads, cfg.head_dim).transpose(0, 1)
        cis = (self.A * F.softplus(self.delta(flat_v)))[0].transpose(0, 1).to(x.dtype)
        q, k = apply_rope(q, positions, inv_freq, amplitude), apply_rope(k, positions, inv_freq, amplitude)
        if cache is not None:
            k, v, cis = torch.cat((cache.k, k), 1), torch.cat((cache.v, v), 1), torch.cat((cache.cis, cis), 1)
        context = SelectionContext(q, k, v, cis, positions, self.layer_idx, settings, inv_freq, amplitude)
        selected = None if settings.dense else selector(context)
        if trace_callback is not None:
            trace_callback(context, selected)
        output = dense_causal_attention(context) if settings.dense else selected_causal_attention(context, selected)
        output = self.o_proj(output.transpose(0, 1).reshape(1, x.shape[1], -1))
        return output, LayerCache(k, v, cis)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = NosaAttention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, *args):
        attention, cache = self.self_attn(self.input_layernorm(x), *args)
        x = x + attention
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class NosaReferenceForCausalLM(nn.Module):
    backend_label = BACKEND_LABEL

    def __init__(self, config, selector: Optional[Callable] = None, settings: Optional[AttentionSettings] = None,
                 trace_callback: Optional[Callable] = None):
        super().__init__()
        self.config = cfg = _config(config)
        self.selector = selector if selector is not None else native_select
        self.settings = settings if settings is not None else AttentionSettings()
        self.trace_callback = trace_callback  # Optional (context, selected) observer; no retained tensors by default.
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.model.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_pretrained(cls, checkpoint_dir, *, device="cpu", dtype=torch.float32,
                        selector=None, settings=None, trace_callback=None):
        """Load local safetensors or weights-only PyTorch, with exact coverage.

        Meta construction plus one-shard-at-a-time loading avoids allocating a
        second full initialized model. Missing A/delta/decoder tensors are fatal.
        """
        directory = Path(checkpoint_dir)
        with (directory / "config.json").open() as stream:
            config = json.load(stream)
        with torch.device("meta"):
            model = cls(config, selector=selector, settings=settings, trace_callback=trace_callback)
        expected = model.state_dict()
        safe_index = directory / "model.safetensors.index.json"
        bin_index = directory / "pytorch_model.bin.index.json"
        if safe_index.exists():
            with safe_index.open() as stream:
                files = sorted(set(json.load(stream)["weight_map"].values()))
        elif (directory / "model.safetensors").exists():
            files = ["model.safetensors"]
        elif bin_index.exists():
            with bin_index.open() as stream:
                files = sorted(set(json.load(stream)["weight_map"].values()))
        elif (directory / "pytorch_model.bin").exists():
            files = ["pytorch_model.bin"]
        else:
            raise FileNotFoundError("no local safetensors or pytorch_model.bin checkpoint found")
        seen = set()

        def assign(name, value):
            if name not in expected or name in seen:
                raise ValueError(f"unexpected or duplicated checkpoint tensor: {name}")
            if not isinstance(value, Tensor) or value.shape != expected[name].shape:
                shape = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
                raise ValueError(f"checkpoint shape mismatch: {name}: {shape} != {tuple(expected[name].shape)}")
            parent_name, leaf = name.rsplit(".", 1)
            owner = model.get_submodule(parent_name)
            setattr(owner, leaf, nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False))
            seen.add(name)

        for filename in files:
            if filename.endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(directory / filename, framework="pt", device="cpu") as shard:
                    for name in shard.keys():
                        assign(name, shard.get_tensor(name))
            elif filename.endswith(".bin"):
                try:
                    state = torch.load(directory / filename, map_location="cpu", weights_only=True, mmap=True)
                except RuntimeError as error:
                    # Legacy torch.save files cannot be memory-mapped. The
                    # restricted weights-only unpickler remains mandatory.
                    if "mmap can only be used" not in str(error):
                        raise
                    state = torch.load(directory / filename, map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    raise ValueError("PyTorch checkpoint must be a flat state_dict")
                for name in list(state):
                    assign(name, state.pop(name))  # Release each CPU tensor after device copy.
                del state
            else:
                raise ValueError(f"unsupported checkpoint shard format: {filename}")
        if model.config.tie_word_embeddings and "lm_head.weight" not in seen:
            model.lm_head.weight = model.model.embed_tokens.weight
            seen.add("lm_head.weight")
        missing = set(expected) - seen
        if missing:
            raise ValueError(f"missing checkpoint tensors ({len(missing)}): {sorted(missing)[:8]}")
        model.load_report = {"backend": BACKEND_LABEL, "loaded_tensors": len(seen),
                             "layers": len(model.model.layers), "checkpoint_dir": str(directory.resolve()),
                             "checkpoint_files": files, "pytorch_weights_only": True,
                             "missing_tensors": [], "unexpected_tensors": []}
        return model.eval()

    def forward(self, input_ids, *, past_key_values=None, use_cache=True, num_logits_to_keep=0):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("reference accepts nonempty unpadded batch-one token IDs")
        if num_logits_to_keep < 0:
            raise ValueError("num_logits_to_keep must be nonnegative")
        if past_key_values is not None and len(past_key_values) != len(self.model.layers):
            raise ValueError("cache layer count does not match model")
        past = 0 if past_key_values is None else past_key_values[0].k.shape[1]
        if past_key_values is not None and any(entry.k.shape[1] != past for entry in past_key_values):
            raise ValueError("cache prefix lengths disagree across layers")
        positions = torch.arange(past, past + input_ids.shape[1], device=input_ids.device)
        inv_freq, amplitude = rope_parameters(self.config, input_ids.device)
        hidden = self.model.embed_tokens(input_ids)
        updated = []
        for idx, layer in enumerate(self.model.layers):
            cache = None if past_key_values is None else past_key_values[idx]
            hidden, new_cache = layer(hidden, positions, inv_freq, amplitude, cache, self.selector,
                                      self.settings, self.trace_callback)
            if use_cache:
                updated.append(new_cache)
        hidden = self.model.norm(hidden)
        if num_logits_to_keep:
            hidden = hidden[:, -num_logits_to_keep:]
        return NosaOutput(self.lm_head(hidden).float(), updated if use_cache else None)

    @torch.inference_mode()
    def prefill(self, input_ids, *, chunk_size=128):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if input_ids.shape[-1] < 1:
            raise ValueError("prefill requires at least one token")
        output = None
        for start in range(0, input_ids.shape[1], chunk_size):
            output = self(input_ids[:, start:start + chunk_size],
                          past_key_values=None if output is None else output.past_key_values,
                          use_cache=True, num_logits_to_keep=1)
        return output

    @torch.inference_mode()
    def greedy_generate(self, input_ids, *, max_new_tokens=32, eos_token_id=None, chunk_size=128):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if max_new_tokens == 0:
            return input_ids.clone()
        eos = getattr(self.config, "eos_token_id", None) if eos_token_id is None else eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else ([] if eos is None else [eos]))
        output = self.prefill(input_ids, chunk_size=chunk_size)
        generated = [input_ids]
        for step in range(max_new_tokens):
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            if token.item() in eos_ids or step + 1 == max_new_tokens:
                break
            output = self(token, past_key_values=output.past_key_values, use_cache=True, num_logits_to_keep=1)
        return torch.cat(generated, -1)
