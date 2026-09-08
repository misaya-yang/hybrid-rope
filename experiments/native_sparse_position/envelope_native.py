"""Cached-prefix adapter for the mean_native Oracle's unchanged exact reader.

Construct Oracle before prefix prefill, call build(oracle, omega=... or K=...)
after prefill, then reset each method against an independent copy of that cache.
Only complete prefix blocks are indexed. If newly generated blocks reach beyond
the local window, fail explicitly rather than silently retaining stale summaries.
Diagnostics are opt-in: they scan complete prefix keys at up to five configured
question positions in first/middle/last attention layers, with separate costs.

Local import of mean_native is deferred so CPU adapter tests do not require the
newer Transformers Qwen3.5 implementation. Those tests do not qualify native
model parity or GPU performance.
"""
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

if __package__:
    from .pair_envelope import build_pair_envelope, score_pair_envelope, build_quest, score_quest
else:
    from pair_envelope import build_pair_envelope, score_pair_envelope, build_quest, score_quest


SELECTORS = ("RoPEMean", "Quest", "PairEnvelope", "RandomPair", "QuestSplit32")
DEFAULT_BUILD = ("Quest", "PairEnvelope", "RandomPair", "QuestSplit32")


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _empty_diagnostic_stats():
    return {"calls": 0, "key_bytes_read": 0, "fp32_key_materialization_bytes": 0, "seconds": 0.0}


class EnvelopeMixin:
    """Mixin over mean_native.Oracle; CPU tests substitute only its base class."""
    def __init__(self, *args, omega=None, K=None, **kwargs):
        self.prefix_post = {}
        self.prefix_shape_info = {}
        self.selector_caches = {}
        self.build_info = {}
        self._prefix_generation = 0
        self._built_generation = -1
        self._configured_omega, self._configured_K = omega, K
        self._diagnostic_config = None
        self.diagnostic_records = []
        self.diagnostic_stats = _empty_diagnostic_stats()
        self._diagnostic_seen = set()
        super().__init__(*args, **kwargs)
        if self.block != 64 or self.local < self.block or self.topk < 0:
            raise ValueError("This adapter requires B=64, local>=64 and nonnegative topk")
        indices = sorted(self.layers)
        if not indices:
            raise ValueError("No supported full-attention layers")
        self.diagnostic_layers = sorted(set((indices[0], indices[len(indices)//2], indices[-1])))
        self._first_attention_layer = indices[0]

    def build(self, omega=None, *, K=None, modes=DEFAULT_BUILD):
        return build(self, omega, K=K, modes=modes)

    def reset(self, mode, prefix_length):
        if mode not in ("Dense",) + SELECTORS:
            raise ValueError("Unsupported envelope mode: " + mode)
        super().reset(mode, prefix_length)
        self._run_prefix_length = int(prefix_length)
        self.diagnostic_records = []
        self.diagnostic_stats = _empty_diagnostic_stats()
        self._diagnostic_seen.clear()

    def configure_diagnostics(self, source_blocks, question_last_position=None, *,
                              query_positions=None, callback=None, layers=None, compare_modes=None,
                              include_query=False, top_competitors=16, row_id=None):
        """Enable bounded last-question diagnostics; pass source_blocks=None to disable.

        question_last_position is the absolute cache index (input_tokens - 1),
        not the count of question tokens. query_positions may add/replace it with
        up to five absolute positions (e.g. queried-key terminal subtokens).
        compare_modes=None diagnoses the
        active selector only (RoPEMean as the score reference on Dense).
        Set compare_modes=SELECTORS to compare all cached descriptors on the
        same failed-trajectory query without changing its selected blocks.
        """
        if source_blocks is None:
            self._diagnostic_config = None
            return
        sources = sorted(set(int(b) for b in source_blocks))
        positions = set(int(p) for p in (() if query_positions is None else query_positions))
        if question_last_position is not None:
            positions.add(int(question_last_position))
        if (any(b < 0 for b in sources) or not positions or min(positions) < 0
                or len(positions) > 5 or top_competitors < 0):
            raise ValueError("Need one to five nonnegative diagnostic positions and valid limits")
        layers = self.diagnostic_layers if layers is None else list(layers)
        if not set(layers).issubset(self.layers):
            raise ValueError("Diagnostic layer is not a supported attention layer")
        if compare_modes is not None and not set(compare_modes).issubset(SELECTORS):
            raise ValueError("Unknown diagnostic selector")
        self._diagnostic_config = SimpleNamespace(
            sources=sources, positions=positions, callback=callback,
            layers=set(layers), compare_modes=None if compare_modes is None else tuple(compare_modes),
            include_query=bool(include_query), competitors=int(top_competitors), row_id=row_id)
        self._diagnostic_seen.clear()

    def pop_diagnostics(self):
        records, self.diagnostic_records = self.diagnostic_records, []
        return records

    def _diagnostic_due(self, layer):
        cfg = self._diagnostic_config
        return (cfg is not None and layer in cfg.layers and self.pos in cfg.positions
                and (self.mode, layer, self.pos) not in self._diagnostic_seen)

    def _cached_scores(self, layer, qs, mode):
        if self._built_generation != self._prefix_generation:
            raise RuntimeError("Call build(oracle, K/omega) after this prefix and before remote selection")
        if self._run_prefix_length != self._built_prefix_length:
            raise RuntimeError("reset prefix length differs from the indexed prefix")
        caches = self.selector_caches[layer]
        if mode not in caches:
            raise RuntimeError("Selector was not built: " + mode)
        cache = caches[mode]
        if mode == "RoPEMean":
            kv, nb, dim = cache.shape
            return torch.einsum("kgd,kbd->kgb", qs.reshape(kv, -1, dim), cache).reshape(qs.shape[0], nb)
        if mode in ("PairEnvelope", "RandomPair"):
            return score_pair_envelope(qs, cache)[0]
        scores = score_quest(qs, cache)[0]
        return scores.reshape(scores.shape[0], -1, 2).amax(-1) if mode == "QuestSplit32" else scores

    def _eligible(self, N, device):
        nb = self._built_prefix_length // self.block if self._built_generation == self._prefix_generation else 0
        remote_complete = max(0, (N - self.local) // self.block)
        if remote_complete > nb:
            raise RuntimeError("Post-prefix blocks reached remote selection; this prefix-only cache must be extended explicitly")
        starts = torch.arange(nb, device=device) * self.block
        return (starts > 0) & (starts + self.block <= N - self.local)

    def _choose(self, scores, eligible):
        count = min(self.topk, int(eligible.sum()))
        return scores.masked_fill(~eligible[None], -torch.inf).topk(count, -1).indices

    def interface(self, module, q, k, v, mask, **kw):
        if module.layer_idx not in self.layers:
            return super().interface(module, q, k, v, mask, **kw)
        layer = module.layer_idx
        if self.mode == "prefix":
            if layer == self._first_attention_layer:
                self.prefix_post.clear()
                self.prefix_shape_info.clear()
                self.selector_caches.clear()
                self.build_info.clear()
                self._prefix_generation += 1
                self._built_generation = -1
            self.prefix_post[layer] = k.detach()  # Reference only; no extra full-K copy.
            self.prefix_shape_info[layer] = {
                "query_heads": q.shape[1], "kv_heads": k.shape[1],
                "key_dim": k.shape[-1], "value_dim": v.shape[-1],
                "key_element_bytes": k.element_size(), "value_element_bytes": v.element_size()}
            return super().interface(module, q, k, v, mask, **kw)
        if q.shape[0] != 1 or q.shape[2] != 1 or k.shape[2] != self.pos + 1:
            raise RuntimeError("Only batch-one, one-token causal continuation is qualified")
        N, H, KV, D = k.shape[2], q.shape[1], k.shape[1], q.shape[-1]
        if H % KV:
            raise RuntimeError("Query heads must be contiguous GQA groups")
        diagnostic = self._diagnostic_due(layer)
        if self.mode == "Dense" and not diagnostic:
            return super().interface(module, q, k, v, mask, **kw)
        qs = q[0, :, 0].float() * float(module.scaling)
        # Qualification before build is safe when every key is mandatory.
        remote_possible = (N - self.local) // self.block > 1
        if remote_possible or diagnostic:
            if self._built_generation != self._prefix_generation:
                raise RuntimeError("Descriptors must be built before remote selection/diagnostics")
            eligible = self._eligible(N, q.device)
            score_mode = "RoPEMean" if self.mode == "Dense" else self.mode
            scores = self._cached_scores(layer, qs, score_mode)
            chosen = self._choose(scores, eligible)
        else:
            eligible = torch.zeros(0, dtype=torch.bool, device=q.device)
            scores = q.new_empty(H, 0, dtype=torch.float32)
            chosen = torch.empty(H, 0, dtype=torch.long, device=q.device)
        if self.mode == "Dense":
            if diagnostic:
                self._diagnose(layer, qs, k, scores, chosen, eligible, dense=True)
            return super().interface(module, q, k, v, mask, **kw)
        self.calls += 1
        remote = (chosen[..., None] * self.block + torch.arange(self.block, device=q.device)).reshape(H, -1)
        mandatory = torch.cat((torch.arange(min(self.block, N), device=q.device),
                               torch.arange(max(self.block, N-self.local), N, device=q.device)))
        ids = torch.cat((remote, mandatory[None].expand(H, -1)), dim=-1).sort(-1).values
        if self.calls <= len(self.layers):
            if not bool((ids[:, 1:] > ids[:, :-1]).all()) or int(ids.max()) >= N:
                raise RuntimeError("Duplicate/future selected index")
        if diagnostic:
            self._diagnose(layer, qs, k, scores, chosen, eligible, dense=False)
        heads = torch.arange(H, device=q.device) // (H // KV)
        # This is the original mean_native gather and exact SDPA reader.
        key = k[0, heads[:, None], ids][None]
        value = v[0, heads[:, None], ids][None]
        out = F.scaled_dot_product_attention(q, key, value, is_causal=False,
                                            dropout_p=0., scale=float(module.scaling))
        return out.transpose(1, 2).contiguous(), None

    def _diagnose(self, layer, qs, k, active_scores, active_chosen, eligible, *, dense):
        cfg = self._diagnostic_config
        self._diagnostic_seen.add((self.mode, layer, self.pos))
        _sync(k)
        start = time.perf_counter()
        KV, N, D = k.shape[1:]
        H, nb = qs.shape[0], eligible.numel()
        cut = nb * self.block
        # ONLY this opt-in path materializes all complete prefix keys in FP32.
        diagnostic_keys = k[0, :, :cut].float().reshape(KV, nb, self.block, D)
        exact = torch.einsum("kgd,kbtd->kgbt", qs.reshape(KV, H//KV, D), diagnostic_keys).reshape(H, nb, self.block)
        maxima = exact.amax(-1)
        logmass = torch.logsumexp(exact, -1)
        # Transfer bounded score arrays once, not one CUDA sync per JSON scalar.
        maxima, logmass = maxima.cpu(), logmass.cpu()
        eligible_cpu = eligible.cpu()
        modes = cfg.compare_modes or (("RoPEMean" if dense else self.mode),)
        record = {"row_id": cfg.row_id, "trajectory_method": self.mode, "layer": layer,
                  "position": self.pos, "source_blocks": cfg.sources,
                  "selection_granularity": "per_query_head_no_head_aggregation",
                  "scanned_complete_prefix_blocks": nb, "comparisons": {}}
        if cfg.include_query:
            record["queries_scaled"] = qs.detach().cpu().tolist()
        for mode in modes:
            scores = active_scores if mode == ("RoPEMean" if dense else self.mode) else self._cached_scores(layer, qs, mode)
            chosen = active_chosen if mode == ("RoPEMean" if dense else self.mode) else self._choose(scores, eligible)
            scores, chosen = scores.cpu(), chosen.cpu()
            rows = []
            for h in range(H):
                selected = set(chosen[h].tolist())
                threshold = float(scores[h, chosen[h]].min()) if selected else None
                def cell(b):
                    if not 0 <= b < nb:
                        return {"block": b, "indexed": False}
                    val = scores[h, b]
                    valid = bool(eligible_cpu[b])
                    rank_min = int(((scores[h] > val) & eligible_cpu).sum()) + 1 if valid else None
                    rank_max = int(((scores[h] >= val) & eligible_cpu).sum()) if valid else None
                    mandatory_tokens = max(0, min((b+1)*self.block, N) - max(b*self.block, N-self.local))
                    if b == 0:
                        mandatory_tokens = min(self.block, N)
                    remote = b in selected
                    active = mode == ("RoPEMean" if dense else self.mode)
                    read_tokens = self.block if remote else mandatory_tokens
                    actual_tokens = self.block if dense else (read_tokens if active else None)
                    mx = float(maxima[h, b])
                    return {"block": b, "indexed": True, "eligible": valid,
                            "score": float(val), "rank_min": rank_min, "rank_max": rank_max,
                            "threshold_margin": float(val)-threshold if threshold is not None and valid else None,
                            "true_max": mx, "true_logmass": float(logmass[h, b]),
                            "upper_bound_inflation": float(val)-mx if mode != "RoPEMean" else None,
                            "true_max_rank_min": int(((maxima[h] > maxima[h, b]) & eligible_cpu).sum())+1 if valid else None,
                            "selected_remote": remote, "selected_token_count": read_tokens,
                            "actual_reader_selected_token_count": actual_tokens,
                            "fully_selected": read_tokens >= self.block}
                rows.append({"head": h, "remote_count": len(selected),
                             "sources": [cell(b) for b in cfg.sources],
                             "selected_competitors": [cell(int(b)) for b in chosen[h, :cfg.competitors]]})
            record["comparisons"][mode] = rows
        _sync(k)
        cost = {"calls": 1, "key_bytes_read": KV*cut*D*k.element_size(),
                "fp32_key_materialization_bytes": 0 if k.dtype == torch.float32 else KV*cut*D*4,
                "seconds": time.perf_counter()-start}
        record["diagnostic_cost"] = cost
        for key, val in cost.items():
            self.diagnostic_stats[key] += val
        self.diagnostic_records.append(record)
        if cfg.callback is not None:
            cfg.callback(record)


def Oracle(model, block=64, local=2048, topk=16, **kwargs):
    """Create before prefix prefill; preserves the reference Oracle's hooks."""
    if __package__:
        from .mean_native import Oracle as MeanOracle
    else:
        from mean_native import Oracle as MeanOracle
    class NativeEnvelopeOracle(EnvelopeMixin, MeanOracle):
        pass
    return NativeEnvelopeOracle(model, block=block, local=local, topk=topk, **kwargs)


def static_read_accounting(oracle, layer, mode, total_length=None):
    """Logical byte counts, not profiling; never scans/uniques selected indices.

    All query heads score all complete-prefix descriptors, even ineligible ones.
    Actual selected-KV union depends on head overlap, so report rigorous bounds
    instead of pretending the gather shares all KV reads. Call off the hot path.
    """
    shape = oracle.prefix_shape_info[layer]
    H, KV = shape["query_heads"], shape["kv_heads"]
    if H % KV:
        raise ValueError("Query heads must be contiguous GQA groups")
    G = H // KV
    N = oracle._built_prefix_length + 1 if total_length is None else int(total_length)
    nb = oracle._built_prefix_length // oracle.block
    if N < oracle._built_prefix_length or (N-oracle.local)//oracle.block > nb:
        raise ValueError("Length outside the prefix-only index contract")
    available = max(0, min(nb, (N-oracle.local)//oracle.block)-1)
    remote = min(oracle.topk, available)
    mandatory = min(oracle.block, N) + max(0, N-max(oracle.block, N-oracle.local))
    token_bytes = shape["key_dim"]*shape["key_element_bytes"] + shape["value_dim"]*shape["value_element_bytes"]
    cache = oracle.selector_caches[layer][mode]
    if mode == "RoPEMean":
        descriptor_bytes = cache.numel()*cache.element_size()
        indices_bytes = 0
    else:
        info = cache.byte_info()
        descriptor_bytes = info["descriptor_bytes"]
        indices_bytes = info["shared_pair_index_bytes"]
    return {
        "scope": "Static logical references at one token/layer; not measured DRAM/HBM traffic",
        "N": N, "query_heads": H, "kv_heads": KV, "queries_per_kv_head": G,
        "indexed_physical_blocks": nb, "eligible_remote_blocks": available,
        "remote_blocks_per_query_head": remote, "mandatory_tokens_per_query_head": mandatory,
        "normal_selector_invoked": available > 0,
        "descriptor_bytes_per_score_query_head_references": descriptor_bytes*G + indices_bytes*H,
        "descriptor_bytes_per_score_unique_union": descriptor_bytes + indices_bytes,
        "descriptor_index_note": "Shared pair-index bytes counted once per logical query head, once in the unique union",
        "selected_kv_bytes_query_head_references": H*(mandatory+remote*oracle.block)*token_bytes,
        "selected_kv_bytes_unique_gqa_union_lower_bound": KV*(mandatory+remote*oracle.block)*token_bytes,
        "selected_kv_bytes_unique_gqa_union_upper_bound": KV*(mandatory+min(available,G*remote)*oracle.block)*token_bytes,
        "selected_union_exact": False,
        "selected_union_exact_formula": "sum_kv [mandatory_tokens+B*size(union of that KV group's selected remote blocks)]*(Dk*key_bytes+Dv*value_bytes)",
        "reader_token_bytes": token_bytes,
        "excludes": "query/score/index-gather output buffers, cache build, optional diagnostics, hardware reuse effects",
    }


@torch.no_grad()
def build(oracle, omega=None, *, K=None, modes=DEFAULT_BUILD):
    """Build once after prefill; build(oracle, K_integer) is also supported.

    All comparisons may coexist in memory for the experiment. Metadata reports
    both that total resident storage and each independently deployable cache.
    RoPEMean is always included to reproduce the failed trajectory cheaply.
    """
    if isinstance(omega, int):
        if K is not None and K != omega:
            raise ValueError("Conflicting K arguments")
        K, omega = omega, None
    if omega is None and K is None:
        omega, K = oracle._configured_omega, oracle._configured_K
    if set(modes) - set(DEFAULT_BUILD):
        raise ValueError("Unknown build selector")
    if set(oracle.prefix_post) != set(oracle.layers):
        raise RuntimeError("Prefill every supported layer before build")
    lengths = {k.shape[2] for k in oracle.prefix_post.values()}
    if len(lengths) != 1:
        raise RuntimeError("Prefix lengths differ across layers")
    prefix_length = lengths.pop()
    if prefix_length < oracle.block:
        raise ValueError("At least one complete prefix block is needed for descriptor build")
    oracle.selector_caches.clear()
    oracle._built_generation = -1
    oracle.build_info = {}
    info = {"prefix_tokens": prefix_length, "block": oracle.block,
            "remote_topk_per_query_head": oracle.topk, "local": oracle.local,
            "sink": oracle.block, "layers": {}, "total_resident_descriptor_bytes": 0,
            "seconds": 0.0, "per_method_build_seconds": {mode: 0.0 for mode in ("RoPEMean",)+tuple(modes)},
            "prefix_post_is_reference": True,
            "cost_boundary": "Build and optional diagnostics excluded from continuation-only timings; report separately"}
    for layer, original_k in sorted(oracle.prefix_post.items()):
        if original_k.shape[0] != 1:
            raise ValueError("Only batch-one prefix caches are supported")
        KV, _, D = original_k.shape[1:]
        nb = prefix_length // oracle.block
        keys = original_k[0, :, :nb*oracle.block].reshape(KV, nb, oracle.block, D)
        _sync(original_k)
        started = time.perf_counter()
        mean_started = time.perf_counter()
        mean = keys.float().mean(2)
        _sync(original_k)
        mean_seconds = time.perf_counter() - mean_started
        info["per_method_build_seconds"]["RoPEMean"] += mean_seconds
        caches = {"RoPEMean": mean}
        layer_info = {"RoPEMean": {"total_tensor_bytes": mean.numel()*mean.element_size(),
                                   "bytes_per_physical_kv_block": D*4, "cache_dtype": "float32",
                                   "build_seconds": mean_seconds}}
        for mode in modes:
            _sync(original_k)
            method_started = time.perf_counter()
            if mode == "Quest":
                cache = build_quest(keys)
            elif mode == "QuestSplit32":
                cache = build_quest(keys.reshape(KV, nb*2, 32, D))
            else:
                cache = build_pair_envelope(keys, omega, K=K,
                    pairing="native" if mode == "PairEnvelope" else "random")
            _sync(original_k)
            method_seconds = time.perf_counter() - method_started
            info["per_method_build_seconds"][mode] += method_seconds
            caches[mode] = cache
            item = cache.byte_info()
            item["build_seconds"] = method_seconds
            item["physical_blocks"] = nb
            item["bytes_per_physical_kv_block"] = item["descriptor_bytes"] / (KV*nb)
            if mode == "QuestSplit32":
                item["aggregation"] = "max of two contiguous 32-token subpage upper bounds; 4D floats/physical64 block"
            layer_info[mode] = item
        _sync(original_k)
        elapsed = time.perf_counter() - started
        oracle.selector_caches[layer] = caches
        info["layers"][layer] = {"methods": layer_info, "seconds": elapsed}
        info["seconds"] += elapsed
        info["total_resident_descriptor_bytes"] += sum(x["total_tensor_bytes"] for x in layer_info.values())
    oracle._built_prefix_length = prefix_length
    oracle._built_generation = oracle._prefix_generation
    info["build_bookkeeping_seconds"] = info["seconds"] - sum(info["per_method_build_seconds"].values())
    info["timing_note"] = "Every method construction is bracketed by device synchronization; total layer time also includes metadata/bookkeeping"
    for layer in oracle.selector_caches:
        for mode, item in info["layers"][layer]["methods"].items():
            item["static_reads_at_first_question_token"] = static_read_accounting(oracle, layer, mode)
    oracle.build_info = info
    return info
