#!/usr/bin/env python3
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

MODEL_PATH = "/opt/dfrope/models_alt/LLM-Research/Meta-Llama-3-8B-Instruct"
HYBRID_LORA_PATH = "/opt/dfrope/results/llama3_hybrid_lora/final_lora"
OUT_DIR = "/opt/dfrope/results/llama3_hybrid_lora_eval"
OUT_JSON = f"{OUT_DIR}/followup_mechanism_p0.json"
OUT_MD = f"{OUT_DIR}/FOLLOWUP_MECHANISM_P0_SUMMARY.md"

SEED = 42
VAL_TOKENS = 1_200_000
N_CHUNKS = 5
MAIN_LENGTHS = [12288, 14336, 16384]
STRATEGIES = ["sequential", "random_start"]
ROBUST_SUBSET_SIZE = 300_000
ROBUST_NUM_SUBSETS = 3
ROBUST_LENGTH = 16384
LOSS_CURVE_WINDOW = 128
ATTN_SINK_PREFIX = 128
ATTN_RECENT_WINDOW = 512
ATTN_LONG_RANGE = 4096
TOK_CACHE = f"{OUT_DIR}/c4_val_tokens_{VAL_TOKENS}.pt"



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def geometric_freq(K, theta):
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))



def anchored_poly_freq(K, theta_base, p=3.9, omf=0.3):
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = k / (K - 1)
    log_omega = math.log(omega_max) + (t ** p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)



def hybrid_freq(freq_a, freq_b, alpha):
    return (1 - alpha) * freq_a + alpha * freq_b



def build_hybrid_inv_freq(head_dim):
    K = head_dim // 2
    geo_100k = geometric_freq(K, 100000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    return hybrid_freq(geo_100k, poly_100k, alpha=0.2)



def patch_model_rope(model):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv = build_hybrid_inv_freq(head_dim)

    cands = []
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        cands.append(model.model.rotary_emb)

    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if hasattr(attn, "rotary_emb"):
            cands.append(attn.rotary_emb)
        if hasattr(attn, "rotary_fn"):
            rf = attn.rotary_fn
            if hasattr(rf, "inv_freq"):
                cands.append(rf)
            if hasattr(rf, "rotary_emb") and hasattr(rf.rotary_emb, "inv_freq"):
                cands.append(rf.rotary_emb)

    seen = set()
    patched = 0
    for rope in cands:
        if id(rope) in seen or not hasattr(rope, "inv_freq"):
            continue
        seen.add(id(rope))
        rope.inv_freq = inv.to(device=rope.inv_freq.device, dtype=rope.inv_freq.dtype)
        if hasattr(rope, "max_seq_len_cached"):
            rope.max_seq_len_cached = 0
        patched += 1

    if patched == 0:
        raise RuntimeError("No rotary module with inv_freq found")



def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok



def load_val_tokens(tokenizer, max_tokens=VAL_TOKENS):
    cache_path = Path(TOK_CACHE)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        cached = cached.to(dtype=torch.long)
        if len(cached) >= max_tokens:
            print(f"[Data] using cached tokens: {cache_path}", flush=True)
            return cached[:max_tokens].clone()

    last_err = None
    for attempt in range(1, 6):
        try:
            ids = []
            ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            for x in ds:
                txt = x.get("text")
                if not txt:
                    continue
                ids.extend(tokenizer.encode(txt, add_special_tokens=False))
                if len(ids) >= max_tokens:
                    break
            tokens = torch.tensor(ids[:max_tokens], dtype=torch.long)
            torch.save(tokens, cache_path)
            print(f"[Data] cached tokens saved: {cache_path}", flush=True)
            return tokens
        except Exception as e:
            last_err = e
            print(f"[Data] C4 load attempt {attempt}/5 failed: {e}", flush=True)
            time.sleep(3 * attempt)

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu").to(dtype=torch.long)
        if len(cached) >= max_tokens:
            print(f"[Data] fallback to cached tokens after failures: {cache_path}", flush=True)
            return cached[:max_tokens].clone()

    raise RuntimeError(f"Failed to load validation tokens after retries: {last_err}")



def load_base_model():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    return model



def load_variant(variant_name):
    model = load_base_model()
    if variant_name in ("base_hybridfreq", "lora_hybridfreq"):
        patch_model_rope(model)
    if variant_name in ("lora_origfreq", "lora_hybridfreq"):
        model = PeftModel.from_pretrained(model, HYBRID_LORA_PATH)
        model.eval()
    return model



def get_starts(total_tokens, L, n_chunks, strategy, seed):
    max_start = total_tokens - L
    if max_start < 0:
        return []

    if strategy == "sequential":
        starts = []
        for i in range(n_chunks):
            s = i * L
            if s <= max_start:
                starts.append(s)
        return starts

    if strategy == "random_start":
        pop = max_start + 1
        k = min(pop, n_chunks)
        rng = random.Random(seed + L + total_tokens)
        if k == pop:
            starts = list(range(pop))
        else:
            starts = rng.sample(range(pop), k)
        starts.sort()
        return starts

    raise ValueError(f"Unknown strategy: {strategy}")



def eval_ppl_on_tokens(model, tokens, lengths, strategies, n_chunks, seed):
    out = {}
    anomalies = []
    device = model.device
    model.eval()

    with torch.no_grad():
        for strategy in strategies:
            out[strategy] = {}
            for L in lengths:
                starts = get_starts(len(tokens), L, n_chunks, strategy, seed)
                losses = []
                bad_batches = []
                for batch_idx, start in enumerate(starts):
                    chunk = tokens[start:start + L].unsqueeze(0).to(device)
                    inp = chunk[:, :-1]
                    tgt = chunk[:, 1:]

                    logits = model(inp).logits
                    logits_finite = bool(torch.isfinite(logits).all().item())

                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                    loss_finite = bool(torch.isfinite(loss).item())
                    loss_val = float(loss.item()) if loss_finite else None

                    if logits_finite and loss_finite:
                        losses.append(loss_val)
                    else:
                        bad = {
                            "length": int(L),
                            "strategy": strategy,
                            "batch_idx": int(batch_idx),
                            "start": int(start),
                            "logits_finite": logits_finite,
                            "loss_finite": loss_finite,
                        }
                        bad_batches.append(bad)
                        anomalies.append(bad)

                    del chunk, inp, tgt, logits, loss

                if losses:
                    mean_loss = sum(losses) / len(losses)
                    ppl = math.exp(mean_loss)
                    std = (sum((math.exp(x) - ppl) ** 2 for x in losses) / len(losses)) ** 0.5
                else:
                    ppl, std = None, None

                out[strategy][str(L)] = {
                    "n_chunks": len(starts),
                    "all_finite": len(bad_batches) == 0,
                    "ppl": round(ppl, 3) if ppl is not None else None,
                    "std": round(std, 3) if std is not None else None,
                    "bad_batches": bad_batches,
                }

    return out, anomalies



def smooth_series(arr, window):
    if len(arr) < window:
        return np.array(arr, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.asarray(arr, dtype=np.float64), kernel, mode="valid")



def summarize_loss_curve(token_nll):
    arr = np.asarray(token_nll, dtype=np.float64)
    smooth = smooth_series(arr, LOSS_CURVE_WINDOW)
    if smooth.size < 3:
        return {
            "n_tokens": int(arr.size),
            "mean_nll": float(arr.mean()) if arr.size else None,
            "abrupt_jump_detected": False,
        }

    diff = np.diff(smooth)
    max_jump_idx = int(np.argmax(diff))
    max_jump = float(diff[max_jump_idx])
    robust_scale = float(np.median(np.abs(diff - np.median(diff))) + 1e-8)
    jump_ratio = float(max_jump / robust_scale)

    head_n = min(2048, arr.size // 3)
    tail_n = min(2048, arr.size // 3)
    head_mean = float(arr[:head_n].mean()) if head_n > 0 else None
    tail_mean = float(arr[-tail_n:].mean()) if tail_n > 0 else None
    tail_over_head = float(tail_mean / head_mean) if (head_mean and tail_mean) else None

    abrupt = bool(jump_ratio > 8.0 and max_jump_idx > 2048 and (tail_over_head is not None and tail_over_head > 1.5))
    return {
        "n_tokens": int(arr.size),
        "mean_nll": float(arr.mean()),
        "head_mean_nll": head_mean,
        "tail_mean_nll": tail_mean,
        "tail_over_head": tail_over_head,
        "max_jump_idx_smoothed": max_jump_idx,
        "max_jump_value": max_jump,
        "max_jump_over_robust_scale": jump_ratio,
        "abrupt_jump_detected": abrupt,
    }



def token_loss_curve(model, tokens, length, start=0):
    device = model.device
    with torch.no_grad():
        chunk = tokens[start:start + length].unsqueeze(0).to(device)
        inp = chunk[:, :-1]
        tgt = chunk[:, 1:]
        logits = model(inp).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)

    out = token_nll.detach().cpu().tolist()
    summary = summarize_loss_curve(out)
    return {
        "length": int(length),
        "start": int(start),
        "summary": summary,
        "token_nll": out,
    }



def model_object_candidates(model):
    cands = [model, getattr(model, "model", None), getattr(model, "base_model", None)]
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        cands.append(getattr(base_model, "model", None))
        inner = getattr(base_model, "model", None)
        if inner is not None:
            cands.append(getattr(inner, "model", None))
    return cands


def locate_layers_module(model):
    for cand in model_object_candidates(model):
        if cand is None:
            continue
        if hasattr(cand, "layers"):
            return cand.layers
        if hasattr(cand, "model") and hasattr(cand.model, "layers"):
            return cand.model.layers

    raise RuntimeError("Could not locate decoder layers for attention probe")


def locate_model_rotary(model):
    for cand in model_object_candidates(model):
        if cand is None:
            continue
        if hasattr(cand, "rotary_emb"):
            return cand.rotary_emb
        if hasattr(cand, "model") and hasattr(cand.model, "rotary_emb"):
            return cand.model.rotary_emb
    return None


def attention_probe(model, tokens, length=ROBUST_LENGTH, start=0):
    device = model.device
    with torch.no_grad():
        chunk = tokens[start:start + length].unsqueeze(0).to(device)
        inp = chunk[:, :-1]
        seq_len = inp.size(1)

        outputs = model(inp, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

    layers = locate_layers_module(model)
    model_rotary = locate_model_rotary(model)
    n_layers = len(layers)
    cand = [0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]
    layer_idxs = sorted(set([max(0, min(n_layers - 1, x)) for x in cand]))

    qpos = sorted(set([max(0, min(seq_len - 1, x)) for x in [seq_len // 4, seq_len // 2, (3 * seq_len) // 4, seq_len - 1]]))
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    by_layer = {}
    global_entropy = []
    global_sink = []
    global_recent = []
    global_long = []
    global_dist = []

    with torch.no_grad():
        for li in layer_idxs:
            layer = layers[li]
            attn = layer.self_attn
            x = hidden_states[li]
            x_ln = layer.input_layernorm(x)

            num_heads = getattr(attn, "num_heads", None)
            if num_heads is None and hasattr(attn, "config"):
                num_heads = getattr(attn.config, "num_attention_heads", None)
            num_kv_heads = getattr(attn, "num_key_value_heads", None)
            if num_kv_heads is None and hasattr(attn, "config"):
                num_kv_heads = getattr(attn.config, "num_key_value_heads", None)
            if num_heads is None:
                num_heads = x_ln.size(-1) // attn.head_dim
            if num_kv_heads is None:
                num_kv_heads = num_heads

            q = attn.q_proj(x_ln).view(1, seq_len, int(num_heads), attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x_ln).view(1, seq_len, int(num_kv_heads), attn.head_dim).transpose(1, 2)

            if hasattr(attn, "rotary_emb"):
                cos, sin = attn.rotary_emb(k, pos_ids)
            elif model_rotary is not None:
                cos, sin = model_rotary(k, pos_ids)
            else:
                raise RuntimeError("No rotary embedding provider found for attention probe")
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if hasattr(attn, "num_key_value_groups"):
                n_rep = int(attn.num_key_value_groups)
            else:
                n_rep = int(int(num_heads) // int(num_kv_heads))
            k = repeat_kv(k, n_rep)

            ent_list = []
            sink_list = []
            recent_list = []
            long_list = []
            dist_list = []

            for q_idx in qpos:
                qv = q[:, :, q_idx, :]
                kv = k[:, :, : q_idx + 1, :]
                scores = torch.einsum("bhd,bhkd->bhk", qv, kv) / math.sqrt(attn.head_dim)
                probs = torch.softmax(scores.float(), dim=-1)
                k_len = probs.size(-1)

                norm = math.log(max(2, k_len))
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1) / norm

                sink_cut = min(ATTN_SINK_PREFIX, k_len)
                sink_mass = probs[:, :, :sink_cut].sum(dim=-1)

                recent_start = max(0, k_len - ATTN_RECENT_WINDOW)
                recent_mass = probs[:, :, recent_start:].sum(dim=-1)

                long_cut = max(0, k_len - ATTN_LONG_RANGE)
                if long_cut > 0:
                    long_mass = probs[:, :, :long_cut].sum(dim=-1)
                else:
                    long_mass = torch.zeros_like(entropy)

                key_idx = torch.arange(k_len, device=device, dtype=torch.float32).view(1, 1, -1)
                d = (float(q_idx) - key_idx).clamp(min=0.0)
                mean_dist = (probs * d).sum(dim=-1)

                ent_list.append(entropy)
                sink_list.append(sink_mass)
                recent_list.append(recent_mass)
                long_list.append(long_mass)
                dist_list.append(mean_dist)

            ent_cat = torch.cat(ent_list, dim=0)
            sink_cat = torch.cat(sink_list, dim=0)
            recent_cat = torch.cat(recent_list, dim=0)
            long_cat = torch.cat(long_list, dim=0)
            dist_cat = torch.cat(dist_list, dim=0)

            by_layer[str(li)] = {
                "entropy_norm_mean": float(ent_cat.mean().item()),
                "entropy_norm_std": float(ent_cat.std().item()),
                "sink_prefix_mass_mean": float(sink_cat.mean().item()),
                "recent_window_mass_mean": float(recent_cat.mean().item()),
                "long_range_mass_mean": float(long_cat.mean().item()),
                "mean_attention_distance": float(dist_cat.mean().item()),
            }

            global_entropy.append(by_layer[str(li)]["entropy_norm_mean"])
            global_sink.append(by_layer[str(li)]["sink_prefix_mass_mean"])
            global_recent.append(by_layer[str(li)]["recent_window_mass_mean"])
            global_long.append(by_layer[str(li)]["long_range_mass_mean"])
            global_dist.append(by_layer[str(li)]["mean_attention_distance"])

            del x, x_ln, q, k, ent_list, sink_list, recent_list, long_list, dist_list
            torch.cuda.empty_cache()

    summary = {
        "layer_indices": layer_idxs,
        "query_positions": qpos,
        "entropy_norm_mean": float(np.mean(global_entropy)),
        "sink_prefix_mass_mean": float(np.mean(global_sink)),
        "recent_window_mass_mean": float(np.mean(global_recent)),
        "long_range_mass_mean": float(np.mean(global_long)),
        "mean_attention_distance": float(np.mean(global_dist)),
    }

    return {
        "summary": summary,
        "by_layer": by_layer,
    }



def make_subsets(tokens, subset_size, n_subsets):
    if subset_size >= len(tokens):
        return [{"name": "subset_0", "offset": 0, "size": int(len(tokens))}]

    max_offset = len(tokens) - subset_size
    if n_subsets <= 1:
        offsets = [0]
    else:
        offsets = [int(round(i * max_offset / (n_subsets - 1))) for i in range(n_subsets)]

    subsets = []
    for i, off in enumerate(offsets):
        subsets.append({"name": f"subset_{i}", "offset": int(off), "size": int(subset_size)})
    return subsets



def write_markdown(results):
    lines = []
    lines.append("# FOLLOWUP MECHANISM P0 SUMMARY")
    lines.append("")
    lines.append(f"- Timestamp: {results['meta']['timestamp']}")
    lines.append(f"- Device: {results['meta']['device']}")
    lines.append(f"- Lengths (2x2): {results['meta']['main_lengths']}")
    lines.append(f"- Strategies: {results['meta']['strategies']}")
    lines.append(f"- Chunks per condition: {results['meta']['n_chunks']}")
    lines.append("")

    lines.append("## 1) 2x2 Factor Ablation (subset_0)")
    lines.append("")
    lines.append("| Variant | Strategy | PPL@12K | PPL@14K | PPL@16K |")
    lines.append("|---|---|---:|---:|---:|")
    for vname, vdata in results["ablation_2x2"].items():
        for st in STRATEGIES:
            p12 = vdata["eval"][st].get("12288", {}).get("ppl")
            p14 = vdata["eval"][st].get("14336", {}).get("ppl")
            p16 = vdata["eval"][st].get("16384", {}).get("ppl")
            lines.append(f"| {vname} | {st} | {p12} | {p14} | {p16} |")
    lines.append("")

    lines.append("## 2) 16K Robustness Across Validation Subsets")
    lines.append("")
    lines.append("| Model | Subset | Strategy | PPL@16K |")
    lines.append("|---|---|---|---:|")
    for model_name, md in results["subset_robustness_16k"].items():
        for subset_name, sd in md.items():
            for st in STRATEGIES:
                ppl = sd["eval"][st][str(ROBUST_LENGTH)]["ppl"]
                lines.append(f"| {model_name} | {subset_name} | {st} | {ppl} |")
    lines.append("")

    lines.append("## 3) 16K Loss Curve Jump Check")
    lines.append("")
    lines.append("| Model | abrupt jump | head mean NLL | tail mean NLL | tail/head | max jump idx | jump/scale |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for model_name, rec in results["loss_curve_16k"].items():
        s = rec["summary"]
        lines.append(
            "| {m} | {a} | {h:.4f} | {t:.4f} | {r:.3f} | {i} | {j:.3f} |".format(
                m=model_name,
                a=s["abrupt_jump_detected"],
                h=s["head_mean_nll"],
                t=s["tail_mean_nll"],
                r=s["tail_over_head"],
                i=s["max_jump_idx_smoothed"],
                j=s["max_jump_over_robust_scale"],
            )
        )
    lines.append("")

    lines.append("## 4) 16K Attention Probe")
    lines.append("")
    lines.append("| Model | entropy mean | sink(128) mean | recent(512) mean | long(>=4k) mean | mean distance |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model_name, rec in results["attention_probe_16k"].items():
        s = rec["summary"]
        lines.append(
            "| {m} | {e:.4f} | {sk:.4f} | {rc:.4f} | {lg:.4f} | {d:.1f} |".format(
                m=model_name,
                e=s["entropy_norm_mean"],
                sk=s["sink_prefix_mass_mean"],
                rc=s["recent_window_mass_mean"],
                lg=s["long_range_mass_mean"],
                d=s["mean_attention_distance"],
            )
        )
    lines.append("")

    lines.append("## Key Takeaways")
    lines.append("")
    lines.append(f"- Numeric anomalies: {len(results['anomalies'])}")
    lines.append(f"- base_orig vs lora_hybridfreq @16K ratio range across subsets/strategies: {results['key_takeaways']['ratio_min']:.3f}x to {results['key_takeaways']['ratio_max']:.3f}x")
    lines.append(f"- lora_origfreq vs base_orig @16K (sequential, subset_0): {results['key_takeaways']['lora_only_ratio_16k_seq_subset0']:.3f}x")
    lines.append(f"- base_hybridfreq vs base_orig @16K (sequential, subset_0): {results['key_takeaways']['freq_only_ratio_16k_seq_subset0']:.3f}x")
    lines.append("")

    Path(OUT_MD).write_text("\n".join(lines))



def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print("[Init] Loading tokenizer and validation tokens...", flush=True)
    tok = load_tokenizer()
    tokens = load_val_tokens(tok)
    print(f"[Init] Loaded tokens: {len(tokens)}", flush=True)

    subsets = make_subsets(tokens, ROBUST_SUBSET_SIZE, ROBUST_NUM_SUBSETS)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
            "device": device,
            "seed": SEED,
            "val_tokens": int(len(tokens)),
            "main_lengths": MAIN_LENGTHS,
            "strategies": STRATEGIES,
            "n_chunks": N_CHUNKS,
            "subset_size": ROBUST_SUBSET_SIZE,
            "subsets": subsets,
            "model_path": MODEL_PATH,
            "lora_path": HYBRID_LORA_PATH,
        },
        "ablation_2x2": {},
        "subset_robustness_16k": {
            "base_orig": {},
            "lora_hybridfreq": {},
        },
        "loss_curve_16k": {},
        "attention_probe_16k": {},
        "anomalies": [],
        "key_takeaways": {},
    }

    variants = ["base_orig", "base_hybridfreq", "lora_origfreq", "lora_hybridfreq"]
    subset0 = subsets[0]
    s0_tokens = tokens[subset0["offset"]: subset0["offset"] + subset0["size"]]

    for vname in variants:
        print(f"\n[Run] Loading variant: {vname}", flush=True)
        model = load_variant(vname)

        print(f"[Run] 2x2 eval for {vname}", flush=True)
        eval_out, anom = eval_ppl_on_tokens(
            model=model,
            tokens=s0_tokens,
            lengths=MAIN_LENGTHS,
            strategies=STRATEGIES,
            n_chunks=N_CHUNKS,
            seed=SEED,
        )
        results["ablation_2x2"][vname] = {
            "eval": eval_out,
            "anomalies": anom,
        }
        results["anomalies"].extend([{"variant": vname, **x} for x in anom])

        if vname in ("base_orig", "lora_hybridfreq"):
            print(f"[Run] 16K subset robustness for {vname}", flush=True)
            for subset in subsets:
                stoks = tokens[subset["offset"]: subset["offset"] + subset["size"]]
                subset_eval, subset_anom = eval_ppl_on_tokens(
                    model=model,
                    tokens=stoks,
                    lengths=[ROBUST_LENGTH],
                    strategies=STRATEGIES,
                    n_chunks=N_CHUNKS,
                    seed=SEED + subset["offset"],
                )
                results["subset_robustness_16k"][vname][subset["name"]] = {
                    "offset": subset["offset"],
                    "size": subset["size"],
                    "eval": subset_eval,
                    "anomalies": subset_anom,
                }
                results["anomalies"].extend([{"variant": vname, "subset": subset["name"], **x} for x in subset_anom])

            print(f"[Run] token-wise 16K loss curve for {vname}", flush=True)
            results["loss_curve_16k"][vname] = token_loss_curve(model, s0_tokens, length=ROBUST_LENGTH, start=0)

            print(f"[Run] 16K attention probe for {vname}", flush=True)
            results["attention_probe_16k"][vname] = attention_probe(model, s0_tokens, length=ROBUST_LENGTH, start=0)

        del model
        torch.cuda.empty_cache()

    ratios = []
    for subset in subsets:
        sname = subset["name"]
        for st in STRATEGIES:
            b = results["subset_robustness_16k"]["base_orig"][sname]["eval"][st][str(ROBUST_LENGTH)]["ppl"]
            h = results["subset_robustness_16k"]["lora_hybridfreq"][sname]["eval"][st][str(ROBUST_LENGTH)]["ppl"]
            if b and h:
                ratios.append(float(b / h))

    b0 = results["ablation_2x2"]["base_orig"]["eval"]["sequential"][str(ROBUST_LENGTH)]["ppl"]
    l0 = results["ablation_2x2"]["lora_origfreq"]["eval"]["sequential"][str(ROBUST_LENGTH)]["ppl"]
    f0 = results["ablation_2x2"]["base_hybridfreq"]["eval"]["sequential"][str(ROBUST_LENGTH)]["ppl"]

    results["key_takeaways"] = {
        "ratio_min": float(min(ratios)) if ratios else None,
        "ratio_max": float(max(ratios)) if ratios else None,
        "lora_only_ratio_16k_seq_subset0": float(b0 / l0) if (b0 and l0) else None,
        "freq_only_ratio_16k_seq_subset0": float(b0 / f0) if (b0 and f0) else None,
    }

    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    write_markdown(results)

    print(f"\n[Done] JSON: {OUT_JSON}", flush=True)
    print(f"[Done] MD:   {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
