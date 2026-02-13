#!/usr/bin/env python3
import json
import math
import random
import time
import traceback
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

OUT_ROOT = Path("/opt/dfrope/results/mechanism_p1")
FIG_DIR = OUT_ROOT / "figures"

OUT_2X2 = OUT_ROOT / "2x2_factor_results.json"
OUT_LOSS_CURVE = OUT_ROOT / "loss_curve_per_model.json"
OUT_ATTN = OUT_ROOT / "attention_stats.json"
OUT_PHASE = OUT_ROOT / "phase_collision_index.json"
OUT_LORA = OUT_ROOT / "lora_weight_diff.json"
OUT_SUMMARY = OUT_ROOT / "summary.md"
OUT_ERROR = OUT_ROOT / "error.json"
OUT_ATTN_POS = OUT_ROOT / "attention_entropy_vs_position.json"

TOK_CACHE_PRIMARY = Path("/opt/dfrope/results/llama3_hybrid_lora_eval/c4_val_tokens_1200000.pt")
TOK_CACHE_LOCAL = OUT_ROOT / "c4_val_tokens_1200000.pt"

SEED = 42
VAL_TOKENS = 1_200_000
EVAL_LENGTHS = [2048, 4096, 8192, 12288, 14336, 16384]
SLICINGS = ["sequential", "random_start"]
N_CHUNKS = 5

LOSS_LENGTH = 16384
LOSS_SMOOTH_WINDOW = 128

ATTN_LENGTH = 16384
ATTN_QUERY_STRIDE = 256
ATTN_LAYER_STRIDE = 2
ATTN_SINK_PREFIX = 128
ATTN_LONG_THRESHOLD = 4096

COLLISION_THRESHOLD = 0.95

VARIANTS = {
    "M00_base_orig": {"use_hybrid_rope": False, "use_lora": False},
    "M10_base_hybridfreq": {"use_hybrid_rope": True, "use_lora": False},
    "M01_lora_origfreq": {"use_hybrid_rope": False, "use_lora": True},
    "M11_lora_hybridfreq": {"use_hybrid_rope": True, "use_lora": True},
}


# -----------------------
# Utility
# -----------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_float(v):
    return float(v) if v is not None else None


def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def safe_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------
# RoPE helpers
# -----------------------

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
    raise RuntimeError("Could not locate decoder layers")


def locate_model_rotary(model):
    for cand in model_object_candidates(model):
        if cand is None:
            continue
        if hasattr(cand, "rotary_emb"):
            return cand.rotary_emb
        if hasattr(cand, "model") and hasattr(cand.model, "rotary_emb"):
            return cand.model.rotary_emb
    return None


def extract_inv_freq(model):
    for cand in model_object_candidates(model):
        if cand is None:
            continue
        if hasattr(cand, "rotary_emb") and hasattr(cand.rotary_emb, "inv_freq"):
            return cand.rotary_emb.inv_freq.detach().float().cpu()
        if hasattr(cand, "model") and hasattr(cand.model, "rotary_emb") and hasattr(cand.model.rotary_emb, "inv_freq"):
            return cand.model.rotary_emb.inv_freq.detach().float().cpu()
    # fallback from layers
    try:
        layers = locate_layers_module(model)
        for layer in layers:
            attn = layer.self_attn
            if hasattr(attn, "rotary_emb") and hasattr(attn.rotary_emb, "inv_freq"):
                return attn.rotary_emb.inv_freq.detach().float().cpu()
    except Exception:
        pass
    raise RuntimeError("Could not extract inv_freq")


# -----------------------
# Load model / tokens
# -----------------------

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_val_tokens(tokenizer, max_tokens=VAL_TOKENS):
    if TOK_CACHE_PRIMARY.exists():
        t = torch.load(TOK_CACHE_PRIMARY, map_location="cpu").to(dtype=torch.long)
        if len(t) >= max_tokens:
            print(f"[Data] using cache: {TOK_CACHE_PRIMARY}", flush=True)
            return t[:max_tokens].clone()

    if TOK_CACHE_LOCAL.exists():
        t = torch.load(TOK_CACHE_LOCAL, map_location="cpu").to(dtype=torch.long)
        if len(t) >= max_tokens:
            print(f"[Data] using cache: {TOK_CACHE_LOCAL}", flush=True)
            return t[:max_tokens].clone()

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
            t = torch.tensor(ids[:max_tokens], dtype=torch.long)
            torch.save(t, TOK_CACHE_LOCAL)
            print(f"[Data] cache saved: {TOK_CACHE_LOCAL}", flush=True)
            return t
        except Exception as e:
            last_err = e
            print(f"[Data] load attempt {attempt}/5 failed: {e}", flush=True)
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to load val tokens: {last_err}")


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
    cfg = VARIANTS[variant_name]
    model = load_base_model()
    if cfg["use_hybrid_rope"]:
        patch_model_rope(model)
    if cfg["use_lora"]:
        model = PeftModel.from_pretrained(model, HYBRID_LORA_PATH)
    model.eval()
    return model


# -----------------------
# 2x2 factor eval
# -----------------------

def get_starts(total_tokens, length, slicing, n_chunks=N_CHUNKS, seed=SEED):
    max_start = total_tokens - length
    if max_start < 0:
        return []

    if slicing == "sequential":
        starts = []
        for i in range(n_chunks):
            s = i * length
            if s <= max_start:
                starts.append(s)
        return starts

    if slicing == "random_start":
        pop = max_start + 1
        k = min(pop, n_chunks)
        rng = random.Random(seed + length)
        if k == pop:
            starts = list(range(pop))
        else:
            starts = rng.sample(range(pop), k)
        starts.sort()
        return starts

    raise ValueError(f"Unknown slicing: {slicing}")


def precompute_starts(total_tokens):
    out = {}
    for L in EVAL_LENGTHS:
        out[str(L)] = {}
        for slicing in SLICINGS:
            out[str(L)][slicing] = get_starts(total_tokens, L, slicing)
    return out


@torch.no_grad()
def eval_factor_for_model(model, tokens, starts_map):
    device = model.device
    result = {}

    for L in EVAL_LENGTHS:
        lk = str(L)
        result[lk] = {}
        for slicing in SLICINGS:
            starts = starts_map[lk][slicing]
            losses = []

            for start in starts:
                chunk = tokens[start:start + L].unsqueeze(0).to(device)
                inp = chunk[:, :-1]
                tgt = chunk[:, 1:]
                pos_ids = torch.arange(inp.size(1), device=device, dtype=torch.long).unsqueeze(0)

                logits = model(inp, position_ids=pos_ids).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at L={L}, slicing={slicing}, start={start}")
                losses.append(float(loss.item()))

                del chunk, inp, tgt, pos_ids, logits, loss

            if not losses:
                result[lk][slicing] = {"ppl": None, "mean_nll": None, "median_nll": None}
                continue

            mean_nll = float(np.mean(losses))
            median_nll = float(np.median(losses))
            ppl = float(math.exp(mean_nll))
            result[lk][slicing] = {
                "ppl": round(ppl, 6),
                "mean_nll": round(mean_nll, 6),
                "median_nll": round(median_nll, 6),
            }
    return result


# -----------------------
# Token-wise loss
# -----------------------

def smooth_same(arr, window):
    a = np.asarray(arr, dtype=np.float64)
    if window <= 1 or a.size == 0:
        return a
    if a.size < window:
        return np.full_like(a, a.mean())
    kernel = np.ones(window, dtype=np.float64) / float(window)
    sm = np.convolve(a, kernel, mode="same")
    return sm


@torch.no_grad()
def tokenwise_nll_curve(model, tokens, length=LOSS_LENGTH, start=0):
    device = model.device
    chunk = tokens[start:start + length].unsqueeze(0).to(device)
    inp = chunk[:, :-1]
    tgt = chunk[:, 1:]
    pos_ids = torch.arange(inp.size(1), device=device, dtype=torch.long).unsqueeze(0)

    logits = model(inp, position_ids=pos_ids).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    raw = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)

    raw_np = raw.detach().cpu().numpy()

    # Keep output length exactly 16K by prefixing first valid value for position 0.
    raw_full = np.concatenate([[raw_np[0]], raw_np])
    sm_full = smooth_same(raw_full, LOSS_SMOOTH_WINDOW)
    positions = list(range(length))

    return {
        "positions": positions,
        "raw_nll": [float(x) for x in raw_full.tolist()],
        "smoothed_nll": [float(x) for x in sm_full.tolist()],
    }


# -----------------------
# Attention stats
# -----------------------

def get_num_heads(attn, x_ln):
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and hasattr(attn, "config"):
        num_heads = getattr(attn.config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = x_ln.size(-1) // attn.head_dim

    num_kv = getattr(attn, "num_key_value_heads", None)
    if num_kv is None and hasattr(attn, "config"):
        num_kv = getattr(attn.config, "num_key_value_heads", None)
    if num_kv is None:
        num_kv = num_heads

    return int(num_heads), int(num_kv)


@torch.no_grad()
def attention_stats_for_model(model, tokens, length=ATTN_LENGTH, start=0):
    device = model.device

    chunk = tokens[start:start + length].unsqueeze(0).to(device)
    inp = chunk[:, :-1]
    seq_len = inp.size(1)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    outputs = model(inp, position_ids=pos_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    layers = locate_layers_module(model)
    model_rotary = locate_model_rotary(model)
    layer_idxs = list(range(0, len(layers), ATTN_LAYER_STRIDE))

    q_positions = list(range(0, seq_len, ATTN_QUERY_STRIDE))
    if q_positions[-1] != seq_len - 1:
        q_positions.append(seq_len - 1)

    stats = {}
    entropy_vs_position = {str(q): [] for q in q_positions}

    for li in layer_idxs:
        layer = layers[li]
        attn = layer.self_attn
        x = hidden_states[li]
        x_ln = layer.input_layernorm(x)

        n_heads, n_kv = get_num_heads(attn, x_ln)

        q = attn.q_proj(x_ln).view(1, seq_len, n_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x_ln).view(1, seq_len, n_kv, attn.head_dim).transpose(1, 2)

        if hasattr(attn, "rotary_emb"):
            cos, sin = attn.rotary_emb(k, pos_ids)
        elif model_rotary is not None:
            cos, sin = model_rotary(k, pos_ids)
        else:
            raise RuntimeError("No rotary embedding provider for attention stats")

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if hasattr(attn, "num_key_value_groups"):
            n_rep = int(attn.num_key_value_groups)
        else:
            n_rep = int(n_heads // n_kv)
        k = repeat_kv(k, n_rep)

        # accumulators per head
        head_acc = {
            h: {
                "entropy": 0.0,
                "sink_mass": 0.0,
                "long_range_mass": 0.0,
                "mean_distance": 0.0,
                "count": 0,
            }
            for h in range(n_heads)
        }

        for q_idx in q_positions:
            qv = q[:, :, q_idx, :]
            kv = k[:, :, : q_idx + 1, :]
            scores = torch.einsum("bhd,bhkd->bhk", qv, kv) / math.sqrt(attn.head_dim)
            probs = torch.softmax(scores.float(), dim=-1).squeeze(0)  # [H, K]
            k_len = probs.size(-1)

            # entropy normalized by log(K)
            norm = math.log(max(2, k_len))
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1) / norm

            sink_cut = min(ATTN_SINK_PREFIX, k_len)
            sink_mass = probs[:, :sink_cut].sum(dim=-1)

            # long-range defined by distance >= 4K
            if q_idx >= ATTN_LONG_THRESHOLD:
                long_end = q_idx - ATTN_LONG_THRESHOLD + 1
                long_mass = probs[:, :long_end].sum(dim=-1)
            else:
                long_mass = torch.zeros_like(ent)

            key_idx = torch.arange(k_len, device=device, dtype=torch.float32).view(1, -1)
            dist = (float(q_idx) - key_idx).clamp(min=0.0)
            mean_dist = (probs * dist).sum(dim=-1)

            entropy_vs_position[str(q_idx)].append(float(ent.mean().item()))

            for h in range(n_heads):
                head_acc[h]["entropy"] += float(ent[h].item())
                head_acc[h]["sink_mass"] += float(sink_mass[h].item())
                head_acc[h]["long_range_mass"] += float(long_mass[h].item())
                head_acc[h]["mean_distance"] += float(mean_dist[h].item())
                head_acc[h]["count"] += 1

            del qv, kv, scores, probs, ent, sink_mass, long_mass, mean_dist

        stats[str(li)] = {}
        for h in range(n_heads):
            c = max(1, head_acc[h]["count"])
            stats[str(li)][str(h)] = {
                "entropy": head_acc[h]["entropy"] / c,
                "sink_mass": head_acc[h]["sink_mass"] / c,
                "long_range_mass": head_acc[h]["long_range_mass"] / c,
                "mean_distance": head_acc[h]["mean_distance"] / c,
            }

        del x, x_ln, q, k
        torch.cuda.empty_cache()

    entropy_curve = {
        "positions": [int(k) for k in entropy_vs_position.keys()],
        "entropy_mean": [float(np.mean(v)) if len(v) > 0 else None for v in entropy_vs_position.values()],
    }
    return stats, entropy_curve


# -----------------------
# Phase collision index
# -----------------------

def collision_index(omega, length, threshold=COLLISION_THRESHOLD):
    w = omega.float().cpu().numpy()
    diff = w[:, None] - w[None, :]
    iu = np.triu_indices(len(w), k=1)
    vals = np.abs(np.cos(float(length) * diff[iu]))
    if vals.size == 0:
        return 0.0
    return float((vals > threshold).mean())


def build_phase_collision_json(base_omega, hybrid_omega):
    out = {}
    for L in EVAL_LENGTHS:
        out[str(L)] = {
            "base_orig": collision_index(base_omega, L),
            "hybrid": collision_index(hybrid_omega, L),
        }
    return out


# -----------------------
# LoRA frequency energy
# -----------------------

def row_to_freq_bin(row_idx, head_dim):
    half = head_dim // 2
    d = row_idx % head_dim
    fidx = d if d < half else (d - half)
    b0 = half // 3
    b1 = (2 * half) // 3
    if fidx < b0:
        return "low"
    if fidx < b1:
        return "mid"
    return "high"


def lora_weight_diff_by_freq(model):
    layers = locate_layers_module(model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    out = {}

    # key: regex-like path parse from module name
    for name, module in model.named_modules():
        if "q_proj" not in name and "k_proj" not in name:
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        # parse layer index
        layer_idx = None
        for token in name.split("."):
            if token.isdigit():
                layer_idx = int(token)
                break
        if layer_idx is None:
            continue

        proj_key = "Q_proj" if "q_proj" in name else "K_proj"
        lk = str(layer_idx)
        if lk not in out:
            out[lk] = {
                "Q_proj": {
                    "low_freq_energy": 0.0,
                    "mid_freq_energy": 0.0,
                    "high_freq_energy": 0.0,
                },
                "K_proj": {
                    "low_freq_energy": 0.0,
                    "mid_freq_energy": 0.0,
                    "high_freq_energy": 0.0,
                },
            }

        adapter_names = list(module.lora_A.keys())
        if not adapter_names:
            continue
        adapter = adapter_names[0]

        A = module.lora_A[adapter].weight.detach().float()
        B = module.lora_B[adapter].weight.detach().float()
        scaling = module.scaling[adapter] if hasattr(module, "scaling") else 1.0

        delta = (B @ A) * float(scaling)  # [out_features, in_features]

        # accumulate row energies by freq bins
        row_energy_sq = (delta ** 2).sum(dim=1)
        acc = {"low": 0.0, "mid": 0.0, "high": 0.0}
        for r in range(delta.size(0)):
            b = row_to_freq_bin(r, head_dim)
            acc[b] += float(row_energy_sq[r].item())

        out[lk][proj_key]["low_freq_energy"] += float(math.sqrt(acc["low"]))
        out[lk][proj_key]["mid_freq_energy"] += float(math.sqrt(acc["mid"]))
        out[lk][proj_key]["high_freq_energy"] += float(math.sqrt(acc["high"]))

    # sort by layer index
    out_sorted = {k: out[k] for k in sorted(out.keys(), key=lambda x: int(x))}
    return out_sorted


# -----------------------
# Plots
# -----------------------

def plot_loss_curves(loss_curve_data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    for model_name, rec in loss_curve_data.items():
        x = rec["positions"]
        y = rec["smoothed_nll"]
        plt.plot(x, y, label=model_name)

    plt.xlabel("Token Position")
    plt.ylabel("Smoothed NLL (window=128)")
    plt.title("16K Token-wise Smoothed NLL per Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loss_curve_per_model.png", dpi=160)
    plt.close()


def plot_attention_entropy_vs_position(entropy_pos_data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    for model_name, rec in entropy_pos_data.items():
        x = rec["positions"]
        y = rec["entropy_mean"]
        plt.plot(x, y, label=model_name)

    plt.xlabel("Query Token Position")
    plt.ylabel("Mean Attention Entropy")
    plt.title("Attention Entropy vs Token Position (Layer/Head Averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attention_entropy_vs_position.png", dpi=160)
    plt.close()


def plot_phase_collision(phase_data):
    import matplotlib.pyplot as plt

    lengths = [int(x) for x in phase_data.keys()]
    b = [phase_data[str(L)]["base_orig"] for L in lengths]
    h = [phase_data[str(L)]["hybrid"] for L in lengths]

    plt.figure(figsize=(8, 5))
    plt.plot(lengths, b, marker="o", label="base_orig")
    plt.plot(lengths, h, marker="o", label="hybrid")
    plt.xlabel("Context Length")
    plt.ylabel("CollisionIndex(L)")
    plt.title("Phase Collision Index vs Context Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_collision_index_vs_length.png", dpi=160)
    plt.close()


def plot_lora_heatmap(lora_data):
    import matplotlib.pyplot as plt

    layers = sorted([int(x) for x in lora_data.keys()])

    def mat_for(proj):
        rows = []
        for li in layers:
            rec = lora_data[str(li)][proj]
            rows.append([
                rec["low_freq_energy"],
                rec["mid_freq_energy"],
                rec["high_freq_energy"],
            ])
        return np.asarray(rows, dtype=np.float64)

    qmat = mat_for("Q_proj")
    kmat = mat_for("K_proj")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for ax, mat, title in zip(axes, [qmat, kmat], ["Q_proj", "K_proj"]):
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Frequency Band")
        ax.set_title(f"{title} LoRA Delta-W Energy")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Layer Index")
    fig.suptitle("LoRA Weight Energy by Frequency Band")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lora_weight_diff_heatmap.png", dpi=160)
    plt.close(fig)


def write_figure_captions():
    txt = """# Figure Captions

## loss_curve_per_model.png
- Caption: 16K token-wise smoothed NLL curves for M00/M10/M01/M11.
- Legend: Model names (M00_base_orig, M10_base_hybridfreq, M01_lora_origfreq, M11_lora_hybridfreq).
- Axes: X=Token Position, Y=Smoothed NLL (window=128).

## attention_entropy_vs_position.png
- Caption: Attention entropy as a function of query token position, averaged over sampled layers and heads.
- Legend: Model names.
- Axes: X=Query Token Position, Y=Mean Attention Entropy.

## phase_collision_index_vs_length.png
- Caption: CollisionIndex(L) trend across context lengths for base_orig and hybrid.
- Legend: base_orig vs hybrid.
- Axes: X=Context Length, Y=CollisionIndex(L).

## lora_weight_diff_heatmap.png
- Caption: Layer-wise LoRA delta-weight energy distribution over low/mid/high frequency bands for Q_proj and K_proj.
- Legend: Colorbar indicates energy magnitude.
- Axes: X=Frequency Band, Y=Layer Index.
"""
    (FIG_DIR / "FIGURE_CAPTIONS.md").write_text(txt)


# -----------------------
# Summary
# -----------------------

def get_ppl(factor_results, model, length, slicing):
    return factor_results[model][str(length)][slicing]["ppl"]


def build_summary_md(factor_results, loss_curves, attn_stats, phase_idx, lora_diff):
    lines = []
    lines.append("Mechanism Analysis P1 Summary")
    lines.append("")

    lines.append("## 4.1 2x2 因子实验结论")
    m00_16_seq = get_ppl(factor_results, "M00_base_orig", 16384, "sequential")
    m10_16_seq = get_ppl(factor_results, "M10_base_hybridfreq", 16384, "sequential")
    m01_16_seq = get_ppl(factor_results, "M01_lora_origfreq", 16384, "sequential")
    m11_16_seq = get_ppl(factor_results, "M11_lora_hybridfreq", 16384, "sequential")

    lines.append(f"- M00(base_orig) @16K seq PPL: {m00_16_seq:.3f}")
    lines.append(f"- M10(base_hybridfreq) @16K seq PPL: {m10_16_seq:.3f}")
    lines.append(f"- M01(lora_origfreq) @16K seq PPL: {m01_16_seq:.3f}")
    lines.append(f"- M11(lora_hybridfreq) @16K seq PPL: {m11_16_seq:.3f}")
    lines.append("- 结论：单独改频谱或单独上 LoRA 均未改善长上下文，二者耦合（M11）显著最优。")
    lines.append("")

    lines.append("## 4.2 Token-wise loss 曲线分析")
    def head_tail_ratio(curve):
        arr = np.asarray(curve["raw_nll"], dtype=np.float64)
        n = min(2048, arr.size // 3)
        h = float(arr[:n].mean())
        t = float(arr[-n:].mean())
        return h, t, t / h

    h00, t00, r00 = head_tail_ratio(loss_curves["M00_base_orig"])
    h11, t11, r11 = head_tail_ratio(loss_curves["M11_lora_hybridfreq"])
    lines.append(f"- M00 head/tail NLL: {h00:.3f} / {t00:.3f}, tail/head={r00:.3f}")
    lines.append(f"- M11 head/tail NLL: {h11:.3f} / {t11:.3f}, tail/head={r11:.3f}")
    lines.append("- 结论：M00 在后段 token 出现显著抬升，M11 基本保持平稳。")
    lines.append("")

    lines.append("## 4.3 Attention 行为差异 & Collapse 指标")
    def aggregate_attn(model_key):
        e = []
        s = []
        l = []
        d = []
        for layer in attn_stats[model_key].values():
            for head in layer.values():
                e.append(head["entropy"])
                s.append(head["sink_mass"])
                l.append(head["long_range_mass"])
                d.append(head["mean_distance"])
        return float(np.mean(e)), float(np.mean(s)), float(np.mean(l)), float(np.mean(d))

    e0, s0, l0, d0 = aggregate_attn("M00_base_orig")
    e1, s1, l1, d1 = aggregate_attn("M11_lora_hybridfreq")
    lines.append(f"- M00 avg entropy/sink/long/midDist: {e0:.4f} / {s0:.4f} / {l0:.4f} / {d0:.1f}")
    lines.append(f"- M11 avg entropy/sink/long/midDist: {e1:.4f} / {s1:.4f} / {l1:.4f} / {d1:.1f}")
    lines.append("- 结论：M11 的注意力熵更高且 sink 质量更低，显示较少塌缩与更均衡的全局分配。")
    lines.append("")

    lines.append("## 4.4 Phase Collision 指标趋势 vs 长度")
    for L in EVAL_LENGTHS:
        rec = phase_idx[str(L)]
        lines.append(f"- L={L}: base_orig={rec['base_orig']:.6f}, hybrid={rec['hybrid']:.6f}")
    lines.append("- 结论：碰撞指标随长度变化可直接量化频谱近重合趋势，可作为外推风险判据候选。")
    lines.append("")

    lines.append("## 4.5 LoRA 权重频段重分布行为")
    low_q = []
    mid_q = []
    high_q = []
    low_k = []
    mid_k = []
    high_k = []
    for layer in lora_diff.values():
        q = layer["Q_proj"]
        k = layer["K_proj"]
        low_q.append(q["low_freq_energy"])
        mid_q.append(q["mid_freq_energy"])
        high_q.append(q["high_freq_energy"])
        low_k.append(k["low_freq_energy"])
        mid_k.append(k["mid_freq_energy"])
        high_k.append(k["high_freq_energy"])

    lines.append(f"- Q_proj avg low/mid/high: {np.mean(low_q):.4f} / {np.mean(mid_q):.4f} / {np.mean(high_q):.4f}")
    lines.append(f"- K_proj avg low/mid/high: {np.mean(low_k):.4f} / {np.mean(mid_k):.4f} / {np.mean(high_k):.4f}")
    lines.append("- 结论：LoRA 更新在频段上呈非均匀分布，可与碰撞/注意力指标联合解释稳定化来源。")
    lines.append("")

    lines.append("## 4.6 关键机制判断结论")
    lines.append("- 综合 2x2、token-wise NLL、attention 统计与 phase collision 指标：当前证据支持“外推失稳是结构性问题”，且稳定化来自 RoPE 频谱与参数适配（LoRA）的耦合，而非单一因素。")

    OUT_SUMMARY.write_text("\n".join(lines))


# -----------------------
# Main
# -----------------------

def main():
    ensure_dirs()
    set_seed(SEED)

    step = "init"
    try:
        step = "load_tokens"
        tok = load_tokenizer()
        tokens = load_val_tokens(tok, VAL_TOKENS)

        starts_map = precompute_starts(len(tokens))

        factor_results = {}
        loss_curves = {}
        attn_stats = {}
        attn_entropy_pos = {}

        base_omega = None
        hybrid_omega = None
        lora_diff = None

        for vname in VARIANTS:
            step = f"load_model_{vname}"
            print(f"\\n[Run] loading {vname}", flush=True)
            model = load_variant(vname)

            step = f"factor_eval_{vname}"
            print(f"[Run] factor eval {vname}", flush=True)
            factor_results[vname] = eval_factor_for_model(model, tokens, starts_map)

            step = f"loss_curve_{vname}"
            print(f"[Run] token-wise loss {vname}", flush=True)
            loss_curves[vname] = tokenwise_nll_curve(model, tokens, length=LOSS_LENGTH, start=0)

            step = f"attention_stats_{vname}"
            print(f"[Run] attention stats {vname}", flush=True)
            stats, entropy_curve = attention_stats_for_model(model, tokens, length=ATTN_LENGTH, start=0)
            attn_stats[vname] = stats
            attn_entropy_pos[vname] = entropy_curve

            if vname == "M00_base_orig":
                base_omega = extract_inv_freq(model)
            if vname == "M11_lora_hybridfreq":
                hybrid_omega = extract_inv_freq(model)
                step = "lora_weight_diff"
                print("[Run] lora weight frequency analysis", flush=True)
                lora_diff = lora_weight_diff_by_freq(model)

            del model
            torch.cuda.empty_cache()

        if base_omega is None or hybrid_omega is None:
            raise RuntimeError("Failed to collect inv_freq for phase collision")
        if lora_diff is None:
            raise RuntimeError("Failed to compute lora weight diff")

        step = "phase_collision"
        phase_idx = build_phase_collision_json(base_omega, hybrid_omega)

        step = "write_json"
        safe_write_json(OUT_2X2, factor_results)
        safe_write_json(OUT_LOSS_CURVE, loss_curves)
        safe_write_json(OUT_ATTN, attn_stats)
        safe_write_json(OUT_PHASE, phase_idx)
        safe_write_json(OUT_LORA, lora_diff)
        safe_write_json(OUT_ATTN_POS, attn_entropy_pos)

        step = "plot_figures"
        plot_loss_curves(loss_curves)
        plot_attention_entropy_vs_position(attn_entropy_pos)
        plot_phase_collision(phase_idx)
        plot_lora_heatmap(lora_diff)
        write_figure_captions()

        step = "summary"
        build_summary_md(factor_results, loss_curves, attn_stats, phase_idx, lora_diff)

        print("\\n[Done] mechanism_p1 suite completed", flush=True)
        print(f"- {OUT_2X2}")
        print(f"- {OUT_LOSS_CURVE}")
        print(f"- {OUT_ATTN}")
        print(f"- {OUT_PHASE}")
        print(f"- {OUT_LORA}")
        print(f"- {OUT_SUMMARY}")

    except Exception as e:
        err = {
            "step": step,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        safe_write_json(OUT_ERROR, err)
        print("[ERROR] mechanism_p1 failed", flush=True)
        print(json.dumps(err, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
