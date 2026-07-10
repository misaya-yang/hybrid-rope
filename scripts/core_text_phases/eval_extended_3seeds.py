#!/usr/bin/env python3
"""Extended PPL eval across 3 seeds: GEO, EVQ, +YaRN at 8K-32K.
Also evaluates 50%/75%/100% checkpoints for training progression analysis.
Reports mean +/- std for paper."""
import sys, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, OrderedDict

DEVICE = "cuda"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Model (mirrors run_gqa_evq_experiment.py MLA architecture)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq, inv_freq):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq)
        self._build(max_seq)
    def _build(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_c", emb.cos(), persistent=False)
        self.register_buffer("sin_c", emb.sin(), persistent=False)
        self._max = seq_len
    def forward(self, L):
        if L > self._max: self._build(L)
        return self.cos_c[:L], self.sin_c[:L]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class MLAttention(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]; self.hd = cfg["head_dim"]
        self.d_rope = cfg.get("d_rope", 32); self.d_nope = self.hd - self.d_rope
        self.d_c = cfg.get("kv_lora_rank", h // 4)
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.kv_down = nn.Linear(h, self.d_c, bias=False)
        self.k_nope_up = nn.Linear(self.d_c, self.nh * self.d_nope, bias=False)
        self.v_up = nn.Linear(self.d_c, self.nh * self.hd, bias=False)
        self.k_rope_proj = nn.Linear(h, self.nh * self.d_rope, bias=False)
        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.rope = rope
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        c_kv = self.kv_down(x)
        k_nope = self.k_nope_up(c_kv).view(B, L, self.nh, self.d_nope).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, L, self.nh, self.d_rope).transpose(1, 2)
        v = self.v_up(c_kv).view(B, L, self.nh, self.hd).transpose(1, 2)
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, m = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate = nn.Linear(h, m, bias=False)
        self.up = nn.Linear(h, m, bias=False)
        self.down = nn.Linear(m, h, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = MLAttention(cfg, rope)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, cfg, inv_freq):
        super().__init__()
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        rope_dim = cfg.get("d_rope", cfg["head_dim"])
        rope = RotaryEmbedding(rope_dim, cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList([Block(cfg, rope) for _ in range(cfg["num_layers"])])
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight
    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))
    def extend_rope(self, L):
        self.blocks[0].attn.rope._build(L)

# ---------------------------------------------------------------------------
# YaRN
# ---------------------------------------------------------------------------

def yarn_inv_freq(orig_inv_freq, scale_factor, train_seq=8192, alpha=1.0, beta=32.0):
    inv = orig_inv_freq.double()
    K = len(inv)
    wavelengths = 2 * math.pi / inv
    low_freq_wavelen = train_seq * beta
    high_freq_wavelen = train_seq * alpha
    new_inv = torch.zeros_like(inv)
    for i in range(K):
        wl = wavelengths[i].item()
        if wl > low_freq_wavelen:
            new_inv[i] = inv[i] / scale_factor
        elif wl < high_freq_wavelen:
            new_inv[i] = inv[i]
        else:
            t = (wl - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
            new_inv[i] = inv[i] / (1.0 + t * (scale_factor - 1.0))
    return new_inv.float()

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ppl(model, val_data, lengths, n_chunks=8, seed=9999):
    model.eval()
    model.extend_rope(max(lengths) + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE)
    rng = np.random.RandomState(seed)
    results = {}
    for L in lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0: continue
        offsets = sorted(rng.choice(max_start, size=min(n_chunks, max_start // L), replace=False))
        for offset in offsets:
            chunk = val_data[offset:offset+L].unsqueeze(0).to(DEVICE)
            try:
                with ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  L={L}: OOM"); del chunk; torch.cuda.empty_cache(); break
                raise
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[L] = round(ppl, 3)
            print(f"  L={L}: PPL={ppl:.3f} ({len(losses)} chunks)")
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    work_dir = Path("/root/autodl-tmp/350m_mla32_500m")
    base = 500000.0
    eval_lengths = [8192, 16384, 20480, 24576, 28672, 32768]
    seeds = [42, 43, 88]
    checkpoints = ["model_50pct.pt", "model_75pct.pt", "model.pt"]  # 50%, 75%, final
    ckpt_labels = {"model_50pct.pt": "50%", "model_75pct.pt": "75%", "model.pt": "100%"}

    cfg = {
        "vocab_size": 50304, "hidden_size": 1024, "num_layers": 24,
        "num_heads": 16, "head_dim": 64, "intermediate_size": 4096,
        "max_position_embeddings": 2048, "attn_type": "mla",
        "d_rope": 32, "kv_lora_rank": 256,
    }

    val_data = torch.load(work_dir / "val_fineweb-edu_5000000.pt", weights_only=True)
    print(f"Val data: {len(val_data)} tokens")

    def evq_cosh_inv_freq(head_dim, tau, base=500000.0):
        K = head_dim // 2
        idx = torch.arange(K, dtype=torch.float64)
        u = idx / (K - 1)
        if abs(tau) < 1e-8:
            phi = 1.0 - u
        else:
            sinh_tau = math.sinh(tau)
            phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * sinh_tau)
        return (base ** phi).reciprocal().float()

    geo_inv = evq_cosh_inv_freq(32, 0.0, base)
    evq_inv = evq_cosh_inv_freq(32, 1.414, base)

    # ================================================================
    # Part 1: Training progression (50% / 75% / 100%)
    # Eval at 8K and 16K only (fast)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  PART 1: Training Progression (50% / 75% / 100%)")
    print(f"{'='*80}")

    progression_eval_lengths = [8192, 16384]
    # progression[tau_name][ckpt_label][seed] = {L: ppl}
    progression = defaultdict(lambda: defaultdict(dict))

    for seed in seeds:
        for tau_name, inv_freq, tau_str in [("GEO", geo_inv, "0.00"), ("EVQ", evq_inv, "1.41")]:
            run_dir = work_dir / f"350m_tau{tau_str}_seed{seed}"
            for ckpt_file in checkpoints:
                ckpt_path = run_dir / ckpt_file
                label = ckpt_labels[ckpt_file]
                if not ckpt_path.exists():
                    print(f"  [SKIP] {ckpt_path}")
                    continue

                print(f"\n--- {tau_name} seed={seed} @ {label} ---")
                model = GPT(cfg, inv_freq)
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(sd, strict=True)
                model = model.to(DEVICE)
                ppl = eval_ppl(model, val_data, progression_eval_lengths)
                progression[tau_name][label][seed] = ppl
                del model; torch.cuda.empty_cache()

    # Print progression table
    print(f"\n{'='*80}")
    print(f"  Training Progression: PPL by checkpoint stage (mean +/- std)")
    print(f"{'='*80}\n")

    for tau_name in ["GEO", "EVQ"]:
        print(f"  {tau_name}:")
        header = f"    {'Stage':<10}" + "".join(f"{'PPL@'+str(L//1024)+'K':>18}" for L in progression_eval_lengths)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for label in ["50%", "75%", "100%"]:
            seed_data = progression[tau_name].get(label, {})
            row = f"    {label:<10}"
            for L in progression_eval_lengths:
                vals = [seed_data[s].get(L, float("nan")) for s in seeds if s in seed_data]
                vals = [v for v in vals if not math.isnan(v)]
                if len(vals) >= 2:
                    m, s = np.mean(vals), np.std(vals, ddof=1)
                    row += f"{m:>10.1f} +/- {s:<5.1f}"
                elif len(vals) == 1:
                    row += f"{vals[0]:>14.1f}    "
                else:
                    row += f"{'N/A':>18}"
            print(row)
        print()

    # EVQ advantage at each stage
    print(f"  EVQ vs GEO (% change):")
    header = f"    {'Stage':<10}" + "".join(f"{'delta@'+str(L//1024)+'K':>18}" for L in progression_eval_lengths)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for label in ["50%", "75%", "100%"]:
        row = f"    {label:<10}"
        for L in progression_eval_lengths:
            geo_vals = [progression["GEO"].get(label, {}).get(s, {}).get(L, float("nan")) for s in seeds]
            evq_vals = [progression["EVQ"].get(label, {}).get(s, {}).get(L, float("nan")) for s in seeds]
            geo_vals = [v for v in geo_vals if not math.isnan(v)]
            evq_vals = [v for v in evq_vals if not math.isnan(v)]
            if geo_vals and evq_vals:
                gm, em = np.mean(geo_vals), np.mean(evq_vals)
                delta = (em / gm - 1) * 100
                row += f"{delta:>+15.1f}%  "
            else:
                row += f"{'N/A':>18}"
        print(row)
    print()

    # ================================================================
    # Part 2: Extended eval on final checkpoints (8K-32K + YaRN)
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  PART 2: Extended Eval (8K-32K + YaRN) on final checkpoints")
    print(f"{'='*80}")

    # all_per_seed[method][seed] = {L: ppl}
    all_per_seed = defaultdict(dict)

    for seed in seeds:
        for tau_name, inv_freq, tau_str in [("GEO", geo_inv, "0.00"), ("EVQ", evq_inv, "1.41")]:
            ckpt_path = work_dir / f"350m_tau{tau_str}_seed{seed}" / "model.pt"
            if not ckpt_path.exists():
                print(f"\n  [SKIP] {ckpt_path}")
                continue

            print(f"\n{'='*60}")
            print(f"  {tau_name} seed={seed}")
            print(f"{'='*60}")

            model = GPT(cfg, inv_freq)
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=True)
            model = model.to(DEVICE)

            # Base eval
            print(f"--- {tau_name} (base) ---")
            ppl = eval_ppl(model, val_data, eval_lengths)
            all_per_seed[tau_name][seed] = ppl

            # YaRN s=2
            tag2 = f"{tau_name}+YaRN(s=2)"
            print(f"--- {tag2} ---")
            y2 = yarn_inv_freq(inv_freq, 2.0)
            orig = model.blocks[0].attn.rope.inv_freq.clone()
            model.blocks[0].attn.rope.inv_freq.copy_(y2)
            model.blocks[0].attn.rope._build(max(eval_lengths) + 100)
            ppl2 = eval_ppl(model, val_data, eval_lengths)
            all_per_seed[tag2][seed] = ppl2
            model.blocks[0].attn.rope.inv_freq.copy_(orig)

            # YaRN s=4
            tag4 = f"{tau_name}+YaRN(s=4)"
            print(f"--- {tag4} ---")
            y4 = yarn_inv_freq(inv_freq, 4.0)
            model.blocks[0].attn.rope.inv_freq.copy_(y4)
            model.blocks[0].attn.rope._build(max(eval_lengths) + 100)
            ppl4 = eval_ppl(model, val_data, eval_lengths)
            all_per_seed[tag4][seed] = ppl4
            model.blocks[0].attn.rope.inv_freq.copy_(orig)

            del model; torch.cuda.empty_cache()

    # ---- Summary tables ----
    print(f"\n{'='*80}")
    print(f"  MULTI-SEED SUMMARY (seeds: {seeds})")
    print(f"  350M MLA-32, 500M tokens @ seq_len=8192")
    print(f"{'='*80}\n")

    methods_order = ["GEO", "GEO+YaRN(s=2)", "GEO+YaRN(s=4)",
                     "EVQ", "EVQ+YaRN(s=2)", "EVQ+YaRN(s=4)"]

    header = f"{'Method':<25}" + "".join(f"{'PPL@'+str(L//1024)+'K':>16}" for L in eval_lengths)
    print(header)
    print("-" * len(header))

    summary = {}
    for method in methods_order:
        seed_data = all_per_seed.get(method, {})
        if not seed_data: continue
        row = f"{method:<25}"
        summary[method] = {}
        for L in eval_lengths:
            vals = [seed_data[s].get(L, float("nan")) for s in seeds if s in seed_data]
            vals = [v for v in vals if not math.isnan(v)]
            if len(vals) >= 2:
                m, s = np.mean(vals), np.std(vals, ddof=1)
                row += f"{m:>9.1f}+/-{s:<5.1f}"
                summary[method][L] = {"mean": round(float(m), 2), "std": round(float(s), 2), "n": len(vals)}
            elif len(vals) == 1:
                row += f"{vals[0]:>12.1f}    "
                summary[method][L] = {"mean": round(vals[0], 2), "std": 0, "n": 1}
            else:
                row += f"{'N/A':>16}"
        print(row)

    # Relative to GEO
    geo_s = summary.get("GEO", {})
    if geo_s:
        print(f"\n--- Relative to GEO mean (%) ---")
        header2 = f"{'Method':<25}" + "".join(f"{'d@'+str(L//1024)+'K':>16}" for L in eval_lengths)
        print(header2)
        print("-" * len(header2))
        for method in methods_order:
            if method == "GEO" or method not in summary: continue
            row = f"{method:<25}"
            for L in eval_lengths:
                gm = geo_s.get(L, {}).get("mean")
                em = summary[method].get(L, {}).get("mean")
                if gm and em:
                    delta = (em / gm - 1) * 100
                    row += f"{delta:>+13.1f}%  "
                else:
                    row += f"{'N/A':>16}"
            print(row)

    # Save everything
    output = {
        "seeds": seeds,
        "eval_lengths": eval_lengths,
        "progression": {},
        "extended": {},
        "summary": {},
    }
    # Progression data
    for tau_name in ["GEO", "EVQ"]:
        output["progression"][tau_name] = {}
        for label in ["50%", "75%", "100%"]:
            seed_data = progression[tau_name].get(label, {})
            output["progression"][tau_name][label] = {
                str(s): {str(k): v for k, v in ppls.items()}
                for s, ppls in seed_data.items()
            }
    # Extended data
    for method, seed_data in all_per_seed.items():
        output["extended"][method] = {
            str(s): {str(k): v for k, v in ppls.items()}
            for s, ppls in seed_data.items()
        }
    # Summary
    for method, mdata in summary.items():
        output["summary"][method] = {
            str(k): {kk: vv for kk, vv in v.items()}
            for k, v in mdata.items()
        }

    out_path = work_dir / "eval_3seeds_full_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
