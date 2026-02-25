"""
Unified RoPE Frequency Distribution Search (3 configs x 3 seeds)

Derived from unified_search.py with minimal changes:
- Same model/data/train/eval code paths
- Restrict configs to 3
- Run seeds [42, 123, 7]
- Report PPL@2048 and PPL@16384 (still computes the full EVAL_LENGTHS list)

This file keeps the original 3-seed default for reproducibility, but supports
overriding seeds via env var `DFROPE_SEEDS` (comma-separated ints or JSON list).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import sys
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# ==================== FIXED CONFIG (DO NOT MODIFY) ====================
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16
WORK_DIR = "/opt/dfrope/results/unified_search_3cfg_3seed"

MODEL_CFG = {
    "vocab_size": 50304,
    "hidden_size": 512,
    "num_layers": 6,
    "num_heads": 8,
    "head_dim": 64,
    "intermediate_size": 2048,
    "max_position_embeddings": 2048,
}

TRAIN_TOKENS = 50_000_000
SEQ_LEN = 2048
BATCH_SIZE = 32
LR = 6e-4
EVAL_LENGTHS = [2048, 4096, 8192, 16384]
EVAL_CHUNKS = 10

SEEDS = [42, 123, 7]

# ==================== FREQUENCY DISTRIBUTIONS ====================


def geometric_freq(K, theta):
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))


def sigmoid_freq(K, theta_base, steepness=8.0, midpoint=0.5, omf=0.3):
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = (k / (K - 1)).numpy()
    s = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    log_omega = np.log(omega_max) + s * (np.log(omega_min) - np.log(omega_max))
    return torch.from_numpy(np.exp(log_omega)).float()


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


# ==================== MODEL ====================


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
        if L > self._max:
            self._build(L)
        return self.cos_c[:L], self.sin_c[:L]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]
        self.hd = cfg["head_dim"]
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.rope = rope

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return self.o(
            F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, -1)
        )


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
        self.attn = Attention(cfg, rope)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, inv_freq):
        super().__init__()
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        rope = RotaryEmbedding(cfg["head_dim"], cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList([Block(cfg, rope) for _ in range(cfg["num_layers"])])
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"  Params: {n/1e6:.1f}M")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def extend_rope(self, L):
        self.blocks[0].attn.rope._build(L)


# ==================== DATA ====================


def load_data(tokenizer, max_tokens, seq_len):
    print(f"  Loading TinyStories ({max_tokens/1e6:.0f}M tokens)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ids = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    n = len(ids) // seq_len
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def load_val(tokenizer, max_tokens=5_000_000):
    print("  Loading validation data...")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids, dtype=torch.long)


# ==================== TRAIN & EVAL ====================


def train_model(model, data):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    steps = len(data) // BATCH_SIZE
    torch.manual_seed(SEED)
    perm = torch.randperm(len(data))
    t0 = time.time()
    for s in range(steps):
        batch = data[perm[s * BATCH_SIZE : (s + 1) * BATCH_SIZE]].to(DEVICE)
        # LR schedule
        warmup = int(steps * 0.02)
        if s < warmup:
            lr = LR * s / max(warmup, 1)
        else:
            lr = LR * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.amp.autocast("cuda", dtype=DTYPE):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1)
            print(f"    step {s}/{steps}  loss={loss.item():.4f}  lr={lr:.2e}  ETA={eta/60:.0f}min")
    return model


@torch.no_grad()
def eval_model(model, val_data, lengths):
    model.eval()
    model.extend_rope(max(lengths) + 100)
    results = {}
    for L in lengths:
        losses = []
        for i in range(EVAL_CHUNKS):
            if (i + 1) * L > len(val_data):
                break
            chunk = val_data[i * L : (i + 1) * L].unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
            losses.append(loss.item())
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[str(L)] = round(ppl, 3)
            print(f"    L={L}: PPL={ppl:.3f}")
    return results


# ==================== MAIN ====================


def run_one(name, inv_freq, train_data, val_data):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  seed={SEED}")
    print(f"  freq range: [{inv_freq.min().item():.2e}, {inv_freq.max().item():.4f}]")
    print(f"{'='*60}")
    torch.manual_seed(SEED)
    model = GPT(MODEL_CFG, inv_freq).to(DEVICE)
    model = train_model(model, train_data)
    results = eval_model(model, val_data, EVAL_LENGTHS)
    del model
    torch.cuda.empty_cache()
    return results


def _fmt_ms(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def _parse_seeds_from_env(default_seeds):
    """
    DFROPE_SEEDS formats:
      - "42,123,7,..."  (comma separated)
      - "[42, 123, 7]"  (JSON list)
    """
    raw = os.environ.get("DFROPE_SEEDS", "").strip()
    if not raw:
        return list(default_seeds)

    try:
        if raw.startswith("[") and raw.endswith("]"):
            seeds = json.loads(raw)
        else:
            seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
        seeds = [int(s) for s in seeds]
    except Exception as e:
        raise ValueError(f"Invalid DFROPE_SEEDS={raw!r}: {e}")

    if not seeds:
        raise ValueError("DFROPE_SEEDS parsed to an empty list.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"DFROPE_SEEDS contains duplicates: {seeds}")
    return seeds


def main():
    seeds = _parse_seeds_from_env(SEEDS)

    # Avoid overwriting the original 3-seed artifacts unless explicitly requested.
    work_dir = os.environ.get("DFROPE_WORK_DIR", "").strip()
    if not work_dir:
        if seeds == SEEDS:
            work_dir = WORK_DIR
        else:
            work_dir = f"/opt/dfrope/results/unified_search_3cfg_{len(seeds)}seed"

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_json = f"{work_dir}/results.json"
    print(f"[multiseed] seeds={seeds}")
    print(f"[multiseed] work_dir={work_dir}")

    # Tokenizer & Data (identical on both machines)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    train_data = load_data(tok, TRAIN_TOKENS, SEQ_LEN)
    val_data = load_val(tok)

    K = MODEL_CFG["head_dim"] // 2  # 32

    # Only run 3 configs.
    configs = [
        ("geo_500k", geometric_freq(K, 500000)),
        ("hybrid_a0.2_t100k", hybrid_freq(geometric_freq(K, 100000), anchored_poly_freq(K, 100000, 3.9, 0.3), 0.2)),
        ("anchpoly_p3.9_omf0.3_t500k", anchored_poly_freq(K, 500000, 3.9, 0.3)),
    ]

    # Optional: restrict which configs to run (time saver).
    # Example: DFROPE_CONFIGS="geo_500k,hybrid_a0.2_t100k"
    raw_cfgs = os.environ.get("DFROPE_CONFIGS", "").strip()
    if raw_cfgs:
        wanted = [x.strip() for x in raw_cfgs.split(",") if x.strip()]
        wanted_set = set(wanted)
        configs = [c for c in configs if c[0] in wanted_set]
        if not configs:
            raise ValueError(f"DFROPE_CONFIGS={raw_cfgs!r} matched no configs. Available: geo_500k, hybrid_a0.2_t100k, anchpoly_p3.9_omf0.3_t500k")
        print(f"[multiseed] restrict configs={wanted}")

    # Results structure:
    # {config: {seed: {"2048":..., "16384":..., ...}}}
    results = {}
    rows = []  # per-run rows for printing

    global SEED
    for cfg_name, freq in configs:
        results[cfg_name] = {}
        for seed in seeds:
            SEED = int(seed)
            run_name = f"{cfg_name}_seed{seed}"
            r = run_one(run_name, freq, train_data, val_data)
            results[cfg_name][str(seed)] = r

            p2 = r.get("2048")
            p16 = r.get("16384")
            rows.append((cfg_name, seed, p2, p16))

            # Save after each (crash safety)
            with open(out_json, "w") as f:
                json.dump({"results": results, "rows": rows}, f, indent=2)

    # Print per-run table
    print("\nConfig              | Seed | PPL@2048 | PPL@16384")
    for cfg_name, seed, p2, p16 in rows:
        print(f"{cfg_name:<19} | {seed:>4} | {p2:>8} | {p16:>9}")

    # Print aggregate table
    print("\nConfig              | PPL@2048 (mean±std) | PPL@16384 (mean±std)")
    for cfg_name, _ in configs:
        vals2 = []
        vals16 = []
        for seed in seeds:
            rr = results[cfg_name][str(seed)]
            if "2048" in rr:
                vals2.append(float(rr["2048"]))
            if "16384" in rr:
                vals16.append(float(rr["16384"]))
        m2 = float(np.mean(vals2))
        s2 = float(np.std(vals2, ddof=0))
        m16 = float(np.mean(vals16))
        s16 = float(np.std(vals16, ddof=0))
        print(f"{cfg_name:<19} | {_fmt_ms(m2, s2):<19} | {_fmt_ms(m16, s16):<19}")

    print(f"\n[done] wrote {out_json}")


if __name__ == "__main__":
    main()

