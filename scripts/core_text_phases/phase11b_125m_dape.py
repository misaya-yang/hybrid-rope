#!/usr/bin/env python3
"""
Phase 11B: 125M L=256 Scaling Law + DAPE Compatibility
1. Scaling law: 125M × {Geo, EVQ τ=2.0, EVQ τ=4.0} × 3 seeds
2. DAPE compatibility: 125M × {Geo, EVQ τ=4.0} × {plain, +DAPE} × 3 seeds

DAPE = Kerple bias + MLP refinement on attention scores

Paper Role:  Fig 3 panel (a) — EVQ vs DAPE learnable PE comparison (Claim C2)
             Table 4 — PE-dominant extreme extrapolation results (125M)
Input:       FineWeb-Edu streaming data (100M tokens)
Output:      results/core_text/phase11b/ (JSON per-seed per-method)
Seeds:       42, 123, 7 (3-seed)
"""

import json, math, os, sys, time, gc, hashlib
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_evq_sweep import (
    GPT, Block, Attention, MLP, RMSNorm, RotaryEmbedding, apply_rope,
    DEVICE, DTYPE, USE_AUTOCAST,
    set_seed, get_batch_from_data,
)

# ── Config ──────────────────────────────────────────────────────────────
BASE = 500_000.0
DIM = 64
SEQ_LEN = 256
TOKENS = 100_000_000  # 100M tokens

EVAL_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
EVAL_CHUNKS = 8

WORK = Path("/root/autodl-tmp/evq_phase11b_125m")
DATA_CACHE_DIR = WORK / "data"

CFG_125M = dict(
    vocab_size=50304,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    head_dim=64,
    intermediate_size=3072,
    max_position_embeddings=SEQ_LEN,
    seq_len=SEQ_LEN,
    train_tokens=TOKENS,
    lr=6e-4,
    batch_size=256,
    micro_batch_size=64,
    grad_accum=4,
)


# ── DAPE Components ────────────────────────────────────────────────────
class KerpleBias(nn.Module):
    """Kerple: -p * log(1 + a * |m-n|), per-head learned p, a."""
    def __init__(self, n_heads):
        super().__init__()
        self.log_p = nn.Parameter(torch.zeros(n_heads, 1, 1))  # (H, 1, 1)
        self.log_a = nn.Parameter(torch.zeros(n_heads, 1, 1))
        nn.init.normal_(self.log_p, mean=0.0, std=0.02)
        nn.init.normal_(self.log_a, mean=0.0, std=0.02)

    def forward(self, seq_len):
        pos = torch.arange(seq_len, device=self.log_p.device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (L, L)
        p = F.softplus(self.log_p)  # ensure positive
        a = F.softplus(self.log_a)
        bias = -p * torch.log1p(a * dist.unsqueeze(0))  # (H, L, L)
        return bias


class DAPERefine(nn.Module):
    """DAPE MLP refinement: MLP(concat(attn_scores, kerple_bias)) → residual."""
    def __init__(self, n_heads, hidden_mult=2):
        super().__init__()
        self.kerple = KerpleBias(n_heads)
        # MLP: input = 2 (attn_score + kerple), output = 1
        hidden = n_heads * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(2 * n_heads, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads),
        )

    def forward(self, attn_scores, seq_len):
        """
        attn_scores: (B, H, L, L)
        Returns refined scores: (B, H, L, L)
        """
        B, H, L, _ = attn_scores.shape
        kerple = self.kerple(seq_len)  # (H, L, L)
        kerple = kerple.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, L, L)

        # Concat along head dim, pass through MLP
        # Reshape: (B, H, L, L) -> (B, L, L, H)
        attn_t = attn_scores.permute(0, 2, 3, 1)  # (B, L, L, H)
        kerple_t = kerple.permute(0, 2, 3, 1)       # (B, L, L, H)
        combined = torch.cat([attn_t, kerple_t], dim=-1)  # (B, L, L, 2H)

        refined = self.net(combined)  # (B, L, L, H)
        refined = refined.permute(0, 3, 1, 2)  # (B, H, L, L)

        return attn_scores + kerple + refined


class DAPEAttention(nn.Module):
    """Attention with optional DAPE refinement."""
    def __init__(self, cfg, rope, use_dape=False):
        super().__init__()
        h = cfg["hidden_size"]
        n = cfg["num_heads"]
        d = cfg["head_dim"]
        self.n_heads = n
        self.head_dim = d
        self.qkv = nn.Linear(h, 3 * n * d, bias=False)
        self.o = nn.Linear(n * d, h, bias=False)
        self.rope = rope
        self.use_dape = use_dape
        if use_dape:
            self.dape = DAPERefine(n)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, D)
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.use_dape:
            # Manual attention with DAPE refinement
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)
            attn_scores = self.dape(attn_scores, L)
            # Causal mask
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v)
        else:
            # Standard SDPA
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.o(out.transpose(1, 2).reshape(B, L, -1))


class DAPEBlock(nn.Module):
    def __init__(self, cfg, rope, use_dape=False):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = DAPEAttention(cfg, rope, use_dape=use_dape)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DAPEGPT(nn.Module):
    """GPT with optional DAPE on all attention layers."""
    def __init__(self, cfg, inv_freq, use_dape=False):
        super().__init__()
        self._num_layers = cfg["num_layers"]
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        rope = RotaryEmbedding(cfg["head_dim"], cfg["max_position_embeddings"], inv_freq)
        self.blocks = nn.ModuleList(
            [DAPEBlock(cfg, rope, use_dape=use_dape) for _ in range(cfg["num_layers"])]
        )
        self.ln = RMSNorm(cfg["hidden_size"])
        self.head = nn.Linear(cfg["vocab_size"], cfg["hidden_size"], bias=False)
        # Fix: head should be (hidden -> vocab), weight tied with emb
        del self.head
        self.head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init)
        residual_scale = 1.0 / math.sqrt(2 * self._num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o.weight, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp.down.weight, std=0.02 * residual_scale)
        n = sum(p.numel() for p in self.parameters())
        print(f"  Model params: {n / 1e6:.1f}M (DAPE={use_dape})")

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


# ── Frequency builders ──────────────────────────────────────────────────
def geometric_inv_freq(dim=DIM, base=BASE):
    n = dim // 2
    return torch.tensor([1.0 / (base ** (2 * i / dim)) for i in range(n)], dtype=torch.float32)


def evq_cosh_inv_freq(dim=DIM, tau=4.0, base=BASE):
    if abs(tau) < 1e-8:
        return geometric_inv_freq(dim, base)
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    freqs = 1.0 / (base ** phi)
    return freqs.float()


# ── Data ────────────────────────────────────────────────────────────────
def load_train_data(seq_len=SEQ_LEN, max_tokens=TOKENS + 5_000_000):
    cache_path = DATA_CACHE_DIR / f"train_fineweb-edu_{max_tokens}_{seq_len}.pt"
    if cache_path.exists():
        print(f"  Loading cached data: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    # Try existing caches on this server
    candidates = [
        Path("/root/autodl-tmp/evq_passkey_mix_10pct/train_fineweb-edu_100000000_2048.pt"),
        Path("/root/autodl-tmp/evq_passkey_mix_5pct/train_fineweb-edu_100000000_2048.pt"),
        Path("/root/autodl-tmp/evq_phase14c/data/train_fineweb-edu_100000000_2048.pt"),
        Path("/root/autodl-tmp/phase16_rsweep/train_fineweb-edu_50000000_2048.pt"),
    ]
    for src in candidates:
        if src.exists():
            print(f"  Loading from {src} and re-chunking to {seq_len}...")
            data = torch.load(src, weights_only=True)
            flat = data.reshape(-1)[:max_tokens]
            n = len(flat) // seq_len
            result = flat[:n * seq_len].reshape(n, seq_len)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, cache_path)
            print(f"  Re-chunked: {data.shape} -> {result.shape}")
            del data, flat
            gc.collect()
            return result

    raise RuntimeError("No cached training data found!")


def load_validation_data():
    candidates = [
        Path("/root/autodl-tmp/evq_passkey_mix_10pct/val_fineweb-edu_5000000.pt"),
        Path("/root/autodl-tmp/evq_passkey_mix_5pct/val_fineweb-edu_5000000.pt"),
        Path("/root/autodl-tmp/phase16_rsweep/val_fineweb-edu_5000000.pt"),
    ]
    for src in candidates:
        if src.exists():
            print(f"  Loading val from {src}")
            return torch.load(src, weights_only=True)
    raise RuntimeError("No cached val data found!")


# ── Eval ────────────────────────────────────────────────────────────────
def eval_ppl(model, val_data, eval_lengths=EVAL_LENGTHS, n_chunks=EVAL_CHUNKS):
    model.eval()
    model.extend_rope(max(eval_lengths) + 100)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    rng = np.random.RandomState(9999)
    results = {}
    for L in eval_lengths:
        losses = []
        max_start = len(val_data) - L
        if max_start <= 0:
            continue
        offsets = sorted(rng.choice(max_start, size=min(n_chunks, max_start // L), replace=False))
        for offset in offsets:
            chunk = val_data[offset:offset + L].unsqueeze(0).to(DEVICE)
            try:
                with torch.no_grad(), ctx:
                    logits = model(chunk[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           chunk[:, 1:].reshape(-1))
                losses.append(loss.item())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    L={L}: OOM, skipping")
                    torch.cuda.empty_cache()
                    break
                raise
            finally:
                del chunk
        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results[str(L)] = round(ppl, 3)
            print(f"    L={L:>5d}: PPL={ppl:.2f}  ({len(losses)} chunks)")
    return results


# ── Training ────────────────────────────────────────────────────────────
def train_model(model, train_data, cfg, seed=42):
    set_seed(seed)
    total_tokens = cfg["train_tokens"]
    bs = cfg["batch_size"]
    mbs = cfg["micro_batch_size"]
    ga = cfg["grad_accum"]
    seq_len = cfg["seq_len"]
    lr = cfg["lr"]

    tokens_per_step = bs * seq_len
    total_steps = total_tokens // tokens_per_step
    warmup_steps = min(200, total_steps // 10)

    print(f"  Training: {total_tokens/1e6:.0f}M tok, bs={bs} (micro={mbs}×ga={ga}), "
          f"L={seq_len}, steps={total_steps}, lr={lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps, eta_min=lr * 0.1)
    ctx = torch.amp.autocast("cuda", dtype=DTYPE) if USE_AUTOCAST else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))

    model.train()
    n_samples = len(train_data)
    perm = torch.randperm(n_samples)
    ptr = 0
    t0 = time.time()
    log_interval = max(1, total_steps // 40)

    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            for pg in opt.param_groups:
                pg["lr"] = lr * step / warmup_steps

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(ga):
            if ptr + mbs > n_samples:
                perm = torch.randperm(n_samples)
                ptr = 0
            indices = perm[ptr:ptr + mbs]
            ptr += mbs
            batch = get_batch_from_data(train_data, indices).to(DEVICE)

            with ctx:
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       batch[:, 1:].reshape(-1))
                loss_scaled = loss / ga

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / ga

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step > warmup_steps:
            sched.step()

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tps = (step * tokens_per_step) / elapsed
            print(f"    step {step:>6d}/{total_steps} ({step/total_steps*100:5.1f}%) | "
                  f"loss={accum_loss:.4f} | lr={opt.param_groups[0]['lr']:.2e} | "
                  f"{tps/1e6:.2f}M tok/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    return elapsed


# ── Run one experiment ──────────────────────────────────────────────────
def run_one(run_id, inv_freq, seed, train_data, val_data, cfg, use_dape=False):
    run_dir = WORK / run_id
    result_path = run_dir / "result.json"

    if result_path.exists():
        print(f"\n  SKIP {run_id} (result exists)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  RUN: {run_id}")
    freq_hash = hashlib.sha256(inv_freq.numpy().tobytes()).hexdigest()[:12]
    print(f"  inv_freq: min={inv_freq.min():.8f} max={inv_freq.max():.6f} hash={freq_hash}")
    print(f"{'='*70}")

    set_seed(seed)
    if use_dape:
        model = DAPEGPT(cfg, inv_freq, use_dape=True).to(DEVICE)
    else:
        model = GPT(cfg, inv_freq).to(DEVICE)

    train_time = train_model(model, train_data, cfg, seed=seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")

    print(f"\n  Evaluating PPL...")
    ppl = eval_ppl(model, val_data)

    result = {
        "run_id": run_id,
        "seed": seed,
        "ppl": ppl,
        "train_time_sec": round(train_time, 1),
        "use_dape": use_dape,
        "inv_freq_hash": freq_hash,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ── Main ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,137,256")
    parser.add_argument("--phase", default="scaling",
                        choices=["scaling", "dape", "all"],
                        help="scaling=125M scaling law, dape=DAPE compat, all=both")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--micro_batch", type=int, default=64,
                        help="64 for plain, 32 for DAPE (manual attn needs more VRAM)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = CFG_125M.copy()
    cfg["train_tokens"] = args.tokens
    cfg["batch_size"] = args.batch_size
    cfg["micro_batch_size"] = args.micro_batch
    cfg["grad_accum"] = args.batch_size // args.micro_batch

    WORK.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 11B: 125M L={SEQ_LEN} (phase={args.phase})")
    print(f"  Seeds: {seeds}, Tokens: {args.tokens/1e6:.0f}M")
    print(f"  Batch: {args.batch_size} (micro={args.micro_batch}×ga={cfg['grad_accum']})")

    # Load data
    print("\n[1] Loading data...")
    train_data = load_train_data(max_tokens=args.tokens + 5_000_000)
    val_data = load_validation_data()
    print(f"  Train: {train_data.shape}, Val: {val_data.shape}")

    # ── Phase 1: Scaling law ────────────────────────────────────────────
    all_results = {}
    if args.phase in ("scaling", "all"):
        print(f"\n{'#'*70}")
        print(f"  PHASE 1: 125M Scaling Law (Geo vs EVQ τ=2.0 vs EVQ τ=4.0)")
        print(f"{'#'*70}")

        methods = {"geo": geometric_inv_freq(), "evq2.0": evq_cosh_inv_freq(tau=2.0),
                   "evq4.0": evq_cosh_inv_freq(tau=4.0)}

        for name, inv_freq in methods.items():
            for seed in seeds:
                r = run_one(f"125m_{name}_seed{seed}", inv_freq, seed,
                           train_data, val_data, cfg)
                all_results[r["run_id"]] = r

    # ── Phase 2: DAPE compatibility ─────────────────────────────────────
    if args.phase in ("dape", "all"):
        print(f"\n{'#'*70}")
        print(f"  PHASE 2: DAPE Compatibility (Geo+DAPE vs EVQ+DAPE)")
        print(f"{'#'*70}")

        dape_methods = {"geo": geometric_inv_freq(), "evq4.0": evq_cosh_inv_freq(tau=4.0)}

        for name, inv_freq in dape_methods.items():
            for seed in seeds:
                r = run_one(f"125m_{name}_dape_seed{seed}", inv_freq, seed,
                           train_data, val_data, cfg, use_dape=True)
                all_results[r["run_id"]] = r

    # Save aggregate
    agg_path = WORK / "all_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  PHASE 11B RESULTS: 125M, L_train={SEQ_LEN}")
    print(f"{sep}")

    def mean_ppl(prefix, L):
        vals = []
        for s in seeds:
            r = all_results.get(f"{prefix}_seed{s}", {})
            v = r.get("ppl", {}).get(str(L))
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    # Print scaling law results
    if args.phase in ("scaling", "all"):
        header = f"  {'Method':>12s}" + "".join(f" {'L='+str(L):>8s}" for L in EVAL_LENGTHS)
        print(header)
        print("  " + "-" * 75)
        for m in ["geo", "evq2.0", "evq4.0"]:
            line = f"  {m:>12s}"
            for L in EVAL_LENGTHS:
                v = mean_ppl(f"125m_{m}", L)
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)

        print(f"\n  DELTA vs Geo:")
        for m in ["evq2.0", "evq4.0"]:
            line = f"  {m:>12s}:"
            for L in EVAL_LENGTHS:
                geo = mean_ppl("125m_geo", L)
                evq = mean_ppl(f"125m_{m}", L)
                if geo and evq:
                    line += f" {(evq/geo-1)*100:>+7.1f}%"
                else:
                    line += f" {'--':>8s}"
            print(line)

    # Print DAPE results
    if args.phase in ("dape", "all"):
        print(f"\n  DAPE Compatibility:")
        header = f"  {'Method':>16s}" + "".join(f" {'L='+str(L):>8s}" for L in EVAL_LENGTHS)
        print(header)
        print("  " + "-" * 80)
        for m in ["geo", "evq4.0", "geo_dape", "evq4.0_dape"]:
            line = f"  {m:>16s}"
            for L in EVAL_LENGTHS:
                v = mean_ppl(f"125m_{m}", L)
                line += f" {v:>8.1f}" if v else f" {'--':>8s}"
            print(line)

        # DAPE additive benefit
        print(f"\n  DAPE benefit (DAPE - plain):")
        for m in ["geo", "evq4.0"]:
            line = f"  {m:>12s}:"
            for L in EVAL_LENGTHS:
                plain = mean_ppl(f"125m_{m}", L)
                dape = mean_ppl(f"125m_{m}_dape", L)
                if plain and dape:
                    line += f" {(dape/plain-1)*100:>+7.1f}%"
                else:
                    line += f" {'--':>8s}"
            print(line)

        # EVQ advantage with and without DAPE
        print(f"\n  EVQ τ=4.0 advantage (vs Geo):")
        for suffix in ["", "_dape"]:
            label = f"plain" if not suffix else "DAPE"
            line = f"  {label:>12s}:"
            for L in EVAL_LENGTHS:
                geo = mean_ppl(f"125m_geo{suffix}", L)
                evq = mean_ppl(f"125m_evq4.0{suffix}", L)
                if geo and evq:
                    line += f" {(evq/geo-1)*100:>+7.1f}%"
                else:
                    line += f" {'--':>8s}"
            print(line)

    print(f"\n  Results saved to {agg_path}")


if __name__ == "__main__":
    main()
