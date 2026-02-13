"""\
350M Final Validation: Hybrid vs Geometric RoPE

Core invariants:
- Frequency computation functions are IDENTICAL to unified_search.py.
- Dataset + tokenizer: TinyStories streaming + gpt-neox-20b encode(add_special_tokens=False).
- BF16 autocast.

Pragmatics:
- For 500M tokens, do NOT hold all tokens in a Python list.
  We stream-tokenize into a uint16 memmap on disk, then train with a shuffled
  chunk index order (matching unified_search's `randperm`-style shuffling).

Outputs:
- /opt/dfrope/results/350m_final/results.json (updated after each experiment)
- /opt/dfrope/results/350m_final/run.log (if you run with tee)
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

# ==================== CONFIG ====================
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

WORK_DIR = "/opt/dfrope/results/350m_final"

# 350M-ish Model (as requested)
MODEL_CFG = {
    "vocab_size": 50304,
    "hidden_size": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "head_dim": 64,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
}

# Training (OOM-safe for A100)
TRAIN_CFG = {
    "seq_len": 2048,
    "total_tokens": 500_000_000,  # 500M
    "micro_batch": 8,
    "grad_accum": 4,  # effective batch=32
    "lr": 3e-4,
    "weight_decay": 0.1,
    "warmup_frac": 0.02,
    "max_grad_norm": 1.0,
}

EVAL_LENGTHS = [2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384]
EVAL_CHUNKS = 10

# ==================== FREQUENCY DISTRIBUTIONS ====================
# IDENTICAL to 50M experiments. DO NOT MODIFY.


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
        print(f"  Model: {n/1e6:.1f}M parameters")

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


# ==================== DATA (stream -> memmap) ====================


def _memmap_paths(out_dir: Path, name: str) -> Tuple[Path, Path]:
    return out_dir / f"{name}.tokens.u16", out_dir / f"{name}.meta.json"


def _write_json(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def build_or_load_token_memmap(
    out_dir: Path,
    *,
    dataset_split: str,
    tokenizer_name: str,
    target_tokens: int,
    seq_len: int,
) -> Tuple[np.memmap, Dict]:
    """Build a uint16 token memmap (vocab<65535) using streaming tokenization."""

    tok_path, meta_path = _memmap_paths(out_dir, f"{dataset_split}_{target_tokens}")

    if tok_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("dataset") == "roneneldan/TinyStories"
            and meta.get("split") == dataset_split
            and meta.get("tokenizer") == tokenizer_name
            and int(meta.get("target_tokens", 0)) == int(target_tokens)
            and int(meta.get("seq_len", 0)) == int(seq_len)
            and int(meta.get("written_tokens", 0)) >= int(target_tokens)
        ):
            mm = np.memmap(tok_path, dtype=np.uint16, mode="r", shape=(int(target_tokens),))
            return mm, meta

    out_dir.mkdir(parents=True, exist_ok=True)
    # Create and fill
    mm = np.memmap(tok_path, dtype=np.uint16, mode="w+", shape=(int(target_tokens),))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = load_dataset("roneneldan/TinyStories", split=dataset_split, streaming=True)

    written = 0
    last_report = 0
    t0 = time.time()
    for ex in ds:
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        if not ids:
            continue
        # Clip to remaining
        rem = int(target_tokens) - written
        if rem <= 0:
            break
        if len(ids) > rem:
            ids = ids[:rem]
        mm[written : written + len(ids)] = np.asarray(ids, dtype=np.uint16)
        written += len(ids)
        if written - last_report >= 10_000_000:
            last_report = written
            dt = time.time() - t0
            print(f"    {written/1e6:.0f}M tokens written ({dt/60:.1f} min)")

    mm.flush()

    # Ensure we have enough tokens to make full chunks.
    n_chunks = written // seq_len
    written_trim = n_chunks * seq_len
    meta = {
        "dataset": "roneneldan/TinyStories",
        "split": dataset_split,
        "tokenizer": tokenizer_name,
        "target_tokens": int(target_tokens),
        "seq_len": int(seq_len),
        "written_tokens": int(written),
        "usable_tokens": int(written_trim),
        "usable_chunks": int(n_chunks),
        "path": str(tok_path),
        "dtype": "uint16",
        "created_at": time.strftime("%Y-%m-%d_%H%M%S"),
    }
    _write_json(meta_path, meta)

    if written < target_tokens:
        print(f"[warn] only wrote {written} tokens (target {target_tokens}); usable {written_trim}")
    mm = np.memmap(tok_path, dtype=np.uint16, mode="r", shape=(int(target_tokens),))
    return mm, meta


def _chunks_from_memmap(mm: np.memmap, chunk_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    # chunk_ids: [B] int64
    # returns [B, seq_len] int64
    # Each chunk is contiguous in the underlying token stream.
    # We copy into a CPU tensor then .to(cuda).
    bs = int(chunk_ids.numel())
    out = torch.empty((bs, seq_len), dtype=torch.long)
    for i in range(bs):
        cid = int(chunk_ids[i].item())
        start = cid * seq_len
        out[i] = torch.from_numpy(np.asarray(mm[start : start + seq_len], dtype=np.uint16)).to(torch.long)
    return out


# ==================== TRAIN ====================


def train_model(model: GPT, train_mm: np.memmap, n_chunks: int, cfg: Dict) -> GPT:
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
    )

    micro_bs = int(cfg["micro_batch"])
    accum = int(cfg["grad_accum"])
    eff_bs = micro_bs * accum

    steps = n_chunks // eff_bs
    warmup = int(steps * float(cfg["warmup_frac"]))

    print(f"  Training: {steps} optimizer steps")
    print(f"  Micro batch: {micro_bs}, Grad accum: {accum}, Effective: {eff_bs}")

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    perm = torch.randperm(n_chunks, generator=gen)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    opt.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0

    # We iterate micro-steps (each consumes micro_bs chunks). Every `accum` micro-steps, do optimizer step.
    total_micro = steps * accum

    for s in range(total_micro):
        start = s * micro_bs
        end = start + micro_bs
        if end > perm.numel():
            break

        chunk_ids = perm[start:end]
        batch_cpu = _chunks_from_memmap(train_mm, chunk_ids, int(cfg["seq_len"]))
        batch = batch_cpu.to(DEVICE, non_blocking=True)

        # LR schedule by optimizer step
        if global_step < warmup:
            lr = cfg["lr"] * global_step / max(warmup, 1)
        else:
            progress = (global_step - warmup) / max(steps - warmup, 1)
            lr = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.amp.autocast("cuda", dtype=DTYPE):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)) / accum

        loss.backward()
        micro_step += 1

        if micro_step % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                real_loss = float(loss.item()) * accum
                elapsed = time.time() - t0
                eta = elapsed / max(global_step, 1) * (steps - global_step)
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"    step {global_step}/{steps}  loss={real_loss:.4f}  lr={lr:.2e}  mem={mem:.1f}GB  ETA={eta/60:.0f}min"
                )

    print(f"  Training complete in {(time.time() - t0)/60:.1f} minutes")
    return model


# ==================== EVAL ====================


@torch.no_grad()
def eval_model(model: GPT, val_tokens: torch.Tensor, lengths: List[int]) -> Dict[str, Dict]:
    model.eval()
    model.extend_rope(max(lengths) + 100)

    results: Dict[str, Dict] = {}
    for L in lengths:
        losses: List[float] = []
        for i in range(EVAL_CHUNKS):
            if (i + 1) * L > val_tokens.numel():
                break
            chunk = val_tokens[i * L : (i + 1) * L].unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
            losses.append(float(loss.item()))

        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            # std of per-chunk perplexities (not mathematically perfect, but stable and readable)
            ppls = [math.exp(l) for l in losses]
            mean_p = sum(ppls) / len(ppls)
            var = sum((x - mean_p) ** 2 for x in ppls) / len(ppls)
            std = math.sqrt(var)
            results[str(L)] = {"ppl": round(ppl, 3), "std": round(std, 3), "n": len(losses)}
            print(f"    L={L:>6}: PPL = {ppl:.3f} ± {std:.3f} (n={len(losses)})")

    return results


# ==================== MAIN ====================


def run_one(name: str, inv_freq: torch.Tensor, train_mm: np.memmap, n_chunks: int, val_tokens: torch.Tensor) -> Dict:
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Freq range: [{inv_freq.min().item():.2e}, {inv_freq.max().item():.4f}]")
    print(f"  Dynamic range: {inv_freq.max().item()/inv_freq.min().item():.1f}x")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    model = GPT(MODEL_CFG, inv_freq).to(DEVICE)
    model = train_model(model, train_mm, n_chunks, TRAIN_CFG)

    print(f"\n  Evaluating {name}...")
    results = eval_model(model, val_tokens, EVAL_LENGTHS)

    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only_hybrid",
        action="store_true",
        help="Run only hybrid experiment (used by guard/resume).",
    )
    args = ap.parse_args()

    out_dir = Path(WORK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  350M FINAL VALIDATION")
    print("  Hybrid(alpha=0.2, theta=100k) vs Geo(theta=500k)")
    print("=" * 70)

    tokenizer_name = "EleutherAI/gpt-neox-20b"

    # Build/load train memmap (streaming)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading TinyStories train ({TRAIN_CFG['total_tokens']/1e6:.0f}M tokens) -> memmap...")
    train_mm, train_meta = build_or_load_token_memmap(
        cache_dir,
        dataset_split="train",
        tokenizer_name=tokenizer_name,
        target_tokens=int(TRAIN_CFG["total_tokens"]),
        seq_len=int(TRAIN_CFG["seq_len"]),
    )
    n_chunks = int(train_meta["usable_chunks"])
    print(f"  Usable train chunks: {n_chunks}")

    # Val tokens (small enough to hold in memory)
    print("  Loading validation data...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    val_mm, val_meta = build_or_load_token_memmap(
        cache_dir,
        dataset_split="validation",
        tokenizer_name=tokenizer_name,
        target_tokens=5_000_000,
        seq_len=1,  # not used for val slicing; store as flat tokens
    )
    val_tokens = torch.from_numpy(np.asarray(val_mm, dtype=np.uint16)).to(torch.long)
    print(f"  Validation tokens: {val_tokens.numel()/1e6:.1f}M")

    K = MODEL_CFG["head_dim"] // 2  # 32

    geo_500k = geometric_freq(K, 500000)
    geo_100k = geometric_freq(K, 100000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    hybrid_100k = hybrid_freq(geo_100k, poly_100k, alpha=0.2)

    experiments: List[Tuple[str, torch.Tensor]] = [("geo_500k", geo_500k), ("hybrid_a0.2_t100k", hybrid_100k)]
    if args.only_hybrid:
        experiments = [("hybrid_a0.2_t100k", hybrid_100k)]

    out_json = out_dir / "results.json"

    if out_json.exists():
        try:
            all_results = json.loads(out_json.read_text())
        except Exception:
            all_results = {}
    else:
        all_results = {}

    all_results.setdefault("timestamp", time.strftime("%Y-%m-%d_%H%M%S"))
    all_results.setdefault("seed", SEED)
    all_results.setdefault("model_cfg", MODEL_CFG)
    all_results.setdefault("train_cfg", TRAIN_CFG)
    all_results.setdefault("train_meta", train_meta)
    all_results.setdefault("val_meta", val_meta)
    all_results.setdefault("experiments", {})

    for name, freq in experiments:
        if name in all_results["experiments"]:
            print(f"[resume] skip already present: {name}")
            continue
        res = run_one(name, freq, train_mm, n_chunks, val_tokens)
        all_results["experiments"][name] = res
        out_json.write_text(json.dumps(all_results, indent=2))

    # Final comparison
    print(f"\n{'='*70}")
    print("  350M FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Length':<8}", end="")
    for name, _ in experiments:
        print(f" | {name:<22}", end="")
    print()
    print(f"  {'-'*60}")

    for L in EVAL_LENGTHS:
        print(f"  {L:<8}", end="")
        for name, _ in experiments:
            r = all_results["experiments"].get(name, {}).get(str(L), {})
            if r:
                print(f" | {r['ppl']:<22.3f}", end="")
            else:
                print(f" | {'N/A':<22}", end="")
        print()

    geo_16k = all_results["experiments"].get("geo_500k", {}).get("16384", {}).get("ppl", 999)
    hyb_16k = all_results["experiments"].get("hybrid_a0.2_t100k", {}).get("16384", {}).get("ppl", 999)

    print(f"\n  PPL@16384: geo_500k={geo_16k:.3f}, hybrid={hyb_16k:.3f}")
    if hyb_16k < geo_16k:
        improvement = (geo_16k - hyb_16k) / geo_16k * 100
        print(f"  *** HYBRID WINS by {improvement:.1f}% ***")
        print("  *** theta=100k Hybrid > theta=500k Geometric ***")
    else:
        print("  Geometric wins. Hybrid advantage does not scale to this setup.")

    print(f"\n  Results saved: {out_json}")


if __name__ == "__main__":
    main()
