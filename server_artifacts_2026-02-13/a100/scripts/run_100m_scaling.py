"""100M scaling sprint: geo_500k vs hybrid_a0.2_t100k.

Notes
- The nominal config (12L/768/12 heads) is ~152M params.
- Micro-batch 32 OOMs on A100 80GB at seq_len=2048.
- We use micro_batch=8 with grad_accum=4 to keep effective batch=32.
- Frequency computation is identical to unified_search.py (DO NOT MODIFY).

Outputs
- /opt/dfrope/results/100m_scaling/results.json
- /opt/dfrope/results/100m_scaling/run.log (via tee)
"""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# ==================== CONFIG ====================
SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

WORK_DIR = "/opt/dfrope/results/100m_scaling"

MODEL_CFG = {
    "vocab_size": 50304,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "head_dim": 64,
    "intermediate_size": 3072,
    "max_position_embeddings": 2048,
}

TRAIN_TOKENS = 50_000_000
SEQ_LEN = 2048
MICRO_BATCH = 8
GRAD_ACCUM = 4
LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC = 0.02
MAX_GRAD_NORM = 1.0

EVAL_LENGTHS = [2048, 16384]
EVAL_CHUNKS = 5

# ==================== FREQUENCIES (IDENTICAL) ====================

def geometric_freq(K: int, theta: float) -> torch.Tensor:
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))


def anchored_poly_freq(K: int, theta_base: float, p: float = 3.9, omf: float = 0.3) -> torch.Tensor:
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = k / (K - 1)
    log_omega = math.log(omega_max) + (t**p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)


def hybrid_freq(freq_a: torch.Tensor, freq_b: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1 - alpha) * freq_a + alpha * freq_b


# ==================== MODEL ====================

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int, inv_freq: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq)
        self._max = 0
        self._build(max_seq)

    def _build(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_c", emb.cos(), persistent=False)
        self.register_buffer("sin_c", emb.sin(), persistent=False)
        self._max = seq_len

    def forward(self, L: int):
        if L > self._max:
            self._build(L)
        return self.cos_c[:L], self.sin_c[:L]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg, rope: RotaryEmbedding):
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]
        self.hd = cfg["head_dim"]
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, rope: RotaryEmbedding):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = Attention(cfg, rope)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, inv_freq: torch.Tensor):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def extend_rope(self, L: int) -> None:
        self.blocks[0].attn.rope._build(L)


# ==================== DATA ====================

def load_train(tokenizer, max_tokens: int, seq_len: int) -> torch.Tensor:
    print(f"  Loading TinyStories train ({max_tokens/1e6:.0f}M tokens)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    ids = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    n = len(ids) // seq_len
    print(f"  Got {n} chunks")
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def load_val(tokenizer, max_tokens: int = 5_000_000) -> torch.Tensor:
    print("  Loading validation data...")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    print(f"  Got {len(ids)/1e6:.1f}M validation tokens")
    return torch.tensor(ids, dtype=torch.long)


# ==================== TRAIN / EVAL ====================

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model: GPT, data: torch.Tensor) -> dict:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    eff_bs = MICRO_BATCH * GRAD_ACCUM
    steps = len(data) // eff_bs
    warmup = int(steps * WARMUP_FRAC)

    print(f"  Training: {steps} steps (micro={MICRO_BATCH}, accum={GRAD_ACCUM}, eff={eff_bs})")
    perm = torch.randperm(len(data))

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    losses = []

    for step in range(steps):
        if step < warmup:
            lr = LR * step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(steps - warmup, 1)
            lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        for a in range(GRAD_ACCUM):
            start = (step * eff_bs) + a * MICRO_BATCH
            idx = perm[start : start + MICRO_BATCH]
            batch = data[idx].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                (loss / GRAD_ACCUM).backward()
            losses.append(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(step + 1, 1) * (steps - step - 1)
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"    step {step}/{steps} loss={losses[-1]:.4f} lr={lr:.2e} mem={mem:.1f}GB ETA={eta/60:.0f}min")

    return {"steps": steps, "mean_loss": float(sum(losses) / max(len(losses), 1))}


@torch.no_grad()
def eval_model(model: GPT, val_data: torch.Tensor) -> dict:
    model.eval()
    model.extend_rope(max(EVAL_LENGTHS) + 100)
    out = {}
    for L in EVAL_LENGTHS:
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
            out[str(L)] = round(ppl, 3)
            print(f"    L={L}: PPL={ppl:.3f}")
    return out


def run_one(name: str, inv_freq: torch.Tensor, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    print(f"\n{'='*70}\n  {name}\n  freq range: [{inv_freq.min().item():.2e}, {inv_freq.max().item():.4f}]\n{'='*70}")
    _set_seed(SEED)
    model = GPT(MODEL_CFG, inv_freq).to(DEVICE)
    train_stats = train_model(model, train_data)
    eval_stats = eval_model(model, val_data)

    del model
    torch.cuda.empty_cache()
    return {"train": train_stats, "eval": eval_stats}


def main() -> None:
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("=" * 70)
    print("  100M SCALING SPRINT: geo_500k vs hybrid_a0.2_t100k")
    print("=" * 70)
    print(f"  torch={torch.__version__}")

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    train_data = load_train(tok, TRAIN_TOKENS, SEQ_LEN)
    val_data = load_val(tok)

    K = MODEL_CFG["head_dim"] // 2
    geo_500k = geometric_freq(K, 500000)
    geo_100k = geometric_freq(K, 100000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    hybrid_100k = hybrid_freq(geo_100k, poly_100k, alpha=0.2)

    experiments = [
        ("geo_500k", geo_500k),
        ("hybrid_a0.2_t100k", hybrid_100k),
    ]

    results = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "seed": SEED,
        "model_cfg": MODEL_CFG,
        "train_cfg": {
            "train_tokens": TRAIN_TOKENS,
            "seq_len": SEQ_LEN,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
        },
        "eval_cfg": {"lengths": EVAL_LENGTHS, "chunks": EVAL_CHUNKS},
        "experiments": {},
    }

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for name, inv in experiments:
        results["experiments"][name] = run_one(name, inv, train_data, val_data)
        with open(f"{WORK_DIR}/results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  SUMMARY (PPL)")
    print("=" * 70)
    for name, _ in experiments:
        e = results["experiments"][name]["eval"]
        print(f"  {name:>18}: 2048={e.get('2048','N/A')}  16384={e.get('16384','N/A')}")


if __name__ == "__main__":
    main()
