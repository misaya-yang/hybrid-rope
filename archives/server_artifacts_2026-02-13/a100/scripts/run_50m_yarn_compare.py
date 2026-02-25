import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

WORK_DIR = "/opt/dfrope/results/50m_yarn_compare"

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
EVAL_LENGTHS = [2048, 4096, 8192, 12288, 16384]
EVAL_CHUNKS = 10


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
        self.position_scale = 1.0
        self._build(max_seq)

    def set_scale(self, scale: float):
        self.position_scale = float(scale)
        self._build(self._max)

    def _build(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # YaRN/PI-style inference scaling: divide positions by scale (>1 extends context)
        t = t / float(self.position_scale)
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

    def set_rope_scale(self, scale: float):
        self.blocks[0].attn.rope.set_scale(scale)

    def extend_rope(self, L):
        self.blocks[0].attn.rope._build(L)


def load_data(tokenizer, max_tokens, seq_len):
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


def load_val(tokenizer, max_tokens=5_000_000):
    print("  Loading validation data...")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids, dtype=torch.long)


def train_model(model, data):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    steps = len(data) // BATCH_SIZE
    warmup = int(steps * 0.02)
    torch.manual_seed(SEED)
    perm = torch.randperm(len(data))
    t0 = time.time()

    for s in range(steps):
        batch = data[perm[s * BATCH_SIZE : (s + 1) * BATCH_SIZE]].to(DEVICE)
        if s < warmup:
            lr = LR * s / max(warmup, 1)
        else:
            lr = LR * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1)))
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.amp.autocast("cuda", dtype=DTYPE):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if s % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (steps - s - 1)
            print(f"    step {s}/{steps}  loss={loss.item():.4f}  lr={lr:.2e}  ETA={eta/60:.0f}min")

    return model


@torch.no_grad()
def eval_model(model, val_data, lengths, rope_scale=1.0):
    model.eval()
    model.set_rope_scale(rope_scale)
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


def main():
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("  50M QUICK RETRAIN + YaRN COMPARE")
    print("=" * 68)

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    train_data = load_data(tok, TRAIN_TOKENS, SEQ_LEN)
    val_data = load_val(tok)

    K = MODEL_CFG["head_dim"] // 2
    geo_500k = geometric_freq(K, 500000)
    geo_100k = geometric_freq(K, 100000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    hybrid = hybrid_freq(geo_100k, poly_100k, alpha=0.2)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "seed": SEED,
        "train_tokens": TRAIN_TOKENS,
        "eval_lengths": EVAL_LENGTHS,
        "experiments": {},
    }

    print("\n[1/3] Train geo_500k")
    torch.manual_seed(SEED)
    geo_model = GPT(MODEL_CFG, geo_500k).to(DEVICE)
    geo_model = train_model(geo_model, train_data)
    geo_ckpt = f"{WORK_DIR}/geo_model.pt"
    torch.save(geo_model.state_dict(), geo_ckpt)
    print(f"  saved {geo_ckpt}")

    print("\n[2/3] Train hybrid_a0.2_t100k")
    torch.manual_seed(SEED)
    hyb_model = GPT(MODEL_CFG, hybrid).to(DEVICE)
    hyb_model = train_model(hyb_model, train_data)
    hyb_ckpt = f"{WORK_DIR}/hybrid_model.pt"
    torch.save(hyb_model.state_dict(), hyb_ckpt)
    print(f"  saved {hyb_ckpt}")

    print("\n[3/3] Evaluate")
    print("  Hybrid @16K (native)")
    hyb_eval = eval_model(hyb_model, val_data, EVAL_LENGTHS, rope_scale=1.0)

    print("\n  Geo @16K (native)")
    geo_native = eval_model(geo_model, val_data, EVAL_LENGTHS, rope_scale=1.0)

    print("\n  Geo @16K (YaRN-style scale=8)")
    geo_yarn = eval_model(geo_model, val_data, EVAL_LENGTHS, rope_scale=8.0)

    all_results["experiments"]["hybrid_native"] = hyb_eval
    all_results["experiments"]["geo_native"] = geo_native
    all_results["experiments"]["geo_yarn_scale8"] = geo_yarn

    out_json = f"{WORK_DIR}/results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 68)
    print("  TABLE: Hybrid vs Geo vs Geo+YaRN")
    print("=" * 68)
    print(f"{'Length':<8} | {'Hybrid':<10} | {'Geo':<10} | {'Geo+YaRN(8)':<12}")
    print("-" * 68)
    for L in EVAL_LENGTHS:
        k = str(L)
        h = hyb_eval.get(k, float("nan"))
        g = geo_native.get(k, float("nan"))
        y = geo_yarn.get(k, float("nan"))
        print(f"{L:<8} | {h:<10.3f} | {g:<10.3f} | {y:<12.3f}")

    print("\n16K summary:")
    print(f"  Hybrid(native): {hyb_eval.get('16384')}")
    print(f"  Geo(native): {geo_native.get('16384')}")
    print(f"  Geo+YaRN(s=8): {geo_yarn.get('16384')}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
