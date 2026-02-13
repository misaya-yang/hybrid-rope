import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 42
DEVICE = "cuda"
DTYPE = torch.bfloat16

WORK_DIR = "/opt/dfrope/results/50m_yarn_compare_v2"
SRC_DIR = "/opt/dfrope/results/50m_yarn_compare"

MODEL_CFG = {
    "vocab_size": 50304,
    "hidden_size": 512,
    "num_layers": 6,
    "num_heads": 8,
    "head_dim": 64,
    "intermediate_size": 2048,
    "max_position_embeddings": 2048,
}

EVAL_LENGTHS = [2048, 4096, 8192, 12288, 16384]
EVAL_CHUNKS = 10
TRAIN_LEN = 2048


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

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def set_rope_scale(self, scale: float):
        self.blocks[0].attn.rope.set_scale(scale)

    def extend_rope(self, L):
        self.blocks[0].attn.rope._build(L)


def load_val(tokenizer, max_tokens=5_000_000):
    print("Loading validation data...")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids = []
    for x in ds:
        ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def eval_progressive_yarn(model, val_data, lengths):
    model.eval()
    results = {}
    for L in lengths:
        # Keep training distribution untouched; progressively scale only beyond train length.
        scale = 1.0 if L <= TRAIN_LEN else min(8.0, L / TRAIN_LEN)
        model.set_rope_scale(scale)
        model.extend_rope(L + 8)

        losses = []
        for i in range(EVAL_CHUNKS):
            if (i + 1) * L > len(val_data):
                break
            chunk = val_data[i * L : (i + 1) * L].unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk[:, 1:].reshape(-1))
            losses.append(loss.item())
        ppl = math.exp(sum(losses) / len(losses))
        results[str(L)] = round(ppl, 3)
        print(f"  progressive_yarn L={L} scale={scale:.2f} ppl={ppl:.3f}")
    return results


@torch.no_grad()
def eval_native(model, val_data, lengths):
    model.eval()
    model.set_rope_scale(1.0)
    model.extend_rope(max(lengths) + 8)
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
        ppl = math.exp(sum(losses) / len(losses))
        results[str(L)] = round(ppl, 3)
        print(f"  native L={L} ppl={ppl:.3f}")
    return results


def main():
    torch.manual_seed(SEED)
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    val = load_val(tok)

    K = MODEL_CFG["head_dim"] // 2
    geo_500k = geometric_freq(K, 500000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    hybrid = hybrid_freq(geometric_freq(K, 100000), poly_100k, alpha=0.2)

    geo = GPT(MODEL_CFG, geo_500k).to(DEVICE)
    hyb = GPT(MODEL_CFG, hybrid).to(DEVICE)

    geo.load_state_dict(torch.load(f"{SRC_DIR}/geo_model.pt", map_location=DEVICE))
    hyb.load_state_dict(torch.load(f"{SRC_DIR}/hybrid_model.pt", map_location=DEVICE))

    print("[Eval] hybrid_native")
    hybrid_native = eval_native(hyb, val, EVAL_LENGTHS)
    print("[Eval] geo_native")
    geo_native = eval_native(geo, val, EVAL_LENGTHS)
    print("[Eval] geo_yarn_progressive")
    geo_yarn_progressive = eval_progressive_yarn(geo, val, EVAL_LENGTHS)

    out = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "note": "Corrected YaRN comparison: no scaling at <=2048, progressive scaling above train length.",
        "experiments": {
            "hybrid_native": hybrid_native,
            "geo_native": geo_native,
            "geo_yarn_progressive": geo_yarn_progressive,
        },
    }

    with open(f"{WORK_DIR}/results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY @16384 ===")
    print("hybrid_native:", hybrid_native.get("16384"))
    print("geo_native:", geo_native.get("16384"))
    print("geo_yarn_progressive:", geo_yarn_progressive.get("16384"))
    print("saved:", f"{WORK_DIR}/results.json")


if __name__ == "__main__":
    import time
    main()
