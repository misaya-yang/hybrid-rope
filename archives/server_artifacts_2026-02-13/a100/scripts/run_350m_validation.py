"""
350M Model Validation: Polynomial vs Geometric RoPE Frequency Distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import os
import numpy as np
from pathlib import Path
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer

# ================= CONFIG =================
DEVICE = "cuda"
DTYPE = torch.bfloat16  # A100 supports bf16 natively
SEED = 42
WORK_DIR = "/opt/dfrope/results/350m_validation"

# Model config: ~350M parameters
MODEL_CFG = {
    "vocab_size": 50304,      # GPT-NeoX tokenizer vocab
    "hidden_size": 1024,
    "num_layers": 24,         # 24 layers
    "num_heads": 16,
    "head_dim": 64,           # 1024 / 16
    "intermediate_size": 4096, # 4x hidden
    "max_position_embeddings": 2048,
    "dropout": 0.0,
}

# Training config
TRAIN_CFG = {
    "seq_len": 2048,
    "total_tokens": 500_000_000,  # 500M tokens
    "batch_size": 32,             # micro batch (adjust if OOM)
    "grad_accum": 1,              # increase if OOM
    "lr": 3e-4,
    "weight_decay": 0.1,
    "warmup_frac": 0.02,
    "max_grad_norm": 1.0,
}

# Eval lengths
EVAL_LENGTHS = [2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384]
EVAL_CHUNKS_PER_LENGTH = 5

# ================= FREQUENCY DISTRIBUTIONS =================

def geometric_freq(K, theta):
    """Standard RoPE geometric frequency distribution."""
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))

def anchored_polynomial_freq(K, p, omega_max=1.0, omega_min=1e-4):
    """
    Anchored polynomial: fixes endpoints, polynomial interpolation in log space.
    
    p=1.0 is equivalent to geometric (linear in log space).
    p>1.0 allocates more channels to intermediate frequencies.
    """
    t = torch.arange(K, dtype=torch.float32) / (K - 1)
    log_omega = math.log(omega_max) + (t ** p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)

# ================= ROTARY EMBEDDING =================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=2048, inv_freq=None):
        super().__init__()
        if inv_freq is None:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq
        self._build_cache(max_seq)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq = seq_len
    
    def forward(self, seq_len):
        if seq_len > self.max_seq:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin

# ================= TRANSFORMER MODEL =================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg, rotary_emb):
        super().__init__()
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.hidden_size = cfg["hidden_size"]
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = rotary_emb
    
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(L)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        return self.o_proj(out)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg, rotary_emb):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.attn = Attention(cfg, rotary_emb)
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg, inv_freq=None):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        
        rotary = RotaryEmbedding(
            cfg["head_dim"],
            max_seq=cfg["max_position_embeddings"],
            inv_freq=inv_freq
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, rotary) for _ in range(cfg["num_layers"])
        ])
        self.ln_f = RMSNorm(cfg["hidden_size"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  Model parameters: {n_params/1e6:.1f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)
    
    def extend_rope(self, new_max_seq):
        for layer in self.layers:
            layer.attn.rotary_emb._build_cache(new_max_seq)

# ================= DATA =================

def get_data(tokenizer, split="train", max_tokens=500_000_000, seq_len=2048):
    """Load and tokenize TinyStories."""
    print(f"  Loading TinyStories ({split})...")
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    
    all_ids = []
    total = 0
    for item in ds:
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        all_ids.extend(ids)
        total += len(ids)
        if total >= max_tokens:
            break
        if total % 10_000_000 == 0:
            print(f"    Tokenized {total/1e6:.0f}M tokens...")
    
    n_chunks = len(all_ids) // seq_len
    all_ids = all_ids[:n_chunks * seq_len]
    chunks = torch.tensor(all_ids, dtype=torch.long).view(n_chunks, seq_len)
    print(f"  Got {n_chunks} chunks of {seq_len} tokens ({len(all_ids)/1e6:.1f}M total)")
    return chunks

# ================= TRAINING =================

def train(model, train_data, cfg, save_path):
    """Train the model."""
    model.train()
    model = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"]
    )
    
    n_chunks = train_data.shape[0]
    steps_per_epoch = n_chunks // cfg["batch_size"]
    total_steps = steps_per_epoch
    warmup_steps = int(total_steps * cfg["warmup_frac"])
    
    print(f"  Training: {total_steps} steps, warmup={warmup_steps}")
    
    scaler = GradScaler()
    t0 = time.time()
    losses = []
    
    torch.manual_seed(SEED)
    perm = torch.randperm(n_chunks)
    
    for step in range(total_steps):
        batch_idx = perm[step * cfg["batch_size"] : (step + 1) * cfg["batch_size"]]
        batch = train_data[batch_idx].to(DEVICE)
        
        if step < warmup_steps:
            lr = cfg["lr"] * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = lr
        
        with autocast("cuda", dtype=DTYPE):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1)
            )
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
        if (step + 1) % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (total_steps - step - 1)
            print(f"    step {step+1}/{total_steps}  loss={avg_loss:.4f}  lr={lr:.2e}  ETA={eta/60:.0f}min")
    
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}")
    return losses

# ================= EVALUATION =================

@torch.no_grad()
def evaluate(model, eval_data, lengths, chunks_per_length=5):
    """Evaluate PPL at multiple sequence lengths."""
    model.eval()
    model = model.to(DEVICE)
    
    max_len = max(lengths)
    model.extend_rope(max_len + 100)
    
    results = {}
    all_ids = eval_data.reshape(-1).tolist()
    
    for L in lengths:
        ppls = []
        for i in range(chunks_per_length):
            start = i * L
            if start + L > len(all_ids):
                break
            chunk = torch.tensor(all_ids[start:start+L], dtype=torch.long, device=DEVICE).unsqueeze(0)
            
            with autocast("cuda", dtype=DTYPE):
                logits = model(chunk[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    chunk[:, 1:].reshape(-1)
                )
            ppls.append(math.exp(loss.item()))
        
        if ppls:
            mean_ppl = sum(ppls) / len(ppls)
            std_ppl = (sum((p - mean_ppl)**2 for p in ppls) / len(ppls)) ** 0.5
            results[str(L)] = {"mean": mean_ppl, "std": std_ppl, "n": len(ppls)}
            print(f"    {L:>6}: PPL = {mean_ppl:.3f} ± {std_ppl:.3f} (n={len(ppls)})")
    
    return results

# ================= MAIN =================

def run_one_experiment(name, inv_freq, train_data, eval_data):
    """Run one complete experiment: train + eval."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*70}")
    
    print(f"  inv_freq range: [{inv_freq.min().item():.6e}, {inv_freq.max().item():.6e}]")
    print(f"  Dynamic range: {inv_freq.max().item() / inv_freq.min().item():.1f}")
    
    torch.manual_seed(SEED)
    model = GPTModel(MODEL_CFG, inv_freq=inv_freq)
    
    save_path = os.path.join(WORK_DIR, f"{name}_model.pt")
    losses = train(model, train_data, TRAIN_CFG, save_path)
    
    print(f"\n  Evaluating {name}...")
    results = evaluate(model, eval_data, EVAL_LENGTHS, EVAL_CHUNKS_PER_LENGTH)
    
    del model
    torch.cuda.empty_cache()
    
    return {"losses": losses[::50], "eval": results}

def main():
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  350M Model Validation: Polynomial vs Geometric RoPE")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_data = get_data(tokenizer, split="train",
                          max_tokens=TRAIN_CFG["total_tokens"],
                          seq_len=TRAIN_CFG["seq_len"])
    eval_data = get_data(tokenizer, split="validation",
                         max_tokens=5_000_000,
                         seq_len=max(EVAL_LENGTHS))
    
    K = MODEL_CFG["head_dim"] // 2
    
    geo_10k_freq = geometric_freq(K, 10000)
    geo_10k_omega_min = geo_10k_freq[-1].item()
    
    experiments = {
        "geo_1k": geometric_freq(K, 1000),
        "geo_10k": geometric_freq(K, 10000),
        "geo_100k": geometric_freq(K, 100000),
        "anchpoly_p3.8_omf0.3": anchored_polynomial_freq(
            K, p=3.8, omega_max=1.0,
            omega_min=geo_10k_omega_min * 0.3
        ),
        "anchpoly_p3.9_omf0.3": anchored_polynomial_freq(
            K, p=3.9, omega_max=1.0,
            omega_min=geo_10k_omega_min * 0.3
        ),
    }
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "model_cfg": MODEL_CFG,
        "train_cfg": TRAIN_CFG,
        "experiments": {}
    }
    
    for name, inv_freq in experiments.items():
        result = run_one_experiment(name, inv_freq, train_data, eval_data)
        all_results["experiments"][name] = result
        
        with open(os.path.join(WORK_DIR, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    
    header = f"{'Length':<8}"
    for name in experiments:
        header += f" | {name:<20}"
    print(header)
    print("-" * len(header))
    
    for L in EVAL_LENGTHS:
        row = f"{L:<8}"
        for name in experiments:
            r = all_results["experiments"][name]["eval"].get(str(L), {})
            if r:
                row += f" | {r['mean']:<20.2f}"
            else:
                row += f" | {'N/A':<20}"
        print(row)
    
    print(f"\nResults saved to {WORK_DIR}/results.json")

if __name__ == "__main__":
    main()
