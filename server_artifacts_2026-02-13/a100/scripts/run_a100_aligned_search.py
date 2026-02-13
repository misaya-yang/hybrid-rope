#!/usr/bin/env python3
"""
A100 Aligned Search - DF-RoPE Frequency Distribution Experiment
Aligned with H800 requirements for fair comparison.

Key requirements:
- tokenizer: EleutherAI/pythia-70m
- lr: 6e-4
- eval: RANDOM start positions (not sequential)
- val_tokens: 2,500,000
- micro_batch: 2, grad_accum: 16 (effective batch 32)
- Model: 6L/512/8heads, vocab=50304
- train_tokens: 50M
- eval lengths: [2048, 16384] (and optionally more)
- seed: 42
- bf16 precision

Configs (14 total):
1. geo_10k_align (baseline)
2-8. sigmoid variations (steep 6-10, mid 0.4-0.6, omf 0.2-0.5, theta 10k)
9-10. sigmoid theta 30k, 50k
11-12. anchored poly theta 100k, 500k
13-14. hybrid mixes
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Enable HF mirror for China access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    seq_len: int = 2048
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    intermediate: int = 2048
    vocab_size: int = 50304
    lr: float = 6e-4
    warmup_ratio: float = 0.02
    micro_batch: int = 2
    grad_accum: int = 16  # effective batch = 32
    max_steps: int = 0  # calculated from tokens
    train_tokens: int = 50_000_000
    val_tokens: int = 2_500_000
    eval_interval: int = 1000
    eval_lengths: List[int] = None
    seed: int = 42
    
    def __post_init__(self):
        if self.eval_lengths is None:
            self.eval_lengths = [2048, 16384]
        tokens_per_step = self.micro_batch * self.grad_accum * self.seq_len
        self.max_steps = self.train_tokens // tokens_per_step


@dataclass  
class RopeConfig:
    kind: str = "standard"  # "standard" or "custom"
    theta: float = 10000.0
    custom_omega: Optional[List[float]] = None


# ============================================================================
# Frequency Functions
# ============================================================================

def geometric_freq(K: int, theta: float) -> np.ndarray:
    """Standard RoPE geometric frequency distribution."""
    idx = np.arange(K, dtype=np.float64)
    return 1.0 / np.power(theta, idx / K)


def sigmoid_freq(
    K: int,
    theta_base: float,
    steepness: float,
    midpoint: float,
    omf: float,  # omega_min_factor
) -> np.ndarray:
    """
    Sigmoid-based frequency allocation.
    omega_max from geometric theta_base, omega_min = omega_max * omf
    """
    # Get omega_max from geometric distribution at first position
    omega_max = 1.0  # Normalized
    omega_min = omega_max * omf
    
    t = np.arange(K, dtype=np.float64) / (K - 1)
    s = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
    log_omega = np.log(omega_max) + s * (np.log(omega_min) - np.log(omega_max))
    omega = np.exp(log_omega)
    
    # Scale to match geometric baseline at theta_base
    geo = geometric_freq(K, theta_base)
    omega = omega * (geo[0] / omega[0])
    
    return omega


def anchored_poly_freq(
    K: int,
    theta_base: float,
    p: float,
    omf: float,
) -> np.ndarray:
    """
    Anchored polynomial frequency distribution.
    omega_max from geometric theta_base, omega_min = omega_max * omf
    """
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0]
    omega_min = omega_max * omf
    
    t = np.arange(K, dtype=np.float64) / (K - 1)
    log_omega = np.log(omega_max) + np.power(t, p) * (np.log(omega_min) - np.log(omega_max))
    return np.exp(log_omega)


def hybrid_any(freq_a: np.ndarray, freq_b: np.ndarray, alpha: float) -> np.ndarray:
    """Hybrid: alpha * freq_a + (1 - alpha) * freq_b"""
    return alpha * freq_a + (1.0 - alpha) * freq_b


# ============================================================================
# Model Definition
# ============================================================================

class CustomRoPE(nn.Module):
    """Custom RoPE with configurable frequency distribution."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 32768, rope_cfg: RopeConfig = None):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_cfg = rope_cfg or RopeConfig()
        
        K = head_dim // 2
        
        if self.rope_cfg.kind == "custom" and self.rope_cfg.custom_omega is not None:
            # Use custom frequencies
            inv_freq = torch.tensor(self.rope_cfg.custom_omega, dtype=torch.float32)
            if len(inv_freq) != K:
                # Interpolate if needed
                inv_freq = torch.nn.functional.interpolate(
                    inv_freq.unsqueeze(0).unsqueeze(0),
                    size=K,
                    mode='linear',
                    align_corners=True
                ).squeeze()
        else:
            # Standard geometric
            inv_freq = 1.0 / (self.rope_cfg.theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CustomGPT2Attention(nn.Module):
    def __init__(self, config: TrainConfig, rope: CustomRoPE):
        super().__init__()
        self.n_head = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.rope = rope
        
        self.c_attn = nn.Linear(config.dim, 3 * config.dim)
        self.c_proj = nn.Linear(config.dim, config.dim)
        
    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(seq_len, hidden_states.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.c_proj(attn_output)


class CustomGPT2Block(nn.Module):
    def __init__(self, config: TrainConfig, rope: CustomRoPE):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = CustomGPT2Attention(config, rope)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.intermediate),
            nn.GELU(),
            nn.Linear(config.intermediate, config.dim),
        )
        
    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class TinyGPT(nn.Module):
    def __init__(self, config: TrainConfig, rope_cfg: RopeConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.dim)
        
        head_dim = config.dim // config.n_heads
        self.rope = CustomRoPE(head_dim=head_dim, max_seq_len=32768, rope_cfg=rope_cfg)
        
        self.h = nn.ModuleList([CustomGPT2Block(config, self.rope) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None):
        hidden_states = self.wte(input_ids)
        
        for block in self.h:
            hidden_states = block(hidden_states)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# ============================================================================
# Data Loading
# ============================================================================

def build_or_load_token_cache(
    out_dir: Path,
    dataset_name: str,
    tokenizer_name: str,
    target_train_tokens: int,
    target_val_tokens: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Build or load tokenized dataset cache."""
    cache_file = out_dir / f"tokens_{tokenizer_name.replace('/', '_')}_t{target_train_tokens}_v{target_val_tokens}_s{seed}.pt"
    meta_file = out_dir / f"tokens_{tokenizer_name.replace('/', '_')}_t{target_train_tokens}_v{target_val_tokens}_s{seed}_meta.json"
    
    if cache_file.exists() and meta_file.exists():
        print(f"[cache] loading from {cache_file}")
        data = torch.load(cache_file)
        meta = json.loads(meta_file.read_text())
        return data["train"], data["val"], meta
    
    print(f"[cache] building token cache for {dataset_name}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    train_ds = load_dataset(dataset_name, split="train", streaming=True)
    val_ds = load_dataset(dataset_name, split="validation", streaming=True)
    
    train_tokens = []
    val_tokens = []
    
    # Tokenize training data
    print("[cache] tokenizing training data...")
    for item in train_ds:
        text = item["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        train_tokens.extend(tokens)
        if len(train_tokens) >= target_train_tokens:
            break
    train_tokens = train_tokens[:target_train_tokens]
    
    # Tokenize validation data
    print("[cache] tokenizing validation data...")
    for item in val_ds:
        text = item["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        val_tokens.extend(tokens)
        if len(val_tokens) >= target_val_tokens:
            break
    val_tokens = val_tokens[:target_val_tokens]
    
    train_tensor = torch.tensor(train_tokens, dtype=torch.long)
    val_tensor = torch.tensor(val_tokens, dtype=torch.long)
    
    torch.save({"train": train_tensor, "val": val_tensor}, cache_file)
    
    meta = {
        "dataset": dataset_name,
        "tokenizer": tokenizer_name,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "vocab_size": len(tokenizer),
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    
    print(f"[cache] saved to {cache_file}")
    return train_tensor, val_tensor, meta


def build_train_order(n_seq: int, seed: int, out_file: Path) -> torch.Tensor:
    """Build random training order."""
    if out_file.exists():
        return torch.load(out_file)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_seq)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(order), out_file)
    return torch.from_numpy(order)


# ============================================================================
# Training
# ============================================================================

def train_one_variant(
    spec: "VariantSpec",
    cfg: TrainConfig,
    train_tokens: torch.Tensor,
    order: torch.Tensor,
    out_dir: Path,
    seed: int = 42,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    save_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Train a single variant."""
    device = torch.device("cuda")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = TinyGPT(cfg, spec.rope_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # Warmup steps
    warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (cfg.max_steps - step) / (cfg.max_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    
    step = 0
    micro_step = 0
    accumulated_loss = 0.0
    
    # Prepare sequences
    n_seq = (train_tokens.numel() - 1) // cfg.seq_len
    
    start_time = time.time()
    
    while step < cfg.max_steps:
        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for accum_step in range(cfg.grad_accum):
            # Get next sequence
            seq_idx = order[step * cfg.grad_accum + accum_step].item()
            start = seq_idx * cfg.seq_len
            end = start + cfg.seq_len + 1
            
            if end > len(train_tokens):
                start = 0
                end = cfg.seq_len + 1
            
            batch = train_tokens[start:end-1].unsqueeze(0).to(device)
            labels = train_tokens[start+1:end].unsqueeze(0).to(device)
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch, labels=labels)
                loss = outputs["loss"] / cfg.grad_accum
            
            accumulated_loss += loss.item()
            
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Gradient clipping
        if use_amp and amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        if use_amp and amp_dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
        
        step += 1
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step * cfg.micro_batch * cfg.grad_accum * cfg.seq_len) / elapsed
            print(f"[{spec.name}] Step {step}/{cfg.max_steps} | Loss: {accumulated_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | {tokens_per_sec:.0f} tok/s")
    
    # Save checkpoint if needed
    ckpt_path = None
    if save_checkpoint:
        ckpt_path = out_dir / f"{spec.name}_final.pt"
        torch.save(model.state_dict(), ckpt_path)
    
    return {
        "train": {
            "steps": step,
            "final_loss": accumulated_loss,
            "checkpoint": str(ckpt_path) if ckpt_path else None,
        }
    }


# ============================================================================
# Evaluation with RANDOM start positions
# ============================================================================

def eval_ppl_lengths(
    model: nn.Module,
    val_tokens: torch.Tensor,
    lengths: List[int],
    n_chunks: int = 5,
    seed: int = 42,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Dict[str, float]]:
    """Evaluate perplexity at different lengths with RANDOM start positions."""
    device = next(model.parameters()).device
    model.eval()
    
    rng = np.random.default_rng(seed)
    results = {}
    
    for length in lengths:
        losses = []
        max_start = len(val_tokens) - length - 1
        
        for _ in range(n_chunks):
            # RANDOM start position (not sequential!)
            start = rng.integers(0, max(1, max_start))
            end = min(start + length + 1, len(val_tokens))
            
            input_ids = val_tokens[start:end-1].unsqueeze(0).to(device)
            labels = val_tokens[start+1:end].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids, labels=labels)
                    losses.append(outputs["loss"].item())
        
        mean_loss = np.mean(losses)
        ppl = math.exp(mean_loss)
        
        results[str(length)] = {
            "mean": ppl,
            "std": np.std([math.exp(l) for l in losses]),
            "n_chunks": n_chunks,
        }
        print(f"[eval] Length {length}: PPL = {ppl:.2f}")
    
    return results


# ============================================================================
# Experiment Setup
# ============================================================================

@dataclass
class VariantSpec:
    name: str
    rope_cfg: RopeConfig
    group: str = ""


def build_configs(K: int) -> List[VariantSpec]:
    """Build all 14 configurations."""
    configs = []
    
    # 1. Baseline: geo_10k_align
    configs.append(VariantSpec(
        name="geo_10k_align",
        rope_cfg=RopeConfig(kind="standard", theta=10000.0),
        group="baseline"
    ))
    
    # 2-8. Sigmoid variations (steep 6-10, mid 0.4-0.6, omf 0.2-0.5, theta 10k)
    sigmoid_params = [
        (6.0, 0.4, 0.2),
        (6.0, 0.5, 0.3),
        (7.0, 0.5, 0.3),
        (8.0, 0.4, 0.2),
        (8.0, 0.5, 0.4),
        (9.0, 0.5, 0.3),
        (10.0, 0.6, 0.5),
    ]
    for steep, mid, omf in sigmoid_params:
        omega = sigmoid_freq(K, theta_base=10000.0, steepness=steep, midpoint=mid, omf=omf)
        configs.append(VariantSpec(
            name=f"sigmoid_t10k_s{int(steep)}_m{mid:.1f}_omf{omf:.1f}",
            rope_cfg=RopeConfig(kind="custom", custom_omega=omega.tolist()),
            group="sigmoid"
        ))
    
    # 9-10. Sigmoid theta 30k, 50k
    for theta in [30000.0, 50000.0]:
        omega = sigmoid_freq(K, theta_base=theta, steepness=8.0, midpoint=0.5, omf=0.3)
        configs.append(VariantSpec(
            name=f"sigmoid_t{int(theta/1000)}k_s8_m0.5_omf0.3",
            rope_cfg=RopeConfig(kind="custom", custom_omega=omega.tolist()),
            group="sigmoid_theta"
        ))
    
    # 11-12. Anchored poly theta 100k, 500k
    for theta in [100000.0, 500000.0]:
        omega = anchored_poly_freq(K, theta_base=theta, p=3.9, omf=0.3)
        configs.append(VariantSpec(
            name=f"anchpoly_t{int(theta/1000)}k_p3.9_omf0.3",
            rope_cfg=RopeConfig(kind="custom", custom_omega=omega.tolist()),
            group="anchored_poly"
        ))
    
    # 13-14. Hybrid mixes
    geo10k = geometric_freq(K, 10000.0)
    anchpoly = anchored_poly_freq(K, theta_base=100000.0, p=3.9, omf=0.3)
    
    for alpha in [0.7, 0.5]:
        omega = hybrid_any(geo10k, anchpoly, alpha)
        configs.append(VariantSpec(
            name=f"hybrid_geo10k_anchpoly100k_a{alpha:.1f}",
            rope_cfg=RopeConfig(kind="custom", custom_omega=omega.tolist()),
            group="hybrid"
        ))
    
    return configs


def run_experiment(
    out_dir: Path,
    cache_dir: Path,
    dataset: str = "roneneldan/TinyStories",
    tokenizer: str = "EleutherAI/pythia-70m",
    train_tokens: int = 50_000_000,
    val_tokens: int = 2_500_000,
    seed: int = 42,
    use_amp: bool = True,
    amp_dtype_str: str = "bf16",
):
    """Run the full experiment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
    
    # Prepare data cache
    print("[prep] Building/loading token cache...")
    train_t, val_t, data_meta = build_or_load_token_cache(
        out_dir=cache_dir,
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        target_train_tokens=train_tokens,
        target_val_tokens=val_tokens,
        seed=seed,
    )
    
    # Build training order
    n_seq = (train_t.numel() - 1) // 2048
    order = build_train_order(n_seq, seed=seed, out_file=cache_dir / f"train_order_seed{seed}.pt")
    
    # Build configs
    K = 32  # head_dim=64 -> K=32
    configs = build_configs(K)
    print(f"[setup] Running {len(configs)} configurations")
    
    # Training config
    cfg = TrainConfig(
        seq_len=2048,
        dim=512,
        n_layers=6,
        n_heads=8,
        intermediate=2048,
        vocab_size=50304,
        lr=6e-4,
        warmup_ratio=0.02,
        micro_batch=2,
        grad_accum=16,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        eval_lengths=[2048, 16384],
        seed=seed,
    )
    
    results_all = {}
    summary = []
    
    for spec in configs:
        print(f"\n{'='*60}")
        print(f"Running: {spec.name}")
        print(f"{'='*60}")
        
        var_dir = out_dir / "variants" / spec.name
        var_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        train_res = train_one_variant(
            spec=spec,
            cfg=cfg,
            train_tokens=train_t,
            order=order,
            out_dir=var_dir,
            seed=seed,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            save_checkpoint=True,
        )
        
        # Load model for eval
        device = torch.device("cuda")
        model = TinyGPT(cfg, spec.rope_cfg).to(device)
        ckpt_path = Path(train_res["train"]["checkpoint"])
        if ckpt_path and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        
        # Evaluate
        ppl = eval_ppl_lengths(
            model=model,
            val_tokens=val_t,
            lengths=cfg.eval_lengths,
            n_chunks=5,
            seed=seed + 1337,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        
        # Cleanup checkpoint
        if ckpt_path and ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)
        
        # Store results
        result = {
            "name": spec.name,
            "group": spec.group,
            "rope": {
                "kind": spec.rope_cfg.kind,
                "theta": spec.rope_cfg.theta,
            },
            "train": train_res["train"],
            "ppl": ppl,
        }
        results_all[spec.name] = result
        
        summary.append({
            "name": spec.name,
            "group": spec.group,
            "ppl_2048": ppl["2048"]["mean"],
            "ppl_16384": ppl["16384"]["mean"],
        })
        
        # Save variant result
        with open(var_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"[done] {spec.name}: PPL@16384 = {ppl['16384']['mean']:.3f}")
        
        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Compute delta vs baseline
    baseline_ppl = results_all.get("geo_10k_align", {}).get("ppl", {}).get("16384", {}).get("mean", float("nan"))
    for s in summary:
        s["vs_baseline_delta"] = s["ppl_16384"] - baseline_ppl if math.isfinite(baseline_ppl) else float("nan")
    
    # Save aggregate results
    payload = {
        "ts": time.strftime("%Y-%m-%d_%H%M%S"),
        "config": {
            "dataset": dataset,
            "tokenizer": tokenizer,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "seed": seed,
            "amp_dtype": amp_dtype_str,
        },
        "baseline_geo_10k_ppl_16384": baseline_ppl,
        "summary": summary,
        "results": results_all,
    }
    
    with open(out_dir / "results_a100_aligned.json", "w") as f:
        json.dump(payload, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    summary_sorted = sorted(summary, key=lambda x: x["ppl_16384"])
    print(f"{'Name':<35} {'PPL@2048':<12} {'PPL@16384':<12} {'Delta':<10}")
    print("-" * 70)
    for row in summary_sorted:
        print(f"{row['name']:<35} {row['ppl_2048']:<12.3f} {row['ppl_16384']:<12.3f} {row['vs_baseline_delta']:+.3f}")
    
    print(f"\n[done] Results saved to {out_dir / 'results_a100_aligned.json'}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/opt/dfrope/results/a100_aligned")
    ap.add_argument("--cache_dir", type=str, default="/opt/dfrope/results/a100_aligned/cache")
    ap.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-70m")
    ap.add_argument("--train_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true", help="Use FP32 instead of AMP")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    args = ap.parse_args()
    
    run_experiment(
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir),
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        seed=args.seed,
        use_amp=not args.fp32,
        amp_dtype_str=args.amp_dtype,
    )


if __name__ == "__main__":
    main()
