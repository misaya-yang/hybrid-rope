"""From-scratch DF-RoPE vs Standard comparison (single GPU).

Implements the requested experiment:
- 4 variants trained from scratch on identical data order and seeds
- PPL evaluation across extrapolation lengths
- Result JSON + terminal table

Usage:
  conda run -n dftorch python /opt/dfrope/from_scratch_dfrope_train_eval.py \
    --data_dataset roneneldan/TinyStories --train_tokens 50000000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


# -----------------------------
# Repro / utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def fmt_num(n: int) -> str:
    return f"{n:,}"


# -----------------------------
# RoPE engine
# -----------------------------

@dataclass
class RopeConfig:
    kind: str  # standard | dfrope | custom
    theta: float
    beta: float = 0.0
    t_mod: float = 100000.0
    lowfreq_only: bool = False
    lowfreq_ratio: float = 0.25
    custom_omega: Optional[List[float]] = None


class RopeEngine:
    def __init__(self, head_dim: int, rope_cfg: RopeConfig):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.half = head_dim // 2
        self.cfg = rope_cfg

        if rope_cfg.kind == "custom":
            if not rope_cfg.custom_omega:
                raise ValueError("custom rope requires custom_omega list")
            if len(rope_cfg.custom_omega) != self.half:
                raise ValueError(
                    f"custom_omega length mismatch: got {len(rope_cfg.custom_omega)}, expected {self.half}"
                )
            inv_freq = torch.tensor(rope_cfg.custom_omega, dtype=torch.float32)
            if torch.any(inv_freq <= 0):
                raise ValueError("custom_omega must be positive")
            if torch.any(inv_freq[:-1] < inv_freq[1:]):
                raise ValueError("custom_omega must be non-increasing")
        else:
            inv_freq = 1.0 / (rope_cfg.theta ** (torch.arange(0, self.half).float() / self.half))
        self.register_inv_freq = inv_freq  # cpu fp32

        if rope_cfg.kind == "dfrope":
            beta_k = torch.full((self.half,), float(rope_cfg.beta), dtype=torch.float32)
            if rope_cfg.lowfreq_only:
                n_low = max(1, int(self.half * rope_cfg.lowfreq_ratio))
                beta_k[:-n_low] = 0.0
            self.register_beta_k = beta_k
        else:
            self.register_beta_k = torch.zeros((self.half,), dtype=torch.float32)

        self._cache: Dict[Tuple[str, torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_phase(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv = self.register_inv_freq.to(device=device)
        base = torch.outer(pos, inv)

        if self.cfg.kind != "dfrope":
            return base

        omega_mod = 2.0 * math.pi / float(self.cfg.t_mod)
        beta_k = self.register_beta_k.to(device=device)
        mod = (torch.cos(omega_mod * pos) - 1.0).unsqueeze(1)
        # phi_k(m) = w_k*m - beta_k*(cos(wm*m)-1)
        return base - mod * beta_k.unsqueeze(0)

    def cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype, seq_len)
        if key in self._cache:
            return self._cache[key]

        phase = self._build_phase(seq_len, device=device)
        cos = torch.cos(phase).to(dtype=dtype)
        sin = torch.sin(phase).to(dtype=dtype)
        self._cache[key] = (cos, sin)
        return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q,k: [B, H, T, D]
    cos,sin: [T, D/2]
    """
    B, H, T, D = q.shape
    half = D // 2

    cos = cos.view(1, 1, T, half)
    sin = sin.view(1, 1, T, half)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


# -----------------------------
# Model
# -----------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.fc_in = nn.Linear(dim, intermediate * 2, bias=False)
        self.fc_out = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc_in(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        return self.fc_out(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: RopeEngine):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = rope

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # B,H,T,D
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        cos, sin = self.rope.cos_sin(T, device=x.device, dtype=x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # full causal attention, no sliding window
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, intermediate: int, rope: RopeEngine):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, rope)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUMLP(dim, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        intermediate: int,
        rope_cfg: RopeConfig,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        rope = RopeEngine(self.head_dim, rope_cfg)

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, intermediate, rope) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)

        # Tie lm head to token embedding matrix.
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return x

    def loss_on_batch(
        self,
        seq_plus_one: torch.Tensor,
        logits_chunk: int = 256,
    ) -> torch.Tensor:
        """
        seq_plus_one: [B, T+1]
        returns mean CE over B*T targets.
        """
        inp = seq_plus_one[:, :-1]
        tgt = seq_plus_one[:, 1:]

        h = self.forward_hidden(inp)  # [B,T,C]
        B, T, _ = h.shape
        V = self.vocab_size

        total_nll = 0.0
        total_tok = 0
        for s in range(0, T, logits_chunk):
            e = min(T, s + logits_chunk)
            hs = h[:, s:e, :]  # [B,tc,C]
            logits = F.linear(hs, self.tok_emb.weight)  # [B,tc,V]
            nll = F.cross_entropy(logits.reshape(-1, V), tgt[:, s:e].reshape(-1), reduction="sum")
            total_nll = total_nll + nll
            total_tok += (e - s) * B

        return total_nll / total_tok


# -----------------------------
# Data
# -----------------------------

@dataclass
class DataCacheMeta:
    dataset: str
    tokenizer: str
    train_tokens: int
    val_tokens: int
    seq_len: int
    seed: int


def build_or_load_token_cache(
    out_dir: Path,
    dataset_name: str,
    tokenizer_name: str,
    target_train_tokens: int,
    target_val_tokens: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, DataCacheMeta]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_p = out_dir / "train_tokens.pt"
    val_p = out_dir / "val_tokens.pt"
    meta_p = out_dir / "meta.json"

    if train_p.exists() and val_p.exists() and meta_p.exists():
        meta = DataCacheMeta(**json.loads(meta_p.read_text()))
        train_tokens = torch.load(train_p, map_location="cpu")
        val_tokens = torch.load(val_p, map_location="cpu")

        cache_ok = (
            meta.dataset == dataset_name
            and meta.tokenizer == tokenizer_name
            and int(train_tokens.numel()) >= target_train_tokens
            and int(val_tokens.numel()) >= target_val_tokens
        )
        if cache_ok:
            if int(train_tokens.numel()) > target_train_tokens:
                train_tokens = train_tokens[:target_train_tokens].contiguous()
            if int(val_tokens.numel()) > target_val_tokens:
                val_tokens = val_tokens[:target_val_tokens].contiguous()
            meta = DataCacheMeta(
                dataset=dataset_name,
                tokenizer=tokenizer_name,
                train_tokens=int(train_tokens.numel()),
                val_tokens=int(val_tokens.numel()),
                seq_len=2048,
                seed=seed,
            )
            return train_tokens, val_tokens, meta

        print("[data] existing cache mismatch with requested setup, rebuilding cache ...")

    print(f"[data] Loading dataset {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0

    train_buf: List[int] = []
    val_buf: List[int] = []

    # Deterministic 95/5 split by index.
    for i, ex in enumerate(ds):
        text = ex.get("text", "")
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            continue
        ids.append(eos)

        is_val = (i % 20 == 0)

        if is_val and len(val_buf) < target_val_tokens:
            need = target_val_tokens - len(val_buf)
            val_buf.extend(ids[:need])
        elif (not is_val) and len(train_buf) < target_train_tokens:
            need = target_train_tokens - len(train_buf)
            train_buf.extend(ids[:need])

        if len(train_buf) >= target_train_tokens and len(val_buf) >= target_val_tokens:
            break

        if i % 20000 == 0 and i > 0:
            print(
                f"[data] scanned={fmt_num(i)} train_tok={fmt_num(len(train_buf))} "
                f"val_tok={fmt_num(len(val_buf))}"
            )

    train_tokens = torch.tensor(train_buf, dtype=torch.int32)
    val_tokens = torch.tensor(val_buf, dtype=torch.int32)

    meta = DataCacheMeta(
        dataset=dataset_name,
        tokenizer=tokenizer_name,
        train_tokens=int(train_tokens.numel()),
        val_tokens=int(val_tokens.numel()),
        seq_len=2048,
        seed=seed,
    )

    torch.save(train_tokens, train_p)
    torch.save(val_tokens, val_p)
    meta_p.write_text(json.dumps(asdict(meta), indent=2))

    print(
        f"[data] cached train={fmt_num(train_tokens.numel())} val={fmt_num(val_tokens.numel())} "
        f"at {out_dir}"
    )

    return train_tokens, val_tokens, meta


def build_train_order(n_sequences: int, seed: int, out_file: Path) -> torch.Tensor:
    if out_file.exists():
        order = torch.load(out_file, map_location="cpu")
        valid = (
            isinstance(order, torch.Tensor)
            and order.ndim == 1
            and order.numel() == n_sequences
            and torch.is_floating_point(order) is False
            and int(order.min().item()) >= 0
            and int(order.max().item()) < n_sequences
        )
        if valid:
            return order.to(torch.int64)
        print("[data] cached train order mismatch with current n_sequences, rebuilding ...")

    g = torch.Generator()
    g.manual_seed(seed)
    order = torch.randperm(n_sequences, generator=g, dtype=torch.int64)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(order, out_file)
    return order


def batch_from_token_stream(
    token_stream: torch.Tensor,
    seq_len: int,
    seq_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Create [B, seq_len+1] from non-overlapping chunks."""
    rows = []
    for idx in seq_indices.tolist():
        s = idx * seq_len
        rows.append(token_stream[s : s + seq_len + 1].to(torch.long))
    x = torch.stack(rows, dim=0).to(device=device, non_blocking=True)
    return x


# -----------------------------
# Train / eval
# -----------------------------

@dataclass
class TrainConfig:
    seq_len: int = 2048
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    intermediate: int = 2048
    vocab_size: int = 50304

    lr: float = 6e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1

    warmup_ratio: float = 0.02
    target_effective_batch: int = 32
    micro_batch: int = 2
    grad_accum: int = 16

    log_every: int = 20
    loss_record_every: int = 100


@dataclass
class VariantSpec:
    name: str
    rope: RopeConfig


def choose_micro_batch(
    cfg: TrainConfig,
    rope_cfg: RopeConfig,
    device: torch.device,
    candidates: List[int],
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> int:
    for mb in candidates:
        try:
            torch.cuda.empty_cache()
            model = TinyGPT(
                vocab_size=cfg.vocab_size,
                dim=cfg.dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                intermediate=cfg.intermediate,
                rope_cfg=rope_cfg,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            x = torch.randint(0, cfg.vocab_size, (mb, cfg.seq_len + 1), device=device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model.loss_on_batch(x)
            loss.backward()
            opt.step()
            del model, opt, x, loss
            torch.cuda.empty_cache()
            return mb
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
    return 1


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_variant(
    spec: VariantSpec,
    cfg: TrainConfig,
    train_tokens: torch.Tensor,
    order: torch.Tensor,
    out_dir: Path,
    seed: int,
    use_amp: bool,
    smoke_steps: int = 0,
    amp_dtype: torch.dtype = torch.float16,
    save_checkpoint: bool = True,
) -> Dict[str, Any]:
    device = torch.device("cuda")
    set_seed(seed)

    n_seq = (train_tokens.numel() - 1) // cfg.seq_len
    order = order[:n_seq]

    # Auto-select micro-batch for this hardware.
    mb = choose_micro_batch(
        cfg,
        spec.rope,
        device,
        candidates=[64, 32, 16, 8, 4, 2, 1],
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    grad_accum = max(1, cfg.target_effective_batch // mb)

    model = TinyGPT(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        intermediate=cfg.intermediate,
        rope_cfg=spec.rope,
    ).to(device)

    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    full_steps = math.ceil(n_seq / (mb * grad_accum))
    total_steps = min(full_steps, smoke_steps) if smoke_steps > 0 else full_steps
    warmup_steps = max(1, int(full_steps * cfg.warmup_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    loss_curve: List[Dict[str, float]] = []
    t0 = time.time()
    ptr = 0

    print(
        f"[train:{spec.name}] n_seq={fmt_num(n_seq)} total_steps={total_steps}/{full_steps} "
        f"micro_batch={mb} grad_accum={grad_accum}"
    )

    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0

        for _ in range(grad_accum):
            if ptr >= n_seq:
                break
            take = min(mb, n_seq - ptr)
            idx = order[ptr : ptr + take]
            ptr += take

            batch = batch_from_token_stream(train_tokens, cfg.seq_len, idx, device=device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model.loss_on_batch(batch) / grad_accum

            scaler.scale(loss).backward()
            step_loss += float(loss.item()) * grad_accum
            step_tokens += int(take * cfg.seq_len)

        # lr schedule
        lr_scale = cosine_lr(step, total_steps, warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = cfg.lr * lr_scale

        scaler.step(opt)
        scaler.update()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            print(
                f"[train:{spec.name}] step={step+1}/{total_steps} "
                f"loss={step_loss:.4f} lr={cfg.lr * lr_scale:.3e} "
                f"tokens={fmt_num(step_tokens)}"
            )
            t0 = time.time()

        if (step + 1) % cfg.loss_record_every == 0 or step == total_steps - 1:
            loss_curve.append({"step": step + 1, "loss": step_loss})

    ckpt = None
    if save_checkpoint:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f"{spec.name}_model.pt"
        torch.save(model.state_dict(), ckpt)

    return {
        "variant": spec.name,
        "rope": asdict(spec.rope),
        "train": {
            "total_steps": total_steps,
            "full_steps": full_steps,
            "micro_batch": mb,
            "grad_accum": grad_accum,
            "effective_batch": mb * grad_accum,
            "loss_curve": loss_curve,
            "checkpoint": (str(ckpt) if ckpt is not None else None),
        },
    }


@torch.inference_mode()
def eval_ppl_lengths(
    model: TinyGPT,
    val_tokens: torch.Tensor,
    lengths: List[int],
    n_chunks: int,
    seed: int,
    logits_chunk: int = 256,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, Dict[str, float]]:
    device = torch.device("cuda")
    model.eval()

    rng = random.Random(seed)
    vt = val_tokens.to(torch.long)

    results: Dict[str, Dict[str, float]] = {}

    for L in lengths:
        needed = L + 1
        if vt.numel() <= needed + 1:
            raise ValueError(f"val tokens too short for length {L}")

        max_start = int(vt.numel() - needed)
        starts = [rng.randint(0, max_start) for _ in range(n_chunks)]

        losses = []
        for s in starts:
            x = vt[s : s + needed].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model.loss_on_batch(x, logits_chunk=logits_chunk)
            losses.append(float(loss.item()))

        ppls = [math.exp(v) for v in losses]
        mean = float(np.mean(ppls))
        std = float(np.std(ppls))
        results[str(L)] = {
            "mean": mean,
            "std": std,
            "n_chunks": n_chunks,
        }
        print(f"[eval] L={L:>5} PPL={mean:.3f} ± {std:.3f}")

    return results


def print_ppl_table(lengths: List[int], per_variant: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    headers = ["Length", "standard", "dfrope_b05", "dfrope_b10", "high_theta"]
    col_w = 14

    line = " | ".join(h.ljust(col_w) for h in headers)
    print("\n" + line)
    print("-" * len(line))

    for L in lengths:
        row = [str(L).ljust(col_w)]
        for v in headers[1:]:
            d = per_variant[v][str(L)]
            row.append(f"{d['mean']:.2f}±{d['std']:.2f}".ljust(col_w))
        print(" | ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dataset", type=str, default="roneneldan/TinyStories")
    ap.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-70m")
    ap.add_argument("--train_tokens", type=int, default=50_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_500_000)
    ap.add_argument("--n_eval_chunks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="/opt/dfrope/results/from_scratch")
    ap.add_argument("--smoke_steps", type=int, default=0, help="If >0, only run this many train steps per variant (quick smoke).")
    ap.add_argument("--fp32", action="store_true", help="Disable AMP")
    ap.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="AMP dtype when AMP is enabled.")
    ap.add_argument("--no_save_ckpt", action="store_true", help="Do not write model checkpoints to disk.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    ckpt_dir = out_dir / "checkpoints"

    # Fixed model/training config from request
    cfg = TrainConfig(
        seq_len=2048,
        dim=512,
        n_layers=6,
        n_heads=8,
        intermediate=2048,
        vocab_size=50304,
        lr=6e-4,
        warmup_ratio=0.02,
        target_effective_batch=32,
    )

    eval_lengths = [2048, 3072, 4096, 5120, 6144, 8192, 12288, 16384]

    variants = [
        VariantSpec("standard", RopeConfig(kind="standard", theta=1000.0)),
        VariantSpec("dfrope_b05", RopeConfig(kind="dfrope", theta=1000.0, beta=0.5, t_mod=100000.0)),
        VariantSpec("dfrope_b10", RopeConfig(kind="dfrope", theta=1000.0, beta=1.0, t_mod=100000.0)),
        VariantSpec("high_theta", RopeConfig(kind="standard", theta=10000.0)),
    ]

    set_seed(args.seed)

    print("[setup] building/loading token cache ...")
    train_tokens, val_tokens, data_meta = build_or_load_token_cache(
        out_dir=cache_dir,
        dataset_name=args.data_dataset,
        tokenizer_name=args.tokenizer,
        target_train_tokens=args.train_tokens,
        target_val_tokens=args.val_tokens,
        seed=args.seed,
    )

    n_seq = (train_tokens.numel() - 1) // cfg.seq_len
    print(f"[setup] train tokens={fmt_num(train_tokens.numel())} -> n_seq={fmt_num(n_seq)}")
    order = build_train_order(n_seq, seed=args.seed, out_file=cache_dir / f"train_order_seed{args.seed}.pt")

    all_results: Dict[str, Any] = {
        "ts": now(),
        "data": asdict(data_meta),
        "train_config": asdict(cfg),
        "eval_lengths": eval_lengths,
        "n_eval_chunks": args.n_eval_chunks,
        "variants": {},
    }

    use_amp = not args.fp32
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    save_checkpoint = not args.no_save_ckpt

    for i, spec in enumerate(variants):
        print(f"\n{'='*90}\n[run] variant={spec.name} rope={asdict(spec.rope)}\n{'='*90}")

        # Train
        res = train_one_variant(
            spec=spec,
            cfg=cfg,
            train_tokens=train_tokens,
            order=order,
            out_dir=ckpt_dir,
            seed=args.seed,
            use_amp=use_amp,
            smoke_steps=args.smoke_steps,
            amp_dtype=amp_dtype,
            save_checkpoint=save_checkpoint,
        )

        # Load model for eval
        device = torch.device("cuda")
        model = TinyGPT(
            vocab_size=cfg.vocab_size,
            dim=cfg.dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            intermediate=cfg.intermediate,
            rope_cfg=spec.rope,
        ).to(device)
        if res["train"]["checkpoint"] is None:
            raise RuntimeError("No checkpoint was saved; cannot eval from checkpoint in main().")
        sd = torch.load(res["train"]["checkpoint"], map_location="cpu")
        model.load_state_dict(sd)

        # PPL eval
        ppl = eval_ppl_lengths(
            model=model,
            val_tokens=val_tokens,
            lengths=eval_lengths,
            n_chunks=args.n_eval_chunks,
            seed=args.seed + i,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        res["ppl"] = ppl
        all_results["variants"][spec.name] = res

        # 2048 sanity check reference (for bug detection)
        d2048 = ppl["2048"]["mean"]
        print(f"[sanity:{spec.name}] PPL@2048={d2048:.3f}")

        # free gpu
        del model
        torch.cuda.empty_cache()

    # sanity spread at 2048
    p2048 = [all_results["variants"][v.name]["ppl"]["2048"]["mean"] for v in variants]
    p_min, p_max = min(p2048), max(p2048)
    spread = (p_max - p_min) / max(1e-6, p_min)
    all_results["sanity_2048_spread_ratio"] = spread
    all_results["sanity_2048_warning"] = spread > 0.20

    out_dir.mkdir(parents=True, exist_ok=True)
    final_json = out_dir / "results.json"
    final_json.write_text(json.dumps(all_results, indent=2))

    # Table
    per_variant = {k: v["ppl"] for k, v in all_results["variants"].items()}
    print_ppl_table(eval_lengths, per_variant)

    if spread > 0.20:
        print(
            f"\n[warning] 2048 PPL spread is {spread*100:.1f}% (>20%). "
            "Training mismatch or implementation bug may exist."
        )
    else:
        print(f"\n[sanity] 2048 PPL spread {spread*100:.1f}% (within 20%).")

    print(f"[done] results written to {final_json}")


if __name__ == "__main__":
    main()
