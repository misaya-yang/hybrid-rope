#!/usr/bin/env python3
"""CHE (Chomsky Hierarchy Evaluation) benchmark with pluggable PE methods.

Reproduces DAPE (NeurIPS 2024) benchmark configuration.
Architecture: Encoder-only, non-causal (bidirectional) attention.
Model: 5 layers, 8 heads, d_model=256, d_head=32.
Train: L in [1,40], 200K steps.  Eval: L=50,100,200,300,500.

PE methods:
  nope       - No positional encoding
  rope_geo   - Standard RoPE (geometric frequencies)
  rope_evq   - EVQ-Cosh RoPE (tau=5.0)
  kerple     - Learned per-head log-distance bias
  dape       - DAPE (Kerple + MLP refinement)
  evq_kerple - EVQ RoPE + Kerple bias (our proposed hybrid)

Usage:
  python run_che.py --task even_pairs --method rope_geo --seed 42
  python run_che.py --task even_pairs --method rope_geo --seed 42 --pilot
"""
import argparse
import json
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASKS_BY_LEVEL = {
    "Regular": ["parity_check", "even_pairs", "modular_arithmetic", "cycle_navigation"],
    "DCF": ["modular_arithmetic_brackets", "reverse_string", "solve_equation",
            "stack_manipulation"],
    "CS": ["binary_addition", "binary_multiplication", "bucket_sort", "compute_sqrt",
           "duplicate_string", "missing_duplicate", "odds_first"],
}
ALL_TASKS = [t for ts in TASKS_BY_LEVEL.values() for t in ts]
ALL_METHODS = ["nope", "rope_geo", "rope_evq", "kerple", "dape", "evq_kerple"]

# ============================================================================
# Task Data Generators
# ============================================================================

class CHETask:
    name: str
    level: str
    input_size: int
    output_size: int

    def output_length(self, L: int) -> int:
        return 1

    def generate_batch(self, B: int, L: int):
        raise NotImplementedError


class ParityCheck(CHETask):
    name, level, input_size, output_size = "parity_check", "Regular", 2, 2

    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        y = (x.sum(1) % 2).unsqueeze(1)
        return x, y, torch.ones(B, 1)


class EvenPairs(CHETask):
    name, level, input_size, output_size = "even_pairs", "Regular", 2, 2

    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        y = ((x[:, :-1] ^ x[:, 1:]).sum(1) % 2).unsqueeze(1) if L > 1 else torch.zeros(B, 1, dtype=torch.long)
        return x, y, torch.ones(B, 1)


class ModularArithmetic(CHETask):
    name, level, input_size, output_size = "modular_arithmetic", "Regular", 9, 5
    MOD = 5

    def generate_batch(self, B, L):
        if L % 2 == 0: L -= 1
        L = max(L, 1)
        n_d, n_o = L // 2 + 1, L // 2
        seqs, results = [], []
        for _ in range(B):
            digits = [random.randint(0, 4) for _ in range(n_d)]
            ops = [random.choice([0, 1, 2]) for _ in range(n_o)]
            seq = []
            for i, d in enumerate(digits):
                seq.append(d)
                if i < n_o: seq.append(ops[i] + 5)
            # Evaluate: * before +/-
            terms, pending = [digits[0]], []
            for i, op in enumerate(ops):
                if op == 2:
                    terms[-1] = (terms[-1] * digits[i+1]) % self.MOD
                else:
                    pending.append(op); terms.append(digits[i+1])
            r = terms[0]
            for i, op in enumerate(pending):
                r = (r + terms[i+1]) % self.MOD if op == 0 else (r - terms[i+1]) % self.MOD
            seqs.append(seq); results.append(r)
        return torch.tensor(seqs), torch.tensor(results).unsqueeze(1), torch.ones(B, 1)


class CycleNavigation(CHETask):
    name, level, input_size, output_size = "cycle_navigation", "Regular", 3, 5

    def generate_batch(self, B, L):
        x = torch.randint(0, 3, (B, L))
        y = ((x.long() - 1).sum(1) % 5).unsqueeze(1)
        return x, y, torch.ones(B, 1)


class ModularArithmeticBrackets(CHETask):
    name, level, input_size, output_size = "modular_arithmetic_brackets", "DCF", 10, 5
    MOD = 5

    def generate_batch(self, B, L):
        seqs, results = [], []
        for _ in range(B):
            s, v = self._gen(L)
            t = self._tok(s)
            if len(t) < L: t += [0] * (L - len(t))
            seqs.append(t[:L]); results.append(v % self.MOD)
        return torch.tensor(seqs), torch.tensor(results).unsqueeze(1), torch.ones(B, 1)

    def _gen(self, L):
        L = max(L, 1)
        if L == 1: d = random.randint(0, 4); return str(d), d
        if L == 2: d = random.randint(0, 4); return f"-{d}", (-d) % self.MOD
        if L == 3: d = random.randint(0, 4); return f"({d})", d
        if L == 4: d = random.randint(0, 4); return f"(-{d})", (-d) % self.MOD
        inner = L - 3
        ll = random.randint(1, max(1, inner - 1)); rl = inner - ll
        if rl < 1: rl = 1; ll = inner - 1
        ls, lv = self._gen(ll); rs, rv = self._gen(rl)
        op = random.choice(['+', '-'])
        v = (lv + rv) % self.MOD if op == '+' else (lv - rv) % self.MOD
        return f"({ls}{op}{rs})", v

    def _tok(self, s):
        m = {str(i): i for i in range(5)}
        m.update({'+': 5, '-': 6, '*': 7, '(': 8, ')': 9})
        return [m[c] for c in s if c in m]


class ReverseString(CHETask):
    name, level, input_size, output_size = "reverse_string", "DCF", 2, 2
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        return x, x.flip(1), torch.ones(B, L)


class SolveEquation(CHETask):
    name, level, input_size, output_size = "solve_equation", "DCF", 11, 5
    MOD = 5

    def generate_batch(self, B, L):
        bg = ModularArithmeticBrackets()
        seqs, results = [], []
        for _ in range(B):
            eL = max(1, L - 2)
            s, v = bg._gen(eL)
            dpos = [i for i, c in enumerate(s) if c.isdigit()]
            if not dpos:
                d = random.randint(0, 4)
                t = [d, 10, v % self.MOD] + [0] * max(0, L - 3)
                seqs.append(t[:L]); results.append(d); continue
            p = random.choice(dpos)
            sol = int(s[p])
            eq = s[:p] + 'x' + s[p+1:] + '=' + str(v % self.MOD)
            m = {str(i): i for i in range(5)}
            m.update({'+': 5, '-': 6, '(': 7, ')': 8, 'x': 9, '=': 10})
            t = [m[c] for c in eq if c in m]
            if len(t) < L: t += [0] * (L - len(t))
            seqs.append(t[:L]); results.append(sol)
        return torch.tensor(seqs), torch.tensor(results).unsqueeze(1), torch.ones(B, 1)


class StackManipulation(CHETask):
    name, level, input_size, output_size = "stack_manipulation", "DCF", 5, 3
    def output_length(self, L): return L + 1

    def generate_batch(self, B, L):
        oL = self.output_length(L)
        seqs, tgts, masks = [], [], []
        for _ in range(B):
            sL = random.randint(1, max(1, L - 1)); aL = L - sL
            stk = [random.randint(0, 1) for _ in range(sL)]
            acts = [random.randint(2, 4) for _ in range(aL)]
            cur = list(stk)
            for a in acts:
                if a == 2 and cur: cur.pop()
                elif a >= 3: cur.append(a - 3)
            res = cur[::-1] + [2]
            m = [1.0] * len(res)
            while len(res) < oL: res.append(0); m.append(0.0)
            seqs.append(stk + acts); tgts.append(res[:oL]); masks.append(m[:oL])
        return torch.tensor(seqs), torch.tensor(tgts), torch.tensor(masks)


def _to_le(n, fL=0):
    if n == 0: bits = [0]
    else:
        bits = []
        while n > 0: bits.append(n & 1); n >>= 1
    if fL > 0:
        while len(bits) < fL: bits.append(0)
        bits = bits[:fL]
    return bits


class BinaryAddition(CHETask):
    name, level, input_size, output_size = "binary_addition", "CS", 3, 3
    def output_length(self, L): return L + 1

    def generate_batch(self, B, L):
        oL = self.output_length(L)
        seqs, tgts, masks = [], [], []
        for _ in range(B):
            if L <= 2:
                n = random.randint(0, max(0, 2**L - 1))
                seqs.append(_to_le(n, L))
                r = _to_le(n, 0) + [2]
                m = [1.0] * len(r)
                while len(r) < oL: r.append(0); m.append(0.0)
                tgts.append(r[:oL]); masks.append(m[:oL]); continue
            nL = random.randint(1, L - 2); mL = L - 1 - nL
            a = random.randint(1, max(1, 2**nL - 1))
            b = random.randint(1, max(1, 2**mL - 1))
            seqs.append(_to_le(a, nL) + [2] + _to_le(b, mL))
            r = _to_le(a + b, 0) + [2]
            m = [1.0] * len(r)
            while len(r) < oL: r.append(0); m.append(0.0)
            tgts.append(r[:oL]); masks.append(m[:oL])
        return torch.tensor(seqs), torch.tensor(tgts), torch.tensor(masks)


class BinaryMultiplication(CHETask):
    name, level, input_size, output_size = "binary_multiplication", "CS", 3, 3
    def output_length(self, L): return L

    def generate_batch(self, B, L):
        oL = self.output_length(L)
        seqs, tgts, masks = [], [], []
        for _ in range(B):
            if L <= 2:
                n = random.randint(0, max(0, 2**L - 1))
                seqs.append(_to_le(n, L))
                r = _to_le(n, 0) + [2]
                m = [1.0] * len(r)
                while len(r) < oL: r.append(0); m.append(0.0)
                tgts.append(r[:oL]); masks.append(m[:oL]); continue
            nL = random.randint(1, L - 2); mL = L - 1 - nL
            a = random.randint(1, max(1, 2**nL - 1))
            b = random.randint(1, max(1, 2**mL - 1))
            seqs.append(_to_le(a, nL) + [2] + _to_le(b, mL))
            r = _to_le(a * b, 0) + [2]
            m = [1.0] * len(r)
            while len(r) < oL: r.append(0); m.append(0.0)
            tgts.append(r[:oL]); masks.append(m[:oL])
        return torch.tensor(seqs), torch.tensor(tgts), torch.tensor(masks)


class BucketSort(CHETask):
    name, level, input_size, output_size = "bucket_sort", "CS", 5, 5
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        x = torch.randint(0, 5, (B, L))
        y, _ = x.sort(dim=1)
        return x, y, torch.ones(B, L)


class ComputeSqrt(CHETask):
    name, level, input_size, output_size = "compute_sqrt", "CS", 2, 2
    def output_length(self, L): return math.ceil(L / 2)

    def generate_batch(self, B, L):
        oL = self.output_length(L)
        seqs, tgts = [], []
        for _ in range(B):
            n = random.randint(1, 2**L - 1)
            bits = []
            tmp = n
            while tmp > 0: bits.append(tmp & 1); tmp >>= 1
            bits.reverse()
            while len(bits) < L: bits.insert(0, 0)
            bits = bits[-L:]
            sq = math.isqrt(n)
            sbits = []
            tmp = sq
            if tmp == 0: sbits = [0]
            else:
                while tmp > 0: sbits.append(tmp & 1); tmp >>= 1
                sbits.reverse()
            while len(sbits) < oL: sbits.insert(0, 0)
            sbits = sbits[-oL:]
            seqs.append(bits); tgts.append(sbits)
        return torch.tensor(seqs), torch.tensor(tgts), torch.ones(B, oL)


class DuplicateString(CHETask):
    name, level, input_size, output_size = "duplicate_string", "CS", 2, 2
    def output_length(self, L): return 2 * L
    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        return x, torch.cat([x, x], 1), torch.ones(B, 2 * L)


class MissingDuplicate(CHETask):
    name, level, input_size, output_size = "missing_duplicate", "CS", 4, 2

    def generate_batch(self, B, L):
        if L < 2:
            return torch.ones(B, 1, dtype=torch.long), torch.ones(B, 1, dtype=torch.long), torch.ones(B, 1)
        half = L // 2
        seqs, tgts = [], []
        for _ in range(B):
            s = [random.randint(0, 1) for _ in range(half)]
            dup = s + s
            idx = random.randint(0, len(dup) - 1)
            val = dup[idx]; dup[idx] = 2
            if L % 2 == 1: dup.append(3)
            seqs.append(dup[:L]); tgts.append(val)
        return torch.tensor(seqs), torch.tensor(tgts).unsqueeze(1), torch.ones(B, 1)


class OddsFirst(CHETask):
    name, level, input_size, output_size = "odds_first", "CS", 2, 2
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        return x, torch.cat([x[:, 1::2], x[:, ::2]], 1), torch.ones(B, L)


TASK_REGISTRY = {
    "parity_check": ParityCheck, "even_pairs": EvenPairs,
    "modular_arithmetic": ModularArithmetic, "cycle_navigation": CycleNavigation,
    "modular_arithmetic_brackets": ModularArithmeticBrackets,
    "reverse_string": ReverseString, "solve_equation": SolveEquation,
    "stack_manipulation": StackManipulation, "binary_addition": BinaryAddition,
    "binary_multiplication": BinaryMultiplication, "bucket_sort": BucketSort,
    "compute_sqrt": ComputeSqrt, "duplicate_string": DuplicateString,
    "missing_duplicate": MissingDuplicate, "odds_first": OddsFirst,
}

# ============================================================================
# Positional Encoding Utilities
# ============================================================================

def geometric_freqs(d_head, base=10000.0):
    k = torch.arange(0, d_head, 2, dtype=torch.float32)
    return 1.0 / (base ** (k / d_head))


def evq_cosh_freqs(d_head, tau=5.0, base=10000.0):
    if abs(tau) < 1e-8:
        return geometric_freqs(d_head, base)
    K = d_head // 2
    u = torch.linspace(1 / d_head, (d_head - 1) / d_head, K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return 1.0 / (base ** phi)


def apply_rope(q, k, freqs, S):
    device = q.device
    pos = torch.arange(S, device=device, dtype=torch.float32)
    ang = torch.outer(pos, freqs.to(device))  # (S, d//2)
    c = ang.cos()[None, None]  # (1,1,S,d//2)
    s = ang.sin()[None, None]
    d2 = q.shape[-1] // 2
    q1, q2 = q[..., :d2].float(), q[..., d2:].float()
    k1, k2 = k[..., :d2].float(), k[..., d2:].float()
    qr = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], -1)
    kr = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], -1)
    return qr.type_as(q), kr.type_as(k)


# ============================================================================
# Model
# ============================================================================

class KerpleBias(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.bias_p = nn.Parameter(torch.rand(n_heads, 1, 1) * 2)
        self.bias_a = nn.Parameter(torch.rand(n_heads, 1, 1))

    def forward(self, S):
        p = self.bias_p.clamp(min=0.01)
        a = self.bias_a.clamp(min=0.01)
        pos = torch.arange(S, device=p.device, dtype=torch.float32)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return -p * torch.log1p(a * dist)  # (H, S, S)


class DAPERefine(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_heads, n_heads),
            nn.GELU(),
            nn.Linear(n_heads, n_heads),
        )

    def forward(self, scores, bias):
        # scores: (B,H,S,S), bias: (H,S,S)
        B, H, S, _ = scores.shape
        b = bias.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([scores, b], 1)  # (B,2H,S,S)
        x = x.permute(0, 2, 3, 1)      # (B,S,S,2H)
        x = self.mlp(x)                 # (B,S,S,H)
        return x.permute(0, 3, 1, 2)    # (B,H,S,S)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, pe_type, rope_freqs=None):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        self.pe_type = pe_type
        self.scale = 1.0 / math.sqrt(d_head)
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        if pe_type in ('rope_geo', 'rope_evq', 'evq_kerple'):
            self.register_buffer('rope_freqs', rope_freqs)
        if pe_type in ('kerple', 'dape', 'evq_kerple'):
            self.kerple = KerpleBias(n_heads)
        if pe_type == 'dape':
            self.dape = DAPERefine(n_heads)

    def forward(self, x):
        B, S, _ = x.shape
        H, d = self.n_heads, self.d_head
        qkv = self.qkv(x).view(B, S, 3, H, d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        if self.pe_type in ('rope_geo', 'rope_evq', 'evq_kerple'):
            q, k = apply_rope(q, k, self.rope_freqs, S)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.pe_type in ('kerple', 'dape', 'evq_kerple'):
            kb = self.kerple(S)
            if self.pe_type == 'dape':
                scores = scores + kb + self.dape(scores, kb)
            else:
                scores = scores + kb

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, H * d)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, pe_type, rope_freqs=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head, pe_type, rope_freqs)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CHEModel(nn.Module):
    """Encoder-only transformer for CHE tasks.

    Input format: [input_tokens, SEP, QUERY * T]
    Output: logits at QUERY positions -> (B, T, output_size)
    """
    def __init__(self, input_size, output_size, d_model=256, n_heads=8,
                 n_layers=5, d_head=32, pe_type='nope'):
        super().__init__()
        self.sep_id = input_size
        self.query_id = input_size + 1
        self.embed = nn.Embedding(input_size + 2, d_model)

        rope_freqs = None
        if pe_type == 'rope_geo':
            rope_freqs = geometric_freqs(d_head)
        elif pe_type in ('rope_evq', 'evq_kerple'):
            rope_freqs = evq_cosh_freqs(d_head, tau=5.0)

        self.layers = nn.ModuleList([
            Block(d_model, n_heads, d_head, pe_type, rope_freqs)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, input_ids, n_out):
        B, L = input_ids.shape
        dev = input_ids.device
        sep = torch.full((B, 1), self.sep_id, dtype=torch.long, device=dev)
        qry = torch.full((B, n_out), self.query_id, dtype=torch.long, device=dev)
        seq = torch.cat([input_ids, sep, qry], 1)
        h = self.embed(seq)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.head(h[:, -n_out:])  # (B, T, output_size)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one(task_name, method, seed, args):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    task = TASK_REGISTRY[task_name]()
    model = CHEModel(
        task.input_size, task.output_size,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_head=args.d_head,
        pe_type=method,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_steps)

    model.train()
    loss_sum, loss_cnt = 0.0, 0
    t0 = time.time()

    for step in range(1, args.n_steps + 1):
        L = random.randint(max(2, args.min_length), args.train_length)
        inp, tgt, msk = task.generate_batch(args.batch_size, L)
        inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
        n_out = tgt.shape[1]

        logits = model(inp, n_out)
        loss = F.cross_entropy(logits.reshape(-1, task.output_size),
                               tgt.reshape(-1), reduction='none')
        loss = (loss.view_as(msk) * msk).sum() / msk.sum().clamp(min=1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        loss_sum += loss.item(); loss_cnt += 1

        if step % 5000 == 0:
            ela = time.time() - t0
            avg = loss_sum / loss_cnt
            spd = step / ela
            eta = (args.n_steps - step) / spd
            print(f"  [{step:>7d}/{args.n_steps}] loss={avg:.4f}  "
                  f"speed={spd:.0f} step/s  ETA={eta:.0f}s")
            loss_sum, loss_cnt = 0.0, 0

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s")
    return model, train_time


@torch.no_grad()
def evaluate(model, task_name, eval_lengths, n_batches=10, batch_size=256):
    model.eval()
    task = TASK_REGISTRY[task_name]()
    results = {}
    for L in eval_lengths:
        correct, total = 0.0, 0.0
        for _ in range(n_batches):
            inp, tgt, msk = task.generate_batch(batch_size, L)
            inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
            n_out = tgt.shape[1]
            try:
                logits = model(inp, n_out)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    # Try with smaller batch
                    half = batch_size // 2
                    if half < 1: break
                    logits1 = model(inp[:half], n_out)
                    logits2 = model(inp[half:], n_out)
                    logits = torch.cat([logits1, logits2], 0)
                else:
                    raise
            preds = logits.argmax(-1)
            correct += ((preds == tgt).float() * msk).sum().item()
            total += msk.sum().item()
        acc = correct / max(total, 1)
        results[L] = round(acc, 4)
        print(f"    L={L:>4d}: acc={acc:.4f}")
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CHE Benchmark")
    parser.add_argument('--task', default='even_pairs')
    parser.add_argument('--method', default='rope_geo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--work_dir', default='./che_results')
    parser.add_argument('--n_steps', type=int, default=200000)
    parser.add_argument('--train_length', type=int, default=40)
    parser.add_argument('--min_length', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--d_head', type=int, default=32)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--eval_lengths', default='50,100,200,300,500')
    parser.add_argument('--pilot', action='store_true', help='Quick pilot (2K steps)')
    parser.add_argument('--resume', action='store_true', help='Skip completed runs')
    args = parser.parse_args()

    if args.pilot:
        args.n_steps = 2000

    eval_lengths = [int(x) for x in args.eval_lengths.split(',')]
    tasks = ALL_TASKS if args.task == 'all' else [args.task]
    methods = ALL_METHODS if args.method == 'all' else [args.method]

    os.makedirs(args.work_dir, exist_ok=True)
    all_results = {}

    # Load existing results if resuming
    ckpt_path = os.path.join(args.work_dir, "results_all.json")
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            all_results = json.load(f)

    for task_name in tasks:
        for method in methods:
            run_id = f"{task_name}__{method}__s{args.seed}"
            result_file = os.path.join(args.work_dir, f"{run_id}.json")

            if args.resume and (run_id in all_results or os.path.exists(result_file)):
                print(f"[SKIP] {run_id}")
                if os.path.exists(result_file) and run_id not in all_results:
                    with open(result_file) as f:
                        all_results[run_id] = json.load(f)
                continue

            print(f"\n{'='*60}")
            print(f"  {run_id}")
            print(f"{'='*60}")

            model, train_time = train_one(task_name, method, args.seed, args)

            print("  Evaluating...")
            eval_res = evaluate(model, task_name, eval_lengths)

            result = {
                "task": task_name, "method": method, "seed": args.seed,
                "train_time_s": round(train_time, 1),
                "eval": {str(k): v for k, v in eval_res.items()},
            }
            all_results[run_id] = result

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            with open(ckpt_path, 'w') as f:
                json.dump(all_results, f, indent=2)

            del model
            torch.cuda.empty_cache()

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for level, task_list in TASKS_BY_LEVEL.items():
        print(f"\n--- {level} ---")
        for task_name in task_list:
            for method in methods:
                run_id = f"{task_name}__{method}__s{args.seed}"
                if run_id in all_results:
                    ev = all_results[run_id]["eval"]
                    accs = " ".join(f"{float(ev.get(str(L), 0))*100:5.1f}" for L in eval_lengths)
                    print(f"  {task_name:30s} {method:12s} | {accs}")

    print(f"\nResults: {ckpt_path}")


if __name__ == "__main__":
    main()
