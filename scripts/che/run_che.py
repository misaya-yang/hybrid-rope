#!/usr/bin/env python3
"""CHE (Chomsky Hierarchy Evaluation) benchmark with pluggable PE methods.

Matches Deletang et al. (ICLR 2023) setup scaled to DAPE (NeurIPS 2024) model size:
  Encoder-only, non-causal (bidirectional) attention.
  5 layers, 8 heads, d_model=256, d_head=32.
  LR=1e-3, Adam, dropout=0.1, grad_clip=1.0.
  Train: L in [1,40], 50K steps.  Eval: L=50,100,200,300,500.

PE methods:
  nope       - No positional encoding
  rope_geo   - Standard RoPE (geometric frequencies)
  rope_evq   - EVQ-Cosh RoPE (tau=5.0)
  kerple     - Learned per-head log-distance bias (Kerple-log)
  dape       - DAPE = Kerple + MLP refinement
  evq_kerple - EVQ RoPE + Kerple bias (our proposed hybrid)

Usage:
  python run_che.py --task even_pairs --method rope_geo --seed 42
  python run_che.py --task even_pairs --method rope_geo --seed 42 --pilot
"""
import argparse, json, math, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F

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
    name: str; level: str; input_size: int; output_size: int
    def output_length(self, L): return 1
    def generate_batch(self, B, L): raise NotImplementedError

class ParityCheck(CHETask):
    name, level, input_size, output_size = "parity_check", "Regular", 2, 2
    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        return x, (x.sum(1) % 2).unsqueeze(1), torch.ones(B, 1)

class EvenPairs(CHETask):
    name, level, input_size, output_size = "even_pairs", "Regular", 2, 2
    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        y = ((x[:, :-1] ^ x[:, 1:]).sum(1) % 2).unsqueeze(1) if L > 1 else torch.zeros(B, 1, dtype=torch.long)
        return x, y, torch.ones(B, 1)

class ModularArithmetic(CHETask):
    name, level, input_size, output_size = "modular_arithmetic", "Regular", 9, 5
    def generate_batch(self, B, L):
        if L % 2 == 0: L -= 1
        L = max(L, 1)
        nd, no = L // 2 + 1, L // 2
        seqs, res = [], []
        for _ in range(B):
            d = [random.randint(0, 4) for _ in range(nd)]
            ops = [random.choice([0,1,2]) for _ in range(no)]
            seq = []
            for i, v in enumerate(d):
                seq.append(v)
                if i < no: seq.append(ops[i] + 5)
            terms, pend = [d[0]], []
            for i, op in enumerate(ops):
                if op == 2: terms[-1] = (terms[-1] * d[i+1]) % 5
                else: pend.append(op); terms.append(d[i+1])
            r = terms[0]
            for i, op in enumerate(pend):
                r = (r + terms[i+1]) % 5 if op == 0 else (r - terms[i+1]) % 5
            seqs.append(seq); res.append(r)
        return torch.tensor(seqs), torch.tensor(res).unsqueeze(1), torch.ones(B, 1)

class CycleNavigation(CHETask):
    name, level, input_size, output_size = "cycle_navigation", "Regular", 3, 5
    def generate_batch(self, B, L):
        x = torch.randint(0, 3, (B, L))
        return x, ((x.long() - 1).sum(1) % 5).unsqueeze(1), torch.ones(B, 1)

class ModularArithmeticBrackets(CHETask):
    name, level, input_size, output_size = "modular_arithmetic_brackets", "DCF", 10, 5
    def generate_batch(self, B, L):
        seqs, res = [], []
        for _ in range(B):
            s, v = self._gen(L); t = self._tok(s)
            if len(t) < L: t += [0]*(L-len(t))
            seqs.append(t[:L]); res.append(v % 5)
        return torch.tensor(seqs), torch.tensor(res).unsqueeze(1), torch.ones(B, 1)
    def _gen(self, L):
        L = max(L, 1)
        if L == 1: d=random.randint(0,4); return str(d), d
        if L == 2: d=random.randint(0,4); return f"-{d}", (-d)%5
        if L == 3: d=random.randint(0,4); return f"({d})", d
        if L == 4: d=random.randint(0,4); return f"(-{d})", (-d)%5
        inner=L-3; ll=random.randint(1,max(1,inner-1)); rl=inner-ll
        if rl<1: rl=1; ll=inner-1
        ls,lv=self._gen(ll); rs,rv=self._gen(rl)
        op=random.choice(['+','-'])
        return f"({ls}{op}{rs})", (lv+rv)%5 if op=='+' else (lv-rv)%5
    def _tok(self, s):
        m={str(i):i for i in range(5)}; m.update({'+':5,'-':6,'*':7,'(':8,')':9})
        return [m[c] for c in s if c in m]

class ReverseString(CHETask):
    name, level, input_size, output_size = "reverse_string", "DCF", 2, 2
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        x = torch.randint(0, 2, (B, L))
        return x, x.flip(1), torch.ones(B, L)

class SolveEquation(CHETask):
    name, level, input_size, output_size = "solve_equation", "DCF", 11, 5
    def generate_batch(self, B, L):
        bg = ModularArithmeticBrackets(); seqs, res = [], []
        for _ in range(B):
            eL = max(1, L-2); s, v = bg._gen(eL)
            dp = [i for i,c in enumerate(s) if c.isdigit()]
            if not dp:
                d=random.randint(0,4); t=[d,10,v%5]+[0]*max(0,L-3)
                seqs.append(t[:L]); res.append(d); continue
            p=random.choice(dp); sol=int(s[p])
            eq=s[:p]+'x'+s[p+1:]+'='+str(v%5)
            m={str(i):i for i in range(5)}; m.update({'+':5,'-':6,'(':7,')':8,'x':9,'=':10})
            t=[m[c] for c in eq if c in m]
            if len(t)<L: t+=[0]*(L-len(t))
            seqs.append(t[:L]); res.append(sol)
        return torch.tensor(seqs), torch.tensor(res).unsqueeze(1), torch.ones(B, 1)

class StackManipulation(CHETask):
    name, level, input_size, output_size = "stack_manipulation", "DCF", 5, 3
    def output_length(self, L): return L+1
    def generate_batch(self, B, L):
        oL=self.output_length(L); seqs,tgts,masks=[],[],[]
        for _ in range(B):
            sL=random.randint(1,max(1,L-1)); aL=L-sL
            stk=[random.randint(0,1) for _ in range(sL)]
            acts=[random.randint(2,4) for _ in range(aL)]
            cur=list(stk)
            for a in acts:
                if a==2 and cur: cur.pop()
                elif a>=3: cur.append(a-3)
            r=cur[::-1]+[2]; m=[1.0]*len(r)
            while len(r)<oL: r.append(0); m.append(0.0)
            seqs.append(stk+acts); tgts.append(r[:oL]); masks.append(m[:oL])
        return torch.tensor(seqs), torch.tensor(tgts), torch.tensor(masks)

def _to_le(n, fL=0):
    if n==0: bits=[0]
    else:
        bits=[]
        while n>0: bits.append(n&1); n>>=1
    if fL>0:
        while len(bits)<fL: bits.append(0)
        bits=bits[:fL]
    return bits

class BinaryAddition(CHETask):
    name, level, input_size, output_size = "binary_addition", "CS", 3, 3
    def output_length(self, L): return L+1
    def generate_batch(self, B, L):
        oL=self.output_length(L); seqs,tgts,masks=[],[],[]
        for _ in range(B):
            if L<=2:
                n=random.randint(0,max(0,2**L-1)); seqs.append(_to_le(n,L))
                r=_to_le(n,0)+[2]; m=[1.0]*len(r)
                while len(r)<oL: r.append(0); m.append(0.0)
                tgts.append(r[:oL]); masks.append(m[:oL]); continue
            nL=random.randint(1,L-2); mL=L-1-nL
            a=random.randint(1,max(1,2**nL-1)); b=random.randint(1,max(1,2**mL-1))
            seqs.append(_to_le(a,nL)+[2]+_to_le(b,mL))
            r=_to_le(a+b,0)+[2]; m=[1.0]*len(r)
            while len(r)<oL: r.append(0); m.append(0.0)
            tgts.append(r[:oL]); masks.append(m[:oL])
        return torch.tensor(seqs), torch.tensor(tgts), torch.tensor(masks)

class BinaryMultiplication(CHETask):
    name, level, input_size, output_size = "binary_multiplication", "CS", 3, 3
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        oL=L; seqs,tgts,masks=[],[],[]
        for _ in range(B):
            if L<=2:
                n=random.randint(0,max(0,2**L-1)); seqs.append(_to_le(n,L))
                r=_to_le(n,0)+[2]; m=[1.0]*len(r)
                while len(r)<oL: r.append(0); m.append(0.0)
                tgts.append(r[:oL]); masks.append(m[:oL]); continue
            nL=random.randint(1,L-2); mL=L-1-nL
            a=random.randint(1,max(1,2**nL-1)); b=random.randint(1,max(1,2**mL-1))
            seqs.append(_to_le(a,nL)+[2]+_to_le(b,mL))
            r=_to_le(a*b,0)+[2]; m=[1.0]*len(r)
            while len(r)<oL: r.append(0); m.append(0.0)
            tgts.append(r[:oL]); masks.append(m[:oL])
        return torch.tensor(seqs), torch.tensor(tgts), torch.tensor(masks)

class BucketSort(CHETask):
    name, level, input_size, output_size = "bucket_sort", "CS", 5, 5
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        x=torch.randint(0,5,(B,L)); y,_=x.sort(dim=1)
        return x, y, torch.ones(B,L)

class ComputeSqrt(CHETask):
    name, level, input_size, output_size = "compute_sqrt", "CS", 2, 2
    def output_length(self, L): return math.ceil(L/2)
    def generate_batch(self, B, L):
        oL=self.output_length(L); seqs,tgts=[],[]
        for _ in range(B):
            n=random.randint(1,2**L-1)
            bits=[]; tmp=n
            while tmp>0: bits.append(tmp&1); tmp>>=1
            bits.reverse()
            while len(bits)<L: bits.insert(0,0)
            bits=bits[-L:]
            sq=math.isqrt(n); sb=[]
            tmp=sq
            if tmp==0: sb=[0]
            else:
                while tmp>0: sb.append(tmp&1); tmp>>=1
                sb.reverse()
            while len(sb)<oL: sb.insert(0,0)
            sb=sb[-oL:]
            seqs.append(bits); tgts.append(sb)
        return torch.tensor(seqs), torch.tensor(tgts), torch.ones(B,oL)

class DuplicateString(CHETask):
    name, level, input_size, output_size = "duplicate_string", "CS", 2, 2
    def output_length(self, L): return 2*L
    def generate_batch(self, B, L):
        x=torch.randint(0,2,(B,L))
        return x, torch.cat([x,x],1), torch.ones(B,2*L)

class MissingDuplicate(CHETask):
    name, level, input_size, output_size = "missing_duplicate", "CS", 4, 2
    def generate_batch(self, B, L):
        if L<2: return torch.ones(B,1,dtype=torch.long), torch.ones(B,1,dtype=torch.long), torch.ones(B,1)
        half=L//2; seqs,tgts=[],[]
        for _ in range(B):
            s=[random.randint(0,1) for _ in range(half)]; dup=s+s
            idx=random.randint(0,len(dup)-1); val=dup[idx]; dup[idx]=2
            if L%2==1: dup.append(3)
            seqs.append(dup[:L]); tgts.append(val)
        return torch.tensor(seqs), torch.tensor(tgts).unsqueeze(1), torch.ones(B,1)

class OddsFirst(CHETask):
    name, level, input_size, output_size = "odds_first", "CS", 2, 2
    def output_length(self, L): return L
    def generate_batch(self, B, L):
        x=torch.randint(0,2,(B,L))
        return x, torch.cat([x[:,1::2],x[:,::2]],1), torch.ones(B,L)

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
# PE Utilities
# ============================================================================

def geometric_freqs(d_head, base=10000.0):
    k = torch.arange(0, d_head, 2, dtype=torch.float32)
    return 1.0 / (base ** (k / d_head))

def evq_cosh_freqs(d_head, tau=5.0, base=10000.0):
    if abs(tau) < 1e-8: return geometric_freqs(d_head, base)
    K = d_head // 2
    u = torch.linspace(1/d_head, (d_head-1)/d_head, K)
    phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
    return 1.0 / (base ** phi)

def apply_rope(q, k, freqs, S):
    pos = torch.arange(S, device=q.device, dtype=torch.float32)
    ang = torch.outer(pos, freqs.to(q.device))
    c, s = ang.cos()[None,None], ang.sin()[None,None]
    d2 = q.shape[-1]//2
    q1,q2 = q[...,:d2].float(), q[...,d2:].float()
    k1,k2 = k[...,:d2].float(), k[...,d2:].float()
    qr = torch.cat([q1*c-q2*s, q1*s+q2*c], -1)
    kr = torch.cat([k1*c-k2*s, k1*s+k2*c], -1)
    return qr.type_as(q), kr.type_as(k)

# ============================================================================
# Model
# ============================================================================

class KerpleBias(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.bias_p = nn.Parameter(torch.rand(H,1,1)*2)
        self.bias_a = nn.Parameter(torch.rand(H,1,1))
    def forward(self, S):
        p = self.bias_p.clamp(min=0.01); a = self.bias_a.clamp(min=0.01)
        pos = torch.arange(S, device=p.device, dtype=torch.float32)
        dist = (pos.unsqueeze(0)-pos.unsqueeze(1)).abs()
        return -p * torch.log1p(a * dist)

class DAPERefine(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2*H, H), nn.GELU(), nn.Linear(H, H))
    def forward(self, scores, bias):
        B,H,S,_ = scores.shape
        b = bias.unsqueeze(0).expand(B,-1,-1,-1)
        x = torch.cat([scores, b], 1).permute(0,2,3,1)
        return self.mlp(x).permute(0,3,1,2)

class Attention(nn.Module):
    def __init__(self, d_model, H, d_head, pe_type, rope_freqs=None, dropout=0.1):
        super().__init__()
        self.H, self.d_head, self.pe_type = H, d_head, pe_type
        self.scale = 1.0/math.sqrt(d_head)
        self.qkv = nn.Linear(d_model, 3*H*d_head, bias=False)
        self.out_proj = nn.Linear(H*d_head, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        if pe_type in ('rope_geo','rope_evq','evq_kerple'):
            self.register_buffer('rope_freqs', rope_freqs)
        if pe_type in ('kerple','dape','evq_kerple'):
            self.kerple = KerpleBias(H)
        if pe_type == 'dape':
            self.dape = DAPERefine(H)

    def forward(self, x):
        B,S,_ = x.shape; H,d = self.H, self.d_head
        qkv = self.qkv(x).view(B,S,3,H,d)
        q,k,v = qkv[:,:,0].transpose(1,2), qkv[:,:,1].transpose(1,2), qkv[:,:,2].transpose(1,2)
        if self.pe_type in ('rope_geo','rope_evq','evq_kerple'):
            q,k = apply_rope(q,k,self.rope_freqs,S)
        scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        if self.pe_type in ('kerple','dape','evq_kerple'):
            kb = self.kerple(S)
            if self.pe_type == 'dape':
                scores = scores + kb + self.dape(scores, kb)
            else:
                scores = scores + kb
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        return self.out_proj(torch.matmul(attn,v).transpose(1,2).reshape(B,S,H*d))

class Block(nn.Module):
    def __init__(self, d_model, H, d_head, pe_type, rope_freqs=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, H, d_head, pe_type, rope_freqs, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CHEModel(nn.Module):
    """Encoder-only transformer for CHE.
    Input: [input_tokens, SEP, QUERY*T] -> predict at QUERY positions.
    """
    def __init__(self, input_size, output_size, d_model=256, n_heads=8,
                 n_layers=5, d_head=32, pe_type='nope', dropout=0.1):
        super().__init__()
        self.sep_id = input_size; self.query_id = input_size + 1
        self.embed = nn.Embedding(input_size+2, d_model)
        self.embed_drop = nn.Dropout(dropout)
        rope_freqs = None
        if pe_type == 'rope_geo': rope_freqs = geometric_freqs(d_head)
        elif pe_type in ('rope_evq','evq_kerple'): rope_freqs = evq_cosh_freqs(d_head, tau=5.0)
        self.layers = nn.ModuleList([
            Block(d_model, n_heads, d_head, pe_type, rope_freqs, dropout)
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, input_ids, n_out):
        B,L = input_ids.shape; dev = input_ids.device
        sep = torch.full((B,1), self.sep_id, dtype=torch.long, device=dev)
        qry = torch.full((B,n_out), self.query_id, dtype=torch.long, device=dev)
        h = self.embed_drop(self.embed(torch.cat([input_ids, sep, qry], 1)))
        for layer in self.layers: h = layer(h)
        return self.head(self.final_norm(h[:, -n_out:]))

# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one(task_name, method, seed, args):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    task = TASK_REGISTRY[task_name]()
    model = CHEModel(
        task.input_size, task.output_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_head=args.d_head,
        pe_type=method, dropout=args.dropout,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  device={DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.n_steps)

    model.train(); loss_sum = 0.0; loss_cnt = 0; t0 = time.time()
    for step in range(1, args.n_steps+1):
        L = random.randint(max(2, args.min_length), args.train_length)
        inp, tgt, msk = task.generate_batch(args.batch_size, L)
        inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
        logits = model(inp, tgt.shape[1])
        loss = F.cross_entropy(logits.reshape(-1, task.output_size), tgt.reshape(-1), reduction='none')
        loss = (loss.view_as(msk)*msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        loss_sum += loss.item(); loss_cnt += 1
        if step % 2000 == 0:
            ela = time.time()-t0; avg = loss_sum/loss_cnt
            print(f"  [{step:>6d}/{args.n_steps}] loss={avg:.4f}  lr={sched.get_last_lr()[0]:.1e}  "
                  f"{step/ela:.0f} step/s  ETA={int((args.n_steps-step)/(step/ela))}s")
            loss_sum = 0.0; loss_cnt = 0
    return model, time.time()-t0

@torch.no_grad()
def evaluate(model, task_name, eval_lengths, n_batches=10, batch_size=256):
    model.eval(); task = TASK_REGISTRY[task_name](); results = {}
    for L in eval_lengths:
        correct = total = 0.0
        for _ in range(n_batches):
            inp, tgt, msk = task.generate_batch(batch_size, L)
            inp, tgt, msk = inp.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)
            try:
                logits = model(inp, tgt.shape[1])
            except RuntimeError:
                torch.cuda.empty_cache()
                h = batch_size//2
                logits = torch.cat([model(inp[:h], tgt.shape[1]), model(inp[h:], tgt.shape[1])], 0)
            correct += ((logits.argmax(-1)==tgt).float()*msk).sum().item()
            total += msk.sum().item()
        results[L] = round(correct/max(total,1), 4)
        print(f"    L={L:>4d}: {results[L]*100:5.1f}%")
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='even_pairs')
    p.add_argument('--method', default='rope_geo')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--work_dir', default='./che_results')
    p.add_argument('--n_steps', type=int, default=50000)
    p.add_argument('--train_length', type=int, default=40)
    p.add_argument('--min_length', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--d_head', type=int, default=32)
    p.add_argument('--eval_lengths', default='50,100,200,300,500')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.pilot: a.n_steps = 2000

    eval_lengths = [int(x) for x in a.eval_lengths.split(',')]
    tasks = ALL_TASKS if a.task == 'all' else [a.task]
    methods = ALL_METHODS if a.method == 'all' else [a.method]
    os.makedirs(a.work_dir, exist_ok=True)

    all_res = {}
    ckpt = os.path.join(a.work_dir, "results_all.json")
    if a.resume and os.path.exists(ckpt):
        with open(ckpt) as f: all_res = json.load(f)

    for tn in tasks:
        for mt in methods:
            rid = f"{tn}__{mt}__s{a.seed}"
            rf = os.path.join(a.work_dir, f"{rid}.json")
            if a.resume and (rid in all_res or os.path.exists(rf)):
                print(f"[SKIP] {rid}")
                if os.path.exists(rf) and rid not in all_res:
                    with open(rf) as f: all_res[rid] = json.load(f)
                continue
            print(f"\n{'='*60}\n  {rid}\n{'='*60}")
            model, tt = train_one(tn, mt, a.seed, a)
            print(f"  Done in {tt:.0f}s. Evaluating...")
            ev = evaluate(model, tn, eval_lengths)
            r = {"task":tn,"method":mt,"seed":a.seed,"time_s":round(tt,1),
                 "eval":{str(k):v for k,v in ev.items()}}
            all_res[rid] = r
            with open(rf,'w') as f: json.dump(r,f,indent=2)
            with open(ckpt,'w') as f: json.dump(all_res,f,indent=2)
            del model; torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for lv, tl in TASKS_BY_LEVEL.items():
        print(f"\n--- {lv} ---")
        for tn in tl:
            for mt in methods:
                rid = f"{tn}__{mt}__s{a.seed}"
                if rid in all_res:
                    ev = all_res[rid]["eval"]
                    accs = " ".join(f"{float(ev.get(str(L),0))*100:5.1f}" for L in eval_lengths)
                    print(f"  {tn:30s} {mt:12s} | {accs}")
    print(f"\nResults: {ckpt}")

if __name__ == "__main__":
    main()
