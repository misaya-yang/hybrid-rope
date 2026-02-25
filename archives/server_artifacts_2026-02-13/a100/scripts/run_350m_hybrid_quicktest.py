import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_CFG = {
    "vocab_size": 50304,
    "hidden_size": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "head_dim": 64,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
    "dropout": 0.0,
}

def get_hybrid_freq(K):
    k = torch.arange(K, dtype=torch.float32)
    geo_10k = 1.0 / (10000.0 ** (2 * k / (2 * K)))
    omega_min = geo_10k[-1].item() * 0.3
    t = k / (K - 1)
    p = 3.9
    log_omega = math.log(1.0) + (t ** p) * (math.log(omega_min) - math.log(1.0))
    poly = torch.exp(log_omega)
    return 0.8 * geo_10k + 0.2 * poly

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, inv_freq):
        super().__init__()
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = 2048
        self._build_cache(2048)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg, rotary):
        super().__init__()
        self.n_head = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.q_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=False)
        self.k_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=False)
        self.v_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=False)
        self.o_proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"], bias=False)
        self.rotary = rotary
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(L)
        q_rot = q * cos[None,None,:,:] + torch.cat([-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]], dim=-1) * sin[None,None,:,:]
        k_rot = k * cos[None,None,:,:] + torch.cat([-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]], dim=-1) * sin[None,None,:,:]
        out = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.up = nn.Linear(cfg["hidden_size"], cfg["intermediate_size"], bias=False)
        self.down = nn.Linear(cfg["intermediate_size"], cfg["hidden_size"], bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg, rotary):
        super().__init__()
        self.ln1 = RMSNorm(cfg["hidden_size"])
        self.ln2 = RMSNorm(cfg["hidden_size"])
        self.attn = Attention(cfg, rotary)
        self.mlp = MLP(cfg)
    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

class GPTModel(nn.Module):
    def __init__(self, cfg, inv_freq):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.rotary = RotaryEmbedding(cfg["head_dim"], inv_freq)
        self.layers = nn.ModuleList([TransformerBlock(cfg, self.rotary) for _ in range(cfg["num_layers"])])
        self.ln_f = RMSNorm(cfg["hidden_size"])
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight
    def forward(self, x):
        x = self.tok_emb(x)
        for l in self.layers: x = l(x)
        return self.lm_head(self.ln_f(x))

def main():
    WORK_DIR = "/opt/dfrope/results/350m_hybrid_quicktest"
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    print("350M Hybrid Quick Test - Loading 10M tokens...")
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tokens = []
    for x in ds:
        tokens.extend(tokenizer.encode(x["text"]))
        if len(tokens) >= 10_000_000: break
    
    n_chunks = len(tokens) // 2048
    data = torch.tensor(tokens[:n_chunks*2048], dtype=torch.long).view(n_chunks, -1)
    print(f"Data ready: {n_chunks} chunks")
    
    K = 32
    inv_freq = get_hybrid_freq(K)
    print(f"Hybrid freq: {inv_freq.min():.2e} - {inv_freq.max():.2f}")
    
    print("Building model...")
    model = GPTModel(MODEL_CFG, inv_freq).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")
    
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = GradScaler()
    
    model.train()
    steps = min(n_chunks // 8, 50)  # Only 50 steps for quick test
    print(f"Quick training {steps} steps...")
    
    import time
    t0 = time.time()
    for s in range(steps):
        batch = data[s*8:(s+1)*8].cuda()
        with autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1), batch[:,1:].flatten())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if s % 10 == 0:
            print(f"Step {s}/{steps} Loss: {loss.item():.4f}")
    
    print(f"Quick test PASSED in {(time.time()-t0)/60:.1f}min!")
    torch.save(model.state_dict(), f"{WORK_DIR}/model_quick.pt")

if __name__ == "__main__":
    main()
