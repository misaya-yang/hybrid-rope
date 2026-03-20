#!/usr/bin/env python
"""Patch run_gqa_evq_experiment.py to add GQA support."""

path = "/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py"
with open(path) as f:
    code = f.read()

# 1. Replace Attention class with GQA-capable version
old_attn = '''class Attention(nn.Module):
    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
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
        return self.o(out.transpose(1, 2).reshape(B, L, -1))'''

new_attn = '''def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for GQA."""
    if n_rep == 1:
        return x
    B, nh, L, hd = x.shape
    return x[:, :, None, :, :].expand(B, nh, n_rep, L, hd).reshape(B, nh * n_rep, L, hd)


class Attention(nn.Module):
    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]          # Q heads
        self.n_kv = cfg.get("n_kv_heads", self.nh)  # KV heads (default=MHA)
        self.hd = cfg["head_dim"]
        self.n_rep = self.nh // self.n_kv   # repeat factor
        assert self.nh % self.n_kv == 0, f"num_heads({self.nh}) must be divisible by n_kv_heads({self.n_kv})"
        # Q: nh * hd, K: n_kv * hd, V: n_kv * hd
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Expand KV heads for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))'''

assert old_attn in code, "Old Attention class not found!"
code = code.replace(old_attn, new_attn)

# 2. Add --n_kv_heads argument (find --compile line)
old_arg = '    parser.add_argument("--compile", action="store_true",'
new_arg = '''    parser.add_argument("--n_kv_heads", type=int, default=None,
                        help="Number of KV heads for GQA (default=num_heads=MHA)")
    parser.add_argument("--compile", action="store_true",'''
assert old_arg in code, "--compile arg not found!"
code = code.replace(old_arg, new_arg)

# 3. Inject n_kv_heads into cfg (find the compile batch_size adjustment)
old_cfg = '    if args.compile and args.batch_size is None:'
new_cfg = '''    # GQA: inject n_kv_heads
    if args.n_kv_heads is not None:
        cfg["n_kv_heads"] = args.n_kv_heads
        print(f"  [GQA] n_kv_heads={args.n_kv_heads} (num_heads={cfg[\'num_heads\']}, ratio={cfg[\'num_heads\']//args.n_kv_heads}x)")
    else:
        cfg["n_kv_heads"] = cfg["num_heads"]  # MHA default

    if args.compile and args.batch_size is None:'''
assert old_cfg in code, "compile batch_size line not found!"
code = code.replace(old_cfg, new_cfg, 1)  # only first occurrence

with open(path, "w") as f:
    f.write(code)

# Verify
with open(path) as f:
    content = f.read()
checks = ["repeat_kv", "n_kv_heads", "q_proj", "k_proj", "v_proj", "n_rep"]
for c in checks:
    assert c in content, f"{c} not found in patched file!"
    print(f"  OK: {c} ({content.count(c)} occurrences)")
print("\nPatch applied successfully!")
