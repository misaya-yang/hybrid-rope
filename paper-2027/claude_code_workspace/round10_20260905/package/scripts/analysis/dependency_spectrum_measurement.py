#!/usr/bin/env python3
"""End-to-end feasibility: measure K(r)=E[(dL/ds_ij)^2 | i-j=r] from an
existing weekend_sweep checkpoint. Analysis-only: no repo files modified.

Verifies the audit's Task-2 claims:
  (1) checkpoint loads and runs forward/backward on CPU (no retraining);
  (2) replacing SDPA (run_evq_sweep.py L347) with explicit softmax is
      numerically equivalent (output match);
  (3) dL/ds_ij is obtainable via autograd and yields a distance-resolved
      loss-sensitivity spectrum K(r).
"""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    REPO / "results/weekend_sweep/L512/50m_tau1.41_seed42")
DATA = REPO / "results/weekend_sweep/L512/train_tinystories_25000000_512.pt"
SEQ = 512

# ---- import the trainer module (defines GPT, Attention, TIER_CONFIGS) ----
spec = importlib.util.spec_from_file_location(
    "res", REPO / "scripts/core_text_phases/run_evq_sweep.py")
res = importlib.util.module_from_spec(spec)
sys.modules["res"] = res
spec.loader.exec_module(res)

cfg = res.TIER_CONFIGS["50m"].copy()
cfg["seq_len"] = SEQ
cfg["max_position_embeddings"] = SEQ

inv_freq = torch.from_numpy(np.load(RUN / "inv_freq.npy")).float()
print(f"[load] inv_freq K={inv_freq.numel()} range=[{inv_freq.min():.3e},{inv_freq.max():.3e}]")

model = res.GPT(cfg, inv_freq)
sd = torch.load(RUN / "model.pt", map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=True), None
print("[load] state_dict loaded strict=True OK")
model.eval()

# ---- data ----
data = torch.load(DATA, map_location="cpu", weights_only=True).long().flatten()
n_chunks = data.numel() // SEQ
data = data[: n_chunks * SEQ].view(n_chunks, SEQ)
print(f"[data] cache -> {tuple(data.shape)} chunks of {SEQ}")
flat = data[:3].flatten()  # 3 consecutive chunks -> 2 shifted windows
xb = flat[: 2 * SEQ].view(2, SEQ)
yb = flat[1: 2 * SEQ + 1].view(2, SEQ)
print(f"[data] batch: x{tuple(xb.shape)} y{tuple(yb.shape)} "
      f"tok-range [{xb.min().item()}, {xb.max().item()}]")


def loss_of(logits):
    return F.cross_entropy(logits.reshape(-1, cfg["vocab_size"]), yb.reshape(-1))


# ---- (1) baseline forward with original SDPA ----
with torch.no_grad():
    out_sdpa = model(xb)
    l_sdpa = loss_of(out_sdpa).item()
print(f"[sdpa] loss={l_sdpa:.6f}")


# ---- (2) patched explicit-softmax attention (the claimed minimal change) ----
def explicit_forward(self, x, capture):
    B, L, _ = x.shape
    qkv = self.qkv(x).view(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    cos, sin = self.rope(L)
    cos, sin = cos[None, None], sin[None, None]
    q, k = res.apply_rope(q, cos, sin), res.apply_rope(k, cos, sin)
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.hd)
    mask = torch.ones(L, L, dtype=torch.bool).triu(1)
    s = s.masked_fill(mask[None, None], float("-inf"))
    s = s.detach().requires_grad_(True)  # leaf: s.grad will be exactly dL/ds_ij
    capture.append(s)
    attn = torch.softmax(s, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
    return self.o(out.transpose(1, 2).reshape(B, L, -1))


capture = []
orig = res.Attention.forward
res.Attention.forward = lambda self, x: explicit_forward(self, x, capture)

with torch.no_grad():
    out_expl = model(xb)
    l_expl = loss_of(out_expl).item()
diff = (out_sdpa - out_expl).abs().max().item()
print(f"[explicit] loss={l_expl:.6f}  max|sdpa-explicit|={diff:.3e}  "
      f"n_logit_tensors={len(capture)}")
assert diff < 1e-4, "explicit softmax does not match SDPA"
assert abs(l_sdpa - l_expl) < 1e-5

# ---- (3) backward -> dL/ds_ij -> distance histogram K(r) ----
res.Attention.forward = orig  # re-patch with fresh capture list (graph-safe)
capture2 = []
res.Attention.forward = lambda self, x: explicit_forward(self, x, capture2)
out2 = model(xb)
loss = loss_of(out2)
loss.backward()

grad_sq = [s.grad.detach() for s in capture2]
B, H, L = 2, cfg["num_heads"], SEQ
hist = np.zeros(L, dtype=np.float64)
cnt = np.zeros(L, dtype=np.float64)
for g in grad_sq:  # g: B,H,L,L
    g2 = g.numpy().astype(np.float64) ** 2
    for r in range(1, L):
        # entries (i, i-r), i=r..L-1: main diagonal of the [r:, :L-r] sub-block
        diag = np.stack([np.diagonal(g2[b, h, r:, :L - r])
                         for b in range(B) for h in range(H)])
        hist[r] += diag.sum()
        cnt[r] += diag.size
Kr = np.where(cnt > 0, hist / np.maximum(cnt, 1), np.nan)
print(f"[K(r)] loss={loss.item():.6f}  layers={len(grad_sq)}")
rs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
print("[K(r)] r : E[(dL/ds)^2 | dist=r]")
for r in rs:
    print(f"       {r:>4} : {Kr[r]:.6e}")
tail = Kr[400:].mean()
head = Kr[1:5].mean()
print(f"[K(r)] mean(r<=4)={head:.4e}  mean(r>=400)={tail:.4e}  ratio={head/max(tail,1e-30):.2f}")
print("FEASIBILITY_OK")

import json
out = Path("/tmp/dep_spectrum_audit/kr_results.json")
out.parent.mkdir(parents=True, exist_ok=True)
store = json.loads(out.read_text()) if out.exists() else {}
store[RUN.name] = {
    "loss": loss.item(),
    "K_r": {str(r): float(Kr[r]) for r in range(1, L)},
    "head_mean_r_le_4": float(head),
    "tail_mean_r_ge_400": float(tail),
}
out.write_text(json.dumps(store, indent=1))
print(f"[save] {out} now has {len(store)} runs")
