"""Activation-level per-slot phase kernel on CPU (no CUDA).

Server receipt: /tmp/cputheory_r0/ on the westc instance (2026-09-13).
Short-doc probe (single text, sinks not excluded); not a task-conditioned census.

Forward a short doc through a small model on CPU with hooks on q_proj/k_proj
(pre-RoPE). For every layer, head, token pair (query m, key n), m>=n, and slot i:

    c_i(n,m) = q_i(n) * conj(k_i(m)),   attention term = Re[c_i e^{-j*Delta*omega_i}], Delta=m-n

Accumulates per slot: mean |c_i| (activation usage) and the Delta-resolved kernel
K_i(bin) = sum_pairs c_i e^{-j Delta omega_i} over log-spaced Delta bins.
"""
import argparse
import json
import math

import numpy as np

B = 500000.0
K = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(32)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = "".join(open(f, errors="ignore").read() for f in
                   ["/tmp/cputheory_r0/cpu_theory_r0.py", "/tmp/cputheory_r0/cpu_kernel_r0.py"])
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(text, return_tensors="pt").input_ids[0][: a.tokens]
    T = int(ids.shape[0])
    print(f"forwarding {T} tokens on CPU ...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map="cpu")
    model.eval()
    layers = model.model.layers
    n_layers, H = len(layers), model.config.num_attention_heads
    KV = getattr(model.config, "num_key_value_heads", None) or H
    hd = model.config.hidden_size // H
    half = hd // 2
    grp = H // KV
    assert half == K

    store = {}
    def mk(name):
        def hook(mod, inp, out):
            store[name] = out.detach()
        return hook
    handles = []
    for l, lyr in enumerate(layers):
        handles.append(lyr.self_attn.q_proj.register_forward_hook(mk((l, "q"))))
        handles.append(lyr.self_attn.k_proj.register_forward_hook(mk((l, "k"))))
    with torch.no_grad():
        model(input_ids=ids[None, :])
    for h in handles:
        h.remove()

    wn = (B ** (-np.arange(K) / K))
    edges = np.unique(np.logspace(0, math.log10(T - 1), 17).astype(int))
    bin_of = np.zeros(T, dtype=int)
    for bi in range(1, len(edges)):
        bin_of[edges[bi - 1]:edges[bi]] = bi - 1
    nb = len(edges) - 1

    abs_c = np.zeros((n_layers, K))
    kern = np.zeros((n_layers, K, nb), dtype=complex)
    cnt = np.zeros((n_layers, K, nb))
    with torch.no_grad():
        for l in range(n_layers):
            q = store[(l, "q")][0].float().numpy().reshape(T, H, hd).transpose(1, 0, 2)
            k = store[(l, "k")][0].float().numpy().reshape(T, KV, hd).transpose(1, 0, 2)
            qc = q[:, :, :half] + 1j * q[:, :, half:]   # [H,T,K]
            kc = k[:, :, :half] + 1j * k[:, :, half:]   # [KV,T,K]
            if grp > 1:  # broadcast each kv head to its q-head group
                kc = np.repeat(kc, grp, axis=0)         # [H,T,K], kv head h//grp
            for m in range(1, T):
                c = qc[:, m:, :] * np.conj(kc[:, : T - m, :])   # [H,T-m,K], Delta=m
                bi = bin_of[m]
                abs_c[l] += np.abs(c).sum(axis=(0, 1))
                ph = np.exp(-1j * m * wn)[None, None, :]
                kern[l][:, bi] += (c * ph).sum(axis=(0, 1))
                cnt[l][:, bi] += c.shape[0] * c.shape[1]
            del store[(l, "q")], store[(l, "k")]
    npairs = T * (T - 1) / 2
    mean_abs_c = (abs_c / npairs).mean(axis=0)
    kern_c = (kern / np.maximum(cnt, 1)).mean(axis=0)          # [K, nb] mean over layers
    contrast = np.abs(kern_c).sum(axis=1)                       # Delta-structure strength
    flat = np.abs(kern_c).mean(axis=1)                          # kernel magnitude (any phase)

    out = {
        "model": a.model, "tokens": T, "n_layers": n_layers, "heads": H, "head_dim": hd,
        "delta_bin_edges": edges.tolist(),
        "mean_abs_c_slot": mean_abs_c.tolist(),
        "kernel_contrast_slot": contrast.tolist(),
        "kernel_abs_slot": flat.tolist(),
        "kernel_by_bin_slot0": [complex(z).real for z in kern_c[0]],
        "kernel_by_bin_slot24": [complex(z).real for z in kern_c[24]],
        "kernel_by_bin_slot44": [complex(z).real for z in kern_c[44]],
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
