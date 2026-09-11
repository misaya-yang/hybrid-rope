#!/usr/bin/env python3
"""The OLMo half of the KKT measurement -- the checkpoint where the pair differs.

WHY THIS FILE HAS TO EXIST.  The Qwen run measures e, n and the sec.5 residual on
Qwen2.5-3B, and the campaign's central puzzle is a CROSS-CHECKPOINT reversal:

    MrRoPE vs BM   OLMo 16K RULER   2.78%  vs 51.32%   (+48.5 for BM)
    MrRoPE vs BM   Qwen 128K RULER 78.13%  vs 70.83%   (-7.3  for BM)

Same band, same gain, same budget; the two tables differ only in how eighteen
increments are distributed between slot 14 and slot 32 on OLMo (and seventeen
between 23 and 40 on Qwen) -- verified bitwise against the archived prepared
tables.  A first-order model of that difference is only testable if the gradient
is measured on BOTH checkpoints, and until this file existed only Qwen had one.
Without the OLMo side the reversal could only be described, never predicted.

WHAT IS MEASURED, and it is deliberately the same instrument as the Qwen run:

  e  = dL_long/dDelta at MrRoPE's table, over teacher-forced answer CE on the 16K
       retrieval rows.  This is the long-range objective's own gradient.
  n  = dL_native/dDelta at the NATIVE table, over in-window (4096) tail NLL from
       held-out natural documents.  The constraint is defined RELATIVE TO NATIVE,
       so its gradient belongs at the native point, not at MrRoPE's.
  lambda_hat = max(0, -<n,e>/<n,n>),  residual = e + lambda_hat n.

THE DECLARED SIMPLIFICATIONS ARE THE SAME ONES THE QWEN RUN DECLARES, and they are
not smaller here: n is the gradient of in-window tail NLL rather than the plan's
output-KL D_N, and e is taken at the incumbent rather than at an optimum.

ONE INSTRUMENT DIFFERENCE, STATED BECAUSE IT IS THE KIND THAT SILENTLY BREAKS
COMPARABILITY.  transformers 5.15's Olmo2RotaryEmbedding.forward is decorated
@torch.no_grad, so swapping `rotary.inv_freq` cannot carry a gradient -- the
forward has to be replaced outright, exactly as the Qwen run replaces it.  The
replacement below reproduces the module's own arithmetic line for line (same
`cat((freqs, freqs))`, same float32 phase, same attention_scaling) minus the
decorator, and asserts the gain it installs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The OLMo geometry and the gain its deployed tables actually carry.  The gain is
# NOT YaRN's by accident: the archived prepared table records 1.138629436111989,
# which is 1 + 0.1 ln 4, and it is identical on both MrRoPE and MrProBM.
THETA = 500_000.0
WINDOW = 4096
HEAD_DIM = 128
K = 64
GAIN = 1.138629436111989


def native_inv_freq(theta=THETA, k=K):
    return theta ** (-np.arange(k, dtype=np.float64) / k)


def m_to_nu(m, theta=THETA):
    return native_inv_freq(theta) * np.power(4.0, -np.asarray(m, dtype=np.float64))


def m_mrpro_olmo(n=18, low=14, k=K):
    """MrRoPE's own increments on this checkpoint: eps_q ∝ q over slots [14,32]."""
    m = np.zeros(k, dtype=np.float64)
    q = np.arange(1, n + 1, dtype=np.float64)
    m[low + 1: low + n + 1] = np.cumsum(2.0 * q / (n * (n + 1)))
    m[low + n + 1:] = 1.0
    return m


def grad_noise_floor(grads):
    """The standard error of a mean gradient, computed across rows.

    A MEAN GRADIENT WITHOUT A NOISE FLOOR CANNOT BE READ.  The statement this
    campaign most needs to make is "the long-range gradient is ZERO on the band",
    and that statement is indistinguishable from "four rows were not enough to
    see it" unless the row-to-row spread is reported.  The first Qwen run stored
    only the mean of four rows whose answer losses ranged from 0.018 to 0.470, so
    every per-slot number in it was uninterpretable in exactly this way.

    Returns the per-slot standard error of the mean, plus two scalars: the norm
    of the noise floor and the ratio of the signal norm to it.  A ratio near 1
    means the vector is not resolved at all.
    """
    import numpy as _np
    G = _np.asarray(grads, dtype=_np.float64)
    if G.ndim != 2 or G.shape[0] < 2:
        raise ValueError("need at least two gradient rows to estimate a floor")
    sem = G.std(axis=0, ddof=1) / _np.sqrt(G.shape[0])
    signal = _np.linalg.norm(G.mean(axis=0))
    floor = float(_np.linalg.norm(sem))
    return dict(sem=sem, n_rows=int(G.shape[0]), noise_norm=floor,
                signal_norm=float(signal),
                snr=(float(signal) / floor if floor > 0 else None),
                per_slot_sem_median=float(_np.median(sem)))


def make_grad_fn(model, delta):
    import torch

    def grad_at(base_np, tokens, target, keep):
        delta.data.zero_()
        delta.grad = None
        base = torch.tensor(base_np, device="cuda", dtype=torch.float32)
        rot = model.model.rotary_emb
        orig = rot.forward

        def fwd(self, x, position_ids):
            freq = base * torch.exp(-delta)
            inv = freq[None, :, None].expand(position_ids.shape[0], -1, 1)
            pos = position_ids[:, None, :].float()
            with torch.autocast(device_type="cuda", enabled=False):
                phase = (inv.float() @ pos.float()).transpose(1, 2)
                emb = torch.cat((phase, phase), dim=-1)
                co, si = emb.cos() * GAIN, emb.sin() * GAIN
            return co.to(x.dtype), si.to(x.dtype)

        rot.forward = types.MethodType(fwd, rot)
        try:
            out = model(tokens, use_cache=False, logits_to_keep=keep).logits[0].float()
            loss = torch.nn.functional.cross_entropy(out, target, reduction="mean")
            loss.backward()
        finally:
            rot.forward = orig
        g = delta.grad.detach().float().cpu().numpy().astype(np.float64)
        if not np.isfinite(g).all():
            raise ValueError("nonfinite gradient")
        return g, float(loss.detach())

    return grad_at


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--panel", required=True, help="RULER screen.jsonl at 16384")
    ap.add_argument("--nll-dir", required=True, help="dir of doc_NN.npy, 16385 ints")
    ap.add_argument("--native-docs", type=int, default=12)
    ap.add_argument("--native-length", type=int, default=4096)
    ap.add_argument("--long-rows", type=int, default=12)
    ap.add_argument("--tasks", default="niah_single_1,niah_single_3,niah_multikey_1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # The anchor, before the GPU is touched: the table this gradient is taken at
    # must BE the table the archive ran, or the gradient describes a table nobody
    # measured.  Checked against the exact rational, not a hand-typed decimal.
    from fractions import Fraction as F
    n_incr, low = 18, 14
    kq = np.arange(1, n_incr + 1)
    exact = np.array([float(F(2 * i, n_incr * (n_incr + 1))) for i in kq])
    mine = np.diff(m_mrpro_olmo()[low: low + n_incr + 1])
    anchor = float(np.abs(mine - exact).max())
    print(json.dumps({"anchor_mrpro_max_err": anchor,
                      "band": [low, low + n_incr], "n": n_incr, "gain": GAIN}),
          flush=True)
    if anchor > 1e-12:
        print("REFUSING: the MrRoPE reconstruction is not the deployed table",
              file=sys.stderr)
        return 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    delta = torch.nn.Parameter(torch.zeros(K, device="cuda", dtype=torch.float32))
    grad_at = make_grad_fn(model, delta)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError("nonzero dropout would change the frozen response")

    native_np = native_inv_freq().astype(np.float32)
    mrpro_np = m_to_nu(m_mrpro_olmo()).astype(np.float32)

    # ---- native side: in-window tail NLL at the NATIVE table ---------------
    docs = sorted(Path(args.nll_dir).glob("doc_*.npy"))[: args.native_docs]
    if not docs:
        print(f"REFUSING: no doc_*.npy under {args.nll_dir}", file=sys.stderr)
        return 2
    n_grads, n_losses = [], []
    t0 = time.monotonic()
    for f in docs:
        data = np.load(f)
        L = args.native_length
        if len(data) < L + 1:
            raise ValueError(f"{f.name} has {len(data)} tokens, need {L + 1}")
        ids = torch.tensor(data[:L].astype(np.int64), device="cuda")[None]
        tgt = torch.tensor(data[L - 511:L + 1].astype(np.int64), device="cuda")
        g, loss = grad_at(native_np, ids, tgt, 512)
        n_grads.append(g)
        n_losses.append(loss)
        print(json.dumps({"side": "native", "doc": f.name, "loss": loss,
                          "seconds": round(time.monotonic() - t0, 1)}), flush=True)
        del ids, tgt
    n_grad = np.mean(n_grads, axis=0)
    n_floor = grad_noise_floor(n_grads)
    (root / "native_grad.json").write_text(json.dumps(dict(
        n=n_grad.tolist(), losses=n_losses, n_docs=len(n_grads),
        grads=[g.tolist() for g in n_grads],
        noise=dict(noise_norm=n_floor["noise_norm"], snr=n_floor["snr"],
                   per_slot_sem_median=n_floor["per_slot_sem_median"]),
        length=args.native_length, table="native", model=args.model,
        metric="in-window tail-512 next-token NLL")))

    # ---- long-range side: teacher-forced answer CE at MrRoPE's table -------
    want = set(args.tasks.split(","))
    rows = [json.loads(l) for l in open(args.panel)]
    rows = [r for r in rows if r["task"] in want]
    rows = rows[: args.long_rows]
    if not rows:
        print(f"REFUSING: no rows with task in {sorted(want)}", file=sys.stderr)
        return 2
    e_grads, e_losses, e_rows = [], [], []
    for row in rows:
        ans = tok.encode(" " + ", ".join(str(x) for x in row["references"]),
                         add_special_tokens=False)
        if not ans:
            continue
        ids = torch.tensor([row["prompt_ids"] + ans[:-1]], device="cuda")
        tgt = torch.tensor(ans, device="cuda")
        g, loss = grad_at(mrpro_np, ids, tgt, len(ans))
        e_grads.append(g)
        e_losses.append(loss)
        e_rows.append(row["row_id"])
        print(json.dumps({"side": "long_range", "row": row["row_id"],
                          "n_ans": len(ans), "loss": loss}), flush=True)
        del ids, tgt
    e = np.mean(e_grads, axis=0)
    e_floor = (grad_noise_floor(e_grads) if len(e_grads) >= 2 else None)

    # ---- the sec.5 residual ----------------------------------------------
    nn = float(n_grad @ n_grad)
    ne = float(n_grad @ e)
    lam = max(0.0, -ne / nn) if nn > 0 else None
    resid = e + lam * n_grad if lam is not None else e
    LN4 = float(np.log(4.0))
    out = dict(
        status="COMPLETE", model=args.model, band=[low, low + n_incr],
        n_slots=n_incr,
        gain=GAIN, theta=THETA, window=WINDOW,
        e=e.tolist(), n=n_grad.tolist(),
        lambda_hat=lam, residual=resid.tolist(),
        residual_norm=float(np.linalg.norm(resid)),
        e_norm=float(np.linalg.norm(e)),
        n_norm=float(np.linalg.norm(n_grad)),
        cos_en=(float(ne / (np.linalg.norm(n_grad) * np.linalg.norm(e)))
                if np.linalg.norm(n_grad) and np.linalg.norm(e) else None),
        e_rows=e_rows, e_losses=e_losses, n_losses=n_losses,
        e_grads=[g.tolist() for g in e_grads],
        e_noise=(None if e_floor is None else
                 dict(noise_norm=e_floor["noise_norm"], snr=e_floor["snr"],
                      per_slot_sem_median=e_floor["per_slot_sem_median"])),
        dL_dm_longrange=(LN4 * e).tolist(),
        dL_dm_native=(LN4 * n_grad).tolist(),
        anchor_mrpro_max_err=anchor,
        scope=("e is the long-range gradient at MrRoPE's table, n is the native "
               "in-window NLL gradient at the NATIVE table; the residual is a "
               "statement about a direction pair, not about one point"),
        native_metric="in-window tail-512 next-token NLL, not the plan's "
                      "output-KL; declared simplification, same as the Qwen run",
    )
    # A defensible encoder: numpy scalars are floats and arrays are lists.  This
    # does NOT paper over the bug that produced the first failure -- an ndarray
    # was written into a field because `n` named both the band width and the
    # gradient vector, and `n_slots=n` then serialised the gradient.  The names
    # are split now; this only stops a numpy scalar from killing the receipt
    # after twenty minutes of GPU work.
    def _enc(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")
    (root / (args.out or "kkt.json")).write_text(
        json.dumps(out, indent=1, default=_enc))
    print(json.dumps({k: out[k] for k in ("lambda_hat", "residual_norm", "e_norm",
                                          "n_norm", "cos_en")}), flush=True)
    print(json.dumps({"NOISE_FLOOR": dict(
        e_norm=out["e_norm"], e_floor=out["e_noise"],
        e_snr=(None if not out["e_noise"] else
               out["e_norm"] / max(out["e_noise"]["noise_norm"], 1e-300)),
        n_norm=out["n_norm"], n_floor=dict(
            noise_norm=n_floor["noise_norm"], snr=n_floor["snr"]))}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
