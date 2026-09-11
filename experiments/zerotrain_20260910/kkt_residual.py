#!/usr/bin/env python3
"""The KKT test: measure BOTH gradients at once, and resolve the sign tension.

THE TENSION THIS FILE EXISTS TO RESOLVE.  The Phase-0 full-model gradients are
the long-range objective's gradient, evaluated at MrRoPE's table, and they say
something that contradicts the published result:

    dL/dm_j  is LARGE and NEGATIVE over the high-frequency slots (mean -21.7 at
    slot 16, |.| up to 27), decaying monotonically to ~0 by slot 40.

A negative derivative means raising m_j -- COMPRESSING the high-frequency slots
-- lowers the long-range loss.  That is the YaRN direction, and YaRN is measured
to be WORSE at long range than MrRoPE.  So the raw gradient cannot be the whole
story, and the missing half is the constraint: MrRoPE holds those slots at m = 0
precisely because moving them costs in-window behaviour, and the plan's problem
is  min L_long  s.t.  D_N <= eps  -- not min L_long.

THE TEST IS THE PLAN'S OWN sec.5 DIAGNOSTIC.  With e the long-range gradient and
n the native constraint's gradient,

    lambda_hat = max(0, -n^T e / (n^T n)),     residual = e + lambda_hat * n

A near-zero residual means the two gradients are antiparallel: the long-range
objective's wish is exactly priced by the native constraint, which is what "MrRoPE
is a KKT point" means. A large residual names a direction that the long-range
objective would still pay to move along at the price the constraint charges --
an unclaimed improvement, and a concrete table to build.

WHERE n IS EVALUATED, AND WHY IT IS NOT e'S EVALUATION POINT.  The constraint is
defined RELATIVE TO NATIVE, so n is the gradient of the native metric at the
NATIVE table (m = 0 everywhere) -- that is the point the checkpoint was trained
at and the point D_N is measured from.  e is the long-range gradient at MrRoPE's
table, because that is the incumbent whose optimality is in question.  They are
not gradients of the same function at the same point and the code keeps them
apart; the residual is a statement about a DIRECTION pair, not about one
Hessian.

THE NATIVE METRIC IS IN-WINDOW TAIL NLL, and that is a declared simplification:
the plan's D_N is an output-KL against the frozen native table (no first-order
term, quadratic form = the Fisher), and the KL is not computed here.  NLL is used
because it is the objective the checkpoint was trained on and because it needs
one forward rather than a stored base-log-prob tensor per document.  The receipt
says so; a KL version is a separate run.
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
from phase1_screen import (GAIN_YARN, QWEN25_3B, anchor_check,  # noqa: E402
                           m_mrpro, m_to_inv_freq, native_inv_freq)


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
    """One forward+backward at a given base table; returns dL/d(delta).

    The rotary patch is installed and removed INSIDE each call rather than left
    in place, because the two sides of this measurement use DIFFERENT base tables
    -- MrRoPE's for the long-range gradient, native for the constraint gradient --
    and a patch that outlived its call would silently evaluate the second side at
    the first side's table.  That failure produces two plausible vectors and a
    meaningless residual, which is the worst kind.
    """
    import types
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
                co, si = emb.cos() * GAIN_YARN, emb.sin() * GAIN_YARN
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
    ap.add_argument("--history", default="/root/autodl-tmp/bm_transfer_20260908")
    ap.add_argument("--native-docs", type=int, default=16)
    ap.add_argument("--native-length", type=int, default=32768)
    ap.add_argument("--long-rows", type=int, default=16,
                    help="teacher-forced rows per gradient; >=2 is required and "
                         "the receipt reports whether the mean is resolved")
    ap.add_argument("--long-length", type=int, default=32768)
    ap.add_argument("--long-tasks", default="niah_multikey_2,niah_multiquery,"
                                            "niah_multikey_1,niah_multivalue")
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ac = anchor_check()
    assert ac["bit_exact"], "anchor is not bit-exact"

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.nongeometric_screen.worker import Worker
    import torch

    w = Worker(root, args.history)
    model = w.model
    theta = QWEN25_3B["theta"]
    native_np = native_inv_freq(theta).astype(np.float32)
    mrpro_np = m_to_inv_freq(m_mrpro(17), theta).astype(np.float32)

    delta = torch.nn.Parameter(torch.zeros(64, device="cuda", dtype=torch.float32))
    grad_at = make_grad_fn(model, delta)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError("nonzero dropout would change the frozen response")

    # ---- native side: in-window tail NLL at the NATIVE table --------------
    docs = w.nll_manifest["docs"]
    n_grads, n_losses = [], []
    t0 = time.monotonic()
    for d in docs[: args.native_docs]:
        data = np.load(w.nll_inputs / d["file"])
        L = args.native_length
        ids = torch.tensor(data[:L].astype(np.int64), device="cuda")[None]
        tgt = torch.tensor(data[L - 511:L + 1].astype(np.int64), device="cuda")
        g, loss = grad_at(native_np, ids, tgt, 512)
        n_grads.append(g)
        n_losses.append(loss)
        print(json.dumps({"side": "native", "doc": d["file"], "loss": loss,
                          "seconds": time.monotonic() - t0}), flush=True)
        del ids, tgt
    n = np.mean(n_grads, axis=0)
    n_floor = grad_noise_floor(n_grads)
    (root / "native_grad.json").write_text(json.dumps(dict(
        n=n.tolist(), losses=n_losses, n_docs=len(n_grads),
        grads=[g.tolist() for g in n_grads],
        noise=dict(noise_norm=n_floor["noise_norm"], snr=n_floor["snr"],
                   per_slot_sem_median=n_floor["per_slot_sem_median"]),
        length=args.native_length, table="native",
        metric="in-window tail-512 next-token NLL",
        seconds=time.monotonic() - t0)))

    # ---- long-range side: reuse the Phase-0 rows' gradient, recomputed here
    # at MrPro so both sides come from one harness in one session ----------
    e_grads, e_losses, e_rows = [], [], []
    # THE ROW SET IS A RESOLUTION LIMIT, NOT A DETAIL.  The first version took
    # the four rows ending _0/_1 of two niah tasks, which is exactly four
    # samples of a quantity whose row-to-row spread was never measured -- and its
    # answer losses ran from 0.018 to 0.470, so the mean was probably one row.
    # `--long-rows` now takes up to N rows spread over every retrieval task at
    # this length, and `grad_noise_floor` reports whether the mean is resolved.
    _want = {s.strip() for s in args.long_tasks.split(",") if s.strip()}
    rows = [r for r in w.screen if r["task"] in _want
            and r["length_cap"] == args.long_length]
    rows.sort(key=lambda r: r["row_id"])
    rows = rows[: args.long_rows]
    if len(rows) < 2:
        print(f"REFUSING: {len(rows)} long-range rows cannot resolve a gradient",
              file=sys.stderr)
        return 2
    for row in rows:
        ans = w.tokenizer.encode(" " + ", ".join(row["references"]),
                                 add_special_tokens=False)
        ids = torch.tensor([row["prompt_ids"] + ans[:-1]], device="cuda")
        tgt = torch.tensor(ans, device="cuda")
        g, loss = grad_at(mrpro_np, ids, tgt, len(ans))
        e_grads.append(g)
        e_losses.append(loss)
        e_rows.append(row["row_id"])
        print(json.dumps({"side": "long_range", "row": row["row_id"],
                          "loss": loss}), flush=True)
        del ids, tgt
    e = np.mean(e_grads, axis=0)
    e_floor = (grad_noise_floor(e_grads) if len(e_grads) >= 2 else None)

    # ---- the sec.5 residual ----------------------------------------------
    nn = float(n @ n)
    ne = float(n @ e)
    lam = max(0.0, -ne / nn) if nn > 0 else None
    resid = e + lam * n if lam is not None else e
    cos = float(ne / (np.linalg.norm(n) * np.linalg.norm(e)))
    LN4 = float(np.log(4.0))
    out = dict(
        status="COMPLETE",
        e=e.tolist(), n=n.tolist(),
        lambda_hat=lam, residual=resid.tolist(),
        residual_norm=float(np.linalg.norm(resid)),
        e_norm=float(np.linalg.norm(e)), n_norm=float(np.linalg.norm(n)),
        cos_en=cos,
        e_rows=e_rows, e_losses=e_losses, n_losses=n_losses,
        e_grads=[g.tolist() for g in e_grads],
        e_noise=(None if e_floor is None else
                 dict(noise_norm=e_floor["noise_norm"], snr=e_floor["snr"],
                      per_slot_sem_median=e_floor["per_slot_sem_median"])),
        # the per-slot reading in the m coordinate, which is what a table speaks
        dL_dm_longrange=(LN4 * e).tolist(),
        dL_dm_native=(LN4 * n).tolist(),
        scope=("e is the long-range gradient at MrRoPE's table, n is the native "
               "in-window NLL gradient at the NATIVE table; the residual is a "
               "statement about a direction pair, not about one point"),
        native_metric="in-window tail-512 next-token NLL, not the plan's "
                      "output-KL; declared simplification",
    )
    (root / "kkt.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("lambda_hat", "residual_norm", "e_norm",
                                          "n_norm", "cos_en")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
