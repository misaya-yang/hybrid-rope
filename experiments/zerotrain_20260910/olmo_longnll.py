#!/usr/bin/env python3
"""A CONTINUOUS long-range instrument on OLMo -- because RULER's is quantized.

THE PROBLEM THIS FIXES.  Every score this campaign has compared is a RULER
exact-match macro average over 350 rows.  Those numbers come out CLUSTERED:

    0.3846  0.3937  0.3969  0.4167   |   0.5001  0.5100  0.5384

with a 0.083 gap in the middle and no table landing in it.  Two consequences:
a "predictor" that only has to separate two clusters will look far better than it
is, and a continuous mechanism will look discontinuous.  The campaign has spent
its last several analyses hunting a continuous scalar to explain a variable that
the INSTRUMENT has quantized.  That is the wrong way round.

The Pro analysis named the mechanism for this: RULER scores an answer by exact
match, so a small continuous logit improvement that crosses an argmax turns a
whole row from 0 to 1.  Thresholds amplify.

WHAT THIS MEASURES INSTEAD.  Teacher-forced next-token NLL over the last 512
positions of the SAME sixteen held-out documents the archived instrument used
(`prepared_nll_02/doc_*.npy`, 16385 tokens each, so the scored positions sit at a
context of about 16k -- well past OLMo's 4096 native window).  No generation.
Every scored position contributes, so an arm gets 16 x 512 = 8192 observations
instead of 350 thresholded rows.

The archived run_nll_01 already did this for exactly two arms and the signal is
large: at 16384 the deployed BM is 0.826 nats BELOW MrRoPE.  So the instrument has
range; it has simply never been pointed at the arms this campaign generated.

WHAT IT CANNOT SAY.  NLL over natural text is not RULER.  An arm can improve NLL
and not retrieve better, or the reverse.  This is a DIFFERENT and much finer
instrument, not a better version of the same one -- and the two should be reported
side by side, never substituted for each other.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OLMO = dict(theta=500_000.0, window=4096, head_dim=128, K=64, low=14, n=18)
GAIN = 1.138629436111989


def build_arms():
    """The arms worth measuring continuously: the whole measured set plus the
    derived controls.  Every one of them has a quantized RULER score already."""
    from experiments.curvature_20260910.tables import (m_C42, m_C42V24,
                                                       m_incr_beta, m_leak,
                                                       m_mrpro, m_step, m_taper,
                                                       m_turns)
    lo, n = OLMO["low"], OLMO["n"]

    def T(a, b):
        return np.asarray(m_turns(a, b, OLMO["theta"], OLMO["window"],
                                  OLMO["head_dim"], ramp="beta1"), float)

    def knife(d, a):
        m = np.asarray(m_taper(d), float).copy()
        if a:
            m[:lo] = np.maximum(m[:lo], a * np.arange(1, lo + 1, dtype=float))
        return np.clip(m, 0.0, 1.0)

    return [
        ("native",    np.zeros(64)),
        ("mrpro",     np.asarray(m_mrpro(n=n, low=lo), float)),
        ("beta_b0p25", np.asarray(m_incr_beta(0.25, n=n, low=lo), float)),
        ("turns_a0p5_b32", T(0.5, 32)),
        ("turns_a1_b16",   T(1.0, 16)),
        ("beta_b0p5", np.asarray(m_incr_beta(0.5, n=n, low=lo), float)),
        ("beta_b1_BM", np.asarray(m_incr_beta(1.0, n=n, low=lo), float)),
        ("beta_b2",   np.asarray(m_incr_beta(2.0, n=n, low=lo), float)),
        ("turns_a2_b32", T(2.0, 32)),
        ("turns_a1_b64", T(1.0, 64)),
        ("knife_taper", knife(0.0063, 0.0)),
        ("knife_leak",  knife(0.0, 0.0063)),
        ("knife_both",  knife(0.0063, 0.0063)),
        ("ctl_C42",   np.asarray(m_C42(), float)),
        ("ctl_C42V24", np.asarray(m_C42V24(), float)),
        ("step_hi25", np.asarray(m_step(25, lo=lo), float)),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True)
    ap.add_argument("--length", type=int, default=16385)
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    files = sorted(Path(args.nll_dir).glob("doc_*.npy"))
    docs = []
    for f in files:
        a = np.load(f)
        if len(a) >= args.length:
            docs.append(a[: args.length])
    if not docs:
        print(f"REFUSING: no doc_*.npy of length >= {args.length}", file=sys.stderr)
        return 2

    def score(values):
        rot = model.model.rotary_emb
        dev = next(model.parameters()).device
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dev)
        rot.attention_scaling = float(GAIN)
        losses, tok = [], []
        with torch.inference_mode():
            for a in docs:
                ids = torch.tensor(a[:-1].astype(np.int64), device=dev)[None]
                tgt = torch.tensor(a[1:].astype(np.int64), device=dev)
                out = model(ids, use_cache=False,
                            logits_to_keep=args.tail).logits[0].float()
                per = torch.nn.functional.cross_entropy(
                    out, tgt[-args.tail:], reduction="none")
                losses.append(float(per.mean()))
                tok.append(per.float().cpu().numpy())
                del ids, tgt, out, per
        return losses, tok

    arms = build_arms()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        arms = [(n_, m) for n_, m in arms if n_ in want]

    rows = []
    for name, m in arms:
        values = (OLMO["theta"] ** (-np.arange(64) / 64)) * np.power(4.0, -m)
        t0 = time.monotonic()
        losses, tok = score(values.astype(np.float32))
        rec = dict(arm=name, sum_m=float(np.asarray(m).sum()),
                   nll=float(np.mean(losses)), per_doc=losses,
                   seconds=round(time.monotonic() - t0, 1))
        (root / f"{name}.npz").parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / f"{name}.npz",
                            token_nll=np.concatenate(tok),
                            per_doc=np.array(losses))
        rows.append(rec)
        with (root / "rows.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"CONT_NLL": rec["arm"], "nll": round(rec["nll"], 6),
                          "sec": rec["seconds"]}), flush=True)

    base = next((r for r in rows if r["arm"] == "native"), None)
    if base:
        print("\n=== continuous long-range NLL (last %d positions, context ~%d) ==="
              % (args.tail, args.length - 1))
        print("%-16s %10s %12s" % ("arm", "NLL", "vs native"))
        for r in sorted(rows, key=lambda z: z["nll"]):
            print("%-16s %10.6f %+12.6f" % (r["arm"], r["nll"],
                                            r["nll"] - base["nll"]))
    (root / "manifest.json").write_text(json.dumps(dict(
        status="COMPLETE", length=args.length, tail=args.tail, docs=len(docs),
        arms=[r["arm"] for r in rows],
        scope=("Teacher-forced next-token NLL over the last %d positions of %d "
               "held-out documents at a context of about %d tokens.  NOT RULER: "
               "continuous and high-resolution, but a different instrument -- "
               "report side by side, never as a substitute."
               % (args.tail, len(docs), args.length - 1))), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
