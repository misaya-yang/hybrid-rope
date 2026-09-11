#!/usr/bin/env python3
"""The cross-model instrument: the same continuous long-range NLL, on Qwen.

WHY THIS EXISTS.  The campaign's cross-model claim rests on a 36-row Qwen RULER
panel whose paired SE is about 7.6 points, and the archived endpoint contrast on
it is

    b=0  MrRoPE       Qwen 128K RULER 78.13%
    b=1  deployed BM  Qwen 128K RULER 70.83%

-- a 7.3 point reversal in favour of MrRoPE, against a 48.5 point gap the other
way on OLMo.  A 7.3 point difference on a panel with a 7.6 point SE is not a
result; it is an instrument that cannot see.  Everything this campaign has
learned on OLMo -- that S is not the operative variable, that the four-corner
simplex has a bad fast vertex, that the optimum sits at an interior ramp -- is
currently untested on any second model family.

WHAT IT DOES.  The identical instrument shape as `olmo_longnll.py`: teacher-forced
tail-512 next-token NLL at a length beyond the model's native window, native
reference measured in the same process with the same batch policy, so every delta
is internally paired.  On Qwen2.5-1.5B (theta 1e6, window 32768, head_dim 128) the
turn rule puts the band on slots [23,40] with n=17, so every arm is rebuilt in
THAT geometry -- the same (alpha, beta) or (a, b) means the same thing on both
checkpoints, which is the whole point of the parameterisation.

THE LENGTH IS 2x, NOT 4x, AND THAT IS A DECLARED COMPROMISE.  The only
Qwen-tokenised long documents on this machine are the 16 x 32769 token arrays in
`bm_transfer_qwen7b_20260908/prepared_nll_01`; there is no raw long-form corpus
to re-tokenise, and the 4x length (131072) would leave only 4 non-overlapping
documents and a paired SE too wide to resolve anything.  Concatenating
non-overlapping pairs gives 8 documents of 65538, so the instrument runs at 2x.
A 2x test is a weaker extrapolation than 4x and the numbers are NOT comparable to
the archived 128K RULER column -- they are comparable to each other, which is all
a ranking instrument needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QWEN = dict(theta=1e6, window=32768, head_dim=128, K=64, low=23, n=17)
GAIN = 1.138629436111989


def build_arms():
    from experiments.curvature_20260910.tables import (m_incr_beta, m_mrpro,
                                                       m_step, m_turns)
    lo, n = QWEN["low"], QWEN["n"]

    def T(a, b):
        return np.asarray(m_turns(a, b, QWEN["theta"], QWEN["window"],
                                  QWEN["head_dim"], ramp="beta1"), float)

    return [
        ("native", np.zeros(64)),
        ("mrpro", np.asarray(m_mrpro(n=n, low=lo), float)),
        ("beta_b0p5", np.asarray(m_incr_beta(0.5, n=n, low=lo), float)),
        ("beta_b1_BM", np.asarray(m_incr_beta(1.0, n=n, low=lo), float)),
        ("beta_b2", np.asarray(m_incr_beta(2.0, n=n, low=lo), float)),
        ("beta_b3", np.asarray(m_incr_beta(3.0, n=n, low=lo), float)),
        ("turns_a1_b16", T(1.0, 16)),
        ("turns_a1_b64", T(1.0, 64)),
        ("turns_a2_b32", T(2.0, 32)),
        # the fourth corner of the forcing simplex, in THIS geometry
        ("corner_a0b1", np.asarray(m_incr_beta(1.0, n=n, low=lo, shape_a=0.0),
                                   float)),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nll-dir", required=True, help="dir of doc_NN.npy, each 32769")
    ap.add_argument("--far", type=int, default=65538)
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
    parts = [np.load(f) for f in files]
    if len(parts) < 2:
        print(f"REFUSING: need at least 2 doc_*.npy under {args.nll_dir}",
              file=sys.stderr)
        return 2
    cat = np.concatenate(parts).astype(np.int32)
    need = args.far + 1
    docs = [cat[i * args.far: i * args.far + need]
            for i in range(len(cat) // args.far)]
    docs = [d for d in docs if len(d) == need]
    if not docs:
        print(f"REFUSING: {len(cat)} tokens cannot make a single {need}-token "
              "document", file=sys.stderr)
        return 2
    print(json.dumps({"phase": "DOCS", "source_docs": len(parts),
                      "total_tokens": int(len(cat)), "far": args.far,
                      "documents": len(docs)}), flush=True)

    rot = model.model.rotary_emb
    dev = next(model.parameters()).device
    nu0 = QWEN["theta"] ** (-np.arange(64) / 64.0)

    def per_doc_nll(values):
        rot.inv_freq = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dev)
        rot.attention_scaling = float(GAIN)
        out = []
        with torch.inference_mode():
            for arr in docs:
                ids = torch.tensor(arr[: args.far].astype(np.int64),
                                   device=dev)[None]
                tgt = torch.tensor(
                    arr[args.far - args.tail + 1: args.far + 1].astype(np.int64),
                    device=dev)
                lg = model(ids, use_cache=False,
                           logits_to_keep=args.tail).logits[0].float()
                out.append(float(torch.nn.functional.cross_entropy(lg, tgt)))
                del ids, tgt, lg
        return np.array(out)

    base = per_doc_nll(nu0)
    print(json.dumps({"phase": "BASELINE", "native_nll": float(base.mean()),
                      "docs": len(docs)}), flush=True)

    arms = build_arms()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        arms = [(n_, m) for n_, m in arms if n_ in want]

    rows = []
    for name, m in arms:
        v = nu0 * np.power(4.0, -np.asarray(m, dtype=float))
        d = per_doc_nll(v) - base
        tD = np.asarray(v) * args.far / (2 * np.pi)
        rec = dict(arm=name, sum_m=float(np.asarray(m).sum()),
                   nll=float(base.mean() + d.mean()), delta=float(d.mean()),
                   se=float(d.std(ddof=1) / np.sqrt(len(d))),
                   n_window=int(((tD >= 0.25) & (tD <= 16)).sum()),
                   per_doc=d.tolist())
        rows.append(rec)
        print(json.dumps({"arm": name, "sum_m": round(rec["sum_m"], 4),
                          "nll": round(rec["nll"], 6),
                          "delta": round(rec["delta"], 6),
                          "se": round(rec["se"], 6),
                          "N_window": rec["n_window"]}), flush=True)
        with (root / "rows.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    (root / "manifest.json").write_text(json.dumps(dict(
        status="COMPLETE", model=args.model, far=args.far, docs=len(docs),
        native_nll=float(base.mean()), arms=[r["arm"] for r in rows],
        scope=("Teacher-forced tail-512 NLL at 2x the native window on Qwen2.5-"
               "1.5B, native reference measured in this process. The 2x length is "
               "a declared compromise: no raw long-form corpus exists on this "
               "machine, so documents are concatenated pairs of the 32769-token "
               "arrays and 4x would leave only 4 documents. These numbers are "
               "comparable to each other, NOT to the archived 128K RULER "
               "column.")), indent=1))

    print("\n=== Qwen 2x long-range: arms by delta vs native ===")
    print("  %-14s %8s %11s %10s %5s" % ("arm", "sum_m", "delta", "se", "N"))
    for r in sorted(rows, key=lambda z: z["delta"]):
        print("  %-14s %8.3f %11.6f %10.6f %5d"
              % (r["arm"], r["sum_m"], r["delta"], r["se"], r["n_window"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
