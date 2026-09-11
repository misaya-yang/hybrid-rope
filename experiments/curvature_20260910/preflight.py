#!/usr/bin/env python3
"""Stage 0: everything that can be wrong before the GPU is touched.

Every check here is either free or needs only the checkpoint's config.json.
Run it first, every session.  Each failure names the thing it invalidates, so a
red result does not need to be diagnosed twice.

  1. the m-coordinate reproduces the DEPLOYED tables (MrPro, YaRN, Native) to
     float32 rounding.  Failure: the algebra is wrong and nothing downstream is
     interpretable.
  2. the checkpoint's config matches the pinned assumptions -- rope_theta,
     head_dim/2 = 64 slots, native window.  Failure: every m value is offset and
     the table that gets installed is not the table that was solved.
  3. the f=1/4..1 tolerance: a solved table whose smallest interior gap goes
     below the float32 resolution near the slow end is numerically degenerate
     even if it is mathematically fine.
  4. the inputs the probes read actually exist, and the harness is intact.
  5. the environment: transformers generation (4.x `rope_scaling` vs 5.x
     `rope_parameters`), SDPA availability, free disk.

  python -m experiments.curvature_20260910.preflight \
      --model /root/autodl-tmp/rope_qwen_baseline_20260907/model
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import tables as T


def check(label, ok, detail, invalidates=None):
    rec = dict(check=label, ok=bool(ok), detail=detail)
    if not ok and invalidates:
        rec["invalidates"] = invalidates
    print(f"[{'ok ' if ok else 'FAIL'}] {label}: {detail}"
          + (f"\n       -> {invalidates}" if (not ok and invalidates) else ""))
    return rec


def check_table_resolution(label, values, tol=10.0):
    """Can float32 hold the gaps this table asks for?

    The slow end of a 64-slot table with theta=1e6 reaches 3.1e-7, where float32
    spacing is ~1.9e-14.  A solved step that puts two slow slots closer than a
    few multiples of that is not a physical table -- the harness will round it
    and the run measures the rounding, not the solution.  Equal slots (gap 0,
    the flat MrRoPE tail) are intentional and excluded.
    """
    nu = np.asarray(values, dtype=np.float64)
    gaps = -np.diff(nu)
    spacing = np.spacing(nu[1:].astype(np.float32)).astype(np.float64)
    moved = gaps > 0
    if not moved.any():
        return check(label, True, "no nonzero gaps (flat table)")
    ratio = float((gaps[moved] / spacing[moved]).min())
    return check(label, ratio > tol,
                 f"smallest gap / float32 spacing = {ratio:.1f} at the tightest slot",
                 "the table is not representable at the precision the harness installs; "
                 "shrink eps or the step will be measured through float32 rounding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", default="analysis/unify_20260910/tables/ground_truth_tables.json")
    ap.add_argument("--npy", default=None, help="long corpus a probe will read")
    ap.add_argument("--harness", default=None, help="harness root, if panel jobs are planned")
    ap.add_argument("--tables", default=None, help="harness prepared_*/tables.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    model = Path(args.model)

    # 1 -- the algebra against what is on the GPU
    try:
        diffs = T.verify(args.gt)
        results.append(check("deployed-table reconstruction", True,
                             json.dumps(diffs),
                             "everything: the m-coordinate or a construction is wrong; "
                             "no probe output can be interpreted"))
    except Exception as exc:
        results.append(check("deployed-table reconstruction", False, repr(exc),
                             "everything downstream"))

    # 2 -- the pinned checkpoint assumptions
    cfg_path = model / "config.json"
    if not cfg_path.exists():
        results.append(check("checkpoint config", False, f"{cfg_path} not found",
                             "the model path is wrong or the checkpoint is not downloaded"))
    else:
        cfg = json.loads(cfg_path.read_text())
        theta = float(cfg.get("rope_theta", cfg.get("rope_scaling", {}).get("rope_theta", 0)) or 0)
        hd = int(cfg.get("head_dim") or cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1))
        W = int(cfg.get("max_position_embeddings", 0))
        want = T.QWEN25_3B
        results.append(check("rope_theta", abs(theta - want["theta"]) < 1.0, f"config {theta:g}",
                             "every m value is offset: the installed table is not the solved one"))
        results.append(check("head_dim/2 = slots", hd // 2 == T.K, f"head_dim {hd} -> {hd//2} slots",
                             "the table has the wrong length and install() will refuse"))
        results.append(check("native window", W == want["window"], f"max_position_embeddings {W}",
                             "the YaRN bands and the f=1/4 reference length both move"))
        results.append(check("architectures", cfg.get("architectures") == ["Qwen2ForCausalLM"],
                             str(cfg.get("architectures")),
                             "model plumbing assumes Qwen2 rotary layout"))
        if abs(theta - want["theta"]) < 1.0:
            results.append(check_table_resolution("MrPro (deployed)",
                                                  T.build("mrpro_n17")["values_float32"]))

    # 3 -- a solved table, if one is present next to the invocation
    for cand in sorted(Path("runs").glob("kkt_*.json")) if Path("runs").exists() else []:
        try:
            rec = json.load(open(cand))
            results.append(check_table_resolution(cand.name,
                                                  np.asarray(rec["nu_new"], dtype=np.float64)))
        except Exception as exc:
            results.append(check(f"solved table {cand.name}", False, repr(exc)))

    # 4 -- inputs
    if args.npy:
        p = Path(args.npy)
        ok = p.exists()
        n = np.load(p, mmap_mode="r").shape[0] if ok else 0
        results.append(check("long corpus", ok and n >= 131072,
                             f"{args.npy}: {n} tokens",
                             "the 128K measurement cannot be made; drop to the longest length "
                             "the corpus supports and record the shortening"))
        # The probe must not be a document the panel is scored on, or the
        # gradient is fitted to the evaluation set one forward at a time.  The
        # panel's long_eval takes the first docs_per_dataset per source.
        lc = p.parent / "manifest.json"
        if lc.exists():
            try:
                man = json.loads(lc.read_text())
                per = 2
                seen, picked = {}, []
                for d in man["docs"]:
                    ds = d["source"]["dataset"]
                    if seen.get(ds, 0) < per:
                        picked.append(d["file"])
                        seen[ds] = seen.get(ds, 0) + 1
                hit = p.name in picked
                results.append(check("probe corpus held out", not hit,
                                     f"{p.name} vs panel documents {picked}",
                                     "the gradient is measured on a document the panel scores; "
                                     "move --npy to another document in the same directory"))
            except Exception as exc:
                results.append(check("probe corpus held out", False, repr(exc)))
    if args.harness:
        h = Path(args.harness)
        results.append(check("harness root", (h / "queue").exists() or h.exists(),
                             str(h), "panel jobs cannot be queued"))
    if args.tables:
        try:
            tabs = json.loads(Path(args.tables).read_text())
            need = {"MrPro"}
            miss = need - set(tabs)
            results.append(check("harness tables", not miss,
                                 f"{sorted(tabs)} missing {sorted(miss)}" if miss else f"{sorted(tabs)}",
                                 "the matched gain cannot be read and the candidate would not "
                                 "be a freq-only change"))
        except Exception as exc:
            results.append(check("harness tables", False, repr(exc)))

    # 5 -- environment
    try:
        import transformers
        import torch
        v = transformers.__version__
        gen = "5.x (rope_parameters)" if int(v.split(".")[0]) >= 5 else "4.x (rope_scaling)"
        results.append(check("transformers", True, f"{v} -> {gen}"))
        results.append(check("torch", True,
                             f"{torch.__version__} cuda={torch.cuda.is_available()} "
                             f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}"))
        results.append(check("sdpa", hasattr(torch.nn.functional, "scaled_dot_product_attention"), "F.scaled_dot_product_attention"))
        try:
            import flash_attn                                            # noqa: F401
            results.append(check("flash_attn", True, "present"))
        except Exception:
            results.append(check("flash_attn", True,
                                 "absent; model.py falls back to SDPA, memory is higher"))
    except Exception as exc:
        results.append(check("environment", False, repr(exc), "nothing runs"))

    for d in ("/root/autodl-tmp", "."):
        try:
            free = shutil.disk_usage(d).free / 2**30
            results.append(check(f"disk {d}", free > 5, f"{free:.1f} GiB free",
                                 "probe receipts are small, but the harness writes token-level "
                                 "NLL rows; below ~5 GiB a long panel run will fail mid-way"))
        except Exception:
            pass

    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(results=results,
                                                  passed=len(results) - len(failed),
                                                  failed=len(failed)), indent=1) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
