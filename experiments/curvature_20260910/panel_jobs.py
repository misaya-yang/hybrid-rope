#!/usr/bin/env python3
"""Turn solved tables into jobs for the existing frozen-weight panel harness.

The harness is `experiments/nongeometric_screen/worker.py`.  It keeps one Qwen
resident, reads explicit JSON jobs from `queue/`, scores NLL and the mixed-RULER
panel against the archived MrPro rows, and refuses to run if any historical
input has drifted.  Nothing here re-implements it: this file only writes job
contracts in the shapes the harness actually dispatches on, so a solved table is
evaluated on exactly the panel every other candidate was evaluated on.

The harness has TWO dispatch paths, and a candidate needs both:

  action='evaluate' (default)   job['spec'] is installed, the RULER panel runs,
                                and NLL is scored on `prepared_nll_01` at
                                `nll_lengths`.  Those documents are 32769 tokens,
                                so the lengths that path can reach are
                                8192/16384/32768 -- the ones the archived MrPro
                                rows in `run_nll_01/` were scored at.
                                `evaluate` also writes `results/<name>/contract.json`.
  action='module'               `experiments.nongeometric_screen.<module>.run`,
                                here `long_eval`: 65536/131072 on `long_inputs`
                                (131073 tokens/document, one contiguous real
                                document prefix, no concatenated filler), paired
                                against MrPro inside the same model load.

`long_eval` recovers each method's spec from `results/<method>/contract.json`,
so the registration job must be processed first.  The worker takes
`sorted(queue/*.json)[0]`, and this file names its two outputs so that the
registration job sorts first -- it raises rather than write a pair that would
run out of order.

Two things this file does that a bare table build would not:

* It reads the harness's own `tables.json` and uses that file's gain, so the
  candidate differs from MrPro in the freqs and nothing else.  Five.6-pro's
  constraint is explicit: gain stays at MrPro's value and is not co-optimised.
  A candidate that quietly changed the gain would be unreadable against the
  archived rows.

* It verifies the reconstruction against the DEPLOYED MrPro table before
  writing anything.  If the m-coordinate or a construction is off by more than
  float32 rounding, the job is not written -- a wrong table that reaches the
  GPU burns hours and produces a number nobody can interpret.

Usage:

  python -m experiments.curvature_20260910.panel_jobs \
      --kkt runs/kkt_mrpro.json --queue-id 0450 \
      --tables /root/autodl-tmp/bm_transfer_20260908/prepared_qwen3_01/tables.json \
      --root /root/autodl-tmp/nongeometric_screen_20260909 \
      --receipts runs/panel_jobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import tables as T


def load_tables(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found.  This file is the harness's own table set and is the "
            "only source of the matched gain; point --tables at the prepared_qwen3_01 "
            "copy on the server.")
    return json.loads(p.read_text())


def reference_table(tables, name="MrPro"):
    if name not in tables:
        raise KeyError(f"{name} missing from the harness tables; have {sorted(tables)}")
    t = tables[name]
    return np.asarray(t["values_float32"], dtype=np.float64), float(t["gain"])


def candidate_from_receipt(path, base_name, gain):
    """A solved table plus the controls it must be read against."""
    rec = json.load(open(path))
    if rec.get("degenerate") or rec.get("G", 0.0) <= 0:
        raise ValueError(f"{path}: degenerate receipt (G <= 0).  The EXPLAINS branch has "
                         "no candidate table to run; write that up instead of a job.")
    d = np.array([float(rec["d"][str(j)]) for j in range(T.K)])
    table = T.from_eps(d, base_name=base_name, gain=gain)
    return table, rec


def check_monotone(values, name):
    """The panel installs whatever it is given; a non-monotone table is still a
    legal rotation grid but it is no longer a frequency ordering, and every
    band statement about it becomes unreadable.  Refuse rather than warn."""
    v = np.asarray(values, dtype=np.float64)
    if not (np.isfinite(v).all() and (v > 0).all()):
        raise ValueError(f"{name}: not strictly positive and finite")
    rise = np.flatnonzero(np.diff(v) > 0)
    if rise.size:
        raise ValueError(f"{name}: slot order is not non-increasing at {rise[:8].tolist()}; "
                         "the solved step left the ordered cone and the band reading of "
                         "this table is meaningless.  Re-solve with the ordering active.")


def spec_of(table):
    return dict(operator="static",
                table=dict(values_float32=[float(x) for x in table["values_float32"]],
                           gain=float(table["gain"])))


def emit_evaluate(root, queue_id, name, table, panel="full", nll_docs=16,
                  nll_lengths=(8192, 16384, 32768)):
    """The registration job: installs the table, runs the RULER panel, and scores
    NLL at the lengths the ARCHIVED MrPro rows were scored at.

    `worker.evaluate` writes `results/<name>/contract.json` before it does any
    work, and `module long_eval` reads that file to recover the spec.  So this job
    is not optional even when only the 64K/128K numbers are wanted: it is what
    registers the table under a name the later job can refer to.
    """
    root = Path(root)
    job = dict(id=name, spec=spec_of(table), panel=panel,
               nll_docs=nll_docs, nll_lengths=list(nll_lengths))
    path = root / "queue" / f"{queue_id}_{name}.json"
    if path.exists() and json.loads(path.read_text()) != job:
        raise ValueError(f"cannot replace an existing job contract at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2) + "\n")
    return path, job


def emit_long(root, queue_id, name, docs_per_dataset=2, lengths=(65536, 131072)):
    """The real extrapolation measurement: `module long_eval` on the held-out
    long-document corpus, paired against MrPro in the same job.

    `nll_lengths` in the evaluate-shaped job cannot reach these lengths -- that
    path reads `prepared_nll_01`, whose documents are 32769 tokens, so a request
    for 131072 would silently truncate to the document length and score an empty
    tail.  The long corpus is `long_inputs/` (131073 tokens/document, one
    contiguous real document prefix, no concatenated filler) and it is read by
    this module only.

    MrPro is passed in `methods` alongside the candidate so the two arms are
    measured in one model load, and `long_eval` reuses the cached MrPro rows
    rather than recomputing them when they already exist.
    """
    root = Path(root)
    job = dict(id=f"{name}_long", action="module", module="long_eval",
               methods=["MrPro", name],
               docs_per_dataset=docs_per_dataset, lengths=list(lengths))
    path = root / "queue" / f"{queue_id}_{name}_long.json"
    if path.exists() and json.loads(path.read_text()) != job:
        raise ValueError(f"cannot replace an existing job contract at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2) + "\n")
    return path, job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kkt", required=True, help="solve_kkt.py receipt for the candidate")
    ap.add_argument("--root", required=True, help="harness root (queue/ lives here)")
    ap.add_argument("--tables", required=True, help="harness prepared_*/tables.json")
    ap.add_argument("--queue-id", required=True, help="e.g. 0450; prefix in the queue filename")
    ap.add_argument("--name", default=None, help="job name; default KKT_<base-table>")
    ap.add_argument("--receipts", default=None, help="directory for the construction receipt")
    ap.add_argument("--gt", default=None,
                    help="ground-truth tables json; when given, the reconstruction is "
                         "checked against the DEPLOYED tables before any job is written")
    ap.add_argument("--panel", default="full", choices=["small", "full"])
    ap.add_argument("--nll-docs", type=int, default=16)
    ap.add_argument("--nll-lengths", default="8192,16384,32768",
                    help="must be the lengths the archived MrPro rows use, or the "
                         "candidate has nothing to be compared against")
    ap.add_argument("--long-lengths", default="65536,131072",
                    help="lengths for the `module long_eval` job, read from long_inputs/")
    ap.add_argument("--docs-per-dataset", type=int, default=2,
                    help="2 matches the archived MrPro long baseline (n=2 per cell); "
                         "changing it makes the comparison unpaired")
    ap.add_argument("--skip-long", action="store_true",
                    help="emit only the registration job (cheaper; no 64K/128K measurement)")
    ap.add_argument("--also-controls", action="store_true",
                    help="emit budget-matched control tables as separate jobs (needs the "
                         "forward_check receipt, not the KKT one)")
    args = ap.parse_args()

    # gate: the algebra must reproduce what is actually on the GPU
    if args.gt:
        diffs = T.verify(args.gt)
        print(json.dumps(dict(reconstruction_max_rel_diff=diffs), indent=1))

    tables = load_tables(args.tables)
    _, gain = reference_table(tables)
    if abs(gain - T.GAIN_YARN) > 1e-12:
        print(f"NOTE: harness MrPro gain is {gain!r}, not YaRN's {T.GAIN_YARN!r}; "
              f"using the harness value so the candidate is a freq-only change")

    rec = json.load(open(args.kkt))
    base_name = rec["base_table"]
    table, rec = candidate_from_receipt(args.kkt, base_name, gain)
    check_monotone(table["values_float32"], "candidate")

    name = args.name or f"KKT_{base_name}"
    lengths = [int(x) for x in args.nll_lengths.split(",") if x]
    path, job = emit_evaluate(args.root, args.queue_id, name, table, panel=args.panel,
                              nll_docs=args.nll_docs, nll_lengths=lengths)

    long_lengths = [int(x) for x in args.long_lengths.split(",") if x]
    long_path, long_job = None, None
    if not args.skip_long:
        if args.queue_id.endswith(("_", ".")):
            raise ValueError("run the long job in the same batch: a queue id ending in "
                             "'_' or '.' would sort the long job BEFORE the registration "
                             "job, and long_eval reads results/<name>/contract.json which "
                             "only the registration job writes")
        long_path, long_job = emit_long(args.root, args.queue_id, name,
                                        docs_per_dataset=args.docs_per_dataset,
                                        lengths=long_lengths)
        if not (str(path) < str(long_path)):
            raise ValueError(f"queue order is wrong: {path.name} must be processed before "
                             f"{long_path.name}; the worker takes sorted(glob)[0]")

    base_nu = np.asarray(T.build(base_name, gain=gain)["values_float32"], dtype=np.float64)
    d_m = np.asarray(rec["m_new"], dtype=np.float64) - np.asarray(rec["m"], dtype=np.float64)
    changed = np.flatnonzero(np.abs(d_m) > 1e-4)

    receipt = dict(
        name=name, job_path=str(path), job=job, base_table=base_name, gain=gain,
        long_job_path=str(long_path) if long_path else None, long_job=long_job,
        kkt_receipt=args.kkt, eps=rec["eps"], G=rec["G"],
        predicted_long_gain=rec["pred_long_gain"],
        predicted_native_kl=rec["pred_native_kl"],
        m=rec["m_new"], delta_m=d_m.tolist(), changed_slots_zero_based=changed.tolist(),
        sum_m_before=float(np.sum(rec["m"])), sum_m_after=float(np.sum(rec["m_new"])),
        per_slot_delta_ln_nu=(np.log(base_nu) - np.log(np.asarray(table["values_float32"],
                                                                 dtype=np.float64))).tolist(),
        monotone=rec.get("monotone"),
        max_abs_rel_freq_change=float(np.abs(np.asarray(table["values_float32"], dtype=np.float64)
                                             / base_nu - 1.0).max()),
        archived_mrpro_long_baseline={"pg19/131072": 2.3077492713928223,
                                      "proofpile/131072": 1.1030299365520477,
                                      "pg19/65536": 2.569820284843445,
                                      "proofpile/65536": 0.6221646964550018,
                                      "source": "done/022_long_nll_baseline.json, n=2 per cell"},
        scope=("One solved whole-table allocation on the frozen Qwen2.5-3B-Instruct MrPro "
               "development panel, gain held at the harness's MrPro value. The step is the "
               "closed-form trust-region solution of the stated objective under a "
               "pre-registered native output-KL budget; its predicted gain is checked "
               "against real forwards in forward_check.py before this job is read. "
               "Historical development inputs, not independent confirmation."),
        comparisons=[f"{base_name} @ same gain", "archived MrPro rows in the same harness"],
    )
    if args.receipts:
        out = Path(args.receipts)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(name=name, queue_path=str(path),
                          long_queue_path=str(long_path) if long_path else None,
                          gain=gain, changed_slots=changed.tolist(),
                          sum_m_after=receipt["sum_m_after"],
                          max_rel_freq_change=receipt["max_abs_rel_freq_change"]), indent=1))


if __name__ == "__main__":
    main()
