"""The long-range side, and the one arithmetic fact that limits what it can say.

WHAT THIS FILE IS.  The constraint side of the solve is the native metric; the
objective side is long-range task risk.  This file turns the prepared RULER rows
into the (θ-independent) skeleton that objective needs: for each row, the token
sequence teacher-forced through prompt + gold reference, the positions that count,
and the group each row belongs to.  It measures nothing about θ -- `bound.py`
supplies b_e(θ) and this file says which rows, with what weight, enter the mean.

THE FACT THAT LIMITS IT, STATED FIRST BECAUSE IT IS EASY TO BURY.
    n_rows = 30,  d_design = 65.

There are 30 rows and 65 design variables.  Every long-range objective this panel
can express is a function of at most 30 linear functionals of θ, because there
are only 30 functions f_e and each contributes one gradient direction:

    L(θ) = mean_e f_e(θ)     =>     ∇L lives in span{∇f_e} ⊆ R^65, dim <= 30.

So the panel constrains θ along at most 30 of its 65 dimensions.  The other >= 35
are null directions of this objective: the solver may move along them at zero
measured long-range cost, and the panel cannot tell whether that helps or hurts.
That is not a caveat on the result -- it is a statement about which results are
results.  Two consequences are enforced here rather than left to discipline:

  1. `span_guard` computes the numerical rank of the row-gradient matrix and the
     dimension of its orthogonal complement, and `loop.py` is required to report
     the null-space component of every accepted step.  A step whose norm is
     mostly null-space is not evidence of a long-range gain even if the objective
     moved, because the movement is inside the panel's blind spot.

  2. The objective enters the quadratic model as GROUP MEANS, never as per-row
     terms.  3 tasks x 2 lengths = 6 groups, so the model carries 6 numbers where
     a per-row fit would carry 30.  `guard` raises if a caller tries to hand the
     solver a per-row weighted objective, and says why.

WHY THIS IS NOT THE FAILED 18-SAMPLE / 64-DOF ROUTE.  V-E2 vetoed "18 samples /
64 degrees of freedom margin-gradient" as a capability-optimisation route -- a
free 64-dim search fitted to 18 behavioural rows.  This runs 30 rows against 65
dof, which is the SAME n < d regime, and the honest response is not to point at
the task count.  The structural differences, each of which is checkable in the
receipts rather than asserted:

  * the step is NOT free.  It is the analytic trust-region step of sec.8 against
    the native output-KL metric, accepted only if the real forward honours the
    model's own prediction (`accept.py`).  The 18-sample route had no constraint
    side, which is why it could fit anything.
  * the objective is a 6-group mean, not 30 free per-row weights, so the effective
    flexibility is 6 and not 30 -- and 6 < 30 < 65.
  * `loo_folds` is mandatory (NEXT_DERIVATION sec.5.3: leave-one-out is the
    minimum), and a group mean that does not survive LOO is reported as not
    surviving rather than as a small effect.
  * every receipt from this panel is stamped `development_only`, and the RULER
    scope here is a 3-task development subset, not the 13-task macro.

None of that makes n=30 into n>65.  It makes the null space visible instead of
silent, which is the only thing available at this sample size.

ROW SCHEMA.  Read from `ruler_prepared_01/rows.jsonl`; every key is validated, so
a schema drift names the missing field instead of producing a shorter panel:

    row_id, task, length_cap, ids (prompt token ids), references (gold strings),
    input_tokens, budget (max_new_tokens), prompt_sha256, upstream_index

`references` is a LIST and is matched by substring against the decoded output
(`ruler_bench.score`).  A teacher-forced bound needs ONE trajectory, so one
reference is chosen and the choice is recorded; rows with several are flagged
rather than silently averaged.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REQUIRED_KEYS = ("row_id", "task", "length_cap", "ids", "references",
                 "input_tokens", "budget", "prompt_sha256")

# The three tasks the prepared subset carries.  Named so a 13-task macro corpus
# cannot be passed in silently and then reported as if it were the same panel:
# the group count changes, and so does what the group mean means.
PREPARED_TASKS = ("niah_single_1", "niah_multikey_3", "vt")

# RULER's own probe strings end immediately before the answer, and the answer is
# matched as a substring of a lowercased decode.  A single space is the separator
# that makes `tokenize(prefix + ref)` the correct continuation for that convention.
DEFAULT_PREFIX = " "

# Per-row forward cost at bf16 on the 3B checkpoint, measured (see RUNBOOK):
# ~4.1 s at 32K and ~33.9 s at 128K, both with logits_to_keep.  Used only to print
# a budget before a pass is started, never to decide anything.
SECONDS_PER_ROW = {4096: 4.1, 8192: 8.0, 32768: 4.1, 65536: 16.0, 131072: 33.9}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_rows(path):
    """Read rows.jsonl with the schema checked per row.

    A missing key is a hard error naming the field and the row id: the failure
    mode this prevents is a corpus that loads, runs, and reports a smaller panel
    than intended without anything noticing.
    """
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in rec]
            if missing:
                raise ValueError(f"{path}:{lineno} missing {missing}")
            if not rec["references"]:
                raise ValueError(f"{path}:{lineno} ({rec.get('row_id')}) has no reference answers")
            if not rec["ids"]:
                raise ValueError(f"{path}:{lineno} ({rec.get('row_id')}) has no prompt ids")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path} has no rows")
    ids = [r["row_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate row_id(s): {dup}")
    return rows


def group_key(row):
    """(length_cap, task).  Length is part of the key because the whole question
    is whether the table helps at the LONG end; pooling the two lengths would
    average the effect this package exists to measure."""
    return (int(row["length_cap"]), str(row["task"]))


def build_groups(rows):
    """-> {group_key: [row, ...]}, plus the axis sizes.

    Raises if the corpus is not a full task x length grid.  An incomplete grid
    means some group mean has a different meaning from its neighbours, and the
    weighted objective would then be comparing quantities that are not the same
    kind of thing.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    lengths = sorted({k[0] for k in groups})
    tasks = sorted({k[1] for k in groups})
    missing = [(L, t) for L in lengths for t in tasks if (L, t) not in groups]
    if missing:
        raise ValueError(f"incomplete task x length grid, empty cells: {missing}")
    return dict(groups=dict(groups), lengths=lengths, tasks=tasks,
                n_groups=len(groups), n_rows=len(rows))


# ---------------------------------------------------------------------------
# reference assembly -- the teacher-forced trajectory sec.3's induction needs
# ---------------------------------------------------------------------------
def reference_ids(tokenizer, references, prefix=DEFAULT_PREFIX, index=0,
                  eos_id=None, include_eos=True):
    """Token ids for ONE gold reference, with the alternatives reported.

    Multi-reference rows (niah_multikey_3 accepts several orderings) are the
    place this can go quietly wrong: the row is scored against all of them, but
    the bound's induction needs one concrete reference prefix.  The chosen index
    is returned alongside every variant's length so a receipt shows what was
    teacher-forced instead of leaving it implied.

    `include_eos` appends the terminator.  RULER's generation stops there, and a
    model that never emits EOS has not produced the reference -- so the final
    target position is part of the event `1{greedy != reference}` and belongs in
    the max unless the caller masks it out (`mask` in bound.py).
    """
    variants = []
    for i, ref in enumerate(references):
        piece = f"{prefix}{ref}" if prefix else str(ref)
        ids = list(tokenizer(piece, add_special_tokens=False)["input_ids"])
        variants.append(dict(index=i, text=str(ref), text_len=len(str(ref)),
                             n_tokens=len(ids), ids=ids))
    if not 0 <= index < len(variants):
        raise IndexError(f"reference index {index} out of range for {len(variants)} references")
    chosen = list(variants[index]["ids"])
    if include_eos:
        if eos_id is None:
            raise ValueError("include_eos=True needs eos_id")
        chosen = chosen + [int(eos_id)]
    return dict(ids=chosen, n_tokens=len(chosen), index=int(index),
                n_references=len(variants),
                variants=[{k: v[k] for k in ("index", "text_len", "n_tokens")}
                          for v in variants],
                multi_reference=len(variants) > 1)


def teacher_forced_record(frozen, row, prefix=DEFAULT_PREFIX, index=0,
                          include_eos=True, reserve=None):
    """Everything the risk objective needs about one row, without any forward.

    Returns the concatenated ids, the number of trailing positions that are
    scored, and the bookkeeping that makes the assembly auditable:

      * `over_cap` -- prompt + reference tokens beyond `length_cap`.  RULER caps
        the INPUT; the gold answer is generated on top of it, so a correct
        teacher-forced sequence legitimately exceeds the cap by the answer
        length.  Reported so an unexpectedly large overshoot is visible.
      * `n_answer` -- positions under the max in `b_e`.
    """
    tok = frozen.tokenizer
    eos = getattr(tok, "eos_token_id", None)
    eos = getattr(frozen.model.config, "eos_token_id", eos) if eos is None else eos
    ref = reference_ids(tok, row["references"], prefix=prefix, index=index,
                        eos_id=eos, include_eos=include_eos)
    prompt = list(row["ids"])
    if reserve is not None and reserve > 0:
        # Only used when a caller wants the whole sequence to stay inside the cap.
        # It TRUNCATES the prompt, which changes the task, so the caller has to
        # ask for it explicitly and the truncation is reported.
        keep = max(len(prompt) - int(reserve), 1)
        prompt = prompt[:keep]
    ids = np.asarray(prompt + ref["ids"], dtype=np.int64)
    cap = int(row["length_cap"])
    return dict(row_id=row["row_id"], task=row["task"], length_cap=cap,
                group=group_key(row), ids=ids, n_tokens=int(ids.size),
                n_prompt=len(prompt), n_answer=ref["n_tokens"],
                prompt_sha256=row["prompt_sha256"],
                budget=int(row["budget"]), input_tokens=int(row["input_tokens"]),
                over_cap=int(max(0, ids.size - cap)),
                reference=ref, truncated=(reserve is not None and reserve > 0))


def answer_mask(record):
    """Which of the trailing positions are scored: all of them, including EOS.

    Returned as a function rather than a constant so the "did you mask EOS?"
    question has one answer in one place.  `mask_positions` below is how a caller
    opts out.
    """
    n = int(record["n_answer"])
    m = np.ones(n, dtype=bool)
    return m


def mask_positions(record, drop_eos=True, keep_first=None):
    """The scored-position mask, with the two opt-outs named.

    Dropping EOS drops the position whose target is the terminator -- the one a
    model most often gets wrong at length, and therefore the one most likely to
    make b_e positive.  That is a real reason to look at both, so it is a switch
    and not a default hidden in the caller.
    """
    n = int(record["n_answer"])
    m = np.ones(n, dtype=bool)
    if drop_eos and n:
        m[-1] = False
    if keep_first is not None:
        m[int(keep_first):] = False
    return m


# ---------------------------------------------------------------------------
# the guard: how many directions this panel can actually see
# ---------------------------------------------------------------------------
def span_guard(row_grads):
    """dim span{∇f_e} and the dimension the panel is blind to.

    `row_grads` is (n_rows, d_design) -- one gradient direction per row, from
    whichever objective is in use (the sec.3 bound's gradient, or a native-NLL
    fallback).  The rank is measured with a relative tolerance so a numerically
    zero singular value is not counted as a direction; the tolerance is reported
    so a rank that sits near it is visible as the judgement call it is.

    Returns a projector onto the row span.  `loop.py` reports ||(I-P)d|| / ||d||
    for every accepted step: a large value means most of the step is a direction
    no row in this panel can price, and any claimed long-range gain from that part
    of the step is unfalsifiable here.
    """
    G = np.atleast_2d(np.asarray(row_grads, dtype=np.float64))
    if G.ndim != 2:
        raise ValueError("row_grads must be (n_rows, d_design)")
    n_rows, d = G.shape
    norms = np.linalg.norm(G, axis=1)
    live = norms > 0
    if not live.any():
        return dict(rank=0, n_rows=n_rows, d_design=d, null_dim=d,
                    tol=None, live_rows=0, basis=None, live_mask=live,
                    note="every row gradient is zero")
    _, s, vt = np.linalg.svd(G[live], full_matrices=False)
    tol = float(s[0]) * max(G.shape) * np.finfo(np.float64).eps
    rank = int((s > tol).sum())
    # G = U S V^T, so the row space is spanned by the leading RIGHT singular
    # vectors.  They are kept as `basis` (rank, d) rather than materialised into a
    # (d, d) projector: the only question anyone asks of it is the null-space
    # FRACTION of a step, and that is a coefficient projection -- cheaper, and it
    # avoids a large matmul whose float64 result is exact anyway.
    basis = vt[:rank].copy()
    return dict(rank=rank, n_rows=n_rows, d_design=d, null_dim=d - rank,
                tol=tol, live_rows=int(live.sum()),
                singular_values=[float(x) for x in s],
                basis=basis,
                note=("the panel sees at most this many directions; the rest of "
                      "the design space is unconstrained by it"))


def projector(guard):
    """The (d, d) orthogonal projector onto the row span, for callers that want
    to move INSIDE the spanned subspace rather than just measure the leak.

    `einsum`, not `@`: numpy 2.0.2 on macOS dispatches this shape to Accelerate,
    which leaves the FP status flags set and makes a correct result emit
    "divide by zero encountered in matmul".  A spurious warning here is worse
    than a slow one -- it hides the real numerical warnings this package relies
    on seeing.
    """
    B = guard.get("basis")
    if B is None:
        return None
    return np.einsum("ri,rj->ij", B, B)


def null_fraction(step, guard):
    """||(I - P) d|| / ||d|| for one step, given a `span_guard` result.

    Zero means the step lies entirely in directions this panel can price.  One
    means it lies entirely in the blind spot -- the objective it was derived from
    cannot distinguish that step from no step, whatever the step's norm.
    """
    d = np.asarray(step, dtype=np.float64).reshape(-1)
    B = guard.get("basis")
    if B is None:
        return float("nan")
    if B.shape[1] != d.size:
        raise ValueError(f"guard was built for d={B.shape[1]}, step has d={d.size}")
    n2 = float(np.dot(d, d))
    if n2 == 0.0:
        return 0.0
    # The null component is subtracted as a VECTOR rather than as a difference of
    # squared norms.  ||d||^2 - ||Bd||^2 is algebraically the same number and
    # numerically much worse: for a step that lies entirely in the span the two
    # norms agree to ~1e-16 relative, so their difference has absolute error
    # ~1e-16*||d||^2 and the reported fraction floors at ~1e-8 instead of 0.
    # Subtracting the projected vector componentwise keeps the floor at 1e-16,
    # which matters because this number is compared against small thresholds.
    # einsum, not `@`, for the reason given in `projector`.
    c = np.einsum("ri,i->r", B, d)
    resid = d - np.einsum("ri,r->i", B, c)
    return float(np.sqrt(max(float(np.dot(resid, resid)), 0.0) / n2))


def guard(n_rows, d_design, n_groups, tasks=None, lengths=None):
    """The structural check that has to pass before a panel becomes an objective.

    Raises on the two configurations that would reproduce the vetoed route under
    a new name:

      * a PER-ROW objective (n_groups == n_rows): 30 free weights fitted against
        a 65-dim design is the 18-sample/64-dof regime with a bigger n and the
        same defect -- nothing constrains the extra directions.
      * more design variables than groups by a wide margin, when the caller has
        not acknowledged the null space.

    Everything it can check is returned as a receipt, including the V-E2 ratio,
    so the acknowledgement is a field in a file rather than a sentence in a
    commit message.
    """
    out = dict(n_rows=int(n_rows), d_design=int(d_design), n_groups=int(n_groups),
               development_only=True)
    if n_rows <= 0 or d_design <= 0:
        raise ValueError("empty panel or empty design")
    if int(n_groups) >= int(n_rows):
        raise ValueError(
            f"per-row objective: {n_groups} groups for {n_rows} rows. The "
            "objective must be a group mean (3 tasks x 2 lengths = 6), not one "
            "free weight per row -- a free per-row fit against a "
            f"{d_design}-dimensional design is the vetoed 18-sample/64-dof route.")
    if int(n_groups) >= int(d_design):
        raise ValueError(
            f"{n_groups} groups vs {d_design} design variables: the group means "
            "alone would over-determine the design. Refusing rather than "
            "relying on the trust region to rescue it.")
    out["groups_per_design"] = int(n_groups) / int(d_design)
    out["rows_per_design"] = int(n_rows) / int(d_design)
    out["n_lt_d"] = bool(int(n_rows) < int(d_design))
    out["vex2_ratio"] = out["rows_per_design"]
    out["vex2_note"] = (
        "30 rows / 65 design variables is the same n<d regime as the vetoed "
        "18-sample/64-dof route. This panel is admissible because the step is "
        "constrained (native output-KL trust region + sec.8 acceptance) and the "
        "objective is a 6-group mean, not 30 free weights -- not because n grew.")
    if tasks is not None:
        out["tasks"] = list(tasks)
    if lengths is not None:
        out["lengths"] = [int(x) for x in lengths]
    return out


def group_weights(groups, scheme="uniform", long_length=None):
    """w_g for L = sum_g w_g * mean_{e in g} f_e.

    `uniform` weights every cell equally, which is the pre-registered default:
    it is the only scheme that needs no data to choose, and the panel's own
    `summarize` reports an unweighted task macro.

    `length_first` puts half the mass on the long cells and half on the short
    ones, so a table that wins at length by losing in-window is not scored as a
    win by a majority vote of cells.  It is off by default and named in the
    receipt when on.
    """
    keys = sorted(groups["groups"])
    if scheme == "uniform":
        return {k: 1.0 / len(keys) for k in keys}
    if scheme == "length_first":
        lengths = sorted({k[0] for k in keys})
        if len(lengths) != 2:
            raise ValueError(f"length_first expects 2 lengths, got {lengths}")
        long_L = int(long_length if long_length is not None else lengths[-1])
        if long_L not in lengths:
            raise ValueError(f"long_length {long_L} not in {lengths}")
        # Half the mass per length, split evenly inside a length.  `long_length`
        # only selects which end is named; both ends get 0.5 by construction.
        return {k: 0.5 / len([x for x in keys if x[0] == k[0]]) for k in keys}
    raise ValueError(f"unknown scheme {scheme!r}")


def loo_folds(groups, order=None):
    """Leave-one-out over the ROWS, then regroup.

    NEXT_DERIVATION sec.5.3 makes leave-one-out the minimum standard for a panel
    this size, and the reason applies here unchanged: at n=30 a group mean can
    move by a whole cell's worth when one row leaves.  Held-out rows are dropped
    BEFORE the group means are formed, not after -- otherwise the held row is
    still inside the mean it is supposed to be held out of.
    """
    keys = sorted(groups["groups"])
    rows = [(k, r) for k in keys for r in groups["groups"][k]]
    n = len(rows)
    folds = []
    for i in range(n):
        held_key, held_row = rows[i]
        train = [r for j, (k, r) in enumerate(rows) if j != i]
        held = [r for (k, r) in rows if k == held_key and r["row_id"] != held_row["row_id"]]
        folds.append(dict(i=i, held=(held_key, held_row["row_id"]),
                          n_train=len(train), n_train_groups=len({group_key(r) for r in train}),
                          n_held_in_group=len(held),
                          singleton_cell=len(held) == 0))
    return folds


def budget_seconds(rows, per_row=None):
    """Print-before-you-spend estimate for a full teacher-forced pass."""
    per_row = per_row or SECONDS_PER_ROW
    total, detail = 0.0, {}
    for r in rows:
        cap = int(r["length_cap"])
        # nearest known length, since only the cap determines the forward cost
        key = min(per_row, key=lambda c: abs(c - cap))
        detail.setdefault(str(cap), dict(n=0, seconds=0.0))["n"] += 1
        detail[str(cap)]["seconds"] += per_row[key]
        total += per_row[key]
    return dict(total_seconds=total, total_minutes=total / 60.0, by_length=detail,
                note="prefill-only (no generation): one forward per row, "
                     "logits_to_keep = answer length + 1")


# ---------------------------------------------------------------------------
def panel_receipt(rows, groups, weights, guard_rec, est=None):
    """Everything a reader needs to know what the long-range number is a number
    ABOUT, assembled in one place so it cannot be split across a receipt and a
    commit message."""
    return dict(
        n_rows=len(rows), n_groups=groups["n_groups"],
        lengths=groups["lengths"], tasks=groups["tasks"],
        cell_sizes={f"{L}/{t}": len(groups["groups"][(L, t)])
                    for (L, t) in sorted(groups["groups"])},
        weights={f"{k[0]}/{k[1]}": float(v) for k, v in sorted(weights.items())},
        guard=guard_rec, estimate=est,
        scope=("3-task RULER development subset (niah_single_1, niah_multikey_3, vt); "
               "not the 13-task RULER macro"),
        task_set_expected=list(PREPARED_TASKS),
        task_set_matches=bool(set(groups["tasks"]) == set(PREPARED_TASKS)),
        multi_reference_rows=[r["row_id"] for r in rows if len(r["references"]) > 1],
        evidence_scope="Historical development inputs; not independent confirmation",
    )


def verify_against_frozen(path, tasks=PREPARED_TASKS):
    """Cheap pre-flight on the rows file alone: counts, grid, cap consistency.

    Runs before the model is loaded so a corpus problem costs nothing.  Deliberately
    does NOT check the prompt token count against `input_tokens` -- that needs the
    tokenizer, and the tokenizer needs the checkpoint, and this is the check that
    is supposed to happen first.
    """
    rows = load_rows(path)
    groups = build_groups(rows)
    out = dict(path=str(path), n_rows=len(rows), n_groups=groups["n_groups"],
               lengths=groups["lengths"], tasks=groups["tasks"],
               cell_sizes={f"{k[0]}/{k[1]}": len(v) for k, v in sorted(groups["groups"].items())},
               tasks_unexpected=[t for t in groups["tasks"] if t not in tasks],
               caps=[int(r["length_cap"]) for r in rows],
               max_budget=max(int(r["budget"]) for r in rows),
               n_multi_reference=sum(len(r["references"]) > 1 for r in rows),
               estimate=budget_seconds(rows))
    out["ok"] = not out["tasks_unexpected"] and len(groups["lengths"]) == 2
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="pre-flight the prepared RULER rows")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--d-design", type=int, default=65)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rec = verify_against_frozen(args.rows)
    rec["guard"] = guard(rec["n_rows"], args.d_design, rec["n_groups"],
                         tasks=rec["tasks"], lengths=rec["lengths"])
    print(json.dumps({k: v for k, v in rec.items() if k != "estimate"}, indent=1))
    print(f"\ntotal forward estimate: {rec['estimate']['total_minutes']:.1f} min "
          f"({rec['estimate']['by_length']})")
    if not rec["ok"]:
        print("STATUS: FAIL -- corpus is not the expected grid")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
