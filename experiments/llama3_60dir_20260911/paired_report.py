"""Paired macro accuracy, source-cluster bootstrap, and power arithmetic.

Plan sections 5.4, 6.3, 6.6.  Kept separate from upgrade_report.py so that the
measurement layer can be tested on fixtures that never touch a model.

Three disciplines are enforced here rather than left to the caller:

1. **Alignment by identity.**  Every arm is joined to the frozen expected-data
   index on `row_id`, and a mismatch is an error, not a warning.  An arm that
   silently scored a subset would otherwise report the accuracy of whichever
   subset it happened to finish.

2. **Equal weights are explicit.**  The main statistic weights tasks equally and
   lengths equally (plan section 5.4).  `macro_accuracy` says so in its return
   value so a reader never has to infer it.

3. **Paired, not independent.**  Differences are computed per row and then
   aggregated.  The standard error is the discordance form
   SE = sqrt((q - delta^2)/N) with q the fraction of discordant pairs, which is
   the right one for two binary scores on the same rows (plan section 6.6).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# plan section 6.6
Z_ALPHA_2 = 1.959963984540054   # two-sided 5%
Z_POWER_80 = 0.8416212335729143


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i + 1}: not JSON ({exc})") from exc
    return rows


def _row_id(r):
    for k in ("row_id", "example_id", "id"):
        if k in r:
            return str(r[k])
    raise KeyError(f"row has no id field: {sorted(r)[:8]}")


def _score(r, metric="partial"):
    """Read one named scoring contract without cross-metric leakage.

    New runner rows carry dedicated fields.  The final fallback to ``correct``
    is only for pre-contract fixture files; a dedicated field always wins.
    """
    fields = {
        "partial": ("partial_score", "correct", "score"),
        "partial_score": ("partial_score", "correct", "score"),
        "strict": ("strict_score", "strict", "correct"),
        "strict_score": ("strict_score", "strict", "correct"),
        "qa_em": ("qa_em", "correct"),
        "qa_f1": ("qa_f1", "correct"),
    }
    if metric not in fields:
        raise ValueError(f"unknown score metric {metric!r}")
    for k in fields[metric]:
        if k in r:
            value = float(r[k])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"score must be finite and in [0,1], got {value!r}")
            return value
    raise KeyError(f"row has no score field: {sorted(r)[:8]}")


def _task(r):
    return str(r.get("task", "?"))


def _length(r):
    for k in ("length_cap", "cap", "length"):
        if k in r:
            return int(r[k])
    return -1


def _cluster(r):
    """The resampling unit: the source document, not the row (plan section 6.6)."""
    for k in ("source_id", "group_id", "doc_id"):
        if k in r:
            return str(r[k])
    return _row_id(r)


def index_arm(rows, name="arm"):
    out = {}
    for r in rows:
        rid = _row_id(r)
        if rid in out:
            raise ValueError(f"{name}: duplicate row_id {rid}")
        out[rid] = r
    return out


def _identity(r):
    """Return the method-independent part of a panel row.

    The campaign has used a few names for the same fields over time.  Normalize
    those aliases here, while treating a field present in only one arm as an
    identity mismatch rather than silently ignoring it.
    """
    def optional(*keys):
        present = [r[k] for k in keys if k in r]
        return (False, None) if not present else (True, present[0])

    has_source, source = optional("source_id", "source_document_id", "doc_id")
    has_group, group = optional("group_id", "semantic_group_id")
    has_prompt, prompt = optional("prompt_sha256", "prompt_input_ids_sha256")
    return {
        "task": _task(r), "length": _length(r),
        "source": (str(source) if has_source else None),
        "group": (str(group) if has_group else None),
        "prompt": (str(prompt) if has_prompt else None),
        "actual_length": r.get("actual_length"),
        "evidence_positions": tuple(r.get("evidence_positions", ())),
        "distractor_positions": tuple(r.get("distractor_positions", ())),
        "gold": r.get("gold"), "scorer_revision": r.get("scorer_revision"),
        "has_source": has_source, "has_group": has_group,
        "has_prompt": has_prompt,
    }


def _check_identity(ref, got, ref_name, got_name):
    """Raise on metadata drift between two rows with the same row_id."""
    a, b = _identity(ref), _identity(got)
    problems = []
    for key in ("task", "length", "source", "group", "prompt",
                "actual_length", "evidence_positions", "distractor_positions",
                "gold", "scorer_revision"):
        has_a = a.get("has_" + key, True)
        has_b = b.get("has_" + key, True)
        if key in ("source", "group", "prompt") and has_a != has_b:
            problems.append(f"{key} presence differs")
        elif has_a and has_b and a[key] != b[key]:
            problems.append(f"{key}: {a[key]!r} != {b[key]!r}")
    if problems:
        raise ValueError(f"identity mismatch for row {_row_id(ref)} between "
                         f"{ref_name} and {got_name}: " + "; ".join(problems))


def _required_cells(rows, required_tasks=None, required_lengths=None):
    tasks = sorted(set(required_tasks) if required_tasks is not None
                   else {_task(r) for r in rows})
    lengths = sorted(set(int(x) for x in required_lengths) if required_lengths is not None
                     else {_length(r) for r in rows})
    return tasks, lengths, {(t, l) for t in tasks for l in lengths}


def _require_complete_cells(rows, required_tasks=None, required_lengths=None,
                            label="panel"):
    tasks, lengths, expected = _required_cells(rows, required_tasks, required_lengths)
    got = {(_task(r), _length(r)) for r in rows}
    missing, extra = sorted(expected - got), sorted(got - expected)
    if missing or extra:
        raise ValueError(f"{label}: incomplete task x length cells; "
                         f"missing={missing[:8]} extra={extra[:8]}")
    return tasks, lengths, expected


def require_aligned(arms: dict, expected_rows=None):
    """Every arm must cover exactly the same rows and identity metadata."""
    if not arms:
        raise ValueError("no arms supplied")
    names = list(arms)
    base = names[0]
    for n, arm in arms.items():
        if not isinstance(arm, dict):
            arm = index_arm(arm, n)
            arms[n] = arm
        # Catch duplicate IDs even when callers pass a pre-indexed mapping.
        if len(arm) == 0:
            raise ValueError(f"{n}: empty arm")
    ref = set(arms[base])
    problems = []
    for n in names[1:]:
        s = set(arms[n])
        if s != ref:
            missing = sorted(ref - s)[:5]
            extra = sorted(s - ref)[:5]
            problems.append({"arm": n, "n_missing": len(ref - s), "n_extra": len(s - ref),
                             "missing_examples": missing, "extra_examples": extra})
    if problems:
        raise ValueError("arms are not aligned on row_id: " + json.dumps(problems))
    for n in names[1:]:
        for rid in sorted(ref):
            _check_identity(arms[base][rid], arms[n][rid], base, n)
    if expected_rows is not None:
        expected_index = index_arm(expected_rows, "expected-data")
        exp = set(expected_index)
        if exp != ref:
            raise ValueError(
                f"arms cover {len(ref)} rows but the frozen plan has {len(exp)}; "
                f"missing {len(exp - ref)}, extra {len(ref - exp)}. "
                "A partial panel must never be reported as the full statistic."
            )
        for rid in sorted(ref):
            _check_identity(expected_index[rid], arms[base][rid],
                            "expected-data", base)
    return sorted(ref)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def macro_accuracy(rows, weights=("task", "length"), required_tasks=None,
                   required_lengths=None, score_key="partial"):
    """Equal-task, equal-length macro accuracy (plan section 5.4).

    When requirements are omitted, the observed Cartesian task x length grid is
    required to be complete.  Production callers should pass the frozen lists.
    """
    rows = list(rows)
    index_arm(rows, "scored panel")
    tasks, lengths, _ = _require_complete_cells(
        rows, required_tasks, required_lengths, "scored panel")
    cells = defaultdict(list)
    for r in rows:
        cells[(_task(r), _length(r))].append(_score(r, score_key))
    cell_mean = {k: float(np.mean(v)) for k, v in cells.items()}
    task_means = {t: float(np.mean([cell_mean[(t, l)] for l in lengths if (t, l) in cell_mean]))
                  for t in tasks}
    overall = float(np.mean(list(task_means.values()))) if task_means else float("nan")
    by_length = {}
    for l in lengths:
        vals = [cell_mean[(t, l)] for t in tasks if (t, l) in cell_mean]
        by_length[l] = float(np.mean(vals)) if vals else float("nan")
    return {
        "macro": overall,
        "by_task": task_means,
        "by_length": by_length,
        "n_cells": len(cell_mean),
        "n_rows": len(rows),
        "weights": {"tasks": "equal", "lengths": "equal"},
    }


def paired_stats(rows_base, rows_cand, use_lengths=None, expected_rows=None,
                 required_tasks=None, score_key="partial"):
    """Paired difference cand - base, equally over complete task x length cells."""
    b = index_arm(rows_base, "base")
    c = index_arm(rows_cand, "cand")
    require_aligned({"base": b, "cand": c}, expected_rows)
    use_lengths = None if use_lengths is None else set(int(x) for x in use_lengths)
    expected_subset = ([r for r in expected_rows if _length(r) in use_lengths]
                       if expected_rows is not None and use_lengths is not None
                       else expected_rows)
    selected_rows = [b[rid] for rid in b
                     if use_lengths is None or _length(b[rid]) in use_lengths]
    tasks, lengths, _ = _require_complete_cells(
        selected_rows, required_tasks, None, "paired panel")
    if expected_subset is not None:
        _require_complete_cells(expected_subset, required_tasks, use_lengths,
                                 "expected paired panel")
    ids = sorted(rid for rid in b
                 if use_lengths is None or _length(b[rid]) in use_lengths)
    cells = defaultdict(list)
    for rid in ids:
        rb, rc = b[rid], c[rid]
        if _task(rb) != _task(rc) or _length(rb) != _length(rc):
            raise ValueError(f"{rid}: task/length disagree between arms")
        cells[(_task(rb), _length(rb))].append(
            _score(rc, score_key) - _score(rb, score_key))
    cell_mean = {k: float(np.mean(v)) for k, v in cells.items()}
    tasks = sorted({k[0] for k in cell_mean})
    by_task = {t: float(np.mean([cell_mean[(t, l)] for l in lengths if (t, l) in cell_mean]))
               for t in tasks}
    by_length = {l: float(np.mean([cell_mean[(t, l)] for t in tasks if (t, l) in cell_mean]))
                 for l in lengths}
    delta = float(np.mean(list(by_task.values()))) if by_task else float("nan")
    vals = np.array([_score(c[rid], score_key) - _score(b[rid], score_key)
                     for rid in ids], dtype=np.float64)
    q = float(np.mean(np.abs(vals) > 1e-12)) if vals.size else float("nan")
    se = math.sqrt(max(q - delta ** 2, 0.0) / vals.size) if vals.size else float("nan")
    return {
        "delta_macro": delta,
        "by_task": by_task,
        "by_length": by_length,
        "n_pairs": int(vals.size),
        "discordance_q": q,
        "se_paired": se,
        "t": (delta / se) if se and se > 0 else float("nan"),
        "delta_equal_task_length": delta,
    }


def mde(q=0.2, n=None, per_length_n=None):
    """Plan section 6.6.

    The plan writes SE = sqrt((q - delta^2)/N) and then quotes, for q = .2,
    "S combined N=96 about 4.56pp, per length N=48 about 6.45pp; H combined
    N=256 about 2.80pp, per length N=128 about 3.95pp".

    Those four numbers are the **standard error**, not the 95% minimum
    detectable effect -- sqrt(0.2/96) = 4.56pp confirms it.  Conflating the two
    is a factor of 1.96, so both are returned under names that cannot be mixed
    up, and the plan's own four values are reproduced by `se`.
    """
    out = {"q": q, "note": "plan section 6.6 quotes `se`; `mde_95` is 1.96 x se"}
    if n is not None:
        se = math.sqrt(q / n)
        out["combined"] = {"n": n, "se": se, "mde_95": Z_ALPHA_2 * se}
    if per_length_n is not None:
        se = math.sqrt(q / per_length_n)
        out["per_length"] = {"n": per_length_n, "se": se, "mde_95": Z_ALPHA_2 * se}
    return out


def n_for_power(delta=0.05, q=0.2, alpha=0.05, power=0.80):
    """Independent-pair approximation for detecting `delta` (plan section 6.6)."""
    za = abs(_norm_ppf(alpha / 2.0))
    zb = abs(_norm_ppf(1.0 - power))
    return int(math.ceil(((za + zb) ** 2) * q / (delta ** 2)))


def _norm_ppf(p):
    # Acklam's rational approximation; adequate for power arithmetic.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------------------
# cluster bootstrap
# --------------------------------------------------------------------------


def cluster_bootstrap(arms_by_rows, contrasts, n_boot=4000, seed=20260911,
                      alpha=0.05, expected_rows=None, required_tasks=None,
                      required_lengths=None):
    """Simultaneous lower bounds for several paired contrasts.

    `arms_by_rows` maps arm name -> {row_id: row} (already aligned).
    `contrasts` is a list of {name, terms: [(arm, +1), ...], lengths: [...]}.
    Clusters are source documents, shared across arms so the pairing is kept.

    Resampling whole source documents (not rows) is what makes the interval
    honest when several rows share a passage; and resampling the SAME cluster
    draw for every arm is what keeps the contrasts paired.

    Simultaneity: the returned bounds use the max-|t| critical value over
    replicate draws, which is the bootstrap analogue of a joint confidence
    region and is what plan section 0.2 asks for ("joint 95% lower bound",
    "same source-paired bootstrap with a simultaneity correction").  Empty-cell
    resamples are rejected and counted, per plan section 9.2.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2 for a bootstrap standard error")
    names = list(arms_by_rows)
    require_aligned(arms_by_rows, expected_rows)
    first = arms_by_rows[names[0]]
    if required_lengths is None:
        ids = sorted(first)
    else:
        wanted_lengths = set(int(x) for x in required_lengths)
        ids = sorted(rid for rid, r in first.items() if _length(r) in wanted_lengths)
    selected_rows = [first[rid] for rid in ids]
    tasks, lengths, required_cells = _require_complete_cells(
        selected_rows, required_tasks, required_lengths, "bootstrap panel")
    # Every contrast must be defined on the same frozen long grid.  A missing
    # length filter would otherwise make it accidentally include an 8K guard.
    for ct in contrasts:
        if required_lengths is not None:
            ct_lengths = ct.get("lengths")
            if ct_lengths is None or set(int(x) for x in ct_lengths) != set(lengths):
                raise ValueError(f"contrast {ct['name']} is not restricted to the required lengths")
    cluster_of = {rid: _cluster(first[rid]) for rid in ids}
    clusters = sorted(set(cluster_of.values()))
    cid = {c: i for i, c in enumerate(clusters)}
    groups = [[] for _ in clusters]
    for rid in ids:
        groups[cid[cluster_of[rid]]].append(rid)

    def _per_row(ct, rid):
        """The per-row value of a contrast.  Two kinds:
        - linear:   sum of weight * score(arm)
        - max_of:   score(subject) - max over the named comparator arms
        max_of is not a linear functional, but it IS a per-row statistic, so the
        cluster bootstrap applies to it unchanged.  It is the contrast that plan
        section 0.2's verdict actually needs: beating MR while losing to YaRN or
        BM must not be a PASS."""
        metric = ct.get("score_key", "partial")
        if ct.get("kind") == "max_of":
            v = _score(arms_by_rows[ct["subject"]][rid], metric)
            return v - max(_score(arms_by_rows[a][rid], metric)
                           for a in ct["comparators"])
        v = 0.0
        for arm, w in ct["terms"]:
            v += w * _score(arms_by_rows[arm][rid], metric)
        return v

    def contrast_value(sel_ids):
        out = {}
        for ct in contrasts:
            lengths = ct.get("lengths")
            per_task = defaultdict(list)
            for rid in sel_ids:
                first = ct["subject"] if ct.get("kind") == "max_of" else ct["terms"][0][0]
                r0 = arms_by_rows[first][rid]
                if lengths is not None and _length(r0) not in lengths:
                    continue
                per_task[_task(r0)].append(_per_row(ct, rid))
            if not per_task:
                out[ct["name"]] = float("nan")
            else:
                out[ct["name"]] = float(np.mean([np.mean(v) for v in per_task.values()]))
        return out

    point = contrast_value(ids)
    rng = np.random.default_rng(seed)
    n_cl = len(clusters)
    reps = defaultdict(list)
    rejected = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_cl, size=n_cl)
        sel = [rid for p in pick for rid in groups[p]]
        # an empty-cell resample (some task/length cell vanishes) is rejected
        cells = {(_task(first[r]), _length(first[r])) for r in sel}
        if cells != required_cells:
            rejected += 1
            continue
        val = contrast_value(sel)
        for k, v in val.items():
            if math.isfinite(v):
                reps[k].append(v - point[k])

    names_ct = [ct["name"] for ct in contrasts]
    crit = float("nan")
    if all(reps[k] for k in names_ct):
        stacked = np.vstack([np.array(reps[k]) for k in names_ct])
        sd = np.array([np.std(reps[k], ddof=1) for k in names_ct])
        sd[sd == 0] = np.inf
        tmax = np.max(np.abs(stacked / sd[:, None]), axis=0)
        crit = float(np.quantile(tmax, 1.0 - alpha))
    else:
        crit = Z_ALPHA_2

    out = {}
    for k in names_ct:
        if reps[k]:
            sd = float(np.std(reps[k], ddof=1))
            out[k] = {
                "point": point[k],
                "se": sd,
                "lower_simultaneous_95": point[k] - crit * sd,
                "upper_simultaneous_95": point[k] + crit * sd,
                "n_replicates": len(reps[k]),
            }
        else:
            out[k] = {"point": point[k], "se": float("nan"),
                      "lower_simultaneous_95": float("nan"),
                      "upper_simultaneous_95": float("nan"), "n_replicates": 0}
    return {
        "contrasts": out,
        "critical_value": crit,
        "n_clusters": n_cl,
        "n_boot_requested": n_boot,
        "n_boot_rejected_empty_cell": rejected,
        "rejection_fraction": rejected / n_boot if n_boot else float("nan"),
        "alpha": alpha,
        "resampling_unit": "source document",
        "simultaneity": "max-|t| bootstrap critical value over all requested contrasts",
    }
