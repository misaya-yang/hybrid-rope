"""The eight-task panel and the P/S/V/H partition (sections 6.2 - 6.4).

Section 6.2 fixes the task mix, and it is not arbitrary: the ledger records that
a development win on one task family does not generalise, and that
`niah_single_3` was an artefact carrier (65% of one +11.83pp panel win came from
it).  So the eight tasks below avoid it, and each row carries the identity
fields section 6.4 requires.

Section 6.4's rules, enforced here rather than documented:

* every row carries `example_id, semantic_group_id, source_document_id,
  prompt_input_ids_sha256, task, cap, actual_length, evidence_positions, gold,
  scorer_revision`, and THESE ARE IDENTICAL ACROSS ALL METHODS;
* `input_tokens + generated_tokens <= cap`, prompts reach at most cap-128;
* padding only ever ADDS irrelevant filler -- evidence, question and template
  are never truncated;
* the same question at two lengths is the same source, not a new sample;
* the four answers of a multivalue row are one observation, not four.

The partition is by SOURCE, never by row.  The ledger records that a held-out
set of 180 rows turned out to hold only 120 unique prompts, and that "two arms
of 96 rows" were 96 paired units rather than 192 independent samples; both
mistakes come from counting rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

# section 6.2
TASKS = {
    "niah_single_2": {"family": "retrieval", "max_new_tokens": 128},
    "niah_multikey_2": {"family": "retrieval", "max_new_tokens": 128},
    "niah_multivalue": {"family": "aggregation", "max_new_tokens": 128},
    "niah_multiquery": {"family": "retrieval", "max_new_tokens": 128},
    "vt": {"family": "tracking", "max_new_tokens": 30},
    "fwe": {"family": "aggregation", "max_new_tokens": 50},
    "qa_1": {"family": "qa", "max_new_tokens": 32},
    "qa_2": {"family": "qa", "max_new_tokens": 32},
}
# section 6.2: this one is NOT a main metric
EXCLUDED_FROM_MAIN = ("niah_single_3",)

# Plan B section 3.2.  These SUPERSEDE the earlier 20x60 plan's 12/32/96.
#   stage | per task x length | 16K+32K total | 8K guard
#     P   |        4          |      64       |    32
#     S   |        8          |     128       |    32
#     V   |       16          |     256       |    64
#     H   |  >=64, locked by the power plan after V | starts at 1024 | 256
# Section 3.2 also says H=1024 is NOT automatically enough: the final size is
# fixed after V, from the variance and a pre-declared effect size (section 11).
STAGE_ROWS = {"P": 4, "S": 8, "V": 16, "H": 64}
STAGE_GUARD_TOTAL = {"P": 32, "S": 32, "V": 64, "H": 256}
STAGE_GUARD = {stage: total // len(TASKS)
               for stage, total in STAGE_GUARD_TOTAL.items()}
H_STARTING_TOTAL = 1024          # section 3.2: 起步 1024
M_DEV_SCENARIOS = 64             # section 3.2
M_H_SCENARIOS = 128              # section 3.2: 128 起
LONG_LENGTHS = (16384, 32768)
GUARD_LENGTH = 8192
EVIDENCE_QUANTILES = (0.10, 0.35, 0.65, 0.90)

# section 2.4: the two deployment protocols, both registered in advance
DEPLOYMENT_PROTOCOLS = {
    "FIXED-4": {"s": 4.0, "lengths": {"8192": 4.0, "16384": 4.0, "32768": 4.0},
                "question": "deploy once to 32K while keeping the short context"},
    "LENGTH-MATCHED": {"s": 4.0, "lengths": {"8192": "native", "16384": 2.0, "32768": 4.0},
                       "question": "the request ceiling is known when choosing the config"},
}

SCORER_REVISION = "ruler-v1-frozen"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


def row_sha(prompt_ids):
    return hashlib.sha256(np.asarray(prompt_ids, dtype=np.int64).tobytes()).hexdigest()


@dataclass
class Row:
    """One evaluation row.  The identity block is shared by every method."""

    example_id: str
    semantic_group_id: str
    source_document_id: str
    task: str
    cap: int
    prompt_ids: list
    evidence_positions: list      # TOKEN indices, never character offsets
    distractor_positions: list
    gold: str
    max_new_tokens: int
    actual_length: int = 0
    layout: str = "spread"        # multivalue / multiquery: "spread" or "mid_clustered"

    @property
    def length_cap(self):
        return self.cap

    def identity(self):
        """Section 6.4's mandatory identity block."""
        return {
            "example_id": self.example_id,
            "semantic_group_id": self.semantic_group_id,
            "source_document_id": self.source_document_id,
            "task": self.task,
            "cap": self.cap,
            "actual_length": self.actual_length or len(self.prompt_ids),
            "prompt_input_ids_sha256": row_sha(self.prompt_ids),
            "evidence_positions": list(self.evidence_positions),
            "distractor_positions": list(self.distractor_positions),
            "gold": self.gold,
            "max_new_tokens": self.max_new_tokens,
            "scorer_revision": SCORER_REVISION,
            "layout": self.layout,
        }


def check_row_invariants(row, tok_len=None):
    """Section 6.4, as assertions rather than prose."""
    problems = []
    n = tok_len if tok_len is not None else len(row.prompt_ids)
    if n + row.max_new_tokens > row.cap:
        problems.append(f"input {n} + new {row.max_new_tokens} exceeds cap {row.cap}")
    # section 6.4: prompt as close to cap-128 as possible, within 128 tokens
    if n < row.cap - row.max_new_tokens - 128:
        problems.append(f"prompt {n} is more than 128 tokens short of cap-128 "
                        f"({row.cap - row.max_new_tokens})")
    if any(p >= n for p in row.evidence_positions):
        problems.append("an evidence position lies outside the prompt")
    return problems


# ---------------------------------------------------------------------------
# the partition: by SOURCE
# ---------------------------------------------------------------------------


@dataclass
class Partition:
    stages: dict          # stage -> list[Row]
    manifest: dict

    def identity_of(self, stage):
        return sorted(r.example_id for r in self.stages[stage])


def _source_of(row):
    return row.source_document_id


def partition(rows, seed=20260911, stages=("P", "S", "V", "H")):
    """Split by source document.  A source appears in exactly one stage.

    This is the whole point.  Splitting rows lets two rows from the same passage
    land in different stages, which makes the stages dependent and turns a
    within-source fluctuation into a "generalisation".  The ledger has already
    paid for this lesson once.
    """
    rng = np.random.default_rng(seed)

    # A source is ALLOCATED WHOLE.  Section 6.4: the same question at two
    # lengths is the same source, not a new sample.  So a source that appears at
    # several lengths keeps its rows together and lands in one stage; shuffling
    # each (task, length) cell independently would split one passage across
    # stages and make them dependent.
    by_cell_source = defaultdict(lambda: defaultdict(list))
    source_locations = defaultdict(set)
    semantic_locations = defaultdict(set)
    for r in rows:
        if r.task not in TASKS or r.cap not in (GUARD_LENGTH, *LONG_LENGTHS):
            raise ValueError(f"unexpected task/cap {r.task}/{r.cap}")
        cell = (r.task, r.cap)
        source = _source_of(r)
        semantic = r.semantic_group_id
        by_cell_source[cell][source].append(r)
        source_locations[source].add(cell)
        semantic_locations[semantic].add(cell)

    # Plan B requires different scenarios at 16K and 32K and partitions by
    # semantic/source cluster.  Reject an input that attempts to reuse either
    # identity across cells instead of counting it twice and relying on a later
    # bootstrap to repair the leak.
    reused_sources = {k: sorted(v) for k, v in source_locations.items() if len(v) > 1}
    reused_semantics = {k: sorted(v) for k, v in semantic_locations.items() if len(v) > 1}
    if reused_sources or reused_semantics:
        raise ValueError(
            "source/semantic cluster appears in multiple task-length cells; "
            f"sources={list(reused_sources)[:3]} semantics={list(reused_semantics)[:3]}")

    out = {s: [] for s in stages}
    used = {}
    cell_counts = defaultdict(lambda: defaultdict(int))
    shortfall = []

    expected_cells = [(task, cap) for task in sorted(TASKS)
                      for cap in (GUARD_LENGTH, *LONG_LENGTHS)]
    for task, cap in expected_cells:
        pool = by_cell_source.get((task, cap), {})
        names = sorted(pool)
        rng.shuffle(names)
        need_of = STAGE_GUARD if cap == GUARD_LENGTH else STAGE_ROWS
        tag = "guard" if cap == GUARD_LENGTH else "long"
        i = 0
        for stage in stages:
            need = need_of[stage]
            take = names[i:i + need]
            i += need
            if len(take) < need:
                shortfall.append({"task": task, "cap": cap, "kind": tag, "stage": stage,
                                  "have": len(take), "need": need})
            for source in take:
                if source in used:
                    raise ValueError(f"source {source} was allocated more than once")
                used[source] = stage
                for row in pool[source]:
                    out[stage].append(row)
                    cell_counts[stage][f"{row.task}|{row.cap}"] += 1
        for source in names[i:]:
            used.setdefault(source, "unused")

    seen = {}
    for stage, rs in out.items():
        for r in rs:
            s = _source_of(r)
            if s in seen and seen[s] != stage:
                raise ValueError(f"REFUSING: source {s} in both {seen[s]} and {stage}")
            seen[s] = stage

    manifest = {
        "seed": seed,
        "rule": ("partition by SOURCE; a source is allocated whole and appears in exactly "
                 "one stage; guard and long sources are disjoint per task"),
        "rows_per_task_per_length": {k: STAGE_ROWS[k] for k in stages},
        "guard_rows_per_task": {k: STAGE_GUARD[k] for k in stages},
        "guard_rows_total": {k: STAGE_GUARD_TOTAL[k] for k in stages},
        "counts": {s: len(v) for s, v in out.items()},
        "cell_counts": {s: dict(d) for s, d in cell_counts.items()},
        "n_sources": len(seen),
        "n_unused_sources": sum(1 for v in used.values() if v == "unused"),
        "shortfalls": shortfall,
        "excluded_tasks": list(EXCLUDED_FROM_MAIN),
        "warning": ("shortfalls mean the panel does not have enough independent sources "
                    "for this stage; section 8.2/8.4 forbid topping up with duplicate "
                    "sources, so the shortage is reported rather than hidden"),
    }
    if shortfall:
        manifest["status"] = "UNDERPOWERED_PANEL"
    return Partition(stages=out, manifest=manifest)


# ---------------------------------------------------------------------------
# statistics (section 8)
# ---------------------------------------------------------------------------


def macro_accuracy(records, long_lengths=LONG_LENGTHS):
    """A_L(M) = (1/8) sum_t (1/n_tL) sum_i s_i, and the two-length average.

    Section 8.1.  Task weights are equal and length weights are equal; both are
    stated in the return value so no reader has to infer them.
    """
    per = defaultdict(lambda: defaultdict(list))
    for r in records:
        per[r["length_cap"]][r["task"]].append(float(r["correct"]))
    task_mean = {L: {t: float(np.mean(v)) for t, v in d.items()} for L, d in per.items()}
    A = {L: float(np.mean(list(d.values()))) for L, d in task_mean.items() if d}
    both = [A[L] for L in long_lengths if L in A]
    return {"A_by_length": A, "task_mean": task_mean,
            "A_long": float(np.mean(both)) if both else float("nan"),
            "weights": {"tasks": "equal", "lengths": "equal"}}


def delta(records_cand, records_mr, long_lengths=LONG_LENGTHS):
    """Delta(M) = mean over the two long lengths of A_L(M) - A_L(MR)."""
    a = macro_accuracy(records_cand, long_lengths)
    b = macro_accuracy(records_mr, long_lengths)
    per = {L: a["A_by_length"][L] - b["A_by_length"][L]
           for L in long_lengths if L in a["A_by_length"] and L in b["A_by_length"]}
    return {"delta": float(np.mean(list(per.values()))) if per else float("nan"),
            "by_length": per, "candidate": a, "mr": b}


def retention_guard(records_cand, records_native, guard_length=GUARD_LENGTH):
    """8K retention relative to the true native run (section 6.3/8.4)."""
    c = macro_accuracy(records_cand, (guard_length,))["A_by_length"].get(guard_length)
    n = macro_accuracy(records_native, (guard_length,))["A_by_length"].get(guard_length)
    if c is None or n is None or n <= 0:
        return None
    return {"candidate": c, "native": n, "ratio": c / n}


def screen_selection(results, max_picks=8, per_direction_cap=2, lines=("I", "II")):
    """Section 8.2's deterministic selection, including its two guarantees.

    * at most one representative per direction first, then the top 8 of those;
    * at most 2 per direction in the final set;
    * each deployment line keeps at least its best 2, "避免全部名额被一个共同校准
      目标占用";
    * ties break on method id -- not on any geometry score.
    """
    rows = [r for r in results if r.get("valid", True)]
    rows.sort(key=lambda r: (-r["delta"], r["method"]))
    by_dir = {}
    for r in rows:
        by_dir.setdefault(r["direction"], []).append(r)
    reps = [v[0] for v in by_dir.values()]
    reps.sort(key=lambda r: (-r["delta"], r["method"]))
    picks = reps[:max_picks]

    per_dir_count = defaultdict(int)
    for r in picks:
        per_dir_count[r["direction"]] += 1
    final = []
    for r in picks:
        if per_dir_count[r["direction"]] <= per_direction_cap:
            final.append(r)
    for line in lines:
        have = [r for r in final if r.get("line") == line]
        if len(have) < 2:
            cands = [r for r in rows if r.get("line") == line and r not in final]
            for r in cands[:2 - len(have)]:
                final.append(r)
    final.sort(key=lambda r: (-r["delta"], r["method"]))
    return {"picked": final, "n_picked": len(final),
            "per_direction": dict(per_dir_count),
            "rule": "one representative per direction, then top-8; ties on method id"}


def sem_arith(q=0.2, n=None):
    """Section 8.5's power reference.  SE = sqrt(q/N) at delta = 0."""
    import math
    if n is None:
        return {}
    return {"n": n, "se": math.sqrt(q / n), "mde_95": 1.959963984540054 * math.sqrt(q / n)}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def write_rows(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.identity(), ensure_ascii=False) + "\n")


def read_results(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
