"""The scorer: two measured numbers per arm, and the cost of getting them.

WHAT IS MEASURED, AND WHY THESE TWO AND NOT OTHERS.

    native_kl   E_u[ KL( p_native || p_table ) ] over the scored positions.
                The plan's CONSTRAINT side.  It is the native-preservation
                metric `curvature_20260910.model.output_kl` already defines --
                no first-order term, quadratic form = the Fisher, a statement
                about behaviour drift rather than weight drift.  Not a
                cross-entropy difference, which would have a non-zero
                first-order term and an indefinite second-order form.

    long_nll    mean next-token NLL over the scored positions, which are all
                BEYOND the checkpoint's native window.  The plan's OBJECTIVE
                side, and the only quantity here that can be improved by a table
                without any training.

Both come out of the SAME forward, which is what makes the screen cheap: one
pass per arm per document, plus a base pass at the native table that every arm
shares.

WHAT THE PAIR CANNOT SETTLE, AND IS NOT ASKED TO.  `long_nll` is language
modelling on held-out documents.  It is not retrieval, not RULER, and not any
task the paper will report.  A table can lower it by making the model more
confident about ordinary continuations while getting every needle wrong; the
converse is also available.  The screen's job is to rank ARMS on a cheap,
pre-registered pair of numbers and to hand the survivors to a task panel -- it
does not establish that the winner is better at anything.  Every receipt carries
that sentence, because a leaderboard is exactly the artefact that gets quoted
without it.

THE COST MODEL, RECORDED RATHER THAN ASSUMED.  `FrozenRoPE` takes
`logits_to_keep`, so the vocabulary head only sees the scored positions and a
128K forward is affordable at all; the archived figure is 4.1 s median at 32K
and 33.9 s at 128K, and `curvature_20260910/RUNBOOK.md:38` records in bold that
those are PROJECTIONS from archived timings rather than measurements by that
package.  So this module measures its own: every arm's receipt carries its own
wall time and peak allocation, and `run_screen.py` prints the first arm's
per-forward cost before committing to the rest.  A screen that inherits a timing
assumption is a screen that can overrun a card reservation.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..curvature_20260910.model import output_kl


class Scorer:
    """Scores tables against a fixed document set.

    `docs` is a list of (1, L) int64 tensors, each LONGER than the native window
    so that the scored positions are genuinely beyond it.  `keep` is how many
    trailing positions are scored; it must be small enough that the base
    log-probability tensor fits -- `keep * vocab * 4` bytes per document, i.e.
    311 MB at keep=512 and a 151936-token vocabulary, so sixteen documents is
    5 GB of cache and that is the number that decides the corpus size.
    """

    def __init__(self, frozen, docs, keep=512, native_gain=1.0, label=""):
        if not docs:
            raise ValueError("no documents")
        self.f = frozen
        self.docs = [d if torch.is_tensor(d) else torch.as_tensor(d) for d in docs]
        for d in self.docs:
            if d.dim() != 2 or d.shape[0] != 1:
                raise ValueError(f"documents must be (1, L), got {tuple(d.shape)}")
            if d.shape[1] <= keep + 1:
                raise ValueError(
                    f"document of {d.shape[1]} tokens cannot be scored with "
                    f"keep={keep}: the scored positions must lie beyond the "
                    "native window, which is the whole point of the exercise")
        self.keep = int(keep)
        self.label = label
        self._base = None
        self._native_table = None
        self.warmup_seconds = None

    # -- one pass at native, shared by every arm ---------------------------
    def warmup(self, native_table=None):
        """Cache the native table's log-probabilities and the base NLL.

        Done ONCE and reused by every arm, which is the difference between
        ~1 forward per arm per document and ~2.  The base NLL is taken from the
        same pass rather than recomputed, and it is what `long_nll` is reported
        against as well as being the NLL of the native arm itself.
        """
        t0 = time.time()
        table = native_table if native_table is not None else dict(
            name="native", m=np.zeros(64), gain=float(self.native_gain),
            theta=None)
        if table.get("theta") is None:
            from ..curvature_20260910 import tables as T
            table = dict(table, theta=float(T.QWEN25_3B["theta"]),
                         values_float32=T.m_to_inv_freq(
                             np.zeros(64), T.QWEN25_3B["theta"]))
        self._native_table = table
        self.f.install_table(table)
        base, nlls = [], []
        with torch.no_grad():
            for d in self.docs:
                lp = self.f.log_probs(d, self.keep)
                base.append(lp.double().cpu())
                lg = self.f.logits(d, self.keep)
                tgt = d[0, -self.keep:]
                nlls.append(float(torch.nn.functional.cross_entropy(
                    lg, tgt, reduction="mean")))
        self._base = base
        self._native_nll = float(np.mean(nlls))
        self.warmup_seconds = time.time() - t0
        return dict(native_nll=self._native_nll,
                    warmup_seconds=self.warmup_seconds, n_docs=len(self.docs),
                    keep=self.keep)

    # -- one arm -----------------------------------------------------------
    def score(self, table, measure_memory=True):
        """Run one arm.  Installs the table, so the caller need not restore."""
        if self._base is None:
            raise RuntimeError("warmup() first -- the base pass is shared by every "
                               "arm and repeating it per arm is the cost the "
                               "screen exists to avoid")
        if measure_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        kls, nlls = [], []
        with torch.no_grad():
            for d, b in zip(self.docs, self._base):
                kls.append(output_kl(self.f, d, self.keep,
                                     b.to(self.f.device), table))
                lg = self.f.logits(d, self.keep)
                nlls.append(float(torch.nn.functional.cross_entropy(
                    lg, d[0, -self.keep:], reduction="mean")))
        secs = time.time() - t0
        peak = (int(torch.cuda.max_memory_allocated())
                if measure_memory and torch.cuda.is_available() else None)
        return dict(
            name=table["name"], gain=float(table["gain"]),
            native_kl=float(np.mean(kls)), long_nll=float(np.mean(nlls)),
            native_kl_per_doc=[float(x) for x in kls],
            long_nll_per_doc=[float(x) for x in nlls],
            seconds=secs, seconds_per_doc=secs / len(self.docs),
            peak_bytes=peak,
            native_nll_at_native=self._native_nll,
            long_nll_gain=float(self._native_nll - np.mean(nlls)),
            note="long_nll is language modelling on held-out documents, not a "
                 "task score; the screen ranks arms and does not establish that "
                 "the winner is better at anything",
        )

    # -- a whole bank ------------------------------------------------------
    def screen(self, tables, progress=None, stop_file=None, budget_seconds=None):
        """Score a list of tables in order, writing a row per table.

        The loop is deliberately dumb: no early stopping on the score, because
        stopping an arm because it is losing is exactly the multi-candidate
        scanning the project forbids.  It stops on TIME or on the presence of a
        stop file, and it reports what it did not reach rather than silently
        returning a shorter table.
        """
        import os
        rows, t0 = [], time.time()
        for i, t in enumerate(tables):
            if stop_file and os.path.exists(stop_file):
                rows.append(dict(name=t["name"], status="not_run",
                                 reason=f"stop file {stop_file} present"))
                break
            if budget_seconds and (time.time() - t0) > budget_seconds:
                rows.append(dict(name=t["name"], status="not_run",
                                 reason=f"budget of {budget_seconds:.0f}s spent "
                                        "before this arm was reached"))
                break
            r = self.score(t)
            r["status"] = "ok"
            r["index"] = i
            r["elapsed_cumulative"] = time.time() - t0
            rows.append(r)
            if progress:
                progress(r)
        return dict(rows=rows, n_ran=sum(r["status"] == "ok" for r in rows),
                    n_declared=len(tables), elapsed=time.time() - t0,
                    warmup_seconds=self.warmup_seconds,
                    keep=self.keep, n_docs=len(self.docs),
                    native_nll_at_native=self._native_nll,
                    label=self.label)
