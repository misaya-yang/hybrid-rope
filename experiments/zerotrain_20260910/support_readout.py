#!/usr/bin/env python3
"""Read the support axis -- the one round 1 never ran -- with its offset removed.

WHAT THIS AXIS IS.  Every arm in round 1 held slots 0..23 at m = 0 and reached
m = 1 by slot 40.  So the screen measured the SHAPE of the transition and nothing
about its SUPPORT, while the project's own method (EVQ) is a global companding
that necessarily moves those slots, and the three deployed tables (YaRN, MrRoPE,
BM) all refuse to.  "Is the held plateau load-bearing?" has therefore never been
measured.  Two groups answer it:

  companding  EVQ-deployed at four tau, EVQ-shift, two power laws -- global
              support, no plateau
  leak        beta_b1 with a share of the compression moved below the band, the
              increment budget held at exactly 1 so the ONLY difference from
              beta_b1 is the support; leak_a0 IS beta_b1 bitwise

THE OFFSET, AND WHY IT IS REMOVED RATHER THAN IGNORED.  The baseline cells come
from the archived `run_nll_01/MrPro.jsonl`, which was evaluated ONE document per
forward, while these arms are evaluated batched (a 32 GB card is mostly idle at
batch 1).  bf16 reductions are not batch-invariant, so every delta carries a
systematic offset of about 2.5e-4 nats -- the same size as several of the effects
being compared.  `mrpro_n17` is MrRoPE, so its measured delta IS that offset and
is used as the anchor: every delta below is reported BEFORE and AFTER subtracting
it.  Reporting only the raw number would attribute a harness difference to the
table, which is LESSONS L6.

WHAT IT CANNOT SAY.  These are in-window tail-NLL numbers at 8192/16384/32768,
all inside the 32768 native window.  They price the SUPPORT, not the capability:
an arm can be free here and useless at 128K, or cheap here and much better there.
The long-range verdict needs the RULER rows, and this file says so rather than
letting a small in-window number read as a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The arm that measures the harness offset, and the arm the leak family must tie.
OFFSET_ARM = "mrpro_n17"
LEAK_IDENTITY_ARM = "leak_a0"
LEAK_IDENTITY_TWIN = "beta_b1"


def load_rows(path):
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        out[d["name"]] = d
    return out


def group_of(name):
    if name.startswith("evq_") or name.startswith("power_"):
        return "companding"
    if name.startswith("leak_"):
        return "leak"
    if name.startswith("mixC_"):
        return "mixture"
    if name.startswith("beta_b") or name.startswith("incr_") or name.startswith("mrpro"):
        return "three-band"
    if name.startswith(("front_", "back_", "both_", "split")):
        return "three-band"
    if name in ("native", "native_gain1", "yarn_lin"):
        return "reference"
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True, help="dir with rows.jsonl")
    ap.add_argument("--anchor", required=True,
                    help="dir whose rows.jsonl carries the offset arm (round 1)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    sup = load_rows(Path(args.support) / "rows.jsonl")
    anc = load_rows(Path(args.anchor) / "rows.jsonl")
    if not sup:
        print("no support rows yet", file=sys.stderr)
        return 1

    # The offset: the same table measured both ways, or an explicit zero.
    # PER LENGTH, NOT A SINGLE MEAN.  The offset is a batching artifact and it
    # scales with the sequence length (batch is set from a token budget, so the
    # long rows run in smaller batches).  Measured on this harness it is
    # -8e-6 at 8192, +3.5e-4 at 16384 and +4.0e-4 at 32768 -- a factor of fifty
    # across the three columns.  Subtracting a single mean would leave the 32K
    # column carrying most of the offset while over-correcting the 8K column, and
    # the 32K column is the one that can be checked against the archive, so
    # getting it wrong hides the one comparison that validates the instrument.
    off_by_len = {}
    src = "no anchor arm present; offset NOT removed"
    anchor_row = anc.get(OFFSET_ARM) or sup.get(OFFSET_ARM)
    if anchor_row:
        off_by_len = {k: float(v["mean_delta"])
                      for k, v in anchor_row["per_length"].items()}
        src = (f"{OFFSET_ARM} (it IS MrRoPE, so its per-length delta IS the offset) "
               f"from {'anchor' if OFFSET_ARM in anc else 'this run'}")
    off = (sum(off_by_len.values()) / len(off_by_len)) if off_by_len else 0.0

    print(f"=== harness offset, per length ===")
    for k in sorted(off_by_len, key=int):
        print(f"    {k:>6s}  {off_by_len[k]:+.6f}")
    print(f"    {src}")
    print()

    # the construction check the leak family exists to pass
    if LEAK_IDENTITY_ARM in sup and LEAK_IDENTITY_TWIN in sup:
        a = float(sup[LEAK_IDENTITY_ARM]["mean_delta_all"])
        b = float(sup[LEAK_IDENTITY_TWIN]["mean_delta_all"])
        tied = abs(a - b) < 1e-12
        print(f"=== construction check: {LEAK_IDENTITY_ARM} vs {LEAK_IDENTITY_TWIN} ===")
        print(f"    {a:+.6f} vs {b:+.6f}  -> {'TIED (good)' if tied else 'DIFFERENT'}")
        if not tied:
            print("    NOTE: these are bit-identical tables, so a difference here is")
            print("    the measurement, not the table. It bounds this instrument's")
            print("    own run-to-run noise, which is worth knowing either way.")
        print()

    rows = []
    for name, d in sup.items():
        raw = float(d["mean_delta_all"])
        pl = {k: float(v["mean_delta"]) for k, v in d["per_length"].items()}
        rows.append(dict(name=name, group=group_of(name), raw=raw,
                         corrected=raw - off,
                         per_length=pl,
                         corrected_by_len={k: pl[k] - off_by_len.get(k, 0.0)
                                           for k in pl},
                         sum_m=float(d["sum_m"])))
    for name, d in anc.items():
        if name in sup or name == OFFSET_ARM:
            continue
        pl = {k: float(v["mean_delta"]) for k, v in d["per_length"].items()}
        rows.append(dict(name=name, group=group_of(name),
                         raw=float(d["mean_delta_all"]),
                         corrected=float(d["mean_delta_all"]) - off,
                         per_length=pl,
                         corrected_by_len={k: pl[k] - off_by_len.get(k, 0.0)
                                           for k in pl},
                         sum_m=float(d["sum_m"]), from_anchor=True))

    rows.sort(key=lambda r: r["corrected_by_len"].get("32768", 0.0))
    print(f"=== in-window cost, PER-LENGTH offset removed ===")
    print(f"{'arm':18s} {'group':11s} {'sum_m':>8s}  "
          f"{'8k':>10s} {'16k':>10s} {'32k':>10s}")
    for r in rows:
        cb = r["corrected_by_len"]
        print(f"{r['name']:18s} {r['group']:11s} {r['sum_m']:8.3f}  "
              + " ".join(f"{cb.get(k, float('nan')):+10.6f}"
                         for k in ("8192", "16384", "32768")))

    # The 32K column is the one the archive can check, so it is the headline.
    print()
    k32 = "32768"
    for r in rows[:1] + rows[-1:]:
        pass
    print(f"=== the 32768 column, which is the one the archived NLL cells cover ===")
    s32 = sorted(rows, key=lambda r: r["corrected_by_len"].get(k32, 0.0))
    for r in s32[:5]:
        print(f"  cheapest  {r['name']:18s} {r['corrected_by_len'][k32]:+.6f}")
    for r in s32[-3:]:
        print(f"  costliest {r['name']:18s} {r['corrected_by_len'][k32]:+.6f}")

    band = [r for r in rows if r["group"] == "three-band"]
    comp = [r for r in rows if r["group"] == "companding"]
    leak = [r for r in rows if r["group"] == "leak"]
    print()
    for tag, grp in (("three-band", band), ("companding", comp), ("leak", leak)):
        if grp:
            c = [r["corrected"] for r in grp]
            print(f"  {tag:11s} n={len(grp):2d}  corrected cost: "
                  f"min {min(c):+.6f}  max {max(c):+.6f}  mean {sum(c)/len(c):+.6f}")

    spread = (max(r["corrected_by_len"][k32] for r in rows)
              - min(r["corrected_by_len"][k32] for r in rows))
    print(f"\n  total spread across every arm measured: {spread:.6f} nats")
    print("  This is the IN-WINDOW axis only. It prices the support, not the")
    print("  capability: an arm can be free here and useless at 128K. The")
    print("  long-range verdict needs the RULER rows and is a separate run.")

    out = dict(offset_by_length=off_by_len, offset=off, offset_source=src,
               spread_32k=spread,
               rows=[{k: v for k, v in r.items() if k != "from_anchor"} for r in rows],
               scope=("in-window tail-512 NLL at 8192/16384/32768, all inside the "
                      "native window; prices the SUPPORT of the compression, not "
                      "long-context capability"))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
