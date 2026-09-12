"""C: the single calibration collection (plan section 5).

51 of the 60 rules need constants that come from real model statistics, and
section 5 is emphatic that this happens ONCE: "一次采集，不能让60条规则各自发明
标定题".  Section 5.4 also forbids the opposite failure -- using C scores as
method results.  C is construction material, never evidence of a gain.

What is collected, and where from
---------------------------------
Section 5.2 fixes the tap point: **Q/K after the projection and before RoPE**,
in the checkpoint's own 2D pair layout, read with the KV group that the Q head
belongs to, averaged within each layer/head over scenarios, then pooled equally
across layers and heads.  Heads are NOT independent samples -- error bars are
resampled by scenario/source group only.

The tap is implemented by wrapping `apply_rotary_pos_emb`, which receives
exactly (q, k) immediately before the rotation.  That makes the collection point
and the deployment point the same place, so they cannot drift apart.

Contents (section 5.1)
----------------------
* 3 relation types x 64 independent scenarios = 192 groups: C-copy, C-bind,
  C-aggregate.  Each has original / semantic / position / format-equivalent
  variants; all native inputs are <= 8192 tokens.
* 64 natural-style unrelated Q/K pairs (four register styles, equally).
* C-carrier subset: 32 groups by ID hash, with stride-r displacements
  r = 128..1024 plus two unfitted checkpoints 2048/4096.
* C-gain subset: 64 natural groups, native short plus 16K/32K long, reading
  real logit rows only -- used by D20 alone.

Two conventions from section 5.3 that are easy to get backwards, so they are
enforced in code rather than in prose: c = conj(q)*k per slot, and E|c|^2 is
never conflated with |Ec|^2.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import core
from adapter import _call_rotary_original

SEED = 20260911
C_SPLIT_HASH_BIT = "lowest bit of sha256(source_id)"   # section 5.3

# section 5.1
RELATION_TYPES = ("copy", "bind", "aggregate")
SCENARIOS_PER_TYPE = 64
EVIDENCE_DISTANCES = (64, 128, 256, 512, 1024, 2048, 4096, 6144)
NATURAL_GROUPS = 64
NATURAL_STYLES = ("prose", "dialogue", "code", "structured")
CARRIER_GROUPS = 32
CARRIER_STRIDES = (128, 256, 384, 512, 640, 768, 896, 1024)
CARRIER_CHECKPOINTS = (2048, 4096)      # never used for fitting
GAIN_SUBSET = 64

MAX_NATIVE_TOKENS = 8192                # section 5.1


# ---------------------------------------------------------------------------
# scenario construction
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One C group: a prompt family with known evidence and distractor spans.

    Offsets are in TOKENS measured after tokenisation, because section 6.4
    forbids using character positions as token positions.
    """

    group_id: str
    relation: str
    distance: int
    style: str
    text_original: str
    text_semantic: str
    text_position: str
    text_format: str
    evidence_marker: str
    distractor_marker: str
    gold: str
    source_document_id: str = ""

    def variants(self):
        return {"original": self.text_original, "semantic": self.text_semantic,
                "position": self.text_position, "format": self.text_format}

    def split(self):
        h = hashlib.sha256(self.source_document_id.encode()).hexdigest()
        return "C-A" if int(h[-1], 16) % 2 == 0 else "C-B"


_FILLER = ("The archive note continues with routine inventory remarks that carry no "
           "relation to the question and exist only to occupy distance. ")


def _pad_to_tokens(text, filler, target_chars):
    if len(text) >= target_chars:
        return text
    reps = int(math.ceil((target_chars - len(text)) / len(filler)))
    return text + filler * reps


def _make_scenario(relation, idx, distance, rng):
    """A synthetic but position-controlled record with a known evidence span."""
    gid = f"C-{relation}-{idx:03d}"
    doc = f"doc-{relation}-{idx:03d}-{distance}"
    key = f"K{rng.integers(1000, 9999)}"
    val = f"V{rng.integers(100000, 999999)}"
    other_key = f"K{rng.integers(1000, 9999)}"
    other_val = f"V{rng.integers(100000, 999999)}"

    if relation == "copy":
        head = f"Record: the identifier {key} holds the value {val}. "
        semantic = f"Record: the identifier {key} holds the value {other_val}. "
        gold = val
        evidence_marker, distractor_marker = val, other_val
    elif relation == "bind":
        head = (f"Entity {key} is bound to {val}; entity {other_key} is bound to "
                f"{other_val}. ")
        semantic = (f"Entity {key} is bound to {other_val}; entity {other_key} is "
                    f"bound to {val}. ")     # swap the binding, keep the value set
        gold = val
        evidence_marker, distractor_marker = val, other_val
    else:  # aggregate
        head = (f"Chain {key} contributes 3 units and chain {other_key} contributes "
                f"5 units, for a recorded total of 8. ")
        semantic = (f"Chain {key} contributes 3 units and chain {other_key} contributes "
                    f"4 units, for a recorded total of 7. ")   # change one quantity
        gold = "8"
        evidence_marker, distractor_marker = "total of 8", "contributes 5"

    question = {
        "copy": f"Question: what value does {key} hold?",
        "bind": f"Question: what value is the entity {key} bound to?",
        "aggregate": "Question: what is the recorded total?",
    }[relation]

    def assemble(record_text, question_first):
        if question_first:
            core_text = question + " " + _FILLER * 2 + record_text
        else:
            core_text = _FILLER * 2 + record_text + " " + question
        # distance in characters as a proxy for the requested token distance;
        # the runner re-measures in real tokens after tokenisation
        approx = int(distance * 4.0)
        return _pad_to_tokens(core_text, _FILLER, approx + len(core_text))

    original = assemble(head, False)
    semantic = assemble(semantic, False)
    # position variant: move the whole record to the other end, order fixed
    position = assemble(head, True)
    # format-equivalent: same semantics, different surface framing
    format_v = ("Note. " + head.replace("Record: ", "").replace("Entity ", "The entity ")
                + " " + _FILLER * 2 + question)
    format_v = _pad_to_tokens(format_v, _FILLER, len(original))

    return Scenario(group_id=gid, relation=relation, distance=distance, style="synthetic",
                    text_original=original, text_semantic=semantic, text_position=position,
                    text_format=format_v, evidence_marker=evidence_marker,
                    distractor_marker=distractor_marker, gold=gold,
                    source_document_id=doc)


def build_c_scenarios(seed=SEED):
    """The 192 synthetic groups, distances balanced by group ID (section 5.1)."""
    rng = np.random.default_rng(seed)
    out = []
    for relation in RELATION_TYPES:
        for idx in range(SCENARIOS_PER_TYPE):
            d = EVIDENCE_DISTANCES[idx % len(EVIDENCE_DISTANCES)]
            out.append(_make_scenario(relation, idx, d, rng))
    return out


def build_natural_groups(seed=SEED):
    """64 natural-style unrelated Q/K groups: prose / dialogue / code / structured."""
    rng = np.random.default_rng(seed + 1)
    out = []
    for i in range(NATURAL_GROUPS):
        style = NATURAL_STYLES[i % len(NATURAL_STYLES)]
        filler = {
            "prose": "The committee reviewed the quarterly summary and noted several "
                     "routine items that require no further action at this time. ",
            "dialogue": "A: did you see the note? B: I did, nothing urgent in it. "
                        "A: agreed, let us move on. ",
            "code": "def process(rows):\n    return [r for r in rows if r.valid]\n"
                    "# no side effects; helper only\n",
            "structured": "field_a: 1\nfield_b: inactive\nfield_c: none\n"
                          "field_d: pending\n",
        }[style]
        text = _pad_to_tokens(f"Note {i}. ", filler, 600 + 80 * i)
        out.append({"group_id": f"C-nat-{i:03d}", "style": style, "text": text,
                    "source_document_id": f"nat-{style}-{i:03d}"})
    return out


def carrier_subset(scenarios, n=CARRIER_GROUPS):
    """Section 5.1: 32 groups chosen by ID hash, for the carrier-frequency fits."""
    ranked = sorted(scenarios,
                    key=lambda s: hashlib.sha256(s.group_id.encode()).hexdigest())
    return ranked[:n]


def collection_coverage(seed=SEED):
    """Return the complete C coverage contract before model work starts.

    The plan declares three auxiliary populations, but the old collector only
    scheduled the 192 relation groups.  Counts are made explicit here so a
    receipt cannot quietly look complete while omitting an auxiliary block.
    The carrier checkpoint text construction and C-gain logit-row schema are
    not specified anywhere in this package; those blocks are therefore
    explicitly BLOCKED rather than filled with an invented proxy.
    """
    relations = build_c_scenarios(seed)
    natural = build_natural_groups(seed)
    carriers = carrier_subset(relations)
    stride_counts = {str(d): sum(s.distance == d for s in carriers)
                     for d in CARRIER_STRIDES}
    styles = {style: sum(g["style"] == style for g in natural)
              for style in NATURAL_STYLES}
    return {
        "schema": "c-coverage-v1",
        "seed": seed,
        "status": "BLOCKED",
        "no_task_labels": True,
        "separation": "C calibration material only; never a method-result panel",
        "blocks": {
            "relations": {
                "status": "READY",
                "groups": len(relations),
                "variants_per_group": 4,
                "records": len(relations) * 4,
                "records_dispatched": 0,
                "relations": list(RELATION_TYPES),
            },
            "natural": {
                "status": "READY_FOR_DISPATCH_BUT_HELD",
                "groups": len(natural),
                "records": len(natural),
                "records_dispatched": 0,
                "styles": styles,
                "required": "raw unrelated Q/K capture; no answer/task labels",
            },
            "carrier": {
                "status": "BLOCKED",
                "groups": len(carriers),
                "fitted_stride_values": list(CARRIER_STRIDES),
                "selected_stride_counts": stride_counts,
                "checkpoint_values": list(CARRIER_CHECKPOINTS),
                "checkpoint_records_expected": len(carriers) * len(CARRIER_CHECKPOINTS),
                "checkpoint_records_dispatched": 0,
                "reason": "checkpoint text/position construction and pairing rule are unspecified",
            },
            "gain": {
                "status": "BLOCKED",
                "groups": GAIN_SUBSET,
                "lengths": ["native_short", 16384, 32768],
                "records_expected": GAIN_SUBSET * 3,
                "records_dispatched": 0,
                "reason": "real logit-row schema, target-token selection, and short/long pairing are unspecified",
            },
        },
        "complete_records_expected": len(relations) * 4 + len(natural)
                               + len(carriers) * len(CARRIER_CHECKPOINTS)
                               + GAIN_SUBSET * 3,
        "complete_records_dispatched": 0,
    }


def _coverage_path(out_path):
    return Path(out_path).with_suffix(".coverage.json")


def _require_complete_coverage(out_path, seed):
    """Write an auditable BLOCKED manifest and refuse partial C output."""
    coverage = collection_coverage(seed)
    _coverage_path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _coverage_path(out_path).write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    blocked = [name for name, block in coverage["blocks"].items()
               if block["status"] == "BLOCKED"]
    if blocked:
        raise RuntimeError(
            "C collection BLOCKED; refusing partial output for blocks: "
            + ", ".join(blocked)
            + f" (coverage manifest: {_coverage_path(out_path)})")
    return coverage


# ---------------------------------------------------------------------------
# the tap: Q/K after projection, before RoPE
# ---------------------------------------------------------------------------


class QKTap:
    """Wrap apply_rotary_pos_emb so the collection point IS the deployment point.

    Section 5.2: take Q/K after normalisation and before RoPE, in the
    checkpoint's own pair layout.  Wrapping the rotation means the same code
    path serves C collection and the deployed operator, so the two cannot
    silently disagree about which tensor is being read.
    """

    def __init__(self, torch, modeling):
        self.torch = torch
        self.modeling = modeling
        self.captured = []
        self._orig = None

    def __enter__(self):
        self._orig = self.modeling.apply_rotary_pos_emb
        tap = self

        def wrapped(q, k, cos, sin, *args, **kwargs):
            tap.captured.append((q.detach().to(tap.torch.float32).cpu().numpy(),
                                 k.detach().to(tap.torch.float32).cpu().numpy()))
            position_ids = kwargs.pop("position_ids", None)
            unsqueeze_dim = kwargs.pop("unsqueeze_dim", 1)
            if args:
                if len(args) >= 2:
                    position_ids, unsqueeze_dim = args[:2]
                elif hasattr(args[0], "shape"):
                    position_ids = args[0]
                else:
                    unsqueeze_dim = args[0]
            return _call_rotary_original(tap._orig, q, k, cos, sin,
                                         position_ids, unsqueeze_dim)

        self.modeling.apply_rotary_pos_emb = wrapped
        return self

    def __exit__(self, *exc):
        self.modeling.apply_rotary_pos_emb = self._orig
        return False

    def take(self):
        out, self.captured = self.captured, []
        return out


# ---------------------------------------------------------------------------
# statistics (section 5.3)
# ---------------------------------------------------------------------------


def per_head_statistics(q, k, query_index, evidence_span, distractor_span, layout="half"):
    """Complex coefficients and the evidence-minus-distractor difference.

    Shapes: q, k are (H, P, 2K) for one layer.  Returns per-head c, Delta c.
    Section 5.3: c = conj(q)*k, and the difference is a MEAN over the fixed
    evidence/distractor sets -- never a max over the most favourable key.
    """
    q = np.asarray(q)
    k = np.asarray(k)
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"expected q/k=(H,P,D), got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0]:
        if q.shape[0] % k.shape[0] != 0:
            raise ValueError(f"GQA head mismatch: Q={q.shape[0]} KV={k.shape[0]}")
        # HF repeat_kv repeats each KV group contiguously for its Q heads.
        k = np.repeat(k, q.shape[0] // k.shape[0], axis=0)
    qv = q[:, query_index, :]                        # (H, 2K)
    ev = k[:, evidence_span[0]:evidence_span[1], :].mean(axis=1)      # (H, 2K)
    ds = k[:, distractor_span[0]:distractor_span[1], :].mean(axis=1)  # (H, 2K)
    c_ev = core.coefficient(qv, ev, layout)
    c_ds = core.coefficient(qv, ds, layout)
    return {"c_evidence": c_ev, "c_distractor": c_ds, "delta_c": c_ev - c_ds}


def pool_layers_heads(per_layer):
    """Equal weight across layers then heads (section 5.2: heads are not samples)."""
    stack = np.stack([np.asarray(x) for x in per_layer], axis=0)     # (L, H, K)
    return stack.mean(axis=0)                                       # (H, K)


def pooled_equal(per_layer):
    """Equal weight over every layer and head at once -> (K,)."""
    return float_mean_stack(per_layer)


def float_mean_stack(per_layer):
    stack = np.stack([np.asarray(x) for x in per_layer], axis=0)
    return stack.reshape(-1, stack.shape[-1]).mean(axis=0)


# ---------------------------------------------------------------------------
# the collection driver
# ---------------------------------------------------------------------------


def tokenize_with_spans(tok, text, marker, distractor, max_tokens=MAX_NATIVE_TOKENS):
    """Tokenise and locate the evidence / distractor spans in TOKEN indices.

    Section 6.4: character positions are never used as token positions.  The
    spans are found by re-tokenising the marker and searching the id sequence.
    """
    ids = tok(text, return_tensors=None, add_special_tokens=True)["input_ids"]
    if len(ids) > max_tokens:
        raise ValueError(f"tokenized scenario exceeds native C limit: {len(ids)} > {max_tokens}")

    def find(needle):
        sub = tok(needle, add_special_tokens=False)["input_ids"]
        if not sub:
            raise ValueError("empty marker is not a valid C span")
        for i in range(len(ids) - len(sub) + 1):
            if ids[i:i + len(sub)] == sub:
                return (i, i + len(sub))
        raise ValueError(f"marker not found after tokenization: {needle!r}")

    return ids, find(marker), find(distractor)


def collect(model_dir, out_path, seed=SEED, limit=None, device="cuda"):
    """Run the single C collection pass and write the frozen statistics."""
    # Validate one-shot completeness before importing/loading any model.  A
    # convenient smoke limit must never produce a plausible-looking partial C
    # artifact.
    scenarios = build_c_scenarios(seed)
    if limit:
        if int(limit) != len(scenarios):
            raise ValueError(
                f"partial C collection refused: limit={limit}, expected {len(scenarios)}; "
                "C statistics must be collected in one complete pass")
        scenarios = scenarios[:limit]

    # Auxiliary populations are part of the one-shot contract.  Do this before
    # importing/loading the model and before writing any statistics.  A caller
    # must not obtain a relation-only C file and mistake it for complete C.
    _require_complete_coverage(out_path, seed)

    import torch
    import transformers
    from transformers.models.llama import modeling_llama as M

    g = core.Geometry.from_config(model_dir)
    g.assert_expected()

    tok = transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    per_group = {}
    with QKTap(torch, M) as tap:
        for s in scenarios:
            for vname, text in s.variants().items():
                ids, ev, ds = tokenize_with_spans(tok, text, s.evidence_marker,
                                                  s.distractor_marker)
                input_ids = torch.tensor([ids], device=model.device)
                with torch.no_grad():
                    model(input_ids=input_ids)
                layers = tap.take()
                stats = []
                for q, k in layers:
                    # (1, H, P, 2K) -> (H, P, 2K)
                    stats.append(per_head_statistics(q[0], k[0], len(ids) - 1, ev, ds))
                per_group[f"{s.group_id}:{vname}"] = {
                    "relation": s.relation, "distance": s.distance, "variant": vname,
                    "split": s.split(), "source_document_id": s.source_document_id,
                    "evidence_span": ev, "distractor_span": ds, "n_tokens": len(ids),
                    "delta_c_per_layer": [x["delta_c"] for x in stats],
                    "c_evidence_per_layer": [x["c_evidence"] for x in stats],
                }
                print(json.dumps({"collected": f"{s.group_id}:{vname}",
                                  "tokens": len(ids)}), flush=True)

    expected_groups = len(scenarios) * 4
    if len(per_group) != expected_groups:
        raise RuntimeError(
            f"partial C collection refused: got {len(per_group)} groups, "
            f"expected {expected_groups}")

    # pool: equal weight over layers and heads -> (K,) complex
    pooled = {}
    for key, rec in per_group.items():
        d = np.array([np.asarray(x) for x in rec["delta_c_per_layer"]])   # (L,H,K)
        pooled[key] = {
            "delta_c": d.reshape(-1, d.shape[-1]).mean(axis=0),
            "split": rec["split"], "relation": rec["relation"],
            "distance": rec["distance"], "variant": rec["variant"],
        }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        keys=np.array(list(pooled), dtype=object),
        delta_c=np.stack([pooled[k]["delta_c"] for k in pooled]),
        split=np.array([pooled[k]["split"] for k in pooled]),
        relation=np.array([pooled[k]["relation"] for k in pooled]),
        distance=np.array([pooled[k]["distance"] for k in pooled]),
        variant=np.array([pooled[k]["variant"] for k in pooled]),
        allow_pickle=True)
    meta = {
        "geometry": {"theta": g.theta, "window": g.window, "head_dim": g.head_dim,
                     "K": g.K, "low": g.low, "high": g.high, "n": g.n,
                     "n_layers": g.n_layers, "n_heads": g.n_heads,
                     "n_kv_heads": g.n_kv_heads},
        "n_groups": len(pooled),
        "relation_types": list(RELATION_TYPES),
        "evidence_distances": list(EVIDENCE_DISTANCES),
        "pooling": "equal weight over layers and heads; heads are not samples",
        "convention": "c = conj(q)*k, d = key_position - query_position",
        "tap_point": "apply_rotary_pos_emb inputs (post-projection, pre-RoPE)",
        "seed": seed,
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({len(pooled)} pooled entries)")
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    collect(a.model, a.out, seed=a.seed, limit=a.limit or None)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
