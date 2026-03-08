
#!/usr/bin/env python3
"""
generate_evq_synth.py

Mechanism-driven synthetic long-context data generator for comparing RoPE variants
(e.g., geometric RoPE vs EVQ-style frequency allocation).

Features
--------
- Single-file, CLI-driven dataset generator
- Streams JSONL output to avoid high memory use
- Supports four task families:
  1) single_hop       : adversarial long-range retrieval
  2) multi_hop        : scattered multi-hop tracing
  3) count            : long-range aggregation / counting
  4) alias_gap        : low-lexical-overlap retrieval
- Optional comparative hardness sampling using two RoPE frequency tables
- Explicit control of evidence position (head / middle / tail)
- Train / val / test generation with separate length regimes
- Hard-gap cache written to disk for reproducibility

Output schema
-------------
Each JSONL row has:
{
  "id": "train_single_hop_000000001",
  "text": "<space-separated tokens>",
  "answer": "val_123",
  "task": "single_hop",
  "split": "train",
  "meta": {
    ...
  }
}

Typical usage
-------------
1) Pure geometric / random-gap generation:
    python generate_evq_synth.py \
        --output-dir out \
        --target-tokens 500000000

2) Comparative hardness generation using your EVQ inv_freq json:
    python generate_evq_synth.py \
        --output-dir out \
        --target-tokens 500000000 \
        --alt-spec-json evq_inv_freq.json \
        --hardness-mode comparative

3) Small smoke test:
    python generate_evq_synth.py \
        --output-dir smoke \
        --target-tokens 50000 \
        --test-lengths 2048,4096 \
        --num-workers-note single-process

Notes
-----
- The generator is single-process for determinism and simplicity.
- Hardness is defined by relative positional collision:
      C(d1, d2) = mean_k cos(omega_k * (d1 - d2))
  and comparative hardness:
      H(d1, d2) = C_base(d1, d2) - C_alt(d1, d2)
- If no --alt-spec-json is supplied, the generator falls back to random-gap mode.
- This script is designed to generate causal-LM training text. The answer token
  is appended in the sequence after the query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------
# Dataclasses and config
# ----------------------------

@dataclass
class RopeSpec:
    name: str
    inv_freq: List[float]


@dataclass
class SplitBudget:
    split: str
    target_tokens: int


@dataclass
class GapPair:
    d_true: int
    d_confuse: int
    hardness: float


@dataclass
class TaskWeights:
    single_hop: float
    multi_hop: float
    count: float
    alias_gap: float

    def normalized_items(self) -> List[Tuple[str, float]]:
        raw = {
            "single_hop": self.single_hop,
            "multi_hop": self.multi_hop,
            "count": self.count,
            "alias_gap": self.alias_gap,
        }
        total = sum(max(v, 0.0) for v in raw.values())
        if total <= 0:
            raise ValueError("Task weights must sum to a positive number.")
        return [(k, v / total) for k, v in raw.items()]


# ----------------------------
# RoPE utilities
# ----------------------------

def geometric_inv_freq(dim_pairs: int, rope_theta: float) -> List[float]:
    """
    Standard RoPE inverse frequencies:
        inv_freq[k] = theta^(-2k/dim)
    where dim_pairs = dim / 2 for RoPE pair dimensions.
    """
    if dim_pairs <= 0:
        raise ValueError("dim_pairs must be positive.")
    dim = dim_pairs * 2
    return [1.0 / (rope_theta ** (2.0 * k / dim)) for k in range(dim_pairs)]


def load_inv_freq_json(path: Path) -> List[float]:
    """
    Accepts a json file in one of these shapes:
      {"inv_freq": [ ... ]}
      [ ... ]
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if "inv_freq" not in obj:
            raise ValueError(f"{path} is missing 'inv_freq'.")
        inv = obj["inv_freq"]
    elif isinstance(obj, list):
        inv = obj
    else:
        raise ValueError(f"{path} must be a JSON list or dict with key 'inv_freq'.")

    if not inv or not all(isinstance(x, (int, float)) for x in inv):
        raise ValueError(f"{path} contains an invalid inv_freq array.")
    return [float(x) for x in inv]


def collision_score(inv_freq: Sequence[float], d1: int, d2: int) -> float:
    """
    Relative positional collision between two distances.
    Higher => more similar in RoPE phase space.
    """
    delta = d1 - d2
    s = 0.0
    for w in inv_freq:
        s += math.cos(w * delta)
    return s / max(1, len(inv_freq))


def hardness_score(base: RopeSpec, alt: RopeSpec, d1: int, d2: int) -> float:
    """
    Comparative hardness:
      H = C_base(d1, d2) - C_alt(d1, d2)
    Large positive H means the distance pair is more confusable under the base
    spec than under the alternative spec.
    """
    return collision_score(base.inv_freq, d1, d2) - collision_score(alt.inv_freq, d1, d2)


def build_hard_gap_pool(
    *,
    base: RopeSpec,
    alt: RopeSpec,
    max_len: int,
    sample_pairs_per_d: int,
    top_quantile: float,
    rng: random.Random,
) -> List[GapPair]:
    """
    Samples candidate (d_true, d_confuse) pairs and keeps only the top quantile
    under comparative hardness.
    """
    if not (0.0 < top_quantile <= 1.0):
        raise ValueError("top_quantile must be in (0, 1].")

    scored: List[GapPair] = []
    for d_true in range(16, max_len + 1):
        upper_delta = min(max_len - 1, max(64, int(0.25 * max_len)))
        for _ in range(sample_pairs_per_d):
            delta = rng.randint(8, upper_delta)
            # sample both sides when possible
            candidates = []
            if d_true + delta <= max_len:
                candidates.append(d_true + delta)
            if d_true - delta >= 8:
                candidates.append(d_true - delta)
            if not candidates:
                continue
            d_confuse = rng.choice(candidates)
            if d_confuse == d_true:
                continue
            h = hardness_score(base, alt, d_true, d_confuse)
            scored.append(GapPair(d_true=d_true, d_confuse=d_confuse, hardness=h))

    if not scored:
        raise RuntimeError("No hard gap candidates were generated.")

    scored.sort(key=lambda x: x.hardness, reverse=True)
    keep = max(1, int(len(scored) * top_quantile))
    return scored[:keep]


# ----------------------------
# Lexicon and token helpers
# ----------------------------

class AliasLexicon:
    """
    Synthetic alias graph to reduce trivial lexical overlap.
    """
    def __init__(
        self,
        *,
        n_keys: int,
        n_vals: int,
        alias_per_key: int,
        alias_per_val: int,
        rng: random.Random,
    ) -> None:
        if n_keys < 100:
            raise ValueError("n_keys should be reasonably large (>=100).")
        if n_vals < 100:
            raise ValueError("n_vals should be reasonably large (>=100).")
        self.rng = rng
        self.key_aliases: Dict[int, List[str]] = {}
        self.val_aliases: Dict[int, List[str]] = {}

        for k in range(n_keys):
            aliases = [f"key_{k}"] + [f"moon_key_{k}_{i}" for i in range(alias_per_key)]
            rng.shuffle(aliases)
            self.key_aliases[k] = aliases

        for v in range(n_vals):
            aliases = [f"val_{v}"] + [f"amber_val_{v}_{i}" for i in range(alias_per_val)]
            rng.shuffle(aliases)
            self.val_aliases[v] = aliases

    def key_base(self, k: int) -> str:
        return self.key_aliases[k][0]

    def key_alias(self, k: int) -> str:
        aliases = self.key_aliases[k]
        if len(aliases) == 1:
            return aliases[0]
        return self.rng.choice(aliases[1:])

    def val_base(self, v: int) -> str:
        return self.val_aliases[v][0]

    def val_alias(self, v: int) -> str:
        aliases = self.val_aliases[v]
        if len(aliases) == 1:
            return aliases[0]
        return self.rng.choice(aliases[1:])


def noise_token(rng: random.Random) -> str:
    return f"noise_{rng.randint(0, 4095)}_{rng.randint(0, 4095)}"


def init_token_canvas(length: int, rng: random.Random) -> List[str]:
    return [noise_token(rng) for _ in range(length)]


def reserve_mask(length: int) -> List[bool]:
    return [False] * length


def chunk_hash(tokens: Sequence[str]) -> str:
    raw = " ".join(tokens).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def place_chunk(
    canvas: List[str],
    used: List[bool],
    start_idx: int,
    chunk: Sequence[str],
) -> bool:
    if start_idx < 0 or start_idx + len(chunk) > len(canvas):
        return False
    if any(used[i] for i in range(start_idx, start_idx + len(chunk))):
        return False
    for i, tok in enumerate(chunk):
        canvas[start_idx + i] = tok
        used[start_idx + i] = True
    return True


def sample_evidence_anchor(length: int, bin_name: str, chunk_len: int, rng: random.Random) -> int:
    """
    Returns a start position for a chunk according to a coarse absolute position bin.
    """
    if chunk_len >= length - 1:
        return 0

    if bin_name == "head":
        lo = 0
        hi = max(0, int(0.20 * length) - chunk_len)
    elif bin_name == "middle":
        lo = max(0, int(0.40 * length))
        hi = max(lo, int(0.60 * length) - chunk_len)
    elif bin_name == "tail":
        lo = max(0, int(0.80 * length))
        hi = max(lo, length - chunk_len - 1)
    else:
        raise ValueError(f"Unknown bin_name: {bin_name}")

    if hi < lo:
        lo, hi = 0, max(0, length - chunk_len - 1)
    return rng.randint(lo, hi) if hi > lo else lo


def choose_position_bin(rng: random.Random, split: str) -> str:
    # Slightly overweight middle on test to expose lost-in-the-middle behavior
    if split == "test":
        bins = ["head", "middle", "tail"]
        weights = [0.25, 0.50, 0.25]
        return rng.choices(bins, weights=weights, k=1)[0]
    return rng.choice(["head", "middle", "tail"])


def choose_length(rng: random.Random, choices: Sequence[int]) -> int:
    return int(rng.choice(list(choices)))


def weighted_choice(items: Sequence[Tuple[str, float]], rng: random.Random) -> str:
    labels = [x[0] for x in items]
    weights = [x[1] for x in items]
    return rng.choices(labels, weights=weights, k=1)[0]


def choose_gap_pair(
    *,
    hard_pool: Optional[Sequence[GapPair]],
    max_gap: int,
    rng: random.Random,
    hardness_mode: str,
) -> GapPair:
    if hardness_mode == "comparative" and hard_pool:
        pair = rng.choice(hard_pool)
        if pair.d_true <= max_gap and pair.d_confuse <= max_gap:
            return pair

        # fall back: resample from valid subset
        valid = [x for x in hard_pool if x.d_true <= max_gap and x.d_confuse <= max_gap]
        if valid:
            return rng.choice(valid)

    # random-gap fallback
    d_true = rng.randint(16, max(24, int(0.80 * max_gap)))
    delta = rng.randint(8, max(12, int(0.20 * max_gap)))
    candidates = []
    if d_true + delta <= max_gap:
        candidates.append(d_true + delta)
    if d_true - delta >= 8:
        candidates.append(d_true - delta)
    d_confuse = rng.choice(candidates) if candidates else max(8, d_true - 8)
    return GapPair(d_true=d_true, d_confuse=d_confuse, hardness=0.0)


def find_query_and_fact_positions(
    *,
    seq_len: int,
    d_true: int,
    fact_chunk_len: int,
    distractor_chunk_len: int,
    query_chunk_len: int,
    evidence_bin: str,
    rng: random.Random,
    max_tries: int = 64,
) -> Optional[Tuple[int, int, int]]:
    """
    Solve for:
      q_start - fact_start ~= d_true
      q_start - distractor_start ~= d_confuse (handled outside)
    Here we only place the fact and query; distractor can be placed separately.
    """
    for _ in range(max_tries):
        fact_start = sample_evidence_anchor(seq_len, evidence_bin, fact_chunk_len, rng)
        q_start = fact_start + d_true
        if q_start + query_chunk_len >= seq_len:
            continue
        if q_start <= fact_start + fact_chunk_len:
            continue
        return fact_start, q_start, q_start - fact_start
    return None


def candidate_starts_at_distance(
    *,
    q_start: int,
    d_confuse: int,
    chunk_len: int,
    seq_len: int,
) -> List[int]:
    cands = []
    left = q_start - d_confuse
    if 0 <= left and left + chunk_len <= seq_len:
        cands.append(left)
    right = q_start + d_confuse
    if 0 <= right and right + chunk_len <= seq_len:
        cands.append(right)
    return cands


# ----------------------------
# Task generators
# ----------------------------

def gen_single_hop(
    *,
    lex: AliasLexicon,
    seq_len: int,
    split: str,
    hard_pool: Optional[Sequence[GapPair]],
    hardness_mode: str,
    rng: random.Random,
) -> Dict:
    canvas = init_token_canvas(seq_len, rng)
    used = reserve_mask(seq_len)

    key_id = rng.randint(0, len(lex.key_aliases) - 1)
    val_id = rng.randint(0, len(lex.val_aliases) - 1)
    distractor_key_id = (key_id + rng.randint(1, 97)) % len(lex.key_aliases)
    distractor_val_id = (val_id + rng.randint(1, 89)) % len(lex.val_aliases)

    fact_key = lex.key_base(key_id)
    query_key = lex.key_base(key_id) if rng.random() < 0.35 else lex.key_alias(key_id)
    fact_val = lex.val_base(val_id)
    distractor_key = lex.key_base(distractor_key_id)
    distractor_val = lex.val_base(distractor_val_id)

    fact_chunk = ["[FACT]", fact_key, "->", fact_val, "."]
    distractor_chunk = ["[FACT]", distractor_key, "->", distractor_val, "."]
    query_chunk = ["[QUERY]", query_key, "->", "?", "[ANSWER]", fact_val]

    max_gap = max(64, seq_len - max(len(fact_chunk), len(query_chunk)) - 4)
    gp = choose_gap_pair(hard_pool=hard_pool, max_gap=max_gap, rng=rng, hardness_mode=hardness_mode)

    evidence_bin = choose_position_bin(rng, split)
    fq = find_query_and_fact_positions(
        seq_len=seq_len,
        d_true=gp.d_true,
        fact_chunk_len=len(fact_chunk),
        distractor_chunk_len=len(distractor_chunk),
        query_chunk_len=len(query_chunk),
        evidence_bin=evidence_bin,
        rng=rng,
    )
    if fq is None:
        # degrade gracefully: place fact near head, query near tail
        fact_start = max(0, seq_len // 8)
        q_start = min(seq_len - len(query_chunk) - 1, fact_start + min(gp.d_true, seq_len // 2))
        true_gap = q_start - fact_start
    else:
        fact_start, q_start, true_gap = fq

    if not place_chunk(canvas, used, fact_start, fact_chunk):
        raise RuntimeError("Failed to place single-hop fact chunk.")
    if not place_chunk(canvas, used, q_start, query_chunk):
        raise RuntimeError("Failed to place single-hop query chunk.")

    cands = candidate_starts_at_distance(
        q_start=q_start,
        d_confuse=gp.d_confuse,
        chunk_len=len(distractor_chunk),
        seq_len=seq_len,
    )
    rng.shuffle(cands)
    distractor_start = None
    for c in cands:
        if c == fact_start:
            continue
        if place_chunk(canvas, used, c, distractor_chunk):
            distractor_start = c
            break

    if distractor_start is None:
        # random fallback placement
        for _ in range(128):
            c = rng.randint(0, seq_len - len(distractor_chunk))
            if c == fact_start or c == q_start:
                continue
            if place_chunk(canvas, used, c, distractor_chunk):
                distractor_start = c
                break

    if distractor_start is None:
        raise RuntimeError("Failed to place single-hop distractor chunk.")

    return {
        "text_tokens": canvas,
        "answer": fact_val,
        "task": "single_hop",
        "meta": {
            "seq_len": seq_len,
            "evidence_bin": evidence_bin,
            "d_true_target": gp.d_true,
            "d_confuse_target": gp.d_confuse,
            "d_true_actual": abs(q_start - fact_start),
            "d_confuse_actual": abs(q_start - distractor_start),
            "hardness": gp.hardness,
            "query_uses_alias": query_key != fact_key,
            "fact_start": fact_start,
            "distractor_start": distractor_start,
            "query_start": q_start,
        },
    }


def gen_alias_gap(
    *,
    lex: AliasLexicon,
    seq_len: int,
    split: str,
    hard_pool: Optional[Sequence[GapPair]],
    hardness_mode: str,
    rng: random.Random,
) -> Dict:
    canvas = init_token_canvas(seq_len, rng)
    used = reserve_mask(seq_len)

    key_id = rng.randint(0, len(lex.key_aliases) - 1)
    val_id = rng.randint(0, len(lex.val_aliases) - 1)
    distractor_key_id = (key_id + rng.randint(1, 53)) % len(lex.key_aliases)
    distractor_val_id = (val_id + rng.randint(1, 47)) % len(lex.val_aliases)

    # low lexical overlap by forcing aliases on both sides
    fact_key = lex.key_alias(key_id)
    query_key = lex.key_base(key_id)
    fact_val = lex.val_alias(val_id)
    distractor_key = lex.key_alias(distractor_key_id)
    distractor_val = lex.val_alias(distractor_val_id)

    fact_chunk = ["[MEM]", fact_key, "means", fact_val, "."]
    distractor_chunk = ["[MEM]", distractor_key, "means", distractor_val, "."]
    query_chunk = ["[ASK]", query_key, "maps_to", "?", "[ANSWER]", fact_val]

    max_gap = max(64, seq_len - max(len(fact_chunk), len(query_chunk)) - 4)
    gp = choose_gap_pair(hard_pool=hard_pool, max_gap=max_gap, rng=rng, hardness_mode=hardness_mode)

    evidence_bin = choose_position_bin(rng, split)
    fq = find_query_and_fact_positions(
        seq_len=seq_len,
        d_true=gp.d_true,
        fact_chunk_len=len(fact_chunk),
        distractor_chunk_len=len(distractor_chunk),
        query_chunk_len=len(query_chunk),
        evidence_bin=evidence_bin,
        rng=rng,
    )
    if fq is None:
        fact_start = max(0, seq_len // 10)
        q_start = min(seq_len - len(query_chunk) - 1, fact_start + min(gp.d_true, seq_len // 2))
    else:
        fact_start, q_start, _ = fq

    if not place_chunk(canvas, used, fact_start, fact_chunk):
        raise RuntimeError("Failed to place alias-gap fact chunk.")
    if not place_chunk(canvas, used, q_start, query_chunk):
        raise RuntimeError("Failed to place alias-gap query chunk.")

    cands = candidate_starts_at_distance(
        q_start=q_start,
        d_confuse=gp.d_confuse,
        chunk_len=len(distractor_chunk),
        seq_len=seq_len,
    )
    rng.shuffle(cands)
    distractor_start = None
    for c in cands:
        if c == fact_start:
            continue
        if place_chunk(canvas, used, c, distractor_chunk):
            distractor_start = c
            break

    if distractor_start is None:
        for _ in range(128):
            c = rng.randint(0, seq_len - len(distractor_chunk))
            if c in {fact_start, q_start}:
                continue
            if place_chunk(canvas, used, c, distractor_chunk):
                distractor_start = c
                break

    if distractor_start is None:
        raise RuntimeError("Failed to place alias-gap distractor chunk.")

    return {
        "text_tokens": canvas,
        "answer": fact_val,
        "task": "alias_gap",
        "meta": {
            "seq_len": seq_len,
            "evidence_bin": evidence_bin,
            "d_true_target": gp.d_true,
            "d_confuse_target": gp.d_confuse,
            "d_true_actual": abs(q_start - fact_start),
            "d_confuse_actual": abs(q_start - distractor_start),
            "hardness": gp.hardness,
            "fact_surface": fact_key,
            "query_surface": query_key,
            "fact_start": fact_start,
            "distractor_start": distractor_start,
            "query_start": q_start,
        },
    }


def gen_multi_hop(
    *,
    seq_len: int,
    split: str,
    rng: random.Random,
    hops: int,
) -> Dict:
    """
    Scattered chain tracing:
      node_a -> node_b
      node_b -> node_c
      ...
      node_z -> attr_foo
      query(node_a) => attr_foo

    This task is intentionally less tied to comparative hardness than single_hop,
    but stresses long-range scattered composition.
    """
    if hops < 2:
        raise ValueError("hops must be >= 2")

    canvas = init_token_canvas(seq_len, rng)
    used = reserve_mask(seq_len)

    node_ids = [rng.randint(0, 999_999) for _ in range(hops + 1)]
    start_alias = f"alias_node_{node_ids[0]}"
    final_attr = f"attr_{rng.randint(0, 99999)}"

    link_chunks: List[List[str]] = []
    for i in range(hops):
        link_chunks.append(["[LINK]", f"node_{node_ids[i]}", "->", f"node_{node_ids[i+1]}", "."])
    attr_chunk = ["[ATTR]", f"node_{node_ids[-1]}", "->", final_attr, "."]
    query_chunk = ["[QUERY]", start_alias, "final_attr", "?", "[ANSWER]", final_attr]

    evidence_bin = choose_position_bin(rng, split)
    # Scatter link chunks with a bias toward larger separations
    # Place the first link roughly according to evidence bin, and the query near the end.
    q_start = min(seq_len - len(query_chunk) - 1, max(0, int(0.85 * seq_len)))
    if not place_chunk(canvas, used, q_start, query_chunk):
        raise RuntimeError("Failed to place multi-hop query chunk.")

    first_link_start = sample_evidence_anchor(seq_len, evidence_bin, len(link_chunks[0]), rng)
    placed_positions: List[int] = []

    if not place_chunk(canvas, used, first_link_start, link_chunks[0]):
        # fallback search
        ok = False
        for _ in range(128):
            s = rng.randint(0, seq_len - len(link_chunks[0]))
            if place_chunk(canvas, used, s, link_chunks[0]):
                first_link_start = s
                ok = True
                break
        if not ok:
            raise RuntimeError("Failed to place multi-hop first link.")
    placed_positions.append(first_link_start)

    # Subsequent links: progressively later but still before the query
    prev = first_link_start
    for chunk in link_chunks[1:]:
        success = False
        for _ in range(256):
            lo = min(seq_len - len(chunk) - 1, prev + max(8, seq_len // (2 * hops)))
            hi = max(lo, q_start - len(chunk) - 8)
            if hi < lo:
                lo, hi = 0, max(0, q_start - len(chunk) - 8)
            s = rng.randint(lo, hi) if hi > lo else lo
            if place_chunk(canvas, used, s, chunk):
                prev = s
                placed_positions.append(s)
                success = True
                break
        if not success:
            # fallback random search
            for _ in range(256):
                s = rng.randint(0, max(0, q_start - len(chunk) - 1))
                if place_chunk(canvas, used, s, chunk):
                    prev = s
                    placed_positions.append(s)
                    success = True
                    break
        if not success:
            raise RuntimeError("Failed to place a multi-hop link chunk.")

    attr_start = None
    for _ in range(256):
        s = rng.randint(0, max(0, q_start - len(attr_chunk) - 1))
        if place_chunk(canvas, used, s, attr_chunk):
            attr_start = s
            break
    if attr_start is None:
        raise RuntimeError("Failed to place multi-hop attr chunk.")

    # Add one distractor edge from the start node to a fake node
    distractor_chunk = ["[LINK]", f"node_{node_ids[0]}", "->", f"node_fake_{rng.randint(0, 99999)}", "."]
    distractor_start = None
    for _ in range(256):
        s = rng.randint(0, max(0, q_start - len(distractor_chunk) - 1))
        if place_chunk(canvas, used, s, distractor_chunk):
            distractor_start = s
            break
    if distractor_start is None:
        raise RuntimeError("Failed to place multi-hop distractor link.")

    return {
        "text_tokens": canvas,
        "answer": final_attr,
        "task": "multi_hop",
        "meta": {
            "seq_len": seq_len,
            "evidence_bin": evidence_bin,
            "hops": hops,
            "first_link_start": first_link_start,
            "link_positions": placed_positions,
            "attr_start": attr_start,
            "distractor_start": distractor_start,
            "query_start": q_start,
            "query_uses_alias": True,
        },
    }


def gen_count(
    *,
    seq_len: int,
    split: str,
    rng: random.Random,
    n_tags: int = 32,
) -> Dict:
    """
    Long-range aggregation/counting.
    The answer is the number of occurrences of a target tag across a long context.
    """
    canvas = init_token_canvas(seq_len, rng)
    used = reserve_mask(seq_len)

    target_tag = f"tag_{rng.randint(0, n_tags - 1)}"
    query_chunk_placeholder = ["[QUERY]", "count", target_tag, "?", "[ANSWER]", "0"]
    q_start = min(seq_len - len(query_chunk_placeholder) - 1, max(0, int(0.88 * seq_len)))
    query_reserved = ["[QUERY]", "count", target_tag, "?", "[ANSWER]", "__COUNT__"]
    if not place_chunk(canvas, used, q_start, query_reserved):
        raise RuntimeError("Failed to place count query chunk.")

    # Place many small items before the query
    count = 0
    item_positions: List[int] = []
    target_positions: List[int] = []

    # density scaled by length; enough events to make counting nontrivial
    n_items = max(32, seq_len // 48)
    for _ in range(n_items):
        tag = f"tag_{rng.randint(0, n_tags - 1)}"
        val = str(rng.randint(0, 999))
        chunk = ["[ITEM]", tag, "=", val, "."]
        placed = False
        for _ in range(128):
            s = rng.randint(0, max(0, q_start - len(chunk) - 1))
            if place_chunk(canvas, used, s, chunk):
                item_positions.append(s)
                if tag == target_tag:
                    count += 1
                    target_positions.append(s)
                placed = True
                break
        if not placed:
            # skip if the canvas is too crowded
            continue

    # Patch the answer token in place
    answer_str = str(count)
    answer_patch = ["[QUERY]", "count", target_tag, "?", "[ANSWER]", answer_str]
    for i, tok in enumerate(answer_patch):
        canvas[q_start + i] = tok

    return {
        "text_tokens": canvas,
        "answer": answer_str,
        "task": "count",
        "meta": {
            "seq_len": seq_len,
            "split": split,
            "query_start": q_start,
            "target_tag": target_tag,
            "true_count": count,
            "n_items_attempted": n_items,
            "n_items_placed": len(item_positions),
            "target_positions": target_positions[:32],  # truncate metadata a bit
        },
    }


# ----------------------------
# Generation orchestration
# ----------------------------

def make_sample(
    *,
    task_name: str,
    lex: AliasLexicon,
    seq_len: int,
    split: str,
    hard_pool: Optional[Sequence[GapPair]],
    hardness_mode: str,
    rng: random.Random,
    multi_hop_range: Tuple[int, int],
) -> Dict:
    if task_name == "single_hop":
        return gen_single_hop(
            lex=lex,
            seq_len=seq_len,
            split=split,
            hard_pool=hard_pool,
            hardness_mode=hardness_mode,
            rng=rng,
        )
    if task_name == "alias_gap":
        return gen_alias_gap(
            lex=lex,
            seq_len=seq_len,
            split=split,
            hard_pool=hard_pool,
            hardness_mode=hardness_mode,
            rng=rng,
        )
    if task_name == "multi_hop":
        hops = rng.randint(multi_hop_range[0], multi_hop_range[1])
        return gen_multi_hop(
            seq_len=seq_len,
            split=split,
            rng=rng,
            hops=hops,
        )
    if task_name == "count":
        return gen_count(
            seq_len=seq_len,
            split=split,
            rng=rng,
        )
    raise ValueError(f"Unknown task: {task_name}")


def stream_jsonl(
    *,
    output_path: Path,
    split: str,
    target_tokens: int,
    length_choices: Sequence[int],
    task_weights: TaskWeights,
    lex: AliasLexicon,
    hard_pool: Optional[Sequence[GapPair]],
    hardness_mode: str,
    rng: random.Random,
    multi_hop_range: Tuple[int, int],
    flush_every: int = 100,
) -> Dict:
    task_items = task_weights.normalized_items()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    emitted_tokens = 0
    task_counter: Dict[str, int] = {k: 0 for k, _ in task_items}
    length_counter: Dict[int, int] = {int(x): 0 for x in length_choices}
    hardness_values: List[float] = []

    with output_path.open("w", encoding="utf-8") as f:
        while emitted_tokens < target_tokens:
            seq_len = choose_length(rng, length_choices)
            task_name = weighted_choice(task_items, rng=rng)

            sample = make_sample(
                task_name=task_name,
                lex=lex,
                seq_len=seq_len,
                split=split,
                hard_pool=hard_pool,
                hardness_mode=hardness_mode,
                rng=rng,
                multi_hop_range=multi_hop_range,
            )
            tokens = sample["text_tokens"]
            row = {
                "id": f"{split}_{task_name}_{n_rows:09d}",
                "text": " ".join(tokens),
                "answer": sample["answer"],
                "task": sample["task"],
                "split": split,
                "meta": sample["meta"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_rows += 1
            emitted_tokens += len(tokens)
            task_counter[task_name] += 1
            length_counter[seq_len] += 1
            if "hardness" in sample["meta"]:
                hardness_values.append(float(sample["meta"]["hardness"]))

            if flush_every and n_rows % flush_every == 0:
                f.flush()

    stats = {
        "output_path": str(output_path),
        "split": split,
        "rows": n_rows,
        "emitted_tokens": emitted_tokens,
        "task_counter": task_counter,
        "length_counter": {str(k): v for k, v in sorted(length_counter.items())},
        "hardness_mean": statistics.mean(hardness_values) if hardness_values else None,
        "hardness_median": statistics.median(hardness_values) if hardness_values else None,
    }
    return stats


def parse_int_list(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer.")
    return vals


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_save_gap_pool(path: Path, gap_pool: Optional[Sequence[GapPair]]) -> None:
    if gap_pool is None:
        return
    payload = [
        {"d_true": x.d_true, "d_confuse": x.d_confuse, "hardness": x.hardness}
        for x in gap_pool
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_budgets(total_tokens: int, val_ratio: float, test_ratio: float) -> List[SplitBudget]:
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")
    test_toks = int(total_tokens * test_ratio)
    val_toks = int(total_tokens * val_ratio)
    train_toks = total_tokens - test_toks - val_toks
    return [
        SplitBudget("train", train_toks),
        SplitBudget("val", val_toks),
        SplitBudget("test", test_toks),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate mechanism-driven synthetic long-context data.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write JSONL files and metadata.")
    p.add_argument("--target-tokens", type=int, default=50_000_000, help="Total token budget across train/val/test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-dims", type=int, default=64, help="RoPE pair dimensions (head_dim/2 style count).")
    p.add_argument("--rope-theta", type=float, default=500_000.0, help="Base geometric RoPE theta.")
    p.add_argument("--alt-spec-json", type=Path, default=None, help="JSON file containing alternative inv_freq.")
    p.add_argument(
        "--hardness-mode",
        type=str,
        choices=["random", "comparative"],
        default="comparative",
        help="comparative = use base-vs-alt hard gap sampling if alt spec is given; random = ignore comparative hardness",
    )
    p.add_argument("--hard-gap-top-quantile", type=float, default=0.10)
    p.add_argument("--hard-gap-samples-per-d", type=int, default=24)
    p.add_argument("--hard-gap-max-len", type=int, default=32768, help="Max distance scanned when building hard-gap pool.")
    p.add_argument("--n-keys", type=int, default=50000)
    p.add_argument("--n-vals", type=int, default=50000)
    p.add_argument("--alias-per-key", type=int, default=3)
    p.add_argument("--alias-per-val", type=int, default=2)
    p.add_argument("--train-lengths", type=str, default="512,1024,2048")
    p.add_argument("--val-lengths", type=str, default="2048")
    p.add_argument("--test-lengths", type=str, default="2048,4096,8192,16384,32768")
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--test-ratio", type=float, default=0.01)
    p.add_argument("--w-single-hop", type=float, default=0.40)
    p.add_argument("--w-multi-hop", type=float, default=0.25)
    p.add_argument("--w-count", type=float, default=0.20)
    p.add_argument("--w-alias-gap", type=float, default=0.15)
    p.add_argument("--multi-hop-min", type=int, default=2)
    p.add_argument("--multi-hop-max", type=int, default=4)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--num-workers-note", type=str, default="single-process", help="Purely informational metadata flag.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    train_lengths = parse_int_list(args.train_lengths)
    val_lengths = parse_int_list(args.val_lengths)
    test_lengths = parse_int_list(args.test_lengths)

    if args.multi_hop_min > args.multi_hop_max:
        raise ValueError("--multi-hop-min must be <= --multi-hop-max")

    base_spec = RopeSpec(
        name="geometric",
        inv_freq=geometric_inv_freq(dim_pairs=args.num_dims, rope_theta=args.rope_theta),
    )

    alt_spec: Optional[RopeSpec] = None
    if args.alt_spec_json is not None:
        alt_spec = RopeSpec(
            name=args.alt_spec_json.stem,
            inv_freq=load_inv_freq_json(args.alt_spec_json),
        )
        if len(alt_spec.inv_freq) != len(base_spec.inv_freq):
            raise ValueError(
                f"Alternative inv_freq length {len(alt_spec.inv_freq)} does not match base length {len(base_spec.inv_freq)}."
            )

    hardness_mode = args.hardness_mode
    if hardness_mode == "comparative" and alt_spec is None:
        print(
            "[warn] --hardness-mode comparative requested but no --alt-spec-json was provided; "
            "falling back to random-gap mode.",
            file=sys.stderr,
        )
        hardness_mode = "random"

    hard_pool: Optional[List[GapPair]] = None
    if hardness_mode == "comparative" and alt_spec is not None:
        gap_rng = random.Random(args.seed + 1337)
        print("[info] Building hard-gap pool ...", file=sys.stderr)
        hard_pool = build_hard_gap_pool(
            base=base_spec,
            alt=alt_spec,
            max_len=args.hard_gap_max_len,
            sample_pairs_per_d=args.hard_gap_samples_per_d,
            top_quantile=args.hard_gap_top_quantile,
            rng=gap_rng,
        )
        print(f"[info] Hard-gap pool size: {len(hard_pool)}", file=sys.stderr)

    lex = AliasLexicon(
        n_keys=args.n_keys,
        n_vals=args.n_vals,
        alias_per_key=args.alias_per_key,
        alias_per_val=args.alias_per_val,
        rng=random.Random(args.seed + 7),
    )

    task_weights = TaskWeights(
        single_hop=args.w_single_hop,
        multi_hop=args.w_multi_hop,
        count=args.w_count,
        alias_gap=args.w_alias_gap,
    )

    meta = {
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "num_dims": args.num_dims,
        "rope_theta": args.rope_theta,
        "base_spec": {
            "name": base_spec.name,
            "inv_freq_head": base_spec.inv_freq[:8],
            "n_inv_freq": len(base_spec.inv_freq),
        },
        "alt_spec": None if alt_spec is None else {
            "name": alt_spec.name,
            "inv_freq_head": alt_spec.inv_freq[:8],
            "n_inv_freq": len(alt_spec.inv_freq),
        },
        "hardness_mode": hardness_mode,
        "hard_gap_top_quantile": args.hard_gap_top_quantile,
        "hard_gap_samples_per_d": args.hard_gap_samples_per_d,
        "hard_gap_max_len": args.hard_gap_max_len,
        "length_choices": {
            "train": train_lengths,
            "val": val_lengths,
            "test": test_lengths,
        },
        "task_weights": asdict(task_weights),
        "multi_hop_range": [args.multi_hop_min, args.multi_hop_max],
        "generator_note": "single-file mechanism-driven synthetic long-context generator",
        "num_workers_note": args.num_workers_note,
    }
    save_json(output_dir / "config.json", meta)
    maybe_save_gap_pool(output_dir / "hard_gap_pool.json", hard_pool)

    budgets = split_budgets(args.target_tokens, args.val_ratio, args.test_ratio)

    split_stats: List[Dict] = []
    for sb in budgets:
        if sb.target_tokens <= 0:
            continue
        if sb.split == "train":
            length_choices = train_lengths
        elif sb.split == "val":
            length_choices = val_lengths
        elif sb.split == "test":
            length_choices = test_lengths
        else:
            raise ValueError(f"Unknown split: {sb.split}")

        print(
            f"[info] Generating {sb.split}: target_tokens={sb.target_tokens}, lengths={length_choices}",
            file=sys.stderr,
        )
        stats = stream_jsonl(
            output_path=output_dir / f"{sb.split}.jsonl",
            split=sb.split,
            target_tokens=sb.target_tokens,
            length_choices=length_choices,
            task_weights=task_weights,
            lex=lex,
            hard_pool=hard_pool,
            hardness_mode=hardness_mode,
            rng=random.Random(args.seed + hash(sb.split) % 10_000),
            multi_hop_range=(args.multi_hop_min, args.multi_hop_max),
            flush_every=args.flush_every,
        )
        split_stats.append(stats)
        print(
            f"[info] Done {sb.split}: rows={stats['rows']}, emitted_tokens={stats['emitted_tokens']}",
            file=sys.stderr,
        )

    summary = {
        "config_path": str(output_dir / "config.json"),
        "hard_gap_pool_path": str(output_dir / "hard_gap_pool.json") if hard_pool is not None else None,
        "splits": split_stats,
    }
    save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
