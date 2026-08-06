#!/usr/bin/env python3
"""Layer-resolved causal readout decomposition of the paired gold-drop traces.

Consumes the frozen 16K logit-lens tensors from
``results/readout_conversion_s42_20260715`` (dense vs ``gold_drop_all``) and the
per-layer/per-head QK probe raw from ``results/lora_sparse_conversion_s42_20260714``.

Sign convention throughout: ``delta = z_full - z_ablated``.  Positive delta on the
gold token means the gold block *supports* that token (deleting the block lowers it).

Every vocabulary-wide statistic is computed in float32 after an explicit bf16 cast.
All distributional statistics (KL, ranks, probabilities) are invariant to the
additive per-layer basis shift that the logit lens is known to introduce; the raw
centred delta is reported alongside them so that basis-sensitive and basis-robust
readings can be compared directly.

Read-only with respect to the repository: loads no model, uses no GPU, downloads
nothing, and writes only inside this directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controls import ARMS, PHASE0_ROOT, TRACE_ROOT, run_controls

OUT_DIR = Path(__file__).resolve().parent
N_LAYERS = 32
N_POS = 3
TOP_K = 32
MAD_SCALE = 1.4826  # MAD -> sigma for a normal reference


# ----------------------------------------------------------------------------
# per-record vocabulary-wide statistics
# ----------------------------------------------------------------------------
def record_metrics(path: Path) -> dict[str, Any]:
    """All per-(position, layer) statistics for one trace record."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    gold_ids = blob["gold_token_ids"].long().tolist()
    full = blob["full_logits"].float()  # [pos, layer, vocab]
    abl = blob["ablated_logits"].float()
    delta = full - abl

    med = delta.median(dim=-1, keepdim=True).values  # [pos, layer, 1]
    mad = (delta - med).abs().median(dim=-1, keepdim=True).values
    centred = delta - med

    gold_idx = torch.tensor(gold_ids).view(N_POS, 1, 1).expand(N_POS, N_LAYERS, 1)
    d_gold = delta.gather(-1, gold_idx).squeeze(-1)  # [pos, layer]
    d_gold_centred = centred.gather(-1, gold_idx).squeeze(-1)
    sigma = (MAD_SCALE * mad).squeeze(-1)
    d_gold_z = d_gold_centred / sigma.clamp_min(1e-12)

    # registered rank rule: 1 + count(delta > delta[gold])
    delta_rank = (delta > d_gold.unsqueeze(-1)).sum(dim=-1) + 1

    # magnitude of the whole causal effect, basis-robust and basis-sensitive
    l2_centred = centred.norm(dim=-1)
    logp_full = torch.log_softmax(full, dim=-1)
    logp_abl = torch.log_softmax(abl, dim=-1)
    p_full = logp_full.exp()
    p_abl = logp_abl.exp()
    kl_full_abl = (p_full * (logp_full - logp_abl)).sum(dim=-1)
    kl_abl_full = (p_abl * (logp_abl - logp_full)).sum(dim=-1)
    tvd = 0.5 * (p_full - p_abl).abs().sum(dim=-1)

    # readout-space position of the gold token
    z_gold_full = full.gather(-1, gold_idx).squeeze(-1)
    z_gold_abl = abl.gather(-1, gold_idx).squeeze(-1)
    rank_full = (full > z_gold_full.unsqueeze(-1)).sum(dim=-1) + 1
    rank_abl = (abl > z_gold_abl.unsqueeze(-1)).sum(dim=-1) + 1
    lp_gold_full = logp_full.gather(-1, gold_idx).squeeze(-1)
    lp_gold_abl = logp_abl.gather(-1, gold_idx).squeeze(-1)

    # gold vs strongest competitor (competitor = argmax over v != gold)
    full_masked = full.scatter(-1, gold_idx, float("-inf"))
    abl_masked = abl.scatter(-1, gold_idx, float("-inf"))
    best_full_val, best_full_id = full_masked.max(dim=-1)
    best_abl_val, best_abl_id = abl_masked.max(dim=-1)
    gap_full = best_full_val - z_gold_full  # >0 means gold loses
    gap_abl = best_abl_val - z_gold_abl

    out = {
        "prompt_sha256": blob["prompt_sha256"],
        "substrate": blob["substrate"],
        "depth_percent": float(blob["depth_percent"]),
        "target_length": int(blob["target_length"]),
        "gold_token_ids": gold_ids,
        "parity_max_abs": float(blob["final_logit_parity_max_abs"]),
        "per_layer": {
            "delta_gold_raw": d_gold.numpy(),
            "delta_gold_centred": d_gold_centred.numpy(),
            "delta_median_vocab": med.squeeze(-1).numpy(),
            "delta_mad_vocab": mad.squeeze(-1).numpy(),
            "delta_gold_z": d_gold_z.numpy(),
            "causal_delta_rank": delta_rank.numpy(),
            "l2_centred": l2_centred.numpy(),
            "kl_full_abl": kl_full_abl.numpy(),
            "kl_abl_full": kl_abl_full.numpy(),
            "tvd": tvd.numpy(),
            "gold_rank_full": rank_full.numpy(),
            "gold_rank_ablated": rank_abl.numpy(),
            "gold_logprob_full": lp_gold_full.numpy(),
            "gold_logprob_ablated": lp_gold_abl.numpy(),
            "gold_logit_full": z_gold_full.numpy(),
            "gold_logit_ablated": z_gold_abl.numpy(),
            "gap_to_best_competitor_full": gap_full.numpy(),
            "gap_to_best_competitor_ablated": gap_abl.numpy(),
            "best_competitor_id_full": best_full_id.numpy(),
            "best_competitor_id_ablated": best_abl_id.numpy(),
        },
    }

    # final-layer (true model output) top-K competition detail, all positions
    final_detail = []
    for t in range(N_POS):
        zf = full[t, N_LAYERS - 1]
        za = abl[t, N_LAYERS - 1]
        cen = centred[t, N_LAYERS - 1]
        top_val, top_id = zf.topk(TOP_K)
        final_detail.append(
            {
                "position": t,
                "gold_id": gold_ids[t],
                "top_ids": top_id.tolist(),
                "top_logits_full": [round(v, 4) for v in top_val.tolist()],
                "top_logits_ablated": [round(v, 4) for v in za[top_id].tolist()],
                "top_delta_centred": [round(v, 4) for v in cen[top_id].tolist()],
                "gold_delta_centred": float(cen[gold_ids[t]]),
            }
        )
    out["final_layer_topk"] = final_detail

    # distribution shape, needed to tell "late layers attenuate the gold signal"
    # apart from "late layers sharpen onto a competitor so a fixed logit delta
    # stops mattering".  Both produce a falling KL; only the first is attenuation.
    entropy = -(p_full * logp_full).sum(dim=-1)
    out["per_layer"]["entropy_full"] = entropy.numpy()
    out["per_layer"]["top1_prob_full"] = p_full.max(dim=-1).values.numpy()
    out["per_layer"]["gold_prob_full"] = p_full.gather(-1, gold_idx).squeeze(-1).numpy()

    del blob, full, abl, delta, centred, logp_full, logp_abl, p_full, p_abl
    return out


# ----------------------------------------------------------------------------
# entry layer / retention classification
# ----------------------------------------------------------------------------
def entry_layer_audit_rule(curve: np.ndarray, min_history: int = 4) -> int | None:
    """l* = min{l : d_l > 3 x MAD(d_0..d_{l-1})}, audit specification."""
    for l in range(min_history, len(curve)):
        hist = curve[:l]
        mad = float(np.median(np.abs(hist - np.median(hist))))
        thresh = np.median(hist) + 3.0 * MAD_SCALE * max(mad, 1e-9)
        if curve[l] > thresh and curve[l] > 0:
            return l
    return None


def entry_layer_vocab_rule(zcurve: np.ndarray, thresh: float = 3.0) -> int | None:
    """First layer whose gold delta exceeds 3 robust sigma of that layer's own
    vocabulary-wide delta distribution.  History-free, self-normalising."""
    for l in range(len(zcurve)):
        if zcurve[l] > thresh:
            return l
    return None


def entry_layer_fraction_rule(curve: np.ndarray, frac: float = 0.1) -> int | None:
    """First layer reaching `frac` of the curve's own maximum (onset measure)."""
    peak = float(curve.max())
    if peak <= 0:
        return None
    for l in range(len(curve)):
        if curve[l] >= frac * peak:
            return l
    return None


# ----------------------------------------------------------------------------
# phase0 per-layer QK profile
# ----------------------------------------------------------------------------
def phase0_layer_profile(
    arm_file: str, prompt_shas: set[str], length: int, frozen_heads: dict[int, list[int]]
) -> dict[str, Any]:
    blob = json.loads((PHASE0_ROOT / arm_file).read_text(encoding="utf-8"))
    entries = [
        e
        for e in blob["results"]
        if int(e["target_length"]) == length
        and (not prompt_shas or e["prompt_sha256"] in prompt_shas)
    ]
    per_case: dict[str, dict[str, np.ndarray]] = {}
    for e in entries:
        hit_all = np.zeros(N_LAYERS)
        hit_frozen = np.full(N_LAYERS, np.nan)
        mass_all = np.zeros(N_LAYERS)
        mass_frozen = np.full(N_LAYERS, np.nan)
        brank_all = np.zeros(N_LAYERS)
        brank_frozen = np.full(N_LAYERS, np.nan)
        for layer_entry in e["layers"]:
            l = int(layer_entry["layer"])
            hits = np.asarray(layer_entry["hit_at"]["16"], dtype=float)
            mass = np.asarray(layer_entry["dense_answer_mass"], dtype=float)
            brank = np.asarray(layer_entry["block_rank"], dtype=float)
            hit_all[l] = hits.mean()
            mass_all[l] = mass.mean()
            brank_all[l] = np.median(brank)
            heads = frozen_heads.get(l)
            if heads:
                hit_frozen[l] = hits[heads].mean()
                mass_frozen[l] = mass[heads].mean()
                brank_frozen[l] = np.median(brank[heads])
        per_case[e["prompt_sha256"]] = {
            "example_id": e["example_id"],
            "hit16_all_heads": hit_all,
            "hit16_frozen_heads": hit_frozen,
            "dense_answer_mass_all_heads": mass_all,
            "dense_answer_mass_frozen_heads": mass_frozen,
            "block_rank_median_all_heads": brank_all,
            "block_rank_median_frozen_heads": brank_frozen,
        }
    return per_case


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata

    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------------
def main() -> int:
    controls = run_controls(verbose=False)
    if not controls["gate_passed"]:
        print("CONTROL GATE FAILED — refusing to run the decomposition.")
        for f in controls["hard_failures"]:
            print("  FAIL", f)
        return 1
    (OUT_DIR / "controls.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pairs = controls["matched_pairs"]
    print(f"gate passed; decomposing {len(pairs)} matched pairs + geo-only controls")

    # ---------------- load every present record -------------------------
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for arm, folder in ARMS.items():
        manifest = json.loads((TRACE_ROOT / folder / "manifest.json").read_text())
        for rec in manifest["records"]:
            path = TRACE_ROOT / folder / rec["file"]
            if not path.is_file():
                continue
            loaded[(arm, rec["prompt_sha256"])] = record_metrics(path)
            print(f"  loaded {arm:>3} {rec['file'].split('/')[-1][:3]} depth={rec['depth_percent']}")

    # ---------------- long-format per-layer CSV -------------------------
    layer_fields = list(next(iter(loaded.values()))["per_layer"].keys())
    rows: list[dict[str, Any]] = []
    for (arm, sha), rec in loaded.items():
        paired = any(p["prompt_sha256"] == sha for p in pairs)
        for t in range(N_POS):
            for l in range(N_LAYERS):
                row = {
                    "arm": arm,
                    "prompt_sha256": sha,
                    "prompt_sha8": sha[:8],
                    "depth_percent": rec["depth_percent"],
                    "matched_pair": int(paired),
                    "answer_position": t,
                    "layer": l,
                    "gold_token_id": rec["gold_token_ids"][t],
                }
                for f in layer_fields:
                    val = rec["per_layer"][f][t, l]
                    row[f] = int(val) if np.issubdtype(val.dtype, np.integer) else float(val)
                rows.append(row)
    csv_path = OUT_DIR / "per_layer_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path.name} ({len(rows)} rows)")

    # ---------------- per-case classification ---------------------------
    per_case: list[dict[str, Any]] = []
    for (arm, sha), rec in loaded.items():
        paired = any(p["prompt_sha256"] == sha for p in pairs)
        for t in range(N_POS):
            centred = rec["per_layer"]["delta_gold_centred"][t]
            zc = rec["per_layer"]["delta_gold_z"][t]
            kl = rec["per_layer"]["kl_full_abl"][t]
            peak = int(np.argmax(centred))
            final = float(centred[-1])
            maxv = float(centred.max())
            per_case.append(
                {
                    "arm": arm,
                    "prompt_sha8": sha[:8],
                    "depth_percent": rec["depth_percent"],
                    "matched_pair": paired,
                    "answer_position": t,
                    "gold_token_id": rec["gold_token_ids"][t],
                    "entry_layer_audit_rule": entry_layer_audit_rule(centred),
                    "entry_layer_vocab_rule": entry_layer_vocab_rule(zc),
                    "entry_layer_10pct_rule": entry_layer_fraction_rule(centred),
                    "peak_layer": peak,
                    "peak_delta_centred": maxv,
                    "final_delta_centred": final,
                    "retention_R": float(final / maxv) if maxv > 0 else float("nan"),
                    "max_delta_gold_z": float(zc.max()),
                    "final_delta_gold_z": float(zc[-1]),
                    "peak_kl_layer": int(np.argmax(kl)),
                    "final_kl": float(kl[-1]),
                    "final_gold_rank_full": int(rec["per_layer"]["gold_rank_full"][t, -1]),
                    "final_gold_rank_ablated": int(rec["per_layer"]["gold_rank_ablated"][t, -1]),
                    "final_causal_delta_rank": int(rec["per_layer"]["causal_delta_rank"][t, -1]),
                    "final_gap_to_competitor_full": float(
                        rec["per_layer"]["gap_to_best_competitor_full"][t, -1]
                    ),
                    "final_gap_to_competitor_ablated": float(
                        rec["per_layer"]["gap_to_best_competitor_ablated"][t, -1]
                    ),
                    "final_best_competitor_id_full": int(
                        rec["per_layer"]["best_competitor_id_full"][t, -1]
                    ),
                }
            )
    with (OUT_DIR / "per_case_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_case[0].keys()))
        writer.writeheader()
        writer.writerows(per_case)
    print(f"wrote per_case_summary.csv ({len(per_case)} rows)")

    # ---------------- final-layer competition detail --------------------
    comp_rows: list[dict[str, Any]] = []
    for (arm, sha), rec in loaded.items():
        for det in rec["final_layer_topk"]:
            t = det["position"]
            gap_f = float(rec["per_layer"]["gap_to_best_competitor_full"][t, -1])
            gap_a = float(rec["per_layer"]["gap_to_best_competitor_ablated"][t, -1])
            closed = gap_a - gap_f  # logits of deficit the gold block removed
            comp_rows.append(
                {
                    "arm": arm,
                    "prompt_sha8": sha[:8],
                    "depth_percent": rec["depth_percent"],
                    "answer_position": t,
                    "gold_id": det["gold_id"],
                    "gold_delta_centred_final": round(det["gold_delta_centred"], 4),
                    "gap_full": round(gap_f, 4),
                    "gap_ablated": round(gap_a, 4),
                    "deficit_closed_by_gold_block": round(closed, 4),
                    "residual_deficit": round(gap_f, 4),
                    "closure_fraction": round(closed / gap_a, 4) if gap_a > 0 else None,
                    "gold_blocks_worth_still_needed": (
                        round(gap_f / closed, 3) if closed > 1e-6 else None
                    ),
                    "median_abs_delta_top32_competitors": round(
                        float(np.median(np.abs(det["top_delta_centred"]))), 4
                    ),
                    "top1_id": det["top_ids"][0],
                    "top1_logit_full": det["top_logits_full"][0],
                    "top_ids_json": json.dumps(det["top_ids"]),
                    "top_delta_centred_json": json.dumps(det["top_delta_centred"]),
                }
            )
    with (OUT_DIR / "final_layer_competition.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comp_rows)
    print(f"wrote final_layer_competition.csv ({len(comp_rows)} rows)")

    # ---------------- phase0 QK cross-reference -------------------------
    summary_blob = json.loads((PHASE0_ROOT / "phase0_summary.json").read_text())
    frozen_heads: dict[int, list[int]] = {}
    for h in summary_blob["retrieval_head_contract"]["heads"]:
        frozen_heads.setdefault(int(h["layer"]), []).append(int(h["head"]))
    matched_shas = {p["prompt_sha256"] for p in pairs}

    qk: dict[str, Any] = {}
    for length, shas in ((16384, matched_shas), (32768, set())):
        for arm, fname in (("evq", "phase0_evq.json"), ("geo", "phase0_geo.json")):
            qk[f"{arm}_{length}"] = phase0_layer_profile(fname, shas, length, frozen_heads)

    qk_rows: list[dict[str, Any]] = []
    for key, per in qk.items():
        arm, length = key.rsplit("_", 1)
        for sha, prof in per.items():
            for l in range(N_LAYERS):
                qk_rows.append(
                    {
                        "arm": arm,
                        "target_length": int(length),
                        "prompt_sha8": sha[:8],
                        "example_id": prof["example_id"],
                        "layer": l,
                        "n_frozen_heads_in_layer": len(frozen_heads.get(l, [])),
                        "hit16_all_heads": float(prof["hit16_all_heads"][l]),
                        "hit16_frozen_heads": float(prof["hit16_frozen_heads"][l]),
                        "dense_answer_mass_all_heads": float(
                            prof["dense_answer_mass_all_heads"][l]
                        ),
                        "dense_answer_mass_frozen_heads": float(
                            prof["dense_answer_mass_frozen_heads"][l]
                        ),
                        "block_rank_median_all_heads": float(
                            prof["block_rank_median_all_heads"][l]
                        ),
                        "block_rank_median_frozen_heads": float(
                            prof["block_rank_median_frozen_heads"][l]
                        ),
                    }
                )
    with (OUT_DIR / "phase0_layer_profile.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(qk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qk_rows)
    print(f"wrote phase0_layer_profile.csv ({len(qk_rows)} rows)")

    # ---------------- QK <-> readout alignment --------------------------
    # readout curve: EVQ centred gold delta at answer position 0 (the position
    # whose query the phase0 probe measures), averaged over the 5 matched cases
    def mean_curve(arm: str, field: str, pos: int = 0) -> np.ndarray:
        stack = [
            loaded[(arm, p["prompt_sha256"])]["per_layer"][field][pos] for p in pairs
        ]
        return np.mean(np.stack(stack), axis=0)

    evq_readout = mean_curve("evq", "delta_gold_centred")
    geo_readout = mean_curve("geo", "delta_gold_centred")
    evq_kl = mean_curve("evq", "kl_full_abl")

    def mean_qk(key: str, field: str) -> np.ndarray:
        per = qk[key]
        return np.nanmean(np.stack([p[field] for p in per.values()]), axis=0)

    qk_adv_hit = mean_qk("evq_16384", "hit16_all_heads") - mean_qk(
        "geo_16384", "hit16_all_heads"
    )
    qk_adv_hit_frozen = mean_qk("evq_16384", "hit16_frozen_heads") - mean_qk(
        "geo_16384", "hit16_frozen_heads"
    )
    qk_adv_mass = mean_qk("evq_16384", "dense_answer_mass_all_heads") - mean_qk(
        "geo_16384", "dense_answer_mass_all_heads"
    )
    qk_adv_hit_32k = mean_qk("evq_32768", "hit16_all_heads") - mean_qk(
        "geo_32768", "hit16_all_heads"
    )

    # cumulative QK advantage up to and including each layer: causal evidence at
    # layer l can only reflect attention that has already happened at layers <= l
    cum_hit = np.cumsum(qk_adv_hit)
    cum_mass = np.cumsum(qk_adv_mass)

    alignment = {
        "note": (
            "readout curve = EVQ mean centred gold delta at answer position 0, the "
            "position whose query the phase0 probe measures "
            "(query_contract=last_prompt_token_predicting_first_answer_token)"
        ),
        "spearman_readout_vs_qk_hit16_all_heads": spearman(evq_readout, qk_adv_hit),
        "spearman_readout_vs_qk_hit16_frozen_heads": spearman(
            evq_readout, qk_adv_hit_frozen
        ),
        "spearman_readout_vs_qk_answer_mass": spearman(evq_readout, qk_adv_mass),
        "spearman_readout_vs_cumulative_qk_hit16": spearman(evq_readout, cum_hit),
        "spearman_readout_vs_cumulative_qk_answer_mass": spearman(evq_readout, cum_mass),
        "spearman_kl_vs_qk_hit16_all_heads": spearman(evq_kl, qk_adv_hit),
        "spearman_kl_vs_cumulative_qk_hit16": spearman(evq_kl, cum_hit),
        "curves": {
            "layer": list(range(N_LAYERS)),
            "evq_readout_delta_centred_pos0": [round(float(v), 5) for v in evq_readout],
            "geo_readout_delta_centred_pos0": [round(float(v), 5) for v in geo_readout],
            "evq_kl_full_abl_pos0": [round(float(v), 6) for v in evq_kl],
            "qk_advantage_hit16_all_heads_16k": [round(float(v), 5) for v in qk_adv_hit],
            "qk_advantage_hit16_frozen_heads_16k": [
                None if not np.isfinite(v) else round(float(v), 5)
                for v in qk_adv_hit_frozen
            ],
            "qk_advantage_answer_mass_16k": [round(float(v), 8) for v in qk_adv_mass],
            "qk_advantage_hit16_all_heads_32k": [
                round(float(v), 5) for v in qk_adv_hit_32k
            ],
            "cumulative_qk_advantage_hit16_16k": [round(float(v), 5) for v in cum_hit],
        },
    }
    (OUT_DIR / "qk_readout_alignment.json").write_text(
        json.dumps(alignment, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote qk_readout_alignment.json")

    # ---------------- competitor identity structure ---------------------
    all_gold = {g for p in pairs for g in p["gold_token_ids"]}
    comp_struct: dict[str, Any] = {"all_case_gold_token_ids": sorted(all_gold)}
    for arm in ("evq", "geo"):
        topsets = []
        for p in pairs:
            det = loaded[(arm, p["prompt_sha256"])]["final_layer_topk"][0]
            topsets.append(set(det["top_ids"]))
        inter = set.intersection(*topsets)
        union = set.union(*topsets)
        comp_struct[arm] = {
            "n_cases": len(topsets),
            "top32_shared_by_all_cases": len(inter),
            "top32_union_size": len(union),
            "shared_fraction_of_32": round(len(inter) / TOP_K, 3),
            "shared_ids_sample": sorted(inter)[:20],
            "top32_entries_that_are_a_case_gold_token": sum(
                len(s & all_gold) for s in topsets
            ),
        }
    (OUT_DIR / "competitor_structure.json").write_text(
        json.dumps(comp_struct, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote competitor_structure.json")

    # ---------------- late-layer trajectory -----------------------------
    # Does the readout ever get closer to the gold than it ends up?  A best rank
    # that is much better than the final rank means the last layers actively move
    # the gold token down, independently of whether the causal delta is retained.
    traj_rows: list[dict[str, Any]] = []
    for (arm, sha), rec in loaded.items():
        for t in range(N_POS):
            pl = rec["per_layer"]
            rank = pl["gold_rank_full"][t]
            prob = pl["gold_prob_full"][t]
            kl = pl["kl_full_abl"][t]
            cen = pl["delta_gold_centred"][t]
            best_layer = int(np.argmin(rank))
            prob_peak = int(np.argmax(prob))
            # restricted to layers >= 20: the logit lens is least trustworthy in
            # early layers, where a spurious rank minimum can mask the late-layer
            # trend.  Reported alongside the unrestricted figure.
            rank_late = rank[20:]
            best_late = int(rank_late.min())
            best_late_layer = int(np.argmin(rank_late)) + 20
            traj_rows.append(
                {
                    "arm": arm,
                    "prompt_sha8": sha[:8],
                    "depth_percent": rec["depth_percent"],
                    "answer_position": t,
                    "best_gold_rank": int(rank.min()),
                    "best_gold_rank_layer": best_layer,
                    "final_gold_rank": int(rank[-1]),
                    "rank_degradation_factor": (
                        round(float(rank[-1] / rank.min()), 3) if rank.min() > 0 else None
                    ),
                    "best_gold_rank_layers_ge20": best_late,
                    "best_gold_rank_layer_ge20": best_late_layer,
                    "rank_degradation_factor_layers_ge20": (
                        round(float(rank[-1] / best_late), 3) if best_late > 0 else None
                    ),
                    "peak_gold_prob": float(prob.max()),
                    "peak_gold_prob_layer": prob_peak,
                    "final_gold_prob": float(prob[-1]),
                    "gold_prob_drop_factor": (
                        round(float(prob.max() / prob[-1]), 3) if prob[-1] > 0 else None
                    ),
                    "peak_kl": float(kl.max()),
                    "peak_kl_layer": int(np.argmax(kl)),
                    "final_kl": float(kl[-1]),
                    "kl_drop_factor": (
                        round(float(kl.max() / kl[-1]), 3) if kl[-1] > 0 else None
                    ),
                    "peak_delta_centred": float(cen.max()),
                    "final_delta_centred": float(cen[-1]),
                    "delta_retention_R": (
                        round(float(cen[-1] / cen.max()), 4) if cen.max() > 0 else None
                    ),
                    "entropy_at_peak_kl_layer": float(pl["entropy_full"][t, int(np.argmax(kl))]),
                    "entropy_final": float(pl["entropy_full"][t, -1]),
                    "top1_prob_final": float(pl["top1_prob_full"][t, -1]),
                }
            )
    with (OUT_DIR / "readout_trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(traj_rows[0].keys()))
        writer.writeheader()
        writer.writerows(traj_rows)
    print(f"wrote readout_trajectory.csv ({len(traj_rows)} rows)")

    # ---------------- headline summary ----------------------------------
    def agg(arm: str, field: str, pos: int | None = 0) -> dict[str, float]:
        vals = [
            r[field]
            for r in per_case
            if r["arm"] == arm
            and r["matched_pair"]
            and (pos is None or r["answer_position"] == pos)
            and r[field] is not None
            and np.isfinite(r[field])
        ]
        return {
            "n": len(vals),
            "median": float(np.median(vals)) if vals else float("nan"),
            "min": float(np.min(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
        }

    geo_all_layers = np.concatenate(
        [loaded[("geo", p["prompt_sha256"])]["per_layer"]["delta_gold_centred"].ravel() for p in pairs]
    )
    evq_all_layers = np.concatenate(
        [loaded[("evq", p["prompt_sha256"])]["per_layer"]["delta_gold_centred"].ravel() for p in pairs]
    )
    geo_z = np.concatenate(
        [loaded[("geo", p["prompt_sha256"])]["per_layer"]["delta_gold_z"].ravel() for p in pairs]
    )

    summary = {
        "n_matched_pairs": len(pairs),
        "depths_present": sorted({p["depth_percent"] for p in pairs}),
        "answer_positions": N_POS,
        "layers": N_LAYERS,
        "geo_negative_control": {
            "max_abs_centred_delta_any_layer_pos_case": float(np.abs(geo_all_layers).max()),
            "median_abs_centred_delta": float(np.median(np.abs(geo_all_layers))),
            "max_abs_robust_z_any_layer": float(np.abs(geo_z).max()),
            "frac_layers_with_z_above_3": float((geo_z > 3).mean()),
            "evq_max_abs_centred_delta_for_scale": float(np.abs(evq_all_layers).max()),
        },
        "evq_pos0": {
            "entry_layer_audit_rule": agg("evq", "entry_layer_audit_rule"),
            "entry_layer_vocab_rule": agg("evq", "entry_layer_vocab_rule"),
            "entry_layer_10pct_rule": agg("evq", "entry_layer_10pct_rule"),
            "peak_layer": agg("evq", "peak_layer"),
            "retention_R": agg("evq", "retention_R"),
            "peak_delta_centred": agg("evq", "peak_delta_centred"),
            "final_delta_centred": agg("evq", "final_delta_centred"),
            "final_gold_rank_full": agg("evq", "final_gold_rank_full"),
            "final_gold_rank_ablated": agg("evq", "final_gold_rank_ablated"),
            "final_causal_delta_rank": agg("evq", "final_causal_delta_rank"),
            "final_gap_to_competitor_full": agg("evq", "final_gap_to_competitor_full"),
        },
        "geo_pos0": {
            "peak_delta_centred": agg("geo", "peak_delta_centred"),
            "final_delta_centred": agg("geo", "final_delta_centred"),
            "final_gold_rank_full": agg("geo", "final_gold_rank_full"),
            "final_causal_delta_rank": agg("geo", "final_causal_delta_rank"),
            "final_gap_to_competitor_full": agg("geo", "final_gap_to_competitor_full"),
        },
        "qk_alignment": {
            k: v for k, v in alignment.items() if k.startswith("spearman")
        },
    }

    def traj_agg(arm: str, field: str, pos: int = 0) -> dict[str, float]:
        vals = [
            r[field]
            for r in traj_rows
            if r["arm"] == arm
            and r["answer_position"] == pos
            and any(p["prompt_sha256"][:8] == r["prompt_sha8"] for p in pairs)
            and r[field] is not None
            and np.isfinite(r[field])
        ]
        return {
            "n": len(vals),
            "median": float(np.median(vals)) if vals else float("nan"),
            "min": float(np.min(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
        }

    summary["evq_pos0_trajectory"] = {
        f: traj_agg("evq", f)
        for f in (
            "best_gold_rank",
            "best_gold_rank_layer",
            "final_gold_rank",
            "rank_degradation_factor",
            "best_gold_rank_layer_ge20",
            "rank_degradation_factor_layers_ge20",
            "peak_gold_prob_layer",
            "gold_prob_drop_factor",
            "peak_kl_layer",
            "kl_drop_factor",
            "entropy_at_peak_kl_layer",
            "entropy_final",
            "top1_prob_final",
        )
    }
    comp = [
        r
        for r in comp_rows
        if r["arm"] == "evq"
        and any(p["prompt_sha256"][:8] == r["prompt_sha8"] for p in pairs)
    ]
    summary["evq_final_competition_by_position"] = {
        f"position_{t}": {
            "median_gap_full_logits": float(
                np.median([r["gap_full"] for r in comp if r["answer_position"] == t])
            ),
            "median_deficit_closed_by_gold_block": float(
                np.median(
                    [r["deficit_closed_by_gold_block"] for r in comp if r["answer_position"] == t]
                )
            ),
            "median_closure_fraction": float(
                np.median(
                    [
                        r["closure_fraction"]
                        for r in comp
                        if r["answer_position"] == t and r["closure_fraction"] is not None
                    ]
                )
            ),
            "median_abs_delta_top32_competitors": float(
                np.median(
                    [
                        r["median_abs_delta_top32_competitors"]
                        for r in comp
                        if r["answer_position"] == t
                    ]
                )
            ),
        }
        for t in range(N_POS)
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote summary.json")

    # stash curves for the figure script
    np.savez(
        OUT_DIR / "curves.npz",
        **{
            f"{arm}__{p['prompt_sha256'][:8]}__{field}": loaded[(arm, p["prompt_sha256"])][
                "per_layer"
            ][field]
            for arm in ("evq", "geo")
            for p in pairs
            for field in (
                "delta_gold_centred",
                "delta_gold_z",
                "kl_full_abl",
                "l2_centred",
                "causal_delta_rank",
                "gold_rank_full",
                "gap_to_best_competitor_full",
                "entropy_full",
                "gold_prob_full",
            )
        },
        qk_adv_hit=qk_adv_hit,
        qk_adv_hit_frozen=qk_adv_hit_frozen,
        qk_adv_mass=qk_adv_mass,
        qk_adv_hit_32k=qk_adv_hit_32k,
        depths=np.array([p["depth_percent"] for p in pairs]),
        shas=np.array([p["prompt_sha256"][:8] for p in pairs]),
    )
    print("wrote curves.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
