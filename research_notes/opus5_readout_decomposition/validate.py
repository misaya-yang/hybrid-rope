#!/usr/bin/env python3
"""Independent validation of the decomposition, and identification of the
final-layer competitors.

Two things are checked here that the decomposition itself cannot check:

V1  My layer-31 quantities are recomputed from the saved lens tensors.  The
    ``rank_16k_v1`` run on the RTX 5090 computed the same quantities online from
    the live model on the same adapters.  If my float32 re-derivation of the
    stored bf16 tensors reproduces that run's ``first_token_rank`` exactly and
    its ``nll_sum`` to bf16 rounding, the whole pipeline (pairing, gold ids,
    gather indices, layer ordering, sign convention) is confirmed end to end.

V2  The competitor tokens cannot be decoded on this machine — no Llama-3
    tokenizer is present and downloading one is out of scope.  But the same run
    recorded the free-running ``generation`` for every case, including the
    decoded ``prediction`` string and ``generated_ids``.  Comparing the top-1
    final-layer competitor against the first generated token identifies the
    competitor without a tokenizer.

Read-only; writes only ``validation.json`` in this directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RANK_RUN = REPO / "results/lora_sparse_conversion_s42_20260714/server_26521/rank_16k_v1"
CHAT_RUN = REPO / "results/lora_sparse_conversion_s42_20260714/server_26521/chatwrap_dense_16k_v1"
NLL_TOL = 0.05  # bf16 storage of the lens tensors vs the run's float32 accumulation


def main() -> int:
    layer31 = {
        (r["arm"], r["prompt_sha8"], int(r["answer_position"])): r
        for r in csv.DictReader((OUT / "per_layer_metrics.csv").open())
        if int(r["layer"]) == 31
    }
    comp = {
        (r["arm"], r["prompt_sha8"], int(r["answer_position"])): r
        for r in csv.DictReader((OUT / "final_layer_competition.csv").open())
    }

    report: dict[str, Any] = {
        "source_run": str(RANK_RUN.relative_to(REPO)),
        "v1_rank_and_nll_parity": [],
        "v2_competitor_identity": [],
        "chat_wrap_control": [],
        "v1_all_ranks_exact": True,
        "v1_all_nll_within_tol": True,
        "v2_top1_competitor_is_first_generated_token_all_cases": True,
    }

    for arm, fname in (("evq", "evq.json"), ("geo", "geo.json")):
        blob = json.loads((RANK_RUN / fname).read_text(encoding="utf-8"))
        # confirm the run used the same adapter as the trace manifests
        report.setdefault("adapter_sha256_in_rank_run", {})[arm] = blob["adapter"][
            "adapter_sha256"
        ]
        for entry in blob["results"]:
            if entry["mode"] != "dense":
                continue
            sha8 = entry["prompt_sha256"][:8]
            if (arm, sha8, 0) not in layer31:
                continue
            mine_rank = int(layer31[(arm, sha8, 0)]["gold_rank_full"])
            mine_nll = -sum(
                float(layer31[(arm, sha8, t)]["gold_logprob_full"]) for t in range(3)
            )
            rank_exact = mine_rank == entry["first_token_rank"]
            nll_ok = abs(mine_nll - entry["nll_sum"]) < NLL_TOL
            report["v1_all_ranks_exact"] &= rank_exact
            report["v1_all_nll_within_tol"] &= nll_ok
            report["v1_rank_and_nll_parity"].append(
                {
                    "arm": arm,
                    "prompt_sha8": sha8,
                    "run_first_token_rank": entry["first_token_rank"],
                    "recomputed_gold_rank_layer31": mine_rank,
                    "rank_exact": rank_exact,
                    "run_nll_sum": round(entry["nll_sum"], 5),
                    "recomputed_nll_sum": round(mine_nll, 5),
                    "nll_abs_diff": round(abs(mine_nll - entry["nll_sum"]), 6),
                    "nll_within_tol": nll_ok,
                }
            )

            gen = entry["generation"]
            top1 = int(comp[(arm, sha8, 0)]["top1_id"])
            top32 = set(json.loads(comp[(arm, sha8, 0)]["top_ids_json"]))
            match = top1 == gen["generated_ids"][0]
            report["v2_top1_competitor_is_first_generated_token_all_cases"] &= match
            report["v2_competitor_identity"].append(
                {
                    "arm": arm,
                    "prompt_sha8": sha8,
                    "gold_answer_string": entry["references"][
                        entry["selected_reference_index"]
                    ],
                    "top1_final_competitor_id": top1,
                    "first_generated_token_id": gen["generated_ids"][0],
                    "top1_competitor_is_first_generated_token": match,
                    "n_top32_competitors_in_generated_32_tokens": len(
                        top32 & set(gen["generated_ids"])
                    ),
                    "free_running_prediction": gen["prediction"][:120],
                    "strict_exact": gen["strict_exact"],
                    "gold_containment": gen["gold_containment"],
                }
            )

    # the same failure under a chat template, i.e. it is not a prompt-format artefact
    for arm, fname in (("evq", "evq.json"), ("geo", "geo.json")):
        blob = json.loads((CHAT_RUN / fname).read_text(encoding="utf-8"))
        for entry in blob["results"]:
            report["chat_wrap_control"].append(
                {
                    "arm": arm,
                    "example_id": entry["example_id"],
                    "first_token_rank": entry.get("first_token_rank"),
                    "strict_exact": entry["generation"]["strict_exact"],
                    "prediction": entry["generation"]["prediction"][:100],
                }
            )

    preds = {
        arm: sorted(
            {
                r["free_running_prediction"]
                for r in report["v2_competitor_identity"]
                if r["arm"] == arm
            }
        )
        for arm in ("evq", "geo")
    }
    report["distinct_free_running_predictions_per_arm"] = {
        arm: {"n_distinct": len(v), "predictions": v} for arm, v in preds.items()
    }

    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("V1 gold rank @ layer 31 exact vs independent GPU run :",
          report["v1_all_ranks_exact"])
    print("V1 answer NLL within bf16 tolerance                  :",
          report["v1_all_nll_within_tol"])
    print("V2 top-1 competitor == first generated token (all)   :",
          report["v2_top1_competitor_is_first_generated_token_all_cases"])
    for arm in ("evq", "geo"):
        n = report["distinct_free_running_predictions_per_arm"][arm]["n_distinct"]
        print(f"    {arm}: {n} distinct free-running prediction(s) across the matched cases")
    print("wrote validation.json")
    return 0 if (report["v1_all_ranks_exact"] and report["v1_all_nll_within_tol"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
