"""DEV-only single-layer C/P/U keep-set intervention from one common prefix.

The probe establishes only a local keep-set effect. It cannot certify that
future queries follow the prefix proxy, or turn a negative result into a
global impossibility. Default is a command preview; --execute runs one row.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import torch

from .adapter import AdapterConfig, PrefillSession
from .run import digest, model_identity, score, write_json


def paired_swap_sets(control, proposal, length, *, seed=20260909, max_swaps=0):
    """Same removed IDs, equal insertion counts, fixed-seed sham per KV head.

Sham insertions come from neither set where possible; no answers or future
queries are consulted. Inputs are not mutated. No-effect cases are explicit.
"""
    if control.shape != proposal.shape or control.ndim != 2:
        raise ValueError("Expected equal-budget [Hkv, B] index sets")
    exact, sham, receipts = [], [], []
    for h, (c, p) in enumerate(zip(control.cpu().tolist(), proposal.cpu().tolist())):
        if len(set(c)) != len(c) or len(set(p)) != len(p):
            raise ValueError("Duplicate cache indices")
        if any(i < 0 or i >= length for i in c + p):
            raise ValueError("Cache index out of range")
        removed, inserted = sorted(set(c) - set(p)), sorted(set(p) - set(c))
        n = len(removed) if max_swaps == 0 else min(max_swaps, len(removed))
        outside = sorted(set(range(length)) - set(c) - set(p))
        n = min(n, len(outside))
        removed, inserted = removed[:n], inserted[:n]
        rng = torch.Generator().manual_seed(seed + h)
        sampled = [outside[i] for i in torch.randperm(len(outside), generator=rng)[:n].tolist()]
        exact.append(sorted((set(c) - set(removed)) | set(inserted)))
        sham.append(sorted((set(c) - set(removed)) | set(sampled)))
        receipts.append(dict(head=h, removed=removed, inserted=inserted,
                             sham_inserted=sampled, swaps=n,
                             available_proposal_swaps=len(set(p) - set(c))))
    return (torch.tensor(exact, device=control.device, dtype=control.dtype),
            torch.tensor(sham, device=control.device, dtype=control.dtype), receipts)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    p.add_argument("--model")
    p.add_argument("--data", required=True)
    p.add_argument("--row-id", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--query-policy", choices=("uniform_prefix", "recent_prefix"), default="uniform_prefix")
    p.add_argument("--horizon", type=int, default=512)
    p.add_argument("--keep-fraction", type=float, default=.25)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--max-swaps", type=int, default=0)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.max_swaps < 0 or a.layer < 0:
        p.error("layer and max-swaps must be nonnegative")
    return a


@torch.inference_mode()
def execute(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root, out = Path(a.root), Path(a.output)
    row = next(r for r in (json.loads(s) for s in Path(a.data).read_text().splitlines()) if r["row_id"] == a.row_id)
    if row["split"] != "dev":
        raise ValueError("Mechanism discovery uses DEV; do not choose interventions on TEST")
    if row["prefix_ids"] + row["suffix_ids"] != row["prompt_ids"]:
        raise ValueError("Tokenization boundary mismatch")
    if len(row["prefix_ids"]) != row["prefix_length"]:
        raise ValueError("Declared prefix length differs from the actual boundary")
    model_path = a.model or str(root / "runs/pm_gpu_ready_20260909_v3/model_view")
    cfg = AdapterConfig(query_policy=a.query_policy, horizon=a.horizon,
                        keep_fraction=a.keep_fraction, seed=a.seed)
    contract = dict(probe="one_prefix_single_layer_keep_set_intervention_v1",
                    model=model_identity(model_path), row_id=a.row_id,
                    input_sha=digest(row["prompt_ids"]), layer=a.layer, config=asdict(cfg),
                    prefix_length=row["prefix_length"],
                    decoding_and_score={key: row.get(key) for key in
                        ("max_new_tokens", "score_contract", "expected", "references")},
                    max_swaps=a.max_swaps, source_sha=digest(Path(__file__).read_text()),
                    dependencies={name: digest(Path(__file__).with_name(name).read_text())
                                  for name in ("adapter.py", "ops.py", "run.py")},
                    interpretation="local DEV intervention; not a deployed method or independent quality test")
    if (out / "contract.json").exists():
        if json.loads((out / "contract.json").read_text()) != contract:
            raise ValueError("Use a new output directory for a changed intervention")
        if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("status") == "COMPLETE":
            print("Already complete; source results reused")
            return
        raise ValueError("Partial probe exists; preserve it and use a new output directory")
    write_json(out / "contract.json", contract)
    write_json(out / "status.json", {"status": "LOADING"})
    started = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        if a.layer >= len(model.model.layers):
            raise ValueError("Layer outside model")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        session = PrefillSession(model, row["prefix_ids"], cfg).prefill()
        csets, psets, usets = [session.keep_indices(arm) for arm in ("C", "P", "U")]
        p_layer, sham_layer, swaps = paired_swap_sets(csets[a.layer], psets[a.layer],
                session.prefix_length, seed=a.seed, max_swaps=a.max_swaps)
        variants = {"C_all": csets}
        for name, target in (("P_layer", p_layer), ("U_full_layer", usets[a.layer]), ("sham_layer", sham_layer)):
            variant = list(csets)
            variant[a.layer] = target
            variants[name] = variant
        write_json(out / "interventions.json", dict(layer=a.layer, swaps=swaps,
                    U_scope="full layer U set; same KV budget, not necessarily same edit count as P",
                    no_P_intervention=all(r["swaps"] == 0 for r in swaps),
                    C_keep_sha=digest([x.cpu().tolist() for x in csets]),
                    full_P_keep_sha=digest([x.cpu().tolist() for x in psets])))
        with (out / "generations.jsonl").open("w") as f:
            for arm, indices in variants.items():
                branch = session.branch("C", indices=indices)
                branch.consume(row["suffix_ids"])
                generated = branch.generate(row["max_new_tokens"], eos)
                result = dict(row_id=a.row_id, arm=arm, layer=a.layer,
                    generated_token_ids=generated["generated_ids"],
                    initial_prefix_kv_bytes=generated["initial_prefix_kv_bytes"],
                    timings=generated["timings"],
                    **score(row, generated["generated_ids"], tokenizer, eos))
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                del branch
        write_json(out / "status.json", {"status": "COMPLETE", "elapsed_seconds": time.perf_counter() - started})
    except BaseException as e:
        write_json(out / "status.json", {"status": "FAILED", "error_type": type(e).__name__, "error": str(e)})
        raise


def main():
    a = parse_args()
    print(json.dumps(vars(a), ensure_ascii=False), flush=True)
    if not a.execute:
        return
    torch.set_num_threads(4)
    with (Path(a.root) / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (Path(a.root) / "STOP").exists():
            raise SystemExit("User STOP exists")
        execute(a)


if __name__ == "__main__":
    main()
