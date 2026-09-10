"""One DEV row, one layer, one last-prompt-token block-swap intervention.

The prompt except its final token is prefetched once using the same candidate.
Control/repair/sham reuse that immutable past; only the selected support at the
specified layer and final prompt token changes. Later decoding uses the same
candidate again. Exact visible attention mass is diagnostic oracle information,
not a deployable selector or evidence about the entire prefill trajectory.

The CLI is a dry run unless --execute is supplied, and execution acquires the
same ROOT/queue.lock used by the experiment queue before loading any model.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from .run import score_output, write_json
from .runtime import (AttentionSettings, BACKEND_LABEL, NosaReferenceForCausalLM,
                      SelectionContext, dense_causal_attention, mandatory_blocks,
                      native_select, selected_causal_attention)
from .selector_controls import BlockSummarySelector


@dataclass
class SwapPlan:
    selected: dict[str, torch.Tensor]
    block_mass: torch.Tensor  # [KV, blocks], exact mean normalized head mass
    records: list[dict]

    def report(self):
        count = sum(item["swap_count"] for item in self.records)
        return {"status": "intervention" if count else "no_intervention",
                "total_head_block_swaps": count, "per_kv_head": self.records,
                "selected_blocks": {k: v[:, 0].tolist() for k, v in self.selected.items()},
                "exact_group_block_mass": self.block_mass.tolist()}


@torch.inference_mode()
def exact_group_block_mass(context: SelectionContext):
    """Exact s=qK/sqrt(d)+CIS probabilities at ONE fixed query position."""
    if context.q.shape[1] != 1 or context.query_positions.numel() != 1:
        raise ValueError("the causal probe accepts exactly one query position")
    kvh, length, dim = context.k.shape
    if context.q.shape[0] % kvh:
        raise ValueError("invalid contiguous GQA mapping")
    group = context.q.shape[0] // kvh
    with torch.autocast(device_type=context.q.device.type, enabled=False):
        q = context.q.reshape(kvh, group, dim).float() / math.sqrt(dim)
        logits = torch.einsum("hgd,htd->hgt", q, context.k.float())
        logits += context.cis.float()[:, None]
        visible = torch.arange(length, device=q.device) <= context.query_positions[0]
        logits.masked_fill_(~visible, -torch.inf)
        probability = logits.softmax(-1)
        blocks = math.ceil(length / context.settings.block_size)
        mass = probability.new_zeros((kvh, group, blocks))
        block_ids = torch.arange(length, device=q.device) // context.settings.block_size
        mass.scatter_add_(-1, block_ids[None, None].expand(kvh, group, -1), probability)
    if not torch.isfinite(mass).all():
        raise FloatingPointError("nonfinite exact diagnostic mass")
    return mass.mean(1)


@torch.inference_mode()
def make_swap_plan(context: SelectionContext, control: torch.Tensor, max_swaps=1):
    """Match repair/sham donors and per-KV-head swap counts exactly.

    Repair inserts high-mass missed blocks. Sham inserts disjoint lowest-mass
    missed blocks with no greater mass than the removed blocks. If a matched
    non-improving sham cannot be formed, that head receives fewer/zero swaps.
    Both interventions leave every mandatory block and total support unchanged.
    """
    if isinstance(max_swaps, bool) or not isinstance(max_swaps, int) or max_swaps < 0:
        raise ValueError("max_swaps must be a nonnegative integer")
    if control.ndim != 3 or control.shape[:2] != (context.k.shape[0], 1):
        raise ValueError("control support must have shape [KV,1,budget]")
    mass = exact_group_block_mass(context)
    protected = mandatory_blocks(context, mass.shape[-1])[0].nonzero().flatten().tolist()
    last_visible = int(context.query_positions[0]) // context.settings.block_size
    selected = {arm: control.detach().clone() for arm in ("control", "repair", "sham")}
    records = []
    # This is one diagnostic event; CPU lists provide explicit reproducible ties.
    weights = mass.cpu().tolist()
    for head in range(context.k.shape[0]):
        original = control[head, 0].tolist()
        kept = [b for b in original if b >= 0]
        if len(kept) != len(set(kept)) or any(b > last_visible for b in kept):
            raise ValueError("control contains duplicate or future blocks")
        if not set(protected).issubset(kept):
            raise ValueError("control failed to retain mandatory blocks")
        donors = sorted(set(kept) - set(protected), key=lambda b: (weights[head][b], b))
        missing = sorted(set(range(last_visible + 1)) - set(kept))
        good = sorted(missing, key=lambda b: (-weights[head][b], b))
        low = sorted(missing, key=lambda b: (weights[head][b], b))
        limit = min(max_swaps, len(donors), len(missing) // 2)
        count = 0
        for trial in range(limit, 0, -1):
            remove, add, sham = donors[:trial], good[:trial], low[:trial]
            if set(add).isdisjoint(sham) and all(
                weights[head][a] > weights[head][d] and weights[head][s] <= weights[head][d]
                for d, a, s in zip(remove, add, sham)
            ):
                count = trial
                break
        remove, add, sham = donors[:count], good[:count], low[:count]
        if count:
            for arm, incoming in (("repair", add), ("sham", sham)):
                changed = sorted((set(kept) - set(remove)) | set(incoming))
                changed += [-1] * (len(original) - len(changed))
                selected[arm][head, 0] = torch.tensor(changed, device=control.device, dtype=control.dtype)
        records.append(dict(kv_head=head, swap_count=count, mandatory_blocks=protected,
            removed=remove, repair_added=add, sham_added=sham,
            removed_mass=sum(weights[head][b] for b in remove),
            repair_added_mass=sum(weights[head][b] for b in add),
            sham_added_mass=sum(weights[head][b] for b in sham),
            no_intervention_reason=None if count else "no_strict_mass_repair_with_disjoint_nonimproving_sham"))
    return SwapPlan(selected, mass, records)


@torch.inference_mode()
def fixed_state_diagnostics(context: SelectionContext, plan: SwapPlan):
    """Attention output errors use the SAME frozen Q/K/V/CIS for all supports."""
    full = dense_causal_attention(context).float()
    baseline = selected_causal_attention(context, plan.selected["control"]).float()
    reports = {}
    for arm, support in plan.selected.items():
        out = baseline if arm == "control" else selected_causal_attention(context, support).float()
        if not torch.isfinite(out).all() or not torch.isfinite(full).all():
            raise FloatingPointError("nonfinite fixed-state attention output")
        difference = out - full
        retained = [float(plan.block_mass[h, ids[ids >= 0]].sum())
                    for h, ids in enumerate(support[:, 0])]
        reports[arm] = dict(exact_retained_mass_per_kv_head=retained,
            exact_retained_mass_mean=sum(retained) / len(retained),
            attention_output_l2_error=float(difference.norm()),
            attention_output_relative_l2_error=float(difference.norm() / full.norm().clamp_min(1e-12)),
            attention_output_max_abs_error=float(difference.abs().max()),
            attention_output_l2_change_vs_control=float((out - baseline).norm()),
            output_scope="attention output before W_O; same fixed Q/K/V/CIS")
    return reports


def _fork_selector(template):
    if template is native_select:
        return native_select
    if not isinstance(template, BlockSummarySelector) or template.mode != "pc2":
        raise ValueError("this probe supports only native and pc2 candidates")
    child = BlockSummarySelector("pc2", rotary_dim=template.rotary_dim,
                                 storage_dtype=template.storage_dtype)
    # Summaries are immutable in the existing selector; concatenation creates
    # new tensors. Copy the dictionary/metrics so branches cannot append to one
    # another's selector state; the common past is not recomputed.
    child.cache = dict(template.cache)
    child.metrics = copy.deepcopy(template.metrics)
    return child


def _signature(context):
    return tuple(t.detach().clone() for t in
                 (context.q, context.k[:, -1:], context.v[:, -1:], context.cis[:, -1:]))


class SinglePointSelector:
    def __init__(self, candidate, layer_idx, position, arm, *, plan=None, signature=None, max_swaps=1):
        self.candidate, self.layer_idx, self.position, self.arm = candidate, layer_idx, position, arm
        self.plan, self.signature, self.max_swaps = plan, signature, max_swaps
        self.hits = 0
        self.diagnostics = None

    def __call__(self, context):
        control = self.candidate(context)
        target = context.layer_idx == self.layer_idx and bool((context.query_positions == self.position).any())
        if not target:
            return control
        if self.hits or context.query_positions.numel() != 1:
            raise ValueError("intervention must occur once at a single last-prompt query")
        self.hits += 1
        if self.plan is None:
            if self.arm != "control":
                raise ValueError("control must establish the frozen-state intervention first")
            self.plan = make_swap_plan(context, control, self.max_swaps)
            self.signature = _signature(context)
            self.diagnostics = fixed_state_diagnostics(context, self.plan)
        else:
            current = (context.q, context.k[:, -1:], context.v[:, -1:], context.cis[:, -1:])
            if not all(torch.equal(a, b) for a, b in zip(current, self.signature)):
                raise RuntimeError("pre-intervention state differs between branches; causal comparison invalid")
            if not torch.equal(control, self.plan.selected["control"]):
                raise RuntimeError("candidate selection changed before intervention")
        return self.plan.selected[self.arm]


@torch.inference_mode()
def _greedy_from_output(model, output, max_new_tokens, eos_ids, device):
    generated = []
    for step in range(max_new_tokens):
        if not torch.isfinite(output.logits).all():
            raise FloatingPointError("nonfinite free-generation logits")
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        output = model(torch.tensor([[token]], device=device),
                       past_key_values=output.past_key_values, num_logits_to_keep=1)
    return generated


@torch.inference_mode()
def run_probe(model, row, tokenizer, eos_ids, *, layer_idx, candidate="pc2", max_swaps=1, chunk_size=128):
    """Run three complete free generations from one common pre-intervention past."""
    if candidate not in ("pc2", "native") or not 0 <= layer_idx < len(model.model.layers):
        raise ValueError("invalid candidate or layer")
    if model.settings.dense:
        raise ValueError("a dense runtime bypasses the intervention selector")
    if row.get("split") != "dev" or len(row["prompt_ids"]) < 2 or row["max_new_tokens"] < 1:
        raise ValueError("one DEV prompt with >=2 tokens and a positive generation cap is required")
    if len(row["prompt_ids"]) + row["max_new_tokens"] > row["length_cap"]:
        raise ValueError("prompt plus generation exceeds the row's unchanged length contract")
    device = next(model.parameters()).device
    ids = torch.tensor([row["prompt_ids"]], device=device, dtype=torch.long)
    original_selector = model.selector
    template = native_select if candidate == "native" else BlockSummarySelector("pc2")
    started = time.perf_counter()
    try:
        model.selector = template
        shared = model.prefill(ids[:, :-1], chunk_size=chunk_size)
        if not torch.isfinite(shared.logits).all():
            raise FloatingPointError("nonfinite shared prefill logits")
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started
        shared_cache = shared.past_key_values
        del shared
        results, plan, signature, diagnostics = {}, None, None, None
        for arm in ("control", "repair", "sham"):
            wrapper = SinglePointSelector(_fork_selector(template), layer_idx, ids.shape[1] - 1,
                                          arm, plan=plan, signature=signature, max_swaps=max_swaps)
            model.selector = wrapper
            arm_start = time.perf_counter()
            output = model(ids[:, -1:], past_key_values=shared_cache, num_logits_to_keep=1)
            if wrapper.hits != 1:
                raise RuntimeError("requested layer/token intervention did not execute")
            if arm == "control":
                plan, signature, diagnostics = wrapper.plan, wrapper.signature, wrapper.diagnostics
            generated = _greedy_from_output(model, output, row["max_new_tokens"], eos_ids, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            results[arm] = {"generated_token_ids": generated, "generated_tokens": len(generated),
                            "last_token_and_decode_seconds": time.perf_counter() - arm_start,
                            "single_point_hits": wrapper.hits,
                            **score_output(row, generated, tokenizer, eos_ids)}
            del output
        return dict(row_id=row["row_id"], task=row.get("task"), split="dev", candidate=candidate,
            layer_idx=layer_idx, intervention_position=ids.shape[1] - 1,
            input_tokens=ids.shape[1], shared_prefill_tokens=ids.shape[1] - 1,
            max_new_tokens=row["max_new_tokens"], score_contract=row["score_contract"],
            shared_prefill_seconds=prefill_seconds, backend=BACKEND_LABEL,
            settings=asdict(model.settings), swap_plan=plan.report(),
            fixed_state_diagnostics=diagnostics, generations=results,
            shared_past_policy="same immutable K/V/CIS tensors reused; independent selector dictionaries",
            pre_intervention_q_current_kv_cis_and_candidate_support_equal=True,
            oracle_scope="current query and all visible K/CIS; no answer or future-generation labels",
            causal_scope="one layer at the last prompt token; does not identify whole-prefill causality",
            timing_scope="diagnostic oracle work included; not deployable-selector timing")
    finally:
        model.selector = original_selector


def _read_row(data, row_id):
    rows = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row.get("row_id") == row_id]
    if len(matches) != 1 or matches[0].get("split") != "dev":
        raise ValueError("row-id must identify exactly one DEV row")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/position_overnight_20260909")
    parser.add_argument("--model", default="/root/autodl-tmp/NOSA-1B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=("pc2", "native"), default="pc2")
    parser.add_argument("--max-swaps", type=int, default=1, help="Maximum matched swaps per KV head")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--select-blocks", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--attention-query-chunk-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    row = _read_row(args.data, args.row_id)
    if args.layer < 0 or args.max_swaps < 0 or args.chunk_size < 1:
        parser.error("layer/swaps must be nonnegative; chunk-size must be positive")
    settings = AttentionSettings(topk=args.topk, select_blocks=args.select_blocks,
                                 attention_query_chunk_size=args.attention_query_chunk_size)
    contract = dict(status="DRY_RUN", row_id=args.row_id, prompt_tokens=len(row["prompt_ids"]),
        layer_idx=args.layer, candidate=args.candidate, max_swaps_per_kv_head=args.max_swaps,
        arms=["control", "repair", "sham"], settings=asdict(settings),
        lock=str(Path(args.root) / "queue.lock"), execute=args.execute,
        model=args.model, data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        scope="one DEV row, one layer, last prompt token; oracle diagnostic, no deployment claim")
    print(json.dumps(contract), flush=True)
    if not args.execute:
        return contract
    root, output = Path(args.root), Path(args.output)
    if not root.is_dir():
        raise ValueError("shared queue root does not exist; refusing a separate accidental lock")
    with (root / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("shared GPU queue is busy; no model loaded or process interrupted") from exc
        if (root / "STOP").exists():
            raise RuntimeError("shared STOP marker is present; no experiment started")
        if output.exists() and any(output.iterdir()):
            raise ValueError("use a fresh output directory; existing causal evidence will not be overwritten")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        output.mkdir(parents=True, exist_ok=True)
        contract.update(status="EXECUTING", pid=os.getpid(), dtype=args.dtype, device=args.device,
                        chunk_size=args.chunk_size,
                        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), Path(__file__).with_name("runtime.py"),
                                      Path(__file__).with_name("selector_controls.py"), Path(__file__).with_name("run.py"))})
        write_json(output / "contract.json", contract)
        write_json(output / "status.json", {"status": "LOADING", "pid": os.getpid()})
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
            cfg = json.loads((Path(args.model) / "config.json").read_text())
            eos = cfg.get("eos_token_id", tokenizer.eos_token_id)
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            model = NosaReferenceForCausalLM.from_pretrained(args.model, device=args.device,
                        dtype=getattr(torch, args.dtype), settings=settings).eval()
            write_json(output / "load_report.json", model.load_report)
            write_json(output / "status.json", {"status": "RUNNING", "pid": os.getpid()})
            result = run_probe(model, row, tokenizer, eos_ids, layer_idx=args.layer,
                               candidate=args.candidate, max_swaps=args.max_swaps, chunk_size=args.chunk_size)
            write_json(output / "result.json", result)
            write_json(output / "status.json", {"status": "COMPLETE", "pid": os.getpid(),
                       "intervention_status": result["swap_plan"]["status"]})
            return result
        except BaseException as error:
            write_json(output / "status.json", {"status": "FAILED", "pid": os.getpid(),
                       "error_type": type(error).__name__, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
