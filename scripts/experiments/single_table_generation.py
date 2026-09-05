#!/usr/bin/env python3
"""Independent physical-context diagnosis: source twins, local oracle, exact EOS.

No historical trainer, generator, RULER scorer, or GPU launch driver is used.
Synthetic lookup/chain scores are not RULER or natural-QA capability evidence.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.lib.rope.generation_contract import (
    token_exact_eos, retention_verdict, paired_retention_intervals,
)

VERSION = "SINGLE_TABLE_SOURCE_TWINS_V2"
NATIVE = 4096
RESERVE = 64
QK = ("q_proj", "k_proj")
QKVO = (*QK, "v_proj", "o_proj")
WEIGHT_SHA = "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def tokenizer_identity(checkpoint):
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    found = {name: sha(checkpoint / name) for name in names if (checkpoint / name).is_file()}
    if "tokenizer.json" not in found:
        raise ValueError("registered tokenizer.json required")
    return found


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def target_score(tokens, gold, eos, decode):
    ended = bool(tokens and tokens[-1] == eos and eos not in tokens[:-1])
    text = decode(tokens[:-1] if ended else tokens)
    expected = decode(gold)
    # Exact components, not substring matches. Extra separators/text fail.
    parts = text.split("|")
    wanted = expected.split("|")
    return {"exact_eos": ended and text == expected,
            "canonical_token_exact_eos": token_exact_eos(tokens, gold, eos),
            "string_exact_eos": ended and text == expected,
            "ended_with_eos": ended,
            "route_exact": len(parts) == 2 and parts[0] == wanted[0],
            "answer_exact": len(parts) == 2 and parts[1] == wanted[1]}


def reference_answer(text):
    """Independent text reader: validate identifiability, not model capability."""
    queries = re.findall(r"Query: give the route and value for key ([0-9]{6})\.", text)
    if len(queries) != 1:
        raise ValueError("query is missing or ambiguous")
    key = queries[0]
    direct = re.findall(r"Record: key " + key + r"; route ([0-9]{6}); value ([0-9]{6})\.", text)
    links = re.findall(r"Record: key " + key + r"; route ([0-9]{6})\.", text)
    answers = list(direct)
    for route in links:
        answers.extend((route, v) for v in re.findall(r"Record: route " + route + r"; value ([0-9]{6})\.", text))
    if len(answers) > 1:
        raise ValueError("more than one source-owned solution")
    return "|".join(answers[0]) if answers else None


def render_case(tokenizer, *, group, task, length, nonce, variant, explicit_format=False):
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    key, r0, r1, v0, v1 = nonce
    route, value = (r0, v0) if variant == 0 else (r1, v1)
    target = f"{route}|{value}"
    gold = encode(target)
    if tokenizer.decode(gold, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target:
        raise ValueError("target tokenizer roundtrip drift")
    if len(gold) + 1 > RESERVE:
        raise ValueError("answer and EOS exceed reserved decode budget")
    separator = next(i+1 for i in range(len(gold)) if tokenizer.decode(gold[:i+1], skip_special_tokens=False, clean_up_tokenization_spaces=False).endswith("|"))
    header = ("Read the records and answer the query. Copy the route and value exactly. "
              "Output ONLY route|value, with no spaces or explanation, then stop.\n")
    if explicit_format:
        header += ("The output must be two six-digit numbers separated by one | character. "
                   "Example of the required syntax: 123456|654321. "
                   "Do not output the words route or value, labels, punctuation, or code fences.\n")
    query = f"\nQuery: give the route and value for key {key}.\nAnswer:"
    marker = "<<SINGLE_TABLE_BODY>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": header + marker + query}],
        tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError("chat template changed the unique body marker")
    left, right = rendered.split(marker)
    prefix, suffix = encode(left), encode(right)
    body_size = length - RESERVE - len(prefix) - len(suffix)
    if body_size < 1024:
        raise ValueError("physical context too small")
    filler = encode(" Background text without any key, route, or value assignment.\n")
    body = (filler * (body_size // len(filler) + 1))[:body_size]
    records = ([f"\nRecord: key {key}; route {route}; value {value}.\n"]
               if task == "lookup" else
               [f"\nRecord: key {key}; route {route}.\n",
                f"\nRecord: route {route}; value {value}.\n"])
    # Query is at the end; source positions are physical, with no position jumps.
    starts = [body_size // 10 // 128 * 128] if task == "lookup" else [body_size // 10 // 128 * 128, body_size * 4 // 10 // 128 * 128]
    # Complete competing records, at a fixed density; never overwrite a target
    # or the reserved local-oracle area. Same distractors in both source twins.
    decoys = random.Random(int(group[:16], 16))
    for offset in range(128, body_size - 640, 128):
        if offset in starts:
            continue
        values = []
        while len(values) < 3:
            value_i = decoys.randrange(100000, 1000000)
            if value_i not in nonce:
                values.append(value_i)
        record_ids = encode(f"\nRecord: key {values[0]}; route {values[1]}; value {values[2]}.\n")
        if len(record_ids) >= 128:
            raise ValueError("distractor record exceeds its reserved slot")
        body[offset:offset + len(record_ids)] = record_ids
    spans = []
    for start, record in zip(starts, records):
        ids = encode(record)
        old = body[start:start + len(ids)]
        body[start:start + len(ids)] = ids
        spans.append({"start": len(prefix) + start, "tokens": ids, "background": old})
    prompt = prefix + body + suffix
    assert len(prompt) + RESERVE == length
    return {"group": group, "task": task, "length": length, "variant": variant,
            "input_ids": prompt, "answer_ids": gold, "answer": target,
            "compact_ids": prefix + [token for record in records for token in encode(record)] + suffix,
            "separator_index": separator,
            "source_spans": spans, "oracle_start": len(prefix) + body_size - 512,
            "query_start": len(prefix) + body_size,
            "eos_token_id": int(tokenizer.eos_token_id), "generation_budget": RESERVE}


def condition_prompt(row, condition):
    result = list(row["input_ids"])
    if condition == "compact":
        if len(row["compact_ids"]) + RESERVE > 2048:
            raise ValueError("compact view exceeds 2K including reserve")
        return list(row["compact_ids"])
    if condition == "remote":
        return result
    if condition == "oracle":
        # Exact token-block exchange preserves the near/far block multiset.
        offset = row["oracle_start"]
        for span in row["source_spans"]:
            start, ids = span["start"], span["tokens"]
            if offset + len(ids) >= row["query_start"]:
                raise ValueError("oracle source overlaps query")
            result[start:start + len(ids)], result[offset:offset + len(ids)] = result[offset:offset + len(ids)], ids
            offset += len(ids) + 8
        return result
    for span in row["source_spans"]:
        start = span["start"]
        result[start:start + len(span["tokens"])] = span["background"]
    if condition != "deleted":
        raise ValueError(condition)
    return result


def validate_pair(pair):
    if len(pair) != 2 or [row["variant"] for row in pair] != [0, 1]:
        raise ValueError("each cell requires two ordered source twins")
    a, b = pair
    for key in ("group", "task", "length", "eos_token_id", "query_start", "oracle_start", "separator_index"):
        if a[key] != b[key]:
            raise ValueError(f"paired {key} drift")
    if a["answer_ids"] == b["answer_ids"] or len(a["answer_ids"]) != len(b["answer_ids"]):
        raise ValueError("source twins must require different answers")
    if condition_prompt(a, "deleted") != condition_prompt(b, "deleted"):
        raise ValueError("source twins changed tokens outside the source")
    for row in pair:
        if len(row["input_ids"]) + RESERVE != row["length"]:
            raise ValueError("physical prompt/decode budget drift")
        if row["eos_token_id"] in row["answer_ids"]:
            raise ValueError("EOS inside gold answer")
        for span in row["source_spans"]:
            i = span["start"]
            if row["input_ids"][i:i + len(span["tokens"])] != span["tokens"]:
                raise ValueError("source location drift")
        condition_prompt(row, "oracle")


def prepare(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    if tok.eos_token_id is None:
        raise ValueError("tokenizer EOS is required")
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    # One globally disjoint nonce pool across train/selection/blind groups.
    total = args.train_pairs + 2 * args.eval_pairs
    pool = rng.sample(range(100000, 1000000), 5 * total)
    config = {"train": (args.train_pairs, (1, 2, 4)),
              "selection": (args.eval_pairs, (1, 2, 4)),
              "blind": (args.eval_pairs, (1, 4, 8, 16, 32))}
    index, files, identities = 0, {}, {}
    for split, (count, factors) in config.items():
        path = out / f"{split}.jsonl"
        groups = []
        with path.open("w") as handle:
            for pair_index in range(count):
                nonce = pool[index * 5:(index + 1) * 5]
                index += 1
                group = canonical({"seed": args.seed, "split": split, "nonce": nonce})
                groups.append(group)
                task = ("lookup", "chain")[pair_index % 2]
                for factor in factors:
                    pair = [render_case(tok, group=group, task=task, length=NATIVE * factor,
                                        nonce=nonce, variant=variant,explicit_format=getattr(args,"explicit_format",False)) for variant in (0, 1)]
                    validate_pair(pair)
                    for row in pair:
                        for condition in ("remote", "oracle", "deleted"):
                            decoded = tok.decode(condition_prompt(row, condition), skip_special_tokens=False, clean_up_tokenization_spaces=False)
                            expected = None if condition == "deleted" else row["answer"]
                            if reference_answer(decoded) != expected:
                                raise ValueError("independent text reader rejected source/target construction")
                        row["split"] = split
                        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        files[split] = {"path": path.name, "sha256": sha(path), "pairs": count,
                        "factors": list(factors)}
        identities[split] = groups
    manifest = {"status": VERSION, "seed": args.seed, "files": files,
                "explicit_format_example":getattr(args,"explicit_format",False),
                "groups": identities, "eos_token_id": tok.eos_token_id,
                "code_sha256": sha(__file__), "tokenizer_class": type(tok).__name__,
                "tokenizer_files": tokenizer_identity(args.checkpoint),
                "checkpoint_config_sha256": sha(args.checkpoint / "config.json"),
                "scope": "synthetic source-use and complete-output diagnosis; not natural QA"}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": "DATA_PREPARED_NOT_MODEL_VALIDATED", "files": files}))


def load_data(root, split, factors):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != VERSION:
        raise ValueError("dataset protocol drift")
    groups = manifest["groups"]
    if any(set(groups[a]) & set(groups[b]) for a, b in
           (("train", "selection"), ("train", "blind"), ("selection", "blind"))):
        raise ValueError("group split leakage")
    entry = manifest["files"][split]
    path = root / entry["path"]
    if sha(path) != entry["sha256"]:
        raise ValueError("data hash drift")
    selected = [row for row in rows(path) if row["length"] in [NATIVE * f for f in factors]]
    cells = {}
    for row in selected:
        if row["group"] not in groups[split] or row["split"] != split:
            raise ValueError("row belongs to another split")
        cells.setdefault((row["group"], row["length"]), []).append(row)
    if len(cells) != entry["pairs"] * len(factors):
        raise ValueError("missing physical factor/group cell")
    for pair in cells.values():
        validate_pair(pair)
    return selected, manifest


def checkpoint_contract(args):
    path=getattr(args,'checkpoint_contract',None)
    if path is None: return {'weight_sha256':WEIGHT_SHA,'model_type':'olmo2','native_context_length':4096}
    value=json.loads(path.read_text())
    if (value.get('status')!='FROZEN_CHECKPOINT_CONTRACT_V1'
            or value['config_sha256']!=sha(args.checkpoint/'config.json')
            or value['tokenizer_files']!=tokenizer_identity(args.checkpoint)):
        raise ValueError('frozen checkpoint contract identity drift')
    return value


def load_runtime(args, *, training=False):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not args.authorized:
        raise ValueError("GPU commands require explicit machine/budget authorization via --authorized")
    contract=checkpoint_contract(args)
    if sha(args.checkpoint / "model.safetensors") != contract['weight_sha256']:
        raise ValueError("registered OLMo checkpoint identity drift")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 30 * 2**30:
        raise RuntimeError("this protocol requires measured >=30 GiB, not a GPU name assumption")
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    if args.data and (args.data / "manifest.json").is_file():
        data_identity = json.loads((args.data / "manifest.json").read_text())
        if (data_identity["tokenizer_files"] != tokenizer_identity(args.checkpoint)
                or data_identity["checkpoint_config_sha256"] != sha(args.checkpoint / "config.json")
                or data_identity["eos_token_id"] != tok.eos_token_id):
            raise ValueError("data/checkpoint/tokenizer identity drift")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    rotary = model.model.rotary_emb
    native = rotary.inv_freq.detach().float().cpu().numpy().copy()
    table = native if args.table is None else np.load(args.table, allow_pickle=False)
    if table.dtype != np.float32 or table.shape != native.shape or not np.isfinite(table).all():
        raise ValueError("invalid float32 table")
    if not np.all(table > 0) or not np.all(table[:-1] > table[1:]):
        raise ValueError("table must be positive and ordered")
    if args.table and (args.table.parent / "manifest.json").is_file():
        exported = json.loads((args.table.parent / "manifest.json").read_text())
        if exported.get("status") == "LOG_P2_FACTOR_FRONTIER_FROZEN_V1":
            actual_native = hashlib.sha256(native.astype("<f4").tobytes()).hexdigest()
            if exported["native"]["float32_sha256"] != actual_native:
                raise ValueError("exporter Native tensor differs from loaded checkpoint")
            matching = [t for t in exported["tables"].values() if t["path"] == args.table.name]
            if len(matching) != 1 or matching[0]["file_sha256"] != sha(args.table):
                raise ValueError("candidate file no longer matches frozen export")
        elif exported.get("status") == "FIXED_NZGY_CONTROLS_FROZEN_V1":
            if exported['native_sha256'] != hashlib.sha256(native.astype('<f4').tobytes()).hexdigest():
                raise ValueError('control export Native tensor differs from loaded checkpoint')
            matching=[a for a in exported['arms'].values() if a['path']==args.table.name]
            if len(matching)!=1 or matching[0]['file_sha256']!=sha(args.table) or matching[0]['rotary_amplitude']!=args.gain:
                raise ValueError('fixed control table/gain drift')
    with torch.no_grad():
        rotary.inv_freq.copy_(torch.from_numpy(table).to(rotary.inv_freq))
        rotary.original_inv_freq = rotary.inv_freq.detach().clone()
        rotary.attention_scaling = args.gain
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).merge_and_unload()
    if model.config.model_type!=contract['model_type'] or model.config.max_position_embeddings!=contract['native_context_length']:
        raise ValueError('checkpoint architecture or actual Native window differs from contract')
    receipt = {"checkpoint_sha256": contract['weight_sha256'],
               "checkpoint_native_context_length":contract['native_context_length'],
               "checkpoint_contract_sha256":sha(args.checkpoint_contract) if getattr(args,'checkpoint_contract',None) else None,
               "checkpoint_config_sha256": sha(args.checkpoint / "config.json"),
               "tokenizer_files": tokenizer_identity(args.checkpoint),
               "table_is_native": args.table is None,
               "table_sha256": hashlib.sha256(table.astype("<f4").tobytes()).hexdigest(),
               "gain": args.gain, "effective_logit_multiplier": args.gain ** 2,
               "adapter_sha256": sha(args.adapter / "adapter_model.safetensors") if args.adapter else None,
               "adapter_config_sha256": sha(args.adapter / "adapter_config.json") if args.adapter else None,
               "gpu": props.name, "memory_bytes": props.total_memory,
               "compute_capability": list(torch.cuda.get_device_capability()),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "transformers": version("transformers"), "peft": version("peft"),
               "attention_implementation": model.config._attn_implementation,
               "code_sha256": sha(__file__)}
    model.config.use_cache = not training
    model.eval()
    return model, tok, receipt


def guard_resources(model, expected, start, args):
    import numpy as np
    import torch
    if time.monotonic() - start > args.max_seconds:
        raise RuntimeError("wall-clock budget exhausted; partial rows retained")
    free, _ = torch.cuda.mem_get_info()
    # Cached but unallocated allocator memory is reusable by this process.
    reusable = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if reusable < args.min_headroom_gib * 2**30:
        raise RuntimeError("measured reusable GPU headroom below contract")
    if shutil.disk_usage(args.output).free < 2 * 2**30:
        raise RuntimeError("less than 2 GiB output free space")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    actual = base.model.rotary_emb.inv_freq.detach().float().cpu().numpy()
    if hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest() != expected["table_sha256"]:
        raise RuntimeError("runtime rotary tensor drift")
    if float(base.model.rotary_emb.attention_scaling) != expected["gain"]:
        raise RuntimeError("runtime gain drift")


def greedy(model, prompt, eos, budget):
    import torch
    output, cache = [], None
    current = torch.tensor([prompt], device="cuda", dtype=torch.long)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(budget):
            result = model.model(input_ids=current, past_key_values=cache, use_cache=True,
                                 return_dict=True)
            logits = model.lm_head(result.last_hidden_state[:, -1:]).float()[:, -1]
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite decoding logits")
            token = int(logits.argmax(-1).item())
            output.append(token)
            cache = result.past_key_values
            if token == eos:
                break
            current = torch.tensor([[token]], device="cuda")
    return output


def gold_prefix_trace(model, prompt, target):
    """Audit all canonical gold-prefix decisions including EOS; no new decoder."""
    import torch
    inputs=torch.tensor([[*prompt,*target[:-1]]],device='cuda')
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
        logits=model.lm_head(hidden[:,len(prompt)-1:]).float()[0]
    gold=torch.tensor(target,device='cuda')
    correct=logits.gather(-1,gold[:,None]).squeeze(-1)
    competitors=logits.scatter(-1,gold[:,None],-torch.inf).amax(-1)
    logp=logits.log_softmax(-1).gather(-1,gold[:,None]).squeeze(-1)
    rank=(logits>correct[:,None]).sum(-1)+1
    if not torch.isfinite(logp).all(): raise RuntimeError('nonfinite gold-prefix trace')
    return {'margins':(correct-competitors).cpu().tolist(),'gold_logprobs':logp.cpu().tolist(),
            'ranks':rank.cpu().tolist(),'canonical_target_ids':target}


def runtime_smoke(args):
    """Actual checkpoint/cache parity, no training and no scientific outcome."""
    import torch
    selected, _ = load_data(args.data, "selection", (1,))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model, tok, identity = load_runtime(args)
    prompt = condition_prompt(selected[0], "oracle")
    cached = greedy(model, prompt, tok.eos_token_id, 8)
    full = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(8):
            ids = torch.tensor([[*prompt, *full]], device="cuda")
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            logits = model.lm_head(hidden[:, -1:]).float()
            token = int(logits[:, -1].argmax(-1).item())
            full.append(token)
            if token == tok.eos_token_id:
                break
    if cached != full:
        raise RuntimeError("cached/reference greedy mismatch; assay not qualified")
    guard_resources(model, identity, time.monotonic(), args)
    write_json(args.output / "smoke.json", {**identity, "cached_reference_exact": True,
               "generated_token_ids": cached, "scope": "runtime parity only, not capability"})
    print("Runtime parity passed; capability and other physical shapes remain untested.")


def evaluate(args):
    import torch
    selected, manifest = load_data(args.data, args.split, args.factors)
    if args.pair_limit:
        chosen = set(manifest["groups"][args.split][:args.pair_limit])
        selected = [row for row in selected if row["group"] in chosen]
    paired = {(r['group'],r['length'],r['variant']):r for r in selected}
    if args.split == "train":
        raise ValueError("training rows cannot serve as evaluation")
    if args.split == "blind" and not args.frozen_selection:
        raise ValueError("blind evaluation requires a frozen selection JSON")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model, tok, identity = load_runtime(args)
    if args.frozen_selection:
        lock = json.loads(args.frozen_selection.read_text())
        for key in ("table_sha256", "gain", "adapter_sha256", "adapter_config_sha256"):
            if lock[key] != identity[key]:
                raise ValueError(f"sealed method {key} drift")
        if lock["data_manifest_sha256"] != sha(args.data / "manifest.json"):
            raise ValueError("sealed data drift")
    write_json(args.output / "run.json", {**identity, "data_manifest_sha256": sha(args.data / "manifest.json"),
               "tokenizer_root": str(args.checkpoint.resolve()),
               "split": args.split, "factors": args.factors, "conditions": args.conditions,
               "expected_rows": len(selected) * len(args.conditions), "audit_margins": args.audit_margins,
               "frozen_selection_sha256": sha(args.frozen_selection) if args.frozen_selection else None})
    start = time.monotonic()
    count = 0
    deleted_cache = {}
    with (args.output / "examples.jsonl").open("w", buffering=1) as handle:
        for row in selected:
            for condition in args.conditions:
                guard_resources(model, identity, start, args)
                prompt = condition_prompt(row, condition)
                prompt_hash = canonical(prompt)
                if condition in ("compact","deleted") and prompt_hash in deleted_cache:
                    generated = deleted_cache[prompt_hash]
                    reused = True
                else:
                    generated = greedy(model, prompt, row["eos_token_id"], RESERVE)
                    reused = False
                    if condition in ("compact","deleted"):
                        deleted_cache[prompt_hash] = generated
                score = target_score(generated, row["answer_ids"], row["eos_token_id"],
                                    lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
                record = {key: row[key] for key in ("group", "task", "length", "variant")}
                record.update(condition=condition, generated_token_ids=generated,
                              answer_token_ids=row["answer_ids"], eos_token_id=row["eos_token_id"],
                              prompt_sha256=prompt_hash, identical_prompt_reused=reused, **score)
                if args.audit_margins:
                    target=[*row['answer_ids'],row['eos_token_id']]
                    trace=gold_prefix_trace(model,prompt,target)
                    twin=paired[(row['group'],row['length'],1-row['variant'])]
                    swapped=gold_prefix_trace(model,condition_prompt(twin,condition),target)
                    record['gold_prefix_trace']=trace
                    record['same_target_source_effect']=sum(a-b for a,b in zip(trace['gold_logprobs'],swapped['gold_logprobs']))/len(target)
                    record['first_canonical_token_difference']=next((i for i,(a,b) in enumerate(zip(generated,target)) if a!=b),
                        min(len(generated),len(target)) if len(generated)!=len(target) else None)
                    record['positive_margin_cache_disagreement']=min(trace['margins'])>.01 and generated!=target
                handle.write(json.dumps(record) + "\n")
                count += 1
                if count % 8 == 0:
                    print(json.dumps({"completed_rows": count, "seconds": time.monotonic()-start}), flush=True)
    guard_resources(model, identity, start, args)
    write_json(args.output / "complete.json", {"rows": count, "examples_sha256": sha(args.output / "examples.jsonl"),
               "run_sha256": sha(args.output / "run.json"), "seconds": time.monotonic()-start,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    summarize(args.output, tok)


def summarize(path, tokenizer=None):
    complete = json.loads((path / "complete.json").read_text())
    data = list(rows(path / "examples.jsonl"))
    run = json.loads((path / "run.json").read_text())
    if sha(path / "run.json") != complete["run_sha256"] or len(data) != run["expected_rows"]:
        raise ValueError("result does not complete the registered panel")
    if sha(path / "examples.jsonl") != complete["examples_sha256"] or len(data) != complete["rows"]:
        raise ValueError("incomplete or changed results")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer_root = Path(run["tokenizer_root"])
        if tokenizer_identity(tokenizer_root) != run["tokenizer_files"]:
            raise ValueError("raw-score tokenizer identity drift")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, trust_remote_code=False)
    cells = {}
    for row in data:
        recomputed = target_score(row["generated_token_ids"], row["answer_token_ids"], row["eos_token_id"],
                                 lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        if any(value != row[key] for key, value in recomputed.items()):
            raise ValueError("stored score disagrees with raw tokens")
        key = (row["task"], row["length"], row["condition"])
        cells.setdefault(key, []).append(row)
    result = {}
    for (task, length, condition), cell in cells.items():
        pairs = {}
        for row in cell:
            variants = pairs.setdefault(row["group"], {})
            if row["variant"] in variants:
                raise ValueError("duplicate result row")
            variants[row["variant"]] = row["exact_eos"]
        if any(set(pair) != {0, 1} for pair in pairs.values()):
            raise ValueError("missing source twin")
        result[f"{task}:{length}:{condition}"] = {
            "rows": len(cell), "independent_groups": len(pairs),
            **{metric: sum(float(r[metric]) for r in cell) / len(cell)
               for metric in ("exact_eos", "route_exact", "answer_exact", "ended_with_eos")},
            "both_twins_exact_eos": sum(all(pair.values()) for pair in pairs.values()) / len(pairs)}
    diagnoses = {}
    for task, length, _ in cells:
        key = f"{task}:{length}"
        panel = {condition: result.get(f"{key}:{condition}") for condition in ("compact", "remote", "oracle", "deleted")}
        if not all(panel.values()) or min(p["independent_groups"] for p in panel.values()) < 8:
            diagnoses[key] = "UNRESOLVED_CONTROLS_OR_SAMPLE_COUNT"
        elif any(r.get('positive_margin_cache_disagreement') for r in data if r['task']==task and r['length']==length):
            diagnoses[key] = "UNRESOLVED_GOLD_CACHE_DISAGREEMENT: inspect positions/mask/numerics before explanation"
        elif panel["compact"]["exact_eos"] < .8:
            diagnoses[key] = "UNRESOLVED_COMPACT: task acquisition or compact compatibility, not isolated long transport"
        elif panel["oracle"]["exact_eos"] < .8:
            diagnoses[key] = "UNRESOLVED_LOCAL_ORACLE: task/format/local-readout failure; no remote-mechanism verdict"
        elif panel["deleted"]["exact_eos"] > .05:
            diagnoses[key] = "UNRESOLVED_SOURCE_CONTROL: inspect leakage or source-independent solution"
        elif panel["remote"]["route_exact"] >= .8 and panel["remote"]["answer_exact"] < .8:
            diagnoses[key] = "ROUTE_OUTPUT_RECOVERED_ANSWER_WEAK: transport/readout hypothesis, not V/O attribution"
        elif panel["remote"]["answer_exact"] >= .8 and panel["remote"]["ended_with_eos"] < .8:
            diagnoses[key] = "TERMINATION_FAILURE"
        elif panel["remote"]["exact_eos"] == 0:
            diagnoses[key] = "RESOLVED_REMOTE_ZERO: stop this candidate and remaining longer cells"
        else:
            diagnoses[key] = "RESOLVED_NONZERO: compare paired exact+EOS and retention before promotion"
    write_json(path / "summary.json", {"cells": result, "diagnoses": diagnoses,
               "scope": "synthetic diagnosis; paired groups, not training seeds; no natural retention claim"})
    print(json.dumps({"cells": result, "diagnoses": diagnoses}, indent=2))


def seal(args):
    """Freeze a qualified method before opening its independent blind groups."""
    summarize(args.selection_result)
    run = json.loads((args.selection_result / "run.json").read_text())
    summary = json.loads((args.selection_result / "summary.json").read_text())
    retention_path = args.retention_result / "retention.json"
    retained = json.loads(retention_path.read_text())
    if not retained.get("strict_pass") or retained["fold"] != "selection":
        raise ValueError("strict Native selection gate must pass before blind evaluation")
    if sha(args.retention_result / "examples.jsonl") != retained["examples_sha256"]:
        raise ValueError("retention raw receipt drift")
    if run["split"] != "selection" or not {1,4} <= set(run["factors"]):
        raise ValueError("selection must include physical 1x and 4x")
    for key in ("table_sha256", "gain", "adapter_sha256", "adapter_config_sha256"):
        if run[key] != retained[key]:
            raise ValueError(f"selection/retention method drift: {key}")
    if any(not text.startswith("RESOLVED_NONZERO") for text in summary["diagnoses"].values()):
        raise ValueError("selection assay not resolved/nonzero; diagnose before far evaluation")
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, {key: run[key] for key in ("table_sha256", "gain", "adapter_sha256", "adapter_config_sha256", "data_manifest_sha256")})
    lock = json.loads(args.output.read_text())
    lock.update(selection_run_sha256=sha(args.selection_result / "run.json"),
                retention_sha256=sha(retention_path), policy="8x then 16x then 32x; no retuning after reveal")
    write_json(args.output, lock)


NATURAL_TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report")


def retention_rows(manifest_path, fold, nll_only=False):
    """Import token assets only. Revalidate lengths and group-split independently.

    Existing assets remain historically exposed even after deterministic splitting.
    A new final-generalization claim needs independently sourced confirmation data.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("tokenization_executed") is not True or manifest["native_context_length"] != NATIVE:
        raise ValueError("tokenized Native-4K asset required")
    entries = {"pg19": manifest["pg19"]}
    for task in (() if nll_only else NATURAL_TASKS):
        cells = manifest["longbench"]["cells"]
        entry = cells.get(f"longbench_e:{task}", cells.get(f"longbench:{task}"))
        if entry is None:
            raise ValueError(f"missing frozen natural task: {task}")
        entries[task] = entry
    selected = []
    for task, entry in entries.items():
        path = manifest_path.parent / entry["rows_path"]
        if sha(path) != entry["rows_sha256"]:
            raise ValueError(f"natural token-asset hash drift: {task}")
        candidates = []
        for row in rows(path):
            if (task == "pg19" and row["multiplier"] != 1) or (task != "pg19" and row["bucket"] != "retention"):
                continue
            group = str(row["anchor_source_sha256"] if task == "pg19" else row["source_id"])
            candidates.append({**row, "task": task, "group": group})
        groups = sorted({r["group"] for r in candidates}, key=lambda g: canonical({"task": task, "group": g}))
        # Both halves are fixed before any new outcome, never first-N after scoring.
        group_set = set(groups[::2] if fold == "selection" else groups[1::2])
        chosen = [r for r in candidates if r["group"] in group_set]
        if len(group_set) < 4:
            raise ValueError(f"too few independent {task} groups for {fold}; supply more data")
        for row in chosen:
            reserve = 0 if task == "pg19" else int(row["generation_reserve"])
            if not row["input_ids"] or len(row["input_ids"]) + reserve > NATIVE:
                raise ValueError("Native prompt plus decode reserve exceeds physical 4K")
            if row.get("tokenizer_sha256") != manifest["tokenizer_sha256"]:
                raise ValueError("row/manifest tokenizer identity drift")
            if task == "pg19" and canonical(row["input_ids"][row["nll_target_start"]:]) != row["nll_target_sha256"]:
                raise ValueError("PG-19 target-token hash drift")
            row["asset_sha256"] = canonical(row)
        selected.extend(chosen)
    return selected


def validate_retention_tokenizer(args):
    root = args.asset_tokenizer_root or args.checkpoint
    entries = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith(".safetensors")]
    manifest = json.loads(args.retention_manifest.read_text())
    if canonical(entries) != manifest["tokenizer_sha256"]:
        raise ValueError("original asset tokenizer tree does not match manifest; provide --asset-tokenizer-root")
    if tokenizer_identity(root) != tokenizer_identity(args.checkpoint):
        raise ValueError("asset and model tokenizer files differ; re-tokenize independently before GPU")


def retention(args):
    import torch
    import torch.nn.functional as F
    from scripts.eval.longbench_metrics import TASK_METRIC_MAP, score_prediction
    validate_retention_tokenizer(args)
    selected = retention_rows(args.retention_manifest, args.retention_fold, args.nll_only)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model, tok, identity = load_runtime(args)
    start = time.monotonic()
    with (args.output / "examples.jsonl").open("w", buffering=1) as handle:
        for row in selected:
            guard_resources(model, identity, start, args)
            record = {key: row[key] for key in ("task", "group", "asset_sha256")}
            if row["task"] == "pg19":
                inputs = torch.tensor([row["input_ids"]], device="cuda")
                begin = int(row["nll_target_start"])
                if begin < 1 or len(row["input_ids"]) - begin != row["nll_target_tokens"]:
                    raise ValueError("NLL target alignment drift")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.model(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
                    logits = model.lm_head(hidden[:, begin-1:-1]).float()
                    nll = F.cross_entropy(logits.flatten(0,1), inputs[:, begin:].flatten())
                record["nll"] = float(nll)
            else:
                tokens = greedy(model, row["input_ids"], tok.eos_token_id, row["generation_reserve"])
                text = tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                score = score_prediction(row["task"], TASK_METRIC_MAP[row["task"]], text.strip(), row["references"], row.get("all_classes", []))
                ended = bool(tokens and tokens[-1] == tok.eos_token_id and tok.eos_token_id not in tokens[:-1])
                record.update(generated_token_ids=tokens, score=score, ended_with_eos=ended,
                              score_eos=score * ended)
            if any(isinstance(v, float) and not math.isfinite(v) for v in record.values()):
                raise RuntimeError("nonfinite retention metric")
            handle.write(json.dumps(record) + "\n")
    guard_resources(model, identity, start, args)
    data = list(rows(args.output / "examples.jsonl"))
    if args.nll_only:
        result = {**identity, "fold": args.retention_fold,
                  "data_sha256": canonical([r["asset_sha256"] for r in selected]),
                  "nll": sum(r["nll"] for r in data)/len(data),
                  "examples_sha256": sha(args.output / "examples.jsonl"),
                  "scope": "NLL constraint screen only; cannot select or promote generated capability"}
        if args.baseline:
            baseline_path=args.baseline / "nll_screen.json"
            if not baseline_path.is_file(): baseline_path=args.baseline / "retention.json"
            baseline = json.loads(baseline_path.read_text())
            native_rows = [r for r in rows(args.baseline / "examples.jsonl") if r["task"] == "pg19"]
            if (not baseline["table_is_native"] or baseline["gain"] != 1. or baseline["adapter_sha256"] is not None
                    or [r["asset_sha256"] for r in native_rows] != [r["asset_sha256"] for r in data]
                    or sha(args.baseline / "examples.jsonl") != baseline["examples_sha256"]):
                raise ValueError("unpaired or non-Native NLL baseline")
            result["ppl_retention"] = math.exp(min(700., baseline["nll"] - result["nll"]))
            result["nll_feasible_088"] = result["ppl_retention"] >= .88
        write_json(args.output / "nll_screen.json", result)
        print(json.dumps(result, indent=2))
        return
    cell = {task: [r for r in data if r["task"] == task] for task in ("pg19", *NATURAL_TASKS)}
    means = {task: sum(r["score"] for r in cell[task])/len(cell[task]) for task in NATURAL_TASKS}
    eos_means = {task: sum(r["score_eos"] for r in cell[task])/len(cell[task]) for task in NATURAL_TASKS}
    result = {**identity, "fold": args.retention_fold, "data_sha256": canonical([r["asset_sha256"] for r in selected]),
              "examples_sha256": sha(args.output / "examples.jsonl"),
              "nll": sum(r["nll"] for r in cell["pg19"])/len(cell["pg19"]),
              "task_macro": sum(means.values())/len(means), "task_means": means,
              "task_eos_macro": sum(eos_means.values())/len(eos_means), "task_eos_means": eos_means,
              "counts": {t: len(c) for t,c in cell.items()},
              "scope": "Native retention on imported assets; historical exposure not erased by splitting"}
    if args.baseline:
        baseline = json.loads((args.baseline / "retention.json").read_text())
        if baseline["adapter_sha256"] is not None or baseline["gain"] != 1. or not baseline["table_is_native"]:
            raise ValueError("retention baseline must be unmodified Native")
        if baseline["data_sha256"] != result["data_sha256"]:
            raise ValueError("unpaired retention rows")
        if sha(args.baseline / "examples.jsonl") != baseline["examples_sha256"]:
            raise ValueError("baseline raw receipt drift")
        result["official_gate"] = retention_verdict(baseline["nll"], result["nll"], baseline["task_macro"], result["task_macro"])
        result["eos_gate"] = retention_verdict(baseline["nll"], result["nll"], baseline["task_eos_macro"], result["task_eos_macro"])
        result["strict_pass"] = result["official_gate"]["strict_088_pass"] and result["eos_gate"]["strict_088_pass"]
        result["paired_uncertainty"] = paired_retention_intervals(list(rows(args.baseline / "examples.jsonl")), data)
    write_json(args.output / "retention.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "smoke", "evaluate", "summarize", "retention", "seal"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-contract",type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--gain", type=float, default=1.)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("selection", "blind"), default="selection")
    parser.add_argument("--factors", nargs="+", type=int, default=[1])
    parser.add_argument("--conditions", nargs="+", choices=("compact", "remote", "oracle", "deleted"), default=["compact", "remote", "oracle", "deleted"])
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--audit-margins", action="store_true")
    parser.add_argument("--retention-manifest", type=Path)
    parser.add_argument("--asset-tokenizer-root", type=Path)
    parser.add_argument("--retention-fold", choices=("selection", "confirmation"), default="selection")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--nll-only", action="store_true")
    parser.add_argument("--selection-result", type=Path)
    parser.add_argument("--retention-result", type=Path)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--train-pairs", type=int, default=128)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--explicit-format",action="store_true",help="fixed format-clarification assay; never a scorer relaxation")
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--min-headroom-gib", type=float, default=1.)
    args = parser.parse_args()
    if args.action == "summarize":
        summarize(args.output)
        return
    if args.action == "seal":
        if not args.selection_result or not args.retention_result:
            parser.error("seal requires --selection-result and --retention-result")
        seal(args)
        return
    if not args.checkpoint or not math.isfinite(args.gain) or args.gain <= 0:
        parser.error("checkpoint and finite positive gain required")
    numeric = (args.max_seconds, args.min_headroom_gib)
    if not all(math.isfinite(v) for v in numeric) or args.max_seconds <= 0 or args.min_headroom_gib < 0 or args.pair_limit < 0:
        parser.error("invalid runtime budget")
    if len(set(args.factors)) != len(args.factors) or not set(args.factors) <= {1,2,4,8,16,32}:
        parser.error("invalid or repeated physical factors")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("repeated conditions")
    if args.action == "prepare":
        if min(args.train_pairs, args.eval_pairs) < 2 or args.train_pairs % 2 or args.eval_pairs % 2:
            parser.error("pair counts must be even and at least two")
        prepare(args)
    elif args.action == "preflight":
        for split, factors in (("train", (1,2,4)), ("selection", (1,2,4)), ("blind", (1,4,8,16,32))):
            selected, _ = load_data(args.data, split, factors)
            print(json.dumps({"split": split, "validated_rows": len(selected), "cuda_loaded": False}))
        if args.retention_manifest:
            validate_retention_tokenizer(args)
            for fold in ("selection", "confirmation"):
                data = retention_rows(args.retention_manifest, fold)
                print(json.dumps({"retention_fold": fold, "rows": len(data), "cuda_loaded": False}))
    elif args.action == "smoke":
        runtime_smoke(args)
    elif args.action == "retention":
        if not args.retention_manifest:
            parser.error("--retention-manifest required")
        retention(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
