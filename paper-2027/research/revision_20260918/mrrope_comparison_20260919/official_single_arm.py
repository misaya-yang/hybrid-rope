#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "accelerate", "datasets", "tqdm"]
# ///
"""One MrRoPE-Pro arm using the archived conference implementation.

Default is an offline plan. --execute is reserved for the later GPU clone.
Run in the existing experiment environment; this script installs nothing.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

OWNER = Path(__file__).resolve().parent
ARCHIVE = OWNER / "ruler_protocol_sources/mrrope_supplement.zip"
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
         "vt", "cwe", "fwe", "qa_1", "qa_2"]
CONTEXT = 131072
DATASET = f"SaylorTwift/RULER-{CONTEXT}-llama-3.1-tokenizer-chat-template"
DATASET_REVISION = "bb8903217d901fb534cc1ca615bd3245f3c1c39a"
SAVED_INPUTS = "/root/autodl-tmp/mrrope_official_20260919/preserved_ruler/today_rope_plan_20260914/tailspline_llama_s16_128k_gate/assets/full13/inputs.jsonl"


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def plan(args):
    return {
        "arm": f"{args.family} + {args.method}",
        "table_file": args.table if args.method == "tailspline" else None,
        "shared_generation_config": args.generation_config,
        "model": args.model,
        "dataset": DATASET if args.official_dataset else "existing upstream-generated RULER gate",
        "dataset_revision": DATASET_REVISION if args.official_dataset else None,
        "inputs_jsonl": None if args.official_dataset else args.inputs, "context": CONTEXT,
        "tasks": TASKS, "samples_per_task": args.samples,
        "expected_rows": len(TASKS) * args.samples,
        "reference_length": 8192, "scale": 16, "dtype": "bfloat16",
        "attention": args.attention, "quantization": None,
        "generation": "Original evaluate_one_task; inherited do_sample; 30 new tokens; EOS or newline",
        "seed": args.seed,
        "differences_from_example": [
            "MrRoPE-Pro (--radix 16), not the Llama YaRN example command",
            "All 13 Table-2 tasks, not the script's default 7; only 128K",
            "Per-row fixed seed for reproducible resume; original script is unseeded",
            ("Dataset pinned to the inspected current revision; historical revision is unknown"
             if args.official_dataset else
             "Reuse existing 13x10 gate prompt IDs unchanged; NOT the author's HF dataset or tokenization path"),
            "Local-only model loading; raw outputs, protocol receipt and aggregate added",
        ],
        "scope": "Four-arm matched input/decoder experiment; old gate scores used a different decoder",
        "model_execution": False,
    }


def unpack(destination):
    # Extract only the required Python modules from the reviewed local archive.
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            path = Path(name)
            if not name.startswith("supplement/") or path.suffix != ".py":
                continue
            if ".." in path.parts:
                raise ValueError(name)
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    root = destination / "supplement"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "eval"))
    return root


class QuietProgress:
    def __init__(self, **kwargs):
        pass

    def update(self, n):
        pass

    def close(self):
        pass


def saved_rows(path, samples):
    grouped = {task: [] for task in TASKS}
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["task"] not in grouped or row["row_id"] in seen:
                raise ValueError("Unknown task or duplicate row identity")
            seen.add(row["row_id"])
            if row.get("selection_uses_model_outputs") is not False:
                raise ValueError("Input selection must be independent of model outputs")
            if not 0 < len(row["prompt_ids"]) <= CONTEXT or len(row["prompt_ids"]) != row["input_tokens"]:
                raise ValueError("Invalid stored input length")
            grouped[row["task"]].append(row)
    for task, rows in grouped.items():
        rows.sort(key=lambda row: row["source_order_index"])
        if len(rows) < samples or len({r["source_order_index"] for r in rows}) != len(rows):
            raise ValueError(f"Incomplete or duplicate source coverage: {task}")
        grouped[task] = rows[:samples]
    return grouped


class StoredInputTokenizer:
    """Reuse exact gate token IDs; delegate decode and stop-token lookup."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.row = None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        import torch
        if self.row is None or text != self.row["row_id"]:
            raise ValueError("Stored input adapter received a different row")
        if kwargs != {"truncation": True, "padding": True, "max_length": CONTEXT, "return_tensors": "pt"}:
            raise ValueError("Unexpected official tokenization call")
        ids = torch.tensor([self.row["prompt_ids"]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def official_evaluator(root):
    """Execute the original three functions without heavyweight unused imports."""
    import gc
    import torch

    tree = ast.parse((root / "eval/ruler.py").read_text())
    names = {"string_match_part", "string_match_all", "evaluate_one_task"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError("Unexpected conference evaluator")
    namespace = {"torch": torch, "gc": gc, "tqdm": QuietProgress}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "conference/ruler.py", "exec"), namespace)
    return namespace


class RecordingModel:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.record = None

    def generate(self, input_ids, **kwargs):
        output = self.model.generate(input_ids, **kwargs)
        new_ids = output[0, input_ids.shape[1]:].detach().cpu().tolist()
        self.record = {
            "input_tokens": input_ids.shape[1],
            "input_ids_sha256": hashlib.sha256(input_ids.detach().cpu().numpy().tobytes()).hexdigest(),
            "output_ids": new_ids,
            "prediction": self.tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
            "generate_kwargs": {k: v for k, v in kwargs.items() if k != "attention_mask"},
        }
        return output


def report(rows, expected):
    scores = {}
    for task in TASKS:
        selected = [r for r in rows if r["task"] == task]
        if selected:
            values = []
            for row in selected:
                hits = [float(ref.lower() in row["prediction"].lower()) for ref in row["references"]]
                values.append(max(hits) if "qa" in task else sum(hits) / len(hits))
            scores[task] = {"n": len(selected), "score": round(100 * sum(values) / len(values), 2)}
    complete = all(scores.get(task, {}).get("n") == expected for task in TASKS)
    return {"status": "COMPLETE" if complete else "PARTIAL", "model_execution": bool(rows),
            "rows": len(rows), "tasks": scores,
            "full13_task_equal_score": sum(v["score"] for v in scores.values()) / 13 if complete else None}


def execute(args, root):
    import torch
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, set_seed
    import model_loader

    def normalize_rope_config(config):
        # Transformers 5 moved this scalar into rope_parameters. Restore the
        # alias expected by the unmodified conference loader, without changing it.
        if not hasattr(config, "rope_theta"):
            params = getattr(config, "rope_parameters", None) or config.to_dict().get("rope_scaling") or {}
            config.rope_theta = params["rope_theta"]
        if not hasattr(config, "rope_scaling"):
            config.rope_scaling = getattr(config, "rope_parameters", None)
        return config

    class CompatibleAutoConfig:
        @staticmethod
        def from_pretrained(*a, **kw):
            return normalize_rope_config(AutoConfig.from_pretrained(*a, **kw))

    model_loader.AutoConfig = CompatibleAutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires the later GPU clone")
    checkpoint = Path(args.model)
    if not checkpoint.is_dir():
        raise RuntimeError("Download the model separately; --model must be an existing local directory")
    config = CompatibleAutoConfig.from_pretrained(checkpoint, local_files_only=True)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if (config.model_type, config.hidden_size, config.num_hidden_layers, dim,
            config.rope_theta, config.max_position_embeddings) != ("llama", 4096, 32, 128, 500000, 131072 if args.family == "llama31" else 8192):
        raise RuntimeError("Checkpoint configuration does not match selected Llama family")
    if args.family == "llama31" and (config.rope_scaling or {}).get("rope_type", (config.rope_scaling or {}).get("type")) != "llama3":
        raise RuntimeError("Expected original Llama-3.1 rope_scaling metadata")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=sys.maxsize,
                                               trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    generation = GenerationConfig.from_pretrained(args.generation_config, local_files_only=True)
    packages = ["torch", "transformers", "accelerate"]
    if args.attention == "flash_attention_2": packages.append("flash-attn")
    if args.official_dataset:
        packages.append("datasets")
    versions = {p: importlib.metadata.version(p) for p in packages}
    prepared = None if args.official_dataset else saved_rows(args.inputs, args.samples)
    receipt = {**plan(args), "config": config.to_dict(), "generation_config": generation.to_dict(),
               "versions": versions, "tokenizer": {"class": type(tokenizer).__name__,
               "eos": tokenizer.eos_token_id, "newline": tokenizer.encode("\n", add_special_tokens=False)[-1]},
               "cuda_device": torch.cuda.get_device_name(0), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "protocol.json"
    if receipt_path.exists():
        if not args.resume or json.loads(receipt_path.read_text()) != receipt:
            raise RuntimeError("Existing output: use --resume with identical protocol/environment")
    else:
        if any(out.iterdir()):
            raise RuntimeError("Output directory is not empty")
        write_json(receipt_path, receipt)
    raw = out / "generations.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r["task"], r["source_index"]): r for r in rows}
    if len(done) != len(rows) or any(t not in TASKS or not 0 <= i < args.samples for t, i in done):
        raise RuntimeError("Invalid/duplicate existing row identities")
    if report(rows, args.samples)["status"] == "COMPLETE":
        print(json.dumps(report(rows, args.samples)))
        return

    options = ["--radix", "16", "--original-max-position-embeddings", "8192"]
    if args.attention == "flash_attention_2": options.append("--flash-attention")
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    loader_args = model_loader.add_args(argparse.ArgumentParser()).parse_args(options)
    # The entire official loader and patch are reused, including BF16, device_map
    # and replacement of model.model.rotary_emb (no stacking of Llama3 scaling).
    model = model_loader.load_model_and_apply_patches(str(checkpoint), loader_args)
    model.generation_config = generation
    model.eval()
    if model.config._attn_implementation != args.attention:
        raise RuntimeError(f"Unexpected attention backend: {model.config._attn_implementation}")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("Official BF16 loading was not honored by this Transformers version")
    rotary = model.model.rotary_emb
    if type(rotary).__name__ != "LlamaMrRoPE" or rotary.inv_freq.dtype != torch.float32:
        raise RuntimeError("Official rotary replacement failed")
    if args.method == "tailspline":
        saved = json.loads(Path(args.table).read_text())
        saved = saved.get("table", saved)
        values = saved.get("inv_freq", saved.get("values_float32"))
        if values is None or len(values) != 64: raise RuntimeError("Expected saved TailSpline 64-pair table")
        rotary.inv_freq = torch.tensor(values, dtype=torch.float32, device=rotary.inv_freq.device)
        rotary.original_inv_freq = rotary.inv_freq
        rotary.attention_scaling = float(saved["gain"])
        rotary.mscale = rotary.attention_scaling
    if any(str(device) in {"cpu", "disk"} for device in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("CPU/disk offload detected; choose a GPU with sufficient memory")
    parameter_devices = sorted({str(p.device) for p in model.parameters()})
    if any(not device.startswith("cuda") for device in parameter_devices):
        raise RuntimeError(f"Model parameters not fully on GPU: {parameter_devices}")
    table = rotary.inv_freq.detach().clone()
    write_json(out / "runtime.json", {"model_execution": False, "inv_freq": table.cpu().tolist(),
        "gain": rotary.attention_scaling,
        "device_map": getattr(model, "hf_device_map", {"parameter_devices": parameter_devices}),
        "effective_generation_config": model.generation_config.to_dict(),
        "attention": model.config._attn_implementation})
    funcs = official_evaluator(root)
    eval_tokenizer = tokenizer if args.official_dataset else StoredInputTokenizer(tokenizer)
    wrapped = RecordingModel(model, eval_tokenizer)
    with raw.open("a", buffering=1) as stream:
        for task_index, task in enumerate(TASKS):
            if args.official_dataset:
                from datasets import load_dataset
                data = load_dataset(DATASET, split=task, revision=DATASET_REVISION)
            else:
                data = prepared[task]
            if len(data) < args.samples:
                raise RuntimeError(f"{task}: only {len(data)} rows")
            for index in range(args.samples):
                item = data[index]
                if args.official_dataset:
                    identity = {"dataset_fingerprint": data._fingerprint,
                                "prompt_sha256": hashlib.sha256(item["input"].encode()).hexdigest(),
                                "references": item["outputs"]}
                else:
                    eval_tokenizer.row = item
                    identity = {"row_id": item["row_id"], "source_order_index": item["source_order_index"],
                                "prompt_sha256": item["prompt_sha256"], "references": item["references"]}
                    item = {"input": item["row_id"], "outputs": item["references"]}
                if not identity["references"] or not all(isinstance(r, str) for r in identity["references"]):
                    raise RuntimeError("Expected nonempty string references")
                if (task, index) in done:
                    if any(done[task, index][key] != value for key, value in identity.items()):
                        raise RuntimeError("Dataset changed during resume")
                    continue
                seed = args.seed + task_index * args.samples + index
                set_seed(seed)
                started = time.monotonic()
                # The actual unmodified official evaluator performs tokenization,
                # generate, decode, cache clearing and scoring, one source row at a time.
                with contextlib.redirect_stdout(io.StringIO()):
                    funcs["evaluate_one_task"](wrapped, eval_tokenizer,
                        {"input": [item["input"]], "outputs": [item["outputs"]]},
                        argparse.Namespace(samples=1), task, CONTEXT)
                if not torch.equal(rotary.inv_freq, table):
                    raise RuntimeError("RoPE table changed during generation")
                row = {"task": task, "source_index": index, "seed": seed, **identity,
                       **wrapped.record, "seconds": time.monotonic() - started}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                summary = report(rows, args.samples)
                write_json(out / "report.json", summary)
                print(json.dumps({"completed": len(rows), "task": task, "index": index,
                                  "task_result": summary["tasks"][task]}), flush=True)
    print(json.dumps(report(rows, args.samples)), flush=True)


def self_test(root):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from scaled_rope.patch import patch_llama_for_mrrope_embeddings

    torch.set_num_threads(2)
    torch.manual_seed(19)
    config = LlamaConfig(vocab_size=32, hidden_size=256, intermediate_size=384,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=131072, rope_theta=500000,
        rope_scaling={"rope_type": "llama3", "factor": 8., "low_freq_factor": 1.,
                      "high_freq_factor": 4., "original_max_position_embeddings": 8192})
    model = LlamaForCausalLM(config).eval()
    patch_llama_for_mrrope_embeddings(model, 128, 16, 500000, 131072, 8192)
    rotary = model.model.rotary_emb
    expected = json.loads((OWNER / "cpu_table_audit/conference_supplement_table.json").read_text())
    assert torch.equal(rotary.inv_freq, torch.tensor(expected["values_float32"]))
    assert rotary.attention_scaling == expected["gain"]
    positions = torch.tensor([[0, 8191, 65535, 131071]])
    x = torch.zeros((1, 4, 256), dtype=torch.bfloat16)
    cosine, sine = rotary(x, positions)
    angle = positions.float()[..., None] * rotary.inv_freq
    angle = torch.cat((angle, angle), dim=-1)
    assert torch.equal(cosine, (angle.cos() * expected["gain"]).bfloat16())
    assert torch.equal(sine, (angle.sin() * expected["gain"]).bfloat16())
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full = model(ids, position_ids=positions).logits
        first = model(ids[:, :3], position_ids=positions[:, :3], use_cache=True)
        tail = model(ids[:, 3:], position_ids=positions[:, 3:], past_key_values=first.past_key_values).logits
    torch.testing.assert_close(full[:, -1:], tail, atol=1e-5, rtol=1e-5)
    funcs = official_evaluator(root)
    # Exercise the original tokenization/generation/decode path with a spy;
    # this checks inherited sampling and stop tokens without pretrained weights.
    class TokenizerSpy:
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            assert kwargs == {"truncation": True, "padding": True,
                              "max_length": CONTEXT, "return_tensors": "pt"}
            return {"input_ids": torch.tensor([[1, 3]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

        def encode(self, text, **kwargs):
            assert text == "\n" and kwargs == {"add_special_tokens": False}
            return [13]

        def decode(self, tokens, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return " Alpha "

    class GenerateSpy:
        def generate(self, input_ids, **kwargs):
            mask = kwargs.pop("attention_mask")
            assert tuple(mask.shape) == tuple(input_ids.shape)
            assert kwargs == {"output_attentions": False, "max_new_tokens": 30,
                "num_beams": 1, "temperature": 0.7, "eos_token_id": [2, 13], "pad_token_id": 2}
            return torch.cat((input_ids, torch.tensor([[4]], device=input_ids.device)), dim=1)

    spy = RecordingModel(GenerateSpy(), TokenizerSpy())
    score = funcs["evaluate_one_task"](spy, spy.tokenizer,
        {"input": ["prompt"], "outputs": [["alpha", "beta"]]},
        argparse.Namespace(samples=1), "niah_single_1", CONTEXT)
    assert score == 50 and spy.record["prediction"] == "Alpha"
    stored = StoredInputTokenizer(TokenizerSpy())
    stored.row = {"row_id": "test", "prompt_ids": [1, 128000, 13, 7]}
    tokens = stored("test", truncation=True, padding=True, max_length=CONTEXT, return_tensors="pt")
    assert tokens["input_ids"].tolist() == [[1, 128000, 13, 7]]
    assert tokens["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert stored.encode("\n", add_special_tokens=False) == [13]
    sample_rows = [{"task": t, "source_index": i, "prediction": p, "references": r}
        for t in TASKS for i, (p, r) in enumerate([("Alpha", ["alpha", "beta"]), ("none", ["beta"])])]
    summary = report(sample_rows, 2)
    for task in TASKS:
        fn = funcs["string_match_part" if "qa" in task else "string_match_all"]
        assert summary["tasks"][task]["score"] == fn(["Alpha", "none"], [["alpha", "beta"], ["beta"]])
    assert report(sample_rows[:-1], 2)["full13_task_equal_score"] is None
    return {"status": "CPU_CHECKS_PASSED", "pretrained_model_execution": False,
            "checks": ["official patch replaces scaled rotary", "64/64 exact CPU frequencies and gain",
                       "BF16 cos/sin at positions up to 131071", "tiny random Llama cached/full forward",
                       "original tokenizer/generate/stop/decode call path with spy",
                       "stored gate input IDs preserved without retokenization",
                       "official scoring equivalence and incomplete-coverage handling"],
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/autodl-tmp/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--family", choices=["llama31", "llama3"], default="llama31")
    parser.add_argument("--method", choices=["mrpro", "tailspline"], default="mrpro")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="flash_attention_2")
    parser.add_argument("--generation-config", default="/root/autodl-tmp/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--table", default=str(Path(SAVED_INPUTS).parents[2] / "tables/tailspline.json"))
    parser.add_argument("--output", default="/root/autodl-tmp/mrrope_official_20260919/results/full13_128k")
    parser.add_argument("--samples", type=int, default=10)
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--inputs", default=SAVED_INPUTS)
    data_mode.add_argument("--official-dataset", action="store_true",
                           help="Use author's HF data instead of existing gate IDs")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not args.execute and not args.self_test:
        print(json.dumps(plan(args), indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="mrrope_official_") as temporary:
        root = unpack(Path(temporary))
        if args.self_test:
            print(json.dumps(self_test(root), indent=2))
        else:
            execute(args, root)


if __name__ == "__main__":
    main()
