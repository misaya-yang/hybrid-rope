"""Numerical and pretrained-model checks for dynamic-position execution semantics.

Run as a module, or copy with cache_schedule_attention.py into an isolated folder.
Short-window results are a diagnostic, never context-extension evidence.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:
    from .cache_schedule_attention import flash, reference, install
except ImportError:
    from cache_schedule_attention import flash, reference, install


def emit(row):
    print(json.dumps(row), flush=True)


@torch.inference_mode()
def kernels():
    torch.manual_seed(9843)
    for dtype in (torch.float16, torch.bfloat16):
        for nk, nq, g, window in ((257, 257, 3, 32), (257, 71, 3, 32),
                                  (257, 1, 3, 32), (257, 71, 1, 32)):
            q = torch.randn(1, 4, nq, 64, device="cuda", dtype=dtype)
            k = torch.randn(1, 2, nk, 64, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            inv = 1 / (1000000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
            expected = reference(q, k, v, nk-nq, inv, g, window, 0.125).float()
            actual = flash(q, k, v, nk-nq, inv, g, window, 0.125).float()
            diff = actual - expected
            rms_relative = (diff.square().mean() / expected.square().mean()).sqrt().item()
            emit(dict(kind="kernel", dtype=str(dtype), nk=nk, nq=nq, group=g,
                      max_abs=diff.abs().max().item(), rms_relative=rms_relative))
            assert rms_relative < (0.002 if dtype == torch.float16 else 0.015)


@torch.inference_mode()
def prefix(model, ids, chunk):
    cache = DynamicCache(config=model.config)
    for start in range(0, ids.shape[1], chunk):
        out = model(ids[:, start:start+chunk], past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
    return out.logits[0, -1].float(), cache


def compare(x, y):
    lp, lq = x.log_softmax(-1), y.log_softmax(-1)
    return dict(max_abs=(x-y).abs().max().item(), rms=(x-y).square().mean().sqrt().item(),
                kl_x_y=(lp.exp()*(lp-lq)).sum().item(),
                argmax_x=x.argmax().item(), argmax_y=y.argmax().item())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/autodl-tmp/qwen25_1p5b_32k")
    p.add_argument("--data")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--local", type=int, default=32)
    p.add_argument("--chunk", type=int, default=127)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--modes", default="native,call,row,fixed")
    args = p.parse_args()
    emit(dict(kind="runtime", torch=torch.__version__, device=torch.cuda.get_device_name(),
              config=vars(args), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              attention_sha256=hashlib.sha256(Path(__file__).with_name("cache_schedule_attention.py").read_bytes()).hexdigest()))
    kernels()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    if args.data:
        row = json.loads(Path(args.data).read_text().splitlines()[args.index])
        ids = row.get("input_ids")
        if ids is None:
            ids = tok.encode(row["input"], add_special_tokens=False)
        ids = torch.tensor([ids], device="cuda")
        emit(dict(kind="input", tokens=ids.shape[1], data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  id=row.get("index", args.index), answers=row.get("outputs", row.get("answer")),
                  token_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()))
    else:
        text = "The archive contains dated records of trade, agriculture, and astronomy. " * 80
        ids = tok(text, return_tensors="pt").input_ids[:, :513].to("cuda")
    # The original SDPA path is only used for the short native-equivalence check.
    if not args.data:
        base, cache = prefix(model, ids, ids.shape[1])
        del cache
        install(model, args.window, args.local, "native", "flash")
        native, cache = prefix(model, ids, ids.shape[1])
        diff = compare(base, native)
        emit(dict(kind="native_equivalence", **diff))
        assert diff["kl_x_y"] < 0.005
        del cache
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for mode in args.modes.split(","):
        install(model, args.window, args.local, mode, "flash", ids.shape[1]+args.max_new_tokens)
        first = None
        for chunk in (ids.shape[1], args.chunk):
            t0 = time.monotonic()
            logits, cache = prefix(model, ids, chunk)
            torch.cuda.synchronize()
            prefill_seconds = time.monotonic()-t0
            current = logits.detach().clone()
            if first is None:
                first = current
            generated = []
            for _ in range(args.max_new_tokens):
                token = logits.argmax().item()
                generated.append(token)
                if token in eos:
                    break
                out = model(torch.tensor([[token]], device="cuda"), past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
                logits = out.logits[0, -1].float()
            emit(dict(kind="model", mode=mode, chunk=chunk, tokens=ids.shape[1],
                      prefill_seconds=prefill_seconds, seconds=time.monotonic()-t0,
                      comparison_to_full=compare(first, current), generated_ids=generated,
                      text=tok.decode(generated, skip_special_tokens=False), terminal_eos=generated[-1] in eos,
                      peak_memory_bytes=torch.cuda.max_memory_allocated()))
            del cache
    emit(dict(kind="complete"))


if __name__ == "__main__":
    main()
