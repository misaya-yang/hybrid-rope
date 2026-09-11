"""Build the screen's corpus: documents longer than the native window.

The screen scores only positions BEYOND the window, so a document must exceed
`window + keep`.  On Qwen2.5-3B that is 32768 + keep, and the prepared corpora the
project already has are mostly 32769 tokens -- one token past the window, which
is not a long-range test.  Three sources, cheapest first:

  1. `--concat K` over an existing tokenised JSONL.  Concatenating K of the
     project's own 32769-token rows gives 65538 (K=2, a 2x test) or 131076 (K=4,
     a 4x test) with no new data and no re-tokenisation.  This is the intended
     path when the server already has `prepared_nll_01` or `long_inputs`.
  2. `--jsonl` of raw text plus the checkpoint's own tokenizer.
  3. `--from-ids` re-chunking an ids JSONL into fixed-length windows.

WHAT THIS REFUSES TO DO.  It will not pad, truncate, or pad-and-truncate to hit a
target length.  Every one of those changes which positions are scored, and the
resulting metric is still a number, so the mistake would be invisible in the
receipt.  A row that is too short is dropped and counted; if too few remain, the
build fails and says how many it had.

The output is JSONL with an `ids` field, which is exactly what `corpus.load_docs`
reads, plus a sidecar receipt recording the source and the lengths so the screen's
numbers can be traced back to the tokens they came from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _rows(path):
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield lineno, json.loads(line)


def _ids_of(rec, lineno, path, tokenizer, text_key="text"):
    if "ids" in rec:
        return [int(x) for x in rec["ids"]]
    if text_key in rec:
        if tokenizer is None:
            raise ValueError(f"{path}:{lineno} has text but no tokenizer was "
                             "supplied (pass --model so the checkpoint's own "
                             "tokenizer is used)")
        return list(tokenizer(rec[text_key], add_special_tokens=False)["input_ids"])
    raise ValueError(f"{path}:{lineno} has neither `ids` nor `{text_key}`; "
                     f"keys are {sorted(rec)}")


def concat_rows(path, k, min_tokens, tokenizer=None, text_key="text", limit=None):
    """Join exactly `k` consecutive rows into one document.

    EXACTLY k, AND A TOO-SHORT RESULT IS DROPPED AND COUNTED.  The obvious
    implementation -- "accumulate until long enough, then emit" -- silently turns
    k into a lower bound and makes the too-short branch unreachable, so a caller
    who asked for k=1 over 32769-token rows would receive 65538-token documents
    and never learn that k was ignored.  That version was written first; this
    test caught it.  Dropping is the honest outcome: padding or truncating would
    move the scored positions, and a document being too short is a fact about the
    source rather than something to repair.
    """
    out, dropped, short = [], 0, 0
    group = []
    for lineno, rec in _rows(path):
        ids = _ids_of(rec, lineno, path, tokenizer, text_key)
        if not ids:
            dropped += 1
            continue
        group.append(ids)
        if len(group) < int(k):
            continue
        joined = [t for g in group for t in g]
        group = []
        if len(joined) < min_tokens:
            short += 1
            continue
        out.append(joined)
        if limit is not None and len(out) >= int(limit):
            break
    return out, dict(source=str(path), mode=f"concat_{k}", dropped=dropped,
                     too_short=short, emitted=len(out),
                     rows_left_over=len(group))


def window_rows(path, length, min_tokens, tokenizer=None, text_key="text",
                limit=None):
    """Re-chunk the stream into fixed-length windows of `length` tokens."""
    buf, out, dropped = [], [], 0
    for lineno, rec in _rows(path):
        ids = _ids_of(rec, lineno, path, tokenizer, text_key)
        if not ids:
            dropped += 1
            continue
        buf.extend(ids)
        while len(buf) >= length:
            out.append(list(buf[:length]))
            del buf[:length]
            if limit is not None and len(out) >= int(limit):
                break
        if limit is not None and len(out) >= int(limit):
            break
    return out, dict(source=str(path), mode=f"window_{length}", dropped=dropped,
                     trailing_leftover=len(buf))


def main(argv=None):
    ap = argparse.ArgumentParser(description="build a beyond-window corpus")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", help="JSONL of rows with `ids` or `text`")
    src.add_argument("--from-ids", help="alias of --jsonl, re-chunked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concat", type=int, default=None,
                    help="join this many consecutive rows into one document")
    ap.add_argument("--length", type=int, default=None,
                    help="re-chunk the stream into windows of this many tokens")
    ap.add_argument("--model", default=None,
                    help="checkpoint whose tokenizer tokenises `text` rows")
    ap.add_argument("--window", type=int, required=True,
                    help="the checkpoint's native window")
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--text-key", default="text")
    args = ap.parse_args(argv)

    if (args.concat is None) == (args.length is None):
        print("pass exactly one of --concat or --length", file=sys.stderr)
        return 2
    min_tokens = args.window + args.keep + 1
    path = args.jsonl or args.from_ids

    tok = None
    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

    if args.concat:
        docs, meta = concat_rows(path, args.concat, min_tokens, tok,
                                 args.text_key, args.limit)
    else:
        docs, meta = window_rows(path, args.length, min_tokens, tok,
                                 args.text_key, args.limit)

    lengths = [len(d) for d in docs]
    if not docs:
        print(f"no documents reached {min_tokens} tokens from {path}", file=sys.stderr)
        return 2
    short = [l for l in lengths if l <= args.window + args.keep]
    if short:
        print(f"REFUSING: {len(short)} documents are not longer than "
              f"window+keep={args.window + args.keep}; the metric would be an "
              "in-window number and would still return a value", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(dict(ids=d)) + "\n")
    receipt = dict(**meta, out=str(out), n_docs=len(docs),
                   min_length=int(min(lengths)), max_length=int(max(lengths)),
                   window=int(args.window), keep=int(args.keep),
                   min_tokens_required=int(min_tokens),
                   all_beyond_window=True,
                   note="documents are never padded or truncated to a target "
                        "length; a row that is too short is dropped and counted")
    (out.parent / (out.name + ".receipt.json")).write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
