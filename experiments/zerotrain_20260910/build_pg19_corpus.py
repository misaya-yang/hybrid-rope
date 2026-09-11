#!/usr/bin/env python3
"""Build a properly-powered long-context NLL corpus from PG19.

RUN ON THE SERVER (writes to /root/autodl-tmp/longtext/).

WHY.  The Qwen 4x point in THE_ANSWER_20260911.md is the one underpowered
number in the table: the existing corpus is 16 x 32769 tokens, and a 4x sample
on Qwen (native window 32768) needs 131073 tokens, so concatenating gives only
FOUR samples.  Four documents cannot settle a sign -- the observed BM-vs-MrRoPE
difference has t=-3.34 but the sign test is 4/4, p=0.125.

PG19 test gives 100 books, 40.4M characters, 16 of them longer than 131k tokens
and the longest 4.5M characters.  Cutting each book into NON-OVERLAPPING
131073-token segments gives roughly 75 independent-enough samples, which is a
4.3x reduction in SE.

Chunks from the same book are not fully independent, so the book id is recorded
in the manifest: a reviewer can cluster by book, and the pre-registration states
which statistic is primary.

Output format matches the existing instruments: doc_NNN.npy, 1-D int32, at least
the requested length (they are exactly the requested length here).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import transformers

SRC = "/root/autodl-tmp/longtext/pg19test.parquet"
OUT = pathlib.Path("/root/autodl-tmp/longtext/prepared_pg19_4x")
TOK = "/root/autodl-tmp/qwen25_1p5b_32k"
LEN = 131073


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = transformers.AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    t = pq.read_table(SRC)
    texts = t.column("text").to_pylist()
    titles = t.column("short_book_title").to_pylist()
    print(f"books: {len(texts)}")

    rows, idx = [], 0
    for bi, (title, s) in enumerate(zip(titles, texts)):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        n = len(ids) // LEN
        for c in range(n):
            seg = np.asarray(ids[c * LEN:(c + 1) * LEN], dtype=np.int32)
            if seg.shape[0] != LEN:
                continue
            p = OUT / f"doc_{idx:03d}.npy"
            np.save(p, seg)
            rows.append(dict(file=p.name, book=bi, title=str(title)[:60],
                             chunk=c, tokens=int(seg.shape[0])))
            idx += 1
        if bi % 20 == 0:
            print(f"  book {bi:3d}  segs so far {idx}", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(
        dict(source=SRC, tokenizer=TOK, length=LEN, n_docs=idx,
             books=len(texts),
             note="non-overlapping chunks; chunks of one book share a book id"),
        indent=1))
    (OUT / "rows.json").write_text(json.dumps(rows))
    print(f"\nwrote {idx} documents of {LEN} tokens to {OUT}")
    print(f"  books contributing: {len({r['book'] for r in rows})}")
    print(f"  SE reduction vs n=4: {np.sqrt(idx/4):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
