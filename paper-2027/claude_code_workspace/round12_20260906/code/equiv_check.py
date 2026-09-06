#!/usr/bin/env python3
"""One-off equivalence check: tokenizers.Tokenizer(tokenizer.json) vs
transformers.AutoTokenizer (OLMo-2), on real PG19/FWE texts + synthetic edges.

data_prep_cpt.py's build stage must avoid importing transformers (~650MB RSS)
because the no-card container has ~1GiB RAM; this check proves the lightweight
path emits identical token ids. Two subprocess stages (RSS resets between):

  dump    : duckdb/pyarrow fetch sample texts → <out>/equiv_texts.jsonl (low RSS)
  compare : load BOTH tokenizers, assert identical ids on every text (~800MB RSS)

Usage:
  python equiv_check.py --tokenizer <model_dir> --pg19-dir <dir> --fwe-dir <dir> \
      --out <dir> [--stage all|dump|compare]
Result: <out>/equiv_result.json  {"verified": true, ...}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SYNTHETIC = [
    ("empty", ""),
    ("space", " "),
    ("ascii", "The quick brown fox jumps over the lazy dog."),
    ("digits", "3.14159265358979 " * 64),
    ("cjk", "旋位置编码的频率分配实验" * 80),
    ("emoji", "😀🎉🚀" * 60),
    ("repeat", "ab" * 40000),
    ("whitespace_mix", "line1\nline2\ttabbed\r\n  indented " * 100),
]


def stage_dump(pg19_dir: Path, fwe_dir: Path, out: Path) -> None:
    import duckdb
    import pyarrow.parquet as pq

    recs: list[dict] = []

    def add(name, text):
        recs.append({"name": name, "text": text[:1_500_000]})

    pg_files = sorted(pg19_dir.glob("train-*.parquet"))
    fwe_files = sorted(fwe_dir.glob("*.parquet"))
    assert pg_files and fwe_files

    def duckdb_fetch(sql: str):
        # Fresh connection per query: PG19 row groups are 365MB compressed and
        # duckdb keeps the ~512-667MB decode buffers on the connection.
        # Container cgroup cap is 2GiB (~1.6GiB usable); spill to disk if needed.
        con = duckdb.connect()
        con.execute("SET memory_limit='1300MB'")
        con.execute(f"SET temp_directory='{out / 'duckdb_tmp'}'")
        con.execute("SET threads=4")
        con.execute("SET preserve_insertion_order=false")
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()

    for i, (text,) in enumerate(duckdb_fetch(
            f"SELECT text FROM read_parquet('{pg_files[0]}') LIMIT 24")):
        add(f"pg19_first_{i}", text)
    for i, (text,) in enumerate(duckdb_fetch(
            f"SELECT text FROM read_parquet('{pg_files[0]}') "
            f"WHERE length(text) >= 32768 LIMIT 16")):
        add(f"pg19_long_{i}", text)

    pf = pq.ParquetFile(fwe_files[0])
    n_long = n_short = 0
    for b in pf.iter_batches(batch_size=512, columns=["text", "token_count"]):
        texts, tcs = b.column("text"), b.column("token_count")
        for i in range(b.num_rows):
            if n_short < 8:
                add(f"fwe_short_{n_short}", texts[i].as_py())
                n_short += 1
            tc = tcs[i].as_py()
            if tc is not None and int(tc) >= 16897 and n_long < 16:
                add(f"fwe_long_{n_long}", texts[i].as_py())
                n_long += 1
            if n_long >= 16 and n_short >= 8:
                break
        if n_long >= 16 and n_short >= 8:
            break

    for name, text in SYNTHETIC:
        add(f"synthetic_{name}", text)

    outp = out / "equiv_texts.jsonl"
    with outp.open("w") as w:
        for r in recs:
            w.write(json.dumps(r) + "\n")
    print(f"dumped {len(recs)} texts to {outp}", flush=True)


def stage_compare(tokenizer_dir: Path, out: Path) -> None:
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tk = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    hf = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    ids_all = hashlib.sha256()
    n = 0
    with (out / "equiv_texts.jsonl").open() as r:
        for line in r:
            rec = json.loads(line)
            text = rec["text"]
            hf_ids = hf.encode(text, add_special_tokens=False)
            tk_ids = tk.encode(text, add_special_tokens=False).ids
            if hf_ids != tk_ids:
                print(f"MISMATCH {rec['name']}: hf={len(hf_ids)} tk={len(tk_ids)}",
                      flush=True)
                for j, (a, b) in enumerate(zip(hf_ids, tk_ids)):
                    if a != b:
                        print(f"  first divergence at {j}: hf={a} tk={b}", flush=True)
                        break
                (out / "equiv_result.json").write_text(json.dumps(
                    {"verified": False, "failed": rec["name"]}, indent=2))
                sys.exit(1)
            ids_all.update(json.dumps(hf_ids).encode())
            n += 1
            print(f"ok {rec['name']} ({len(hf_ids)} ids)", flush=True)

    result = {"verified": True, "n_texts": n,
              "ids_sha256": ids_all.hexdigest(),
              "transformers": transformers.__version__,
              "tokenizers": tokenizers.__version__,
              "tokenizer_dir": str(tokenizer_dir),
              "call": "encode(text, add_special_tokens=False), both libraries"}
    (out / "equiv_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pg19-dir", required=True)
    ap.add_argument("--fwe-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "dump", "compare"])
    args = ap.parse_args()

    tokenizer_dir, pg19_dir, fwe_dir, out = (Path(args.tokenizer), Path(args.pg19_dir),
                                             Path(args.fwe_dir), Path(args.out))
    out.mkdir(parents=True, exist_ok=True)

    stages = ["dump", "compare"] if args.stage == "all" else [args.stage]
    for st in stages:
        if args.stage == "all":
            cmd = [sys.executable, __file__, "--tokenizer", str(tokenizer_dir),
                   "--pg19-dir", str(pg19_dir), "--fwe-dir", str(fwe_dir),
                   "--out", str(out), "--stage", st]
            print(f"=== stage {st} ===", flush=True)
            subprocess.run(cmd, check=True)
        else:
            {"dump": lambda: stage_dump(pg19_dir, fwe_dir, out),
             "compare": lambda: stage_compare(tokenizer_dir, out)}[st]()


if __name__ == "__main__":
    main()
