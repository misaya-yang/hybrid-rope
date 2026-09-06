#!/usr/bin/env python3
"""Track B Phase-A CPT data prep: 2048 real 16K sequences (1024 PG19 + 1024 FineWeb-Edu).

THREE-STAGE LOW-RESOURCE PIPELINE for the no-card container
(cgroup: 2GiB RAM, 0.5 CPU core; platform may kill long CPU-saturating
processes — every stage is a separate resumable subprocess, all heavy
lifting is vectorized/IO-bound, never per-row Python loops over big data):

  stage pg19 : duckdb (SQL char-length filter) → candidate books → jsonl on disk
  stage fwe  : duckdb (SQL token_count prefilter) → candidate docs → jsonl
  stage build: tokenizers-only (tokenizer.json, no transformers), RESUMABLE
               (append-mode segment bin + progress sidecar per source/split),
               encode candidates, gate >= SEQ_LEN+1 tokens, take first
               1024 train + 64 val per source (file-disjoint), freeze npy+manifest
  stage all  : run the three stages as separate subprocesses (RSS resets between)

Determinism: scans use threads=1 + preserve_insertion_order=true → candidate
order = parquet row order; candidates files are the frozen source of truth for
build (their sha256 is recorded in the manifest).

Selection rules from the plan (Section 7.3), enforced here:
  - 1024 sequences from PG19 train books + 1024 from FineWeb-Edu long documents.
  - Train/validation split BY SOURCE FILE: validation documents come only from
    the LAST sorted parquet file per source; no document crosses splits.
  - Documents shorter than SEQ_LEN+1 tokens are dropped, never padded/repeated
    into fake long context. Each accepted document contributes one contiguous
    SEQ_LEN+1-token segment from its head.
  - Plain causal LM packing: stored array is SEQ_LEN+1 token ids;
    input = [:-1], targets = [1:]. No artificial EOS at truncation points.

The tokenizer used in `build` is `tokenizers.Tokenizer.from_file(tokenizer.json)`;
its equivalence with `AutoTokenizer.encode(..., add_special_tokens=False)` was
verified by code/equiv_check.py on 72 texts (result recorded in manifest).

Usage (unchanged CLI):
  python data_prep_cpt.py \
      --tokenizer /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
      --pg19-dir /root/autodl-tmp/claude_round12_20260906/datasets/pg19/data \
      --fwe-dir /root/autodl-tmp/fineweb_edu/sample/10BT \
      --out /root/autodl-tmp/claude_round12_20260906/data/cpt \
      [--stage all|pg19|fwe|build]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SEQ_LEN = 16384
N_TRAIN_PER_SOURCE = 1024
N_VAL_PER_SOURCE = 64
CHAR_MIN = SEQ_LEN * 2                 # cheap char prefilter (token gate is authoritative)
FWE_TOKENCOUNT_MIN = SEQ_LEN + 1 + 512  # external token_count prefilter (gate still authoritative)
CAND_TRAIN = 1792                      # candidate margin over 1024 (char gate is loose)
CAND_VAL = 128                         # candidate margin over 64
TEXT_CAP_CHARS = 1_200_000             # never store/encode more chars than this (~300K tokens;
                                       # we only ever use the first 16385, far below)
SEG_BYTES = (SEQ_LEN + 1) * 4          # int32 segment size


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_caches() -> None:
    """Reclaim page cache inside the 2GiB cgroup between stages (best effort)."""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        print("drop_caches: ok", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"drop_caches skipped: {e}", flush=True)


def duckdb_scan(path: Path, where: str, limit: int, tmp_dir: Path):
    """Stream texts from one parquet file, deterministic row order."""
    import duckdb

    con = duckdb.connect()
    # PG19 row groups are 365MB compressed (1000 books): decoding one column
    # chunk needs ~512MB+256MB buffers. Container cgroup cap is 2GiB.
    con.execute("SET memory_limit='1300MB'")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET threads=1")                # 0.5-core quota; order determinism
    con.execute("SET preserve_insertion_order=true")
    try:
        cur = con.execute(
            f"SELECT text FROM read_parquet('{path}') WHERE {where} LIMIT {limit}")
        while True:
            rows = cur.fetchmany(16)
            if not rows:
                break
            for (text,) in rows:
                yield text
    finally:
        con.close()


# --------------------------------------------------------------------------
# stage pg19: candidate books → jsonl
# --------------------------------------------------------------------------
def stage_pg19(pg19_dir: Path, out: Path) -> None:
    files = sorted(pg19_dir.glob("train-*.parquet"))
    assert len(files) >= 2, "need >=2 PG19 shards for file-disjoint val"
    train_files, val_files = files[:-1], files[-1:]
    where = f"length(text) >= {CHAR_MIN}"

    for split, fl, cap in (("train", train_files, CAND_TRAIN),
                           ("validation", val_files, CAND_VAL)):
        outp = out / f"candidates_pg19_{split}.jsonl"
        n = 0
        with outp.open("w") as w:
            for f in fl:
                if n >= cap:
                    break
                for text in duckdb_scan(f, where, cap - n, out / "duckdb_tmp"):
                    rec = {"source": "pg19", "split": split, "file": f.name,
                           "doc_index_eligible": n, "n_chars": len(text),
                           "text": text[:TEXT_CAP_CHARS]}
                    w.write(json.dumps(rec) + "\n")
                    n += 1
                    if n >= cap:
                        break
                print(f"pg19/{split}: {n}/{cap} candidates after {f.name}", flush=True)
        assert n > 0, f"no PG19 candidates for {split}"
        print(f"pg19/{split}: wrote {outp} ({n} candidates)", flush=True)


# --------------------------------------------------------------------------
# stage fwe: candidate docs → jsonl (duckdb SQL filter; FWE row groups tiny)
# --------------------------------------------------------------------------
def stage_fwe(fwe_dir: Path, out: Path) -> None:
    files = sorted(fwe_dir.glob("*.parquet"))
    assert len(files) >= 2, "need >=2 FWE shards for file-disjoint val"
    train_files, val_files = files[:-1], files[-1:]
    where = f"token_count >= {FWE_TOKENCOUNT_MIN} AND length(text) >= {CHAR_MIN}"

    for split, fl, cap in (("train", train_files, CAND_TRAIN),
                           ("validation", val_files, CAND_VAL)):
        outp = out / f"candidates_fwe_{split}.jsonl"
        n = 0
        with outp.open("w") as w:
            for f in fl:
                if n >= cap:
                    break
                for text in duckdb_scan(f, where, cap - n, out / "duckdb_tmp"):
                    rec = {"source": "fineweb_edu", "split": split, "file": f.name,
                           "doc_index_eligible": n, "n_chars": len(text),
                           "text": text[:TEXT_CAP_CHARS]}
                    w.write(json.dumps(rec) + "\n")
                    n += 1
                    if n >= cap:
                        break
                print(f"fwe/{split}: {n}/{cap} candidates after {f.name}", flush=True)
        assert n > 0, f"no FWE candidates for {split}"
        print(f"fwe/{split}: wrote {outp} ({n} candidates)", flush=True)


# --------------------------------------------------------------------------
# stage build: tokenizers-only encode + gate + freeze (RESUMABLE)
# --------------------------------------------------------------------------
def _encode_candidates(tok, out: Path, source: str, prefix: str, split: str, quota: int):
    """Encode candidates for one (source, split); resumable via append-mode
    segment bin + progress sidecar. Returns (segments list, provenance list)."""
    cand_path = out / f"candidates_{prefix}_{split}.jsonl"
    bin_path = out / f"partial_{prefix}_{split}.bin"
    prog_path = out / f"partial_{prefix}_{split}.progress.jsonl"

    # ---- recover state (crash-consistent: count only fully-paired records)
    done = 0
    progress: list[dict] = []
    if prog_path.exists():
        with prog_path.open() as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                try:
                    progress.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # torn tail write; ignore the rest
    n_bin = bin_path.stat().st_size // SEG_BYTES if bin_path.exists() else 0
    done = min(len(progress), n_bin)
    progress = progress[:done]
    if bin_path.exists() and bin_path.stat().st_size > done * SEG_BYTES:
        with bin_path.open("r+b") as b:  # truncate torn tail segment
            b.truncate(done * SEG_BYTES)
    accepted_lines = {p["cand_line"] for p in progress}
    if done:
        print(f"{source}/{split}: resuming with {done}/{quota} segments done", flush=True)
    if done >= quota:
        segs = np.fromfile(bin_path, dtype=np.int32, count=done * (SEQ_LEN + 1))
        return [segs[i * (SEQ_LEN + 1):(i + 1) * (SEQ_LEN + 1)] for i in range(done)], \
               [p["prov"] for p in progress]

    if done:
        prev = np.fromfile(bin_path, dtype=np.int32).reshape(done, SEQ_LEN + 1)
        bucket: list[np.ndarray] = [prev[i] for i in range(done)]
    else:
        bucket = []
    prov = [p["prov"] for p in progress]

    with cand_path.open() as r, bin_path.open("ab") as b, prog_path.open("a") as pr:
        for i, line in enumerate(r):
            if len(bucket) >= quota:
                break
            if i in accepted_lines:
                continue  # already encoded in a previous (killed) run
            rec = json.loads(line)
            ids = tok.encode(rec["text"], add_special_tokens=False).ids
            if len(ids) < SEQ_LEN + 1:
                print(f"{source}/{split}: cand {i} short ({len(ids)} tok) — dropped",
                      flush=True)
                continue
            seg = np.asarray(ids[:SEQ_LEN + 1], dtype=np.int32)
            b.write(seg.tobytes())
            b.flush()
            p = {"source": source, "split": split, "file": rec["file"],
                 "doc_index_eligible": rec["doc_index_eligible"],
                 "doc_tokens": len(ids), "segment_start": 0,
                 "segment_len": SEQ_LEN + 1}
            pr.write(json.dumps({"cand_line": i, "prov": p}) + "\n")
            pr.flush()
            bucket.append(seg)
            prov.append(p)
            print(f"{source}/{split}: {len(bucket)}/{quota} (cand {i}, {len(ids)} tok)",
                  flush=True)
    assert len(bucket) == quota, \
        f"{source}/{split}: only {len(bucket)}/{quota} candidates passed token gate"
    return bucket, prov


def stage_build(tokenizer_dir: Path, pg19_dir: Path, fwe_dir: Path, out: Path) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    pg_train, pg_train_prov = _encode_candidates(tok, out, "pg19", "pg19", "train",
                                                 N_TRAIN_PER_SOURCE)
    pg_val, pg_val_prov = _encode_candidates(tok, out, "pg19", "pg19", "validation",
                                             N_VAL_PER_SOURCE)
    fwe_train, fwe_train_prov = _encode_candidates(tok, out, "fineweb_edu", "fwe", "train",
                                                   N_TRAIN_PER_SOURCE)
    fwe_val, fwe_val_prov = _encode_candidates(tok, out, "fineweb_edu", "fwe", "validation",
                                               N_VAL_PER_SOURCE)

    def dump(name, docs):
        arr = np.stack(docs).astype(np.int32)
        np.save(out / f"{name}.npy", arr)
        return arr.shape, sha256_file(out / f"{name}.npy")

    tr_shape, tr_sha = dump("train_2048x16385", pg_train + fwe_train)
    va_shape, va_sha = dump("validation", pg_val + fwe_val)

    equiv_path = out / "equiv_result.json"
    manifest = {
        "status": "CPT_DATA_FROZEN_V1",
        "seq_len": SEQ_LEN, "layout": "array[t, SEQ_LEN+1]; input=[ :-1], targets=[1:]",
        "train": {"shape": list(tr_shape), "sha256": tr_sha,
                  "order": "pg19 docs then fineweb_edu docs (first-eligible order)"},
        "validation": {"shape": list(va_shape), "sha256": va_sha,
                       "order": "pg19 val docs then fineweb_edu val docs"},
        "split_rule": "validation candidates come only from the last sorted parquet file "
                      "per source; no document crosses splits",
        "provenance": pg_train_prov + fwe_train_prov + pg_val_prov + fwe_val_prov,
        "pg19_files_sha256": {f.name: sha256_file(f)
                              for f in sorted(pg19_dir.glob("train-*.parquet"))},
        "fwe_files_used": [f.name for f in sorted(fwe_dir.glob("*.parquet"))],
        "candidates_sha256": {p.name: sha256_file(p)
                              for p in sorted(out.glob("candidates_*.jsonl"))},
        "tokenizer": str(tokenizer_dir),
        "tokenizer_equiv": json.loads(equiv_path.read_text()) if equiv_path.exists()
                           else "equiv_result.json missing",
        "prepared_via": "3-stage low-resource pipeline (duckdb scan x2 / tokenizers-only "
                        "resumable encode); selection rules identical to the pre-registered "
                        "in-process collect()",
        "note": "documents below SEQ_LEN+1 tokens dropped; one segment per document; "
                "candidate texts stored truncated to 1.2M chars (far beyond the 16385 used)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"train": list(tr_shape), "validation": list(va_shape)}, indent=2),
          flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pg19-dir", required=True)
    ap.add_argument("--fwe-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "pg19", "fwe", "build"])
    args = ap.parse_args()

    tokenizer_dir, pg19_dir, fwe_dir, out = (Path(args.tokenizer), Path(args.pg19_dir),
                                             Path(args.fwe_dir), Path(args.out))
    out.mkdir(parents=True, exist_ok=True)

    stages = ["pg19", "fwe", "build"] if args.stage == "all" else [args.stage]
    for st in stages:
        drop_caches()
        if args.stage == "all":
            # separate subprocess per stage → RSS resets between stages
            cmd = [sys.executable, __file__, "--tokenizer", str(tokenizer_dir),
                   "--pg19-dir", str(pg19_dir), "--fwe-dir", str(fwe_dir),
                   "--out", str(out), "--stage", st]
            print(f"=== stage {st} ===", flush=True)
            subprocess.run(cmd, check=True)
        else:
            {"pg19": lambda: stage_pg19(pg19_dir, out),
             "fwe": lambda: stage_fwe(fwe_dir, out),
             "build": lambda: stage_build(tokenizer_dir, pg19_dir, fwe_dir, out)}[st]()
        (out / f".{st}.done").touch()
        print(f"=== stage {st} done ===", flush=True)


if __name__ == "__main__":
    main()
