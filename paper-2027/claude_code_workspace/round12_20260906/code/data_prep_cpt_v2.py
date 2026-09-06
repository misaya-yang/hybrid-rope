#!/usr/bin/env python3
"""500M-token CPT data prep V2 (R12_CPT_DATA_500M_V1).

PG19-only training corpus, stream-encoded (no candidate text stored on disk):
  - train shards: every train-*.parquet in --pg19-dir EXCEPT the V1
    validation shard (--exclude-shard, default train-00002-of-00023.parquet)
    — file-disjoint from the frozen V1 validation split.
  - each eligible book (>= SEQ_LEN+1 tokens) contributes up to
    MAX_SEGMENTS_PER_BOOK contiguous non-overlapping head segments:
    segment k = ids[k*SEQ_LEN : (k+1)*SEQ_LEN + 1], so input=[ :-1] and
    targets=[1:] exactly as in V1 (array[t, SEQ_LEN+1] int32).
  - deterministic order: sorted shard order, parquet row order (duckdb
    threads=1 + preserve_insertion_order), first-eligible, stop at quota.
  - validation: REUSES the frozen V1 validation.npy (path + sha256 recorded
    in the manifest); no new validation documents are built.
  - resumable: append-mode segment bin + progress sidecar + per-shard done
    markers; deterministic scan order makes row-skip on resume exact.

Quota: 30,592 segments x 16,384 input tokens = 501,219,328 tokens >= 500M.

The tokenizer is tokenizers.Tokenizer.from_file(tokenizer.json); its
equivalence with AutoTokenizer was already frozen in the V1 equiv_result.json
(referenced, not re-run).

Usage:
  python data_prep_cpt_v2.py \
      --tokenizer /root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct \
      --pg19-dir /root/autodl-tmp/claude_round12_20260906/datasets/pg19/data \
      --v1-data /root/autodl-tmp/claude_round12_20260906/data/cpt \
      --out /root/autodl-tmp/claude_round12_20260906/data/cpt_500m
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEQ_LEN = 16384
QUOTA_SEGMENTS = 30592                 # x 16384 = 501,219,328 tokens >= 500M
MAX_SEGMENTS_PER_BOOK = 8
CHAR_MIN = SEQ_LEN * 3                 # loose char prefilter (token gate authoritative)
SEG_BYTES = (SEQ_LEN + 1) * 4          # int32 segment size
TEXT_CAP_CHARS = 480_000               # 8 segments need <=131,073 tokens ~ <=450K chars;
                                       # cap never drops a segment the cap-8 rule allows
                                       # for any text with >= 3.66 chars/token


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duckdb_scan(path: Path, where: str, tmp_dir: Path):
    """Stream texts from one parquet file, deterministic row order."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit='1300MB'")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    try:
        cur = con.execute(f"SELECT text FROM read_parquet('{path}') WHERE {where}")
        while True:
            rows = cur.fetchmany(16)
            if not rows:
                break
            for (text,) in rows:
                yield text
    finally:
        con.close()


def recover_state(bin_path: Path, prog_path: Path):
    """Crash-consistent recovery: pair bin segments with progress records."""
    progress: list[dict] = []
    if prog_path.exists():
        for line in prog_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                progress.append(json.loads(line))
            except json.JSONDecodeError:
                break  # torn tail write; ignore the rest
    n_bin = bin_path.stat().st_size // SEG_BYTES if bin_path.exists() else 0
    n_seg_prog = sum(p["n_segments"] for p in progress if p.get("kind") == "book")
    done_files = {p["file"] for p in progress if p.get("kind") == "file_done"}
    n_valid = min(n_bin, n_seg_prog)
    # truncate torn tail (segment written but progress not flushed)
    if bin_path.exists() and bin_path.stat().st_size > n_valid * SEG_BYTES:
        with bin_path.open("r+b") as b:
            b.truncate(n_valid * SEG_BYTES)
    # roll progress back to the last fully paired book
    cum = 0
    keep = []
    for p in progress:
        if p.get("kind") == "book":
            if cum + p["n_segments"] > n_valid:
                break
            cum += p["n_segments"]
        keep.append(p)
    return keep, n_valid, done_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pg19-dir", required=True)
    ap.add_argument("--v1-data", required=True,
                    help="V1 frozen data dir (validation.npy + manifest + equiv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-shard", default="train-00002-of-00023.parquet",
                    help="V1 validation shard; never used for train")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp_dir = out / "duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))

    shards = sorted(Path(args.pg19_dir).glob("train-*.parquet"))
    assert len(shards) >= 4, f"expected >=4 PG19 shards, got {len(shards)}"
    train_shards = [s for s in shards if s.name != args.exclude_shard]
    assert all(s.name != args.exclude_shard for s in train_shards)

    bin_path = out / "partial_train.bin"
    prog_path = out / "partial_train.progress.jsonl"
    progress, n_done, done_files = recover_state(bin_path, prog_path)
    if n_done >= QUOTA_SEGMENTS:
        print(f"quota already met ({n_done}/{QUOTA_SEGMENTS}); freezing", flush=True)
    else:
        if n_done:
            print(f"resuming: {n_done}/{QUOTA_SEGMENTS} segments, "
                  f"{len(done_files)} shards done", flush=True)
        processed_this_shard = {}
        for p in progress:
            if p.get("kind") == "book":
                processed_this_shard[p["file"]] = \
                    max(processed_this_shard.get(p["file"], 0), p["book_seq"] + 1)
        total = n_done
        with bin_path.open("ab") as b, prog_path.open("a") as pr:
            for shard in train_shards:
                if total >= QUOTA_SEGMENTS:
                    break
                if shard.name in done_files:
                    continue
                skip = processed_this_shard.get(shard.name, 0)
                seen = 0
                for text in duckdb_scan(shard, f"length(text) >= {CHAR_MIN}", tmp_dir):
                    if seen < skip:
                        seen += 1
                        continue
                    ids = tok.encode(text[:TEXT_CAP_CHARS],
                                     add_special_tokens=False).ids
                    n_seg = 0
                    if len(ids) >= SEQ_LEN + 1:
                        n_seg = min(MAX_SEGMENTS_PER_BOOK,
                                    (len(ids) - SEQ_LEN - 1) // SEQ_LEN + 1)
                        n_seg = min(n_seg, QUOTA_SEGMENTS - total)
                        for k in range(n_seg):
                            seg = np.asarray(
                                ids[k * SEQ_LEN:k * SEQ_LEN + SEQ_LEN + 1],
                                dtype=np.int32)
                            b.write(seg.tobytes())
                        b.flush()
                    total += n_seg
                    pr.write(json.dumps({"kind": "book", "file": shard.name,
                                         "book_seq": seen, "n_tokens": len(ids),
                                         "n_segments": n_seg}) + "\n")
                    pr.flush()
                    seen += 1
                    if total % 256 < n_seg or total >= QUOTA_SEGMENTS:
                        print(f"{shard.name}: {total}/{QUOTA_SEGMENTS} segments "
                              f"(book {seen}, {len(ids)} tok, +{n_seg})", flush=True)
                    if total >= QUOTA_SEGMENTS:
                        break
                pr.write(json.dumps({"kind": "file_done", "file": shard.name,
                                     "books_scanned": seen}) + "\n")
                pr.flush()
                done_files.add(shard.name)
                print(f"shard done {shard.name}: {seen} books scanned, "
                      f"total {total}/{QUOTA_SEGMENTS}", flush=True)
        assert total >= QUOTA_SEGMENTS, \
            f"corpus exhausted: only {total}/{QUOTA_SEGMENTS} segments; " \
            "more shards or higher MAX_SEGMENTS_PER_BOOK needed"

    # ---- freeze -------------------------------------------------------------
    print("freezing npy...", flush=True)
    arr = np.fromfile(bin_path, dtype=np.int32).reshape(-1, SEQ_LEN + 1)
    assert arr.shape == (QUOTA_SEGMENTS, SEQ_LEN + 1), arr.shape
    train_path = out / f"train_{QUOTA_SEGMENTS}x{SEQ_LEN + 1}.npy"
    np.save(train_path, arr)
    train_sha = sha256_file(train_path)

    v1 = Path(args.v1_data)
    v1_manifest = json.loads((v1 / "manifest.json").read_text())
    assert v1_manifest["status"] == "CPT_DATA_FROZEN_V1"
    val_path = v1 / "validation.npy"
    val_sha = sha256_file(val_path)
    equiv = json.loads((v1 / "equiv_result.json").read_text()) \
        if (v1 / "equiv_result.json").exists() else None

    prov_path = out / "provenance_train.jsonl"
    with prov_path.open("w") as w:
        for p in progress:
            if p.get("kind") == "book" and p["n_segments"] > 0:
                w.write(json.dumps(p) + "\n")

    manifest = {
        "status": "CPT_DATA_FROZEN_V2_500M",
        "seq_len": SEQ_LEN,
        "layout": "array[t, SEQ_LEN+1]; input=[ :-1], targets=[1:]",
        "n_segments": QUOTA_SEGMENTS,
        "total_train_tokens": QUOTA_SEGMENTS * SEQ_LEN,
        "train": {"path": train_path.name, "shape": list(arr.shape),
                  "sha256": train_sha,
                  "order": "sorted PG19 shards (excluding V1 val shard), parquet row "
                           "order, up to MAX_SEGMENTS_PER_BOOK contiguous head segments "
                           "per book, first-eligible stop at quota"},
        "validation": {"path": str(val_path), "sha256": val_sha,
                       "source": "frozen V1 validation.npy (CPT_DATA_FROZEN_V1), "
                                 "built from the excluded shard + last FWE shard"},
        "max_segments_per_book": MAX_SEGMENTS_PER_BOOK,
        "shards_used": [s.name for s in train_shards],
        "val_shard_excluded": args.exclude_shard,
        "provenance_path": prov_path.name,
        "provenance_sha256": sha256_file(prov_path),
        "shard_file_sha256": {s.name: sha256_file(s) for s in train_shards},
        "tokenizer": args.tokenizer,
        "tokenizer_equiv": equiv or "V1 equiv_result.json missing",
        "text_cap_chars": TEXT_CAP_CHARS,
        "note": "documents below SEQ_LEN+1 tokens dropped; segments are contiguous "
                "non-overlapping heads of each book; no document crosses splits; "
                "book text truncated to 480K chars before encoding (>= 8x16K segments; "
                "lossless for >=3.66 chars/token text)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"train": list(arr.shape), "tokens": QUOTA_SEGMENTS * SEQ_LEN,
                      "validation_sha256": val_sha}, indent=2), flush=True)
    print("CPT_DATA_V2_FROZEN", flush=True)


if __name__ == "__main__":
    main()
