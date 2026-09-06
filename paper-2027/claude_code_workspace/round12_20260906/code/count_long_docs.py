#!/usr/bin/env python3
"""CPU inventory: how many >=16K-token docs exist per source (500M CPT packing)."""
import glob
import json
import time

import duckdb

SEQ_MIN = 16385
out = {}

con = duckdb.connect()
con.execute("SET memory_limit='1300MB'")
con.execute("SET temp_directory='/root/autodl-tmp/claude_round12_20260906/data/duckdb_tmp'")
con.execute("SET threads=1")

t0 = time.time()
n_fwe = 0
tok_sum_fwe = 0
for f in sorted(glob.glob("/root/autodl-tmp/fineweb_edu/sample/10BT/*.parquet")):
    r = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(token_count),0) FROM read_parquet(?) "
        "WHERE token_count >= " + str(SEQ_MIN), [f]).fetchone()
    n_fwe += int(r[0])
    tok_sum_fwe += int(r[1])
    print(f"fwe {f}: cum_long_docs={n_fwe} cum_long_tokens={tok_sum_fwe} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
out["fwe_long_docs"] = n_fwe
out["fwe_long_tokens"] = tok_sum_fwe

n_pg = 0
for f in sorted(glob.glob(
        "/root/autodl-tmp/claude_round12_20260906/datasets/pg19/data/train-*.parquet")):
    # char heuristic gate (token gate authoritative at build): >=3 chars/token avg
    r = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?) WHERE length(text) >= "
        + str(SEQ_MIN * 3), [f]).fetchone()
    n_pg += int(r[0])
    print(f"pg19 {f}: cum_books_charpass={n_pg} elapsed={time.time()-t0:.0f}s", flush=True)
out["pg19_books_charpass"] = n_pg

json.dump(out, open("/root/autodl-tmp/claude_round12_20260906/data/long_doc_inventory.json",
                   "w"), indent=2)
print("INVENTORY_DONE", json.dumps(out), flush=True)
