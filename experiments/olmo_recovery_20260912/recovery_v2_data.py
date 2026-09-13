#!/usr/bin/env python3
"""Assemble the eight-pool recovery-v2 manifest from real prepared/public data."""

from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil

from experiments.olmo_recovery_20260912.expand_public_sft import stable_id


SHORT_LM_TARGET=32*1024*1024
SYNTHETIC_FAMILIES=("lookup","linked_lookup","latest_update","attribute_binding")


def rows_in(path):
    with path.open() as stream:return sum(bool(line.strip()) for line in stream)


def entry(path,fmt,rows=None):
    return {"path":str(path.resolve()),"format":fmt,"rows":rows}


def discover_sft(directories):
    result={"short":[],8192:[],16384:[]};missing=[]
    for directory in directories:
        if not directory.is_dir():missing.append(str(directory));continue
        for path in sorted(directory.glob("*.jsonl")):
            if path.stat().st_size==0:continue
            name=path.name
            if name in ("native_short_sft_train.jsonl",):result["short"].append(path)
            elif name in ("long_sft_8192_train.jsonl","longcite_8192_train.jsonl"):result[8192].append(path)
            elif name in ("long_sft_16384_train.jsonl","longcite_16384_train.jsonl"):result[16384].append(path)
    for key in result:result[key]=list(dict.fromkeys(result[key]))
    return result,missing


def prepare_short_lm(parquets,tokenizer,path):
    import pyarrow.parquet as pq
    selected=0;seen=set();tokens=predictions=read=0
    with path.open("x") as output:
        for parquet_path in parquets:
            parquet=pq.ParquetFile(parquet_path)
            if "text" not in parquet.schema.names:raise ValueError(f"FineWeb parquet lacks text: {parquet_path}")
            for batch in parquet.iter_batches(columns=["text"],batch_size=128):
                for text in batch.column(0).to_pylist():
                    read+=1
                    if tokens>=SHORT_LM_TARGET:break
                    if not isinstance(text,str) or not text.strip():continue
                    source=stable_id(text);split_value=int(source[:8],16)%10
                    if split_value>=8 or source in seen:continue
                    ids=list(tokenizer.encode(text,add_special_tokens=False))
                    if len(ids)<256:continue
                    ids=ids[:4097];seen.add(source)
                    row={"id":"fineweb:"+source,"source_id":"fineweb:"+source,"family":"fineweb_short_lm",
                         "task":"causal_lm","split":"train","input_ids":ids,"target_start":1,
                         "prompt_ids":[],"references":[],"input_tokens":len(ids),
                         "prediction_tokens":len(ids)-1,"terminal_policy":"natural document tokens; no artificial EOS"}
                    output.write(json.dumps(row)+"\n");selected+=1;tokens+=len(ids);predictions+=len(ids)-1
                if tokens>=SHORT_LM_TARGET:break
            if tokens>=SHORT_LM_TARGET:break
    if tokens<SHORT_LM_TARGET:raise ValueError(f"FineWeb sources supply only {tokens} eligible short-LM tokens")
    return {"rows":selected,"input_tokens":tokens,"prediction_tokens":predictions,
            "target_input_tokens":SHORT_LM_TARGET,"overshoot_input_tokens":tokens-SHORT_LM_TARGET,
            "raw_rows_read":read,"source_split":"content hash modulo10; train buckets0-7","max_source_visits":1}


def synthetic_rows(tokenizer,length,seeds,split,labels):
    from experiments.rope_fast_5090_20260912.source_counterfactual import build_group
    for seed in seeds:
        for family in SYNTHETIC_FAMILIES:
            for row in build_group(tokenizer,family,length,seed,labels):
                answer_ids=list(tokenizer.encode(row["answer"],add_special_tokens=False));full=row["prompt_ids"]+answer_ids+[tokenizer.eos_token_id]
                if len(answer_ids)+1>row["max_new_tokens"] or len(full)>length:raise ValueError("synthetic answer/EOS exceeds frozen physical reserve")
                yield {**row,"id":row["row_id"],"source_id":"synthetic:"+row["group_id"],"split":split,
                       "input_ids":full,"target_start":len(row["prompt_ids"]),"references":[row["answer"]],
                       "answer_tokens":len(answer_ids)+1,"training_length":length,"max_new_tokens":16,
                       "provenance":"new training-only source counterfactual namespace; real contiguous prompt, no position gap"}


def write_synthetic(tokenizer,output):
    from experiments.rope_fast_5090_20260912.source_counterfactual import equal_token_labels
    labels=equal_token_labels(tokenizer);paths={};counts={};tokens={};targets={}
    namespaces={"train":range(2027100000,2027100128),"dev":range(2027200000,2027200016),"test":range(2027300000,2027300016)}
    for split,seeds in namespaces.items():
        for length in (8192,16384):
            path=output/f"long_synthetic_{length}_{split}.jsonl";count=total=0
            supervised=0
            with path.open("x") as stream:
                for row in synthetic_rows(tokenizer,length,seeds,split,labels):stream.write(json.dumps(row)+"\n");count+=1;total+=len(row["input_ids"]);supervised+=len(row["input_ids"])-row["target_start"]
            expected=1024 if split=="train" else 128
            if count!=expected:raise AssertionError(f"unexpected synthetic {split}/{length} rows: {count}")
            paths[split,length]=path;counts[f"{split}/{length}"]=count;tokens[f"{split}/{length}"]=total;targets[f"{split}/{length}"]=supervised
    return paths,{"rows":counts,"input_tokens":tokens,"supervised_tokens":targets,"seed_namespaces":{key:[values.start,values.stop-1] for key,values in namespaces.items()},
                  "families":list(SYNTHETIC_FAMILIES),"worlds_per_group":2,"train_source_visits_upper_bound":2}


def sft_stats(paths):
    families=Counter();tokens=Counter();rows={}
    for path in paths:
        count=0
        with path.open() as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line);family=row.get("family","unknown");families[family]+=1;tokens[family]+=len(row["input_ids"]);count+=1
        rows[str(path.resolve())]=count
    return rows,dict(families),dict(tokens)


def sft_file_stats(paths):
    rows={};families=Counter();tokens=Counter();fallback=[]
    for path in paths:
        manifest_path=path.parent/"manifest.json"
        if not manifest_path.is_file():fallback.append(path);continue
        manifest=json.loads(manifest_path.read_text());row_meta=manifest.get("rows",{});token_meta=manifest.get("unique_input_tokens",manifest.get("input_tokens",{}))
        if path.name.startswith("native_short"):
            suffix="/native/train"
        elif "8192" in path.name:suffix="/8192/train"
        elif "16384" in path.name:suffix="/16384/train"
        else:fallback.append(path);continue
        matched={key:value for key,value in row_meta.items() if key.endswith(suffix)}
        if not matched:fallback.append(path);continue
        count=sum(matched.values());rows[str(path.resolve())]=count
        family_override="longcite" if path.name.startswith("longcite_") else None
        for key,value in matched.items():families[family_override or key.split("/",1)[0]]+=value
        for key,value in token_meta.items():
            if key.endswith(suffix):tokens[family_override or key.split("/",1)[0]]+=value
    if fallback:
        old_rows,old_families,old_tokens=sft_stats(fallback);rows.update(old_rows);families.update(old_families);tokens.update(old_tokens)
    return rows,dict(families),dict(tokens)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--fineweb-parquets",type=Path,nargs="*")
    parser.add_argument("--sft-dirs",type=Path,nargs="*");parser.add_argument("--execute",action="store_true")
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    defaults=[root/f"data_expanded_{name}" for name in ("longalign","longalpaca","ultrachat","longcite")]
    sft,missing_dirs=discover_sft([path.resolve() for path in (args.sft_dirs or defaults)])
    pg_sources=root/"sources_pg19_expanded";existing_pg=root/"pg19_multiscale"
    default_fineweb=[root.parent/"rotary_budget_20260908/data/000_00000.parquet",root.parent/"rotary_budget_20260908/data/004_00000.parquet"]
    fineweb=[path.resolve() for path in (args.fineweb_parquets or default_fineweb)]
    pending=[]
    if not sft["short"]:pending.append("short_sft: native_short_sft_train.jsonl")
    for length in (8192,16384):
        if not sft[length]:pending.append(f"long_sft_{length}: prepared LongAlign/LongAlpaca/LongCite train file")
    pg_ready=all((existing_pg/name).is_file() for name in ("cpt_train_8192.npy","cpt_train.npy","lm_validation.npy","lm_test.npy","pg19_manifest.json"))
    pg_source_ready=False
    if (pg_sources/"pg19_books.json").is_file() and (pg_sources/"expansion_receipt.json").is_file():
        pg_source_ready=json.loads((pg_sources/"expansion_receipt.json").read_text()).get("status")=="COMPLETE"
    if not pg_ready and not pg_source_ready:pending.append("long_lm: complete prepared pg19_multiscale or sources_pg19_expanded receipt")
    if not fineweb or any(not path.is_file() for path in fineweb):pending.append("short_lm: --fineweb-parquets existing files")
    metadata={"status":"METADATA_ONLY","execute":False,"pending":pending,"missing_optional_sft_dirs":missing_dirs,
              "discovered_sft":{str(key):[str(path) for path in value] for key,value in sft.items()},
              "would_generate":{"short_lm_target_input_tokens":SHORT_LM_TARGET,"synthetic_train_rows_per_length":1024,"synthetic_dev_test_rows_per_length":128}}
    if not args.execute:print(json.dumps(metadata,sort_keys=True));return
    if pending:raise FileNotFoundError("v2 source inputs pending: "+"; ".join(pending))
    final_manifest=output/"manifest.json"
    if final_manifest.is_file():
        existing=json.loads(final_manifest.read_text())
        if existing.get("status")=="READY":print(json.dumps({"status":"REUSED_READY","manifest":str(final_manifest)}));return
    output.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model.resolve(),local_files_only=True)
    pg_stage=output/"pg19_stage.json"
    # Prefer the expanded 512-book source whenever its receipt is complete.
    # The legacy prepared arrays are the smaller R0 pool and are only a
    # fallback when expanded sources are unavailable.
    if pg_ready and not pg_source_ready:
        pg=existing_pg;receipt=json.loads((pg/"pg19_manifest.json").read_text())
        pg_stats={"train_windows_by_length":{length:len(receipt["rows"][f"train_{length}"]) for length in ("8192","16384")},
                  "train_prediction_tokens_by_length":receipt["training_prediction_tokens_by_length"],
                  "heldout_books":{split:len(receipt["rows"][split]) for split in ("validation","test")},
                  "reuse":"existing prepared multiscale arrays; no retokenization"}
    else:
        pg=output/"pg19_multiscale"
        pg_files=("cpt_train_8192.npy","cpt_train.npy","lm_validation.npy","lm_test.npy","pg19_manifest.json")
        if pg_stage.is_file() and all((pg/name).is_file() for name in pg_files) and json.loads(pg_stage.read_text()).get("status")=="COMPLETE":
            pg_stats=json.loads(pg_stage.read_text())["statistics"]
        else:
            if pg.exists():shutil.rmtree(pg)
            pg.mkdir();from experiments.olmo_recovery_20260912.data_multiscale import prepare_pg19_multiscale
            pg_stats=prepare_pg19_multiscale(pg_sources,pg,tokenizer)
            pg_stage.write_text(json.dumps({"status":"COMPLETE","statistics":pg_stats},indent=2)+"\n")
    if not pg_stage.is_file():pg_stage.write_text(json.dumps({"status":"COMPLETE","statistics":pg_stats},indent=2)+"\n")
    short_path=output/"short_lm_train.jsonl";short_stage=output/"short_lm_stage.json"
    if short_stage.is_file() and short_path.is_file() and json.loads(short_stage.read_text()).get("status")=="COMPLETE":short_stats=json.loads(short_stage.read_text())["statistics"]
    else:
        if short_path.exists():short_path.unlink()
        short_stats=prepare_short_lm(fineweb,tokenizer,short_path);short_stage.write_text(json.dumps({"status":"COMPLETE","statistics":short_stats},indent=2)+"\n")
    synthetic_stage=output/"synthetic_stage.json"
    expected_synthetic=[output/f"long_synthetic_{length}_{split}.jsonl" for split in ("train","dev","test") for length in (8192,16384)]
    if synthetic_stage.is_file() and all(path.is_file() for path in expected_synthetic) and json.loads(synthetic_stage.read_text()).get("status")=="COMPLETE":
        saved=json.loads(synthetic_stage.read_text());synthetic_stats=saved["statistics"]
        synthetic={(split,length):output/f"long_synthetic_{length}_{split}.jsonl" for split in ("train","dev","test") for length in (8192,16384)}
    else:
        for path in expected_synthetic:
            if path.exists():path.unlink()
        synthetic,synthetic_stats=write_synthetic(tokenizer,output);synthetic_stage.write_text(json.dumps({"status":"COMPLETE","statistics":synthetic_stats},indent=2)+"\n")
    all_sft=sft["short"]+sft[8192]+sft[16384];sft_rows,family_rows,family_tokens=sft_file_stats(all_sft)
    pools={
        "short_lm":[entry(short_path,"jsonl",short_stats["rows"])],
        "short_sft":[entry(path,"jsonl",sft_rows[str(path.resolve())]) for path in sft["short"]],
        "long_lm_8192":[entry(pg/"cpt_train_8192.npy","npy",pg_stats["train_windows_by_length"]["8192"])],
        "long_lm_16384":[entry(pg/"cpt_train.npy","npy",pg_stats["train_windows_by_length"]["16384"])],
        "long_sft_8192":[entry(path,"jsonl",sft_rows[str(path.resolve())]) for path in sft[8192]],
        "long_sft_16384":[entry(path,"jsonl",sft_rows[str(path.resolve())]) for path in sft[16384]],
        "long_synthetic_8192":[entry(synthetic["train",8192],"jsonl",1024)],
        "long_synthetic_16384":[entry(synthetic["train",16384],"jsonl",1024)]}
    panels={split:[str(synthetic[split,length].resolve()) for length in (8192,16384)] for split in ("dev","test")}
    synthetic_train_tokens=sum(value for key,value in synthetic_stats["input_tokens"].items() if key.startswith("train/"))
    five_rows={**family_rows,"source_counterfactual":2048};five_tokens={**family_tokens,"source_counterfactual":synthetic_train_tokens}
    lm_prediction_tokens=short_stats["prediction_tokens"]+sum(pg_stats["train_prediction_tokens_by_length"].values())
    sft_input_tokens=sum(family_tokens.values())+synthetic_train_tokens
    synthetic_supervised_tokens=sum(value for key,value in synthetic_stats["supervised_tokens"].items() if key.startswith("train/"))
    manifest={"status":"READY","asset_identity_policy":"user-attested sources/no SHA scan","model":str(args.model.resolve()),
        "pools":pools,"evaluation_panels":panels,"lm_evaluation":{"dev":str((pg/"lm_validation.npy").resolve()),"test":str((pg/"lm_test.npy").resolve())},
        "statistics":{"short_lm":short_stats,"pg19":pg_stats,"synthetic":synthetic_stats,"sft_rows_by_file":sft_rows,
                      "five_sft_families_rows":five_rows,"five_sft_families_input_tokens":five_tokens,
                      "train_only_lm_prediction_tokens":lm_prediction_tokens,"train_only_sft_input_tokens":sft_input_tokens,
                      "train_only_synthetic_supervised_tokens":synthetic_supervised_tokens,"recommended_max_source_visits":2,
                      "two_visit_upper_bound":{"lm_prediction_tokens":2*lm_prediction_tokens,"sft_input_tokens":2*sft_input_tokens}},
        "boundaries":["Existing expanded SFT files are referenced, not copied or rebuilt.","FineWeb is retokenized from raw text; no NeoX token stream is reused.",
                      "Synthetic train/dev/test seed namespaces are disjoint from E3/E7-source evaluation.","LM prediction tokens, SFT input tokens, and supervised answer tokens remain separate accounting fields."]}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":"READY","pools":{key:sum(x["rows"] for x in value) for key,value in pools.items()},"lm_prediction_tokens":lm_prediction_tokens,"sft_input_tokens":sft_input_tokens},sort_keys=True))


if __name__=="__main__":main()
