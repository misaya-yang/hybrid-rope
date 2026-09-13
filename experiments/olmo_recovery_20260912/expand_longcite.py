#!/usr/bin/env python3
"""Stream intact LongCite-45k rows into source-separated 8K/16K SFT pools."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
import math
from pathlib import Path
import re

from experiments.evq_recovery.data import write_json
from experiments.evq_recovery.prepare import shingles
from experiments.olmo_recovery_20260912.expand_public_sft import (
    complete_chat, english_target, qasper_denied, stable_id, stable_split,
)


BUCKETS=(8192,16384)
SPLITS=("train","dev","test")
MARKER=re.compile(r"<C(\d+)>")
CITE=re.compile(r"<cite>\s*\[([^\]]+)\]",re.IGNORECASE)


def document_sections(prompt):
    matches=list(MARKER.finditer(prompt));sections={}
    for index,match in enumerate(matches):
        end=matches[index+1].start() if index+1<len(matches) else len(prompt)
        sections[int(match.group(1))]=(match.end(),end)
    return matches,sections


def citation_ids(response):
    found=[];invalid=[]
    for raw in CITE.findall(response):
        for part in raw.split(","):
            part=part.strip()
            match=re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?",part)
            if not match:
                invalid.append(part);continue
            left=int(match.group(1));right=int(match.group(2) or left)
            if right<left or right-left>256:invalid.append(part);continue
            found.extend(range(left,right+1))
    return sorted(set(found)),invalid


def map_token_spans(tokenizer,prompt,row,sections):
    rendered=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded["input_ids"])
    if ids!=row["prompt_ids"]:raise ValueError("offset tokenizer differs from native chat prompt")
    base=rendered.find(prompt)
    if base<0 or rendered.find(prompt,base+1)>=0:raise ValueError("original prompt is absent or ambiguous in native chat rendering")
    result={}
    for source_id,(start,end) in sections.items():
        absolute=(base+start,base+end);touched=[i for i,(left,right) in enumerate(encoded["offset_mapping"]) if right>absolute[0] and left<absolute[1]]
        if touched:result[source_id]=[touched[0],touched[-1]+1]
    return result


_WORKER_TOKENIZER=None
_WORKER_DENIED=None


def init_worker(model_path,denied,buckets):
    global _WORKER_TOKENIZER,_WORKER_DENIED,BUCKETS
    from transformers import AutoTokenizer
    _WORKER_TOKENIZER=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    _WORKER_DENIED=denied
    BUCKETS=tuple(buckets)


def process_record(payload):
    index,line=payload;tokenizer=_WORKER_TOKENIZER
    if tokenizer is None or _WORKER_DENIED is None:raise RuntimeError("LongCite worker is uninitialized")
    try:raw=json.loads(line)
    except json.JSONDecodeError:return "invalid_json",None
    prompt=str(raw.get("prompt", ""));response=str(raw.get("response", ""))
    if not prompt.strip() or not response.strip():return "missing_prompt_or_response",None
    if not english_target(prompt+"\n"+response):return "non_english_over_1pct_cjk",None
    markers,sections=document_sections(prompt)
    if not markers:return "missing_document_marker",None
    first_start,first_end=sections[int(markers[0].group(1))]
    document_material=" ".join(prompt[first_start:first_end].split())[:8192]
    if not document_material:return "empty_first_document",None
    if len(shingles(prompt,stride=1)&_WORKER_DENIED)>=3:return "qasper_overlap",None
    try:row=complete_chat(tokenizer,[{"role":"user","content":prompt},{"role":"assistant","content":response}])
    except ValueError:return "invalid_native_chat",None
    bucket=next((value for value in BUCKETS if row["prompt_tokens"]>=math.ceil(.75*value) and len(row["input_ids"])<=value),None)
    if bucket is None:return "outside_true_long_buckets",None
    cited,invalid=citation_ids(response);missing_ids=[value for value in cited if value not in sections]
    spans=map_token_spans(tokenizer,prompt,row,sections);mapped=[]
    for value in cited:
        if value in spans:mapped.append({"source_marker":value,"source_token_span":spans[value],"distance_to_answer_start":row["target_start"]-spans[value][1]})
    status=("no_citation_tag" if not cited and not invalid else "invalid_citation_syntax" if invalid else
            "missing_prompt_source_marker" if missing_ids else "mapped_unverified")
    row_id="longcite:"+stable_id(line.rstrip("\n"));group="longcite:"+stable_id(document_material)
    row.update(id=row_id,source_id=group,source_row=index,family="longcite",task="citation_qa",split=stable_split(document_material),length_bucket=bucket,
        original_prompt=prompt,original_response=response,references=[response],input_tokens=len(row["input_ids"]),
        supervised_tokens=len(row["input_ids"])-row["target_start"],citation_dataset_identity="LongCite-45k raw prompt/response",
        citation_parse={"status":status,"cited_marker_ids":cited,"invalid_ranges":invalid,"missing_marker_ids":missing_ids,"mapped_source_spans":mapped,
                        "verification":"marker/range mapping only; citation correctness and answer truth are not human-verified here"},
        provenance="LongCite published response with statement/cite tags preserved intact; no truncation or padding")
    return "selected",row


def main():
    global BUCKETS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--legacy-sources",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--buckets",type=int,nargs=2,default=list(BUCKETS))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    BUCKETS=tuple(args.buckets)
    if tuple(sorted(set(BUCKETS)))!=BUCKETS:raise ValueError("buckets must be two strictly increasing lengths")
    if not 1<=args.workers<=20:raise ValueError("--workers must be between 1 and 20")
    if args.out.exists():raise FileExistsError(args.out)
    denied,missing=qasper_denied(args.source.resolve().parent,args.legacy_sources.resolve())
    if missing:raise FileNotFoundError("complete QASPER exclusion sources required: "+", ".join(missing))
    args.out.mkdir(parents=True);handles={(bucket,split):(args.out/f"longcite_{bucket}_{split}.jsonl").open("x") for bucket in BUCKETS for split in SPLITS}
    counts=Counter();tokens=Counter();targets=Counter();citation_status=Counter();groups={};seen_rows=set()
    try:
        with args.source.open() as stream,ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                initargs=(str(args.model.resolve()),denied,BUCKETS)) as pool:
            index=0
            while True:
                lines=list(islice(stream,32))
                if not lines:break
                batch=[(index+offset,line) for offset,line in enumerate(lines)]
                for reason,row in pool.map(process_record,batch):
                    index+=1
                    if reason!="selected":counts[reason]+=1
                    else:
                        group=row["source_id"];split=row["split"];bucket=row["length_bucket"]
                        if groups.setdefault(group,split)!=split:raise ValueError("one LongCite document group crossed splits")
                        if row["id"] in seen_rows:counts["duplicate_row"]+=1;continue
                        seen_rows.add(row["id"]);citation_status[row["citation_parse"]["status"]]+=1
                        handles[bucket,split].write(json.dumps(row)+"\n")
                        key=f"selected/{bucket}/{split}";counts[key]+=1;tokens[key]+=row["input_tokens"];targets[key]+=row["supervised_tokens"]
                    if index%500==0:print(json.dumps({"rows_read":index,"selected":sum(counts[key] for key in counts if key.startswith("selected/"))}),flush=True)
    finally:
        for handle in handles.values():handle.close()
    train_tokens=sum(value for key,value in tokens.items() if key.endswith("/train"))
    write_json(args.out/"manifest.json",{"status":"COMPLETE","source":str(args.source.resolve()),"model":str(args.model.resolve()),
        "asset_identity_policy":"user-attested download; no file SHA scan","family":"longcite","workers":args.workers,"bounded_batch_rows":32,"rows":dict(counts),"input_tokens":dict(tokens),
        "supervised_tokens":dict(targets),"train_only_input_tokens":train_tokens,"two_pass_train_input_tokens":2*train_tokens,
        "source_groups":len(groups),"citation_parse_status":dict(citation_status),
        "split_policy":"first real <Cnumber> document normalized prefix identifies the source group; stable 80/10/10 group split",
        "length_contract":f"intact native-chat prompt >=75% of {BUCKETS} and complete response+terminal <= bucket; no truncation or unrelated padding",
        "citation_policy":"original statement/cite markup retained; source spans and distances reported only when marker/range parsing maps; no truth verification claim",
        "qasper_exclusion":"32-word shingles against QASPER dev/test documents"})
    print(json.dumps({"status":"COMPLETE","selected":sum(counts[key] for key in counts if key.startswith("selected/")),"train_only_input_tokens":train_tokens},sort_keys=True))


if __name__=="__main__":main()
