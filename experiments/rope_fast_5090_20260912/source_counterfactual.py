#!/usr/bin/env python3
"""Prepare, run, and score the bounded E7-source counterfactual mechanism panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

FAMILIES=("lookup","linked_lookup","latest_update","attribute_binding")
LENGTHS=(4096,16384)
SEEDS=tuple(range(2026091201,2026091217))
ARMS=("native_g1","bm_g4","mrpro_g4")
LABELS=("amber","birch","cedar","coral","denim","ivory","jade","lilac",
        "maple","olive","pearl","ruby","sable","teal","umber","violet")
MAX_NEW_TOKENS=16


def stable_id(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def write_json(path,value):
    temporary=path.with_name(path.name+".incomplete")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    os.replace(temporary,path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalize_answer(text):
    value=re.sub(r"^answer\s*:\s*","",text.strip().lower())
    return value.strip(" \t\r\n.!,;:\"'`")


def equal_token_labels(tokenizer):
    groups=defaultdict(list)
    for word in LABELS:
        signature=(len(tokenizer.encode(" "+word,add_special_tokens=False)),
                   len(tokenizer.encode(" "+word+".",add_special_tokens=False)))
        groups[signature].append(word)
    candidates=max(groups.values(),key=len)
    if len(candidates)<8:
        raise ValueError("tokenizer has too few equal-token label words")
    return candidates


def token_spans(tokenizer,rendered,char_spans,prompt_ids):
    encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded["input_ids"]);offsets=encoded["offset_mapping"]
    if ids!=prompt_ids:
        raise ValueError("offset-tokenization differs from frozen prompt ids")
    result=[]
    for start,end,role in char_spans:
        touched=[index for index,(left,right) in enumerate(offsets) if right>start and left<end]
        if not touched:
            raise ValueError("evidence has no token span")
        result.append({"role":role,"char_span":[start,end],"token_span":[touched[0],touched[-1]+1]})
    return result


def build_group(tokenizer,family,length,seed,labels):
    rng=random.Random(int(stable_id([family,length,seed])[:16],16))
    numbers=rng.sample(range(10000,99999),7000);item="R"+str(numbers[0]);holder="H"+str(numbers[1])
    answers=rng.sample(labels,2);old=[word for word in labels if word not in answers][:2]
    placement=(.2,.8)[rng.randrange(2)];background=[]
    for number in numbers[2:]:
        key="R"+str(number);label=rng.choice(labels)
        if family=="lookup":line=f"Item {key}: label {label}."
        elif family=="linked_lookup":
            h="H"+str(number);line=f"Item {key}: holder {h}. Holder {h}: label {label}."
        elif family=="latest_update":line=f"Update for item {key}: label {label}."
        else:
            group,material=rng.choice((("north","wood"),("south","glass"),("east","stone")))
            line=f"Item {key}: group {group}; material {material}; label {label}."
        background.append(line)
    if family=="lookup":
        question=f"What is the label of item {item}?";instruction="Find the record for the exact requested item."
    elif family=="linked_lookup":
        question=f"What is the label of the holder of item {item}?";instruction="First find the item holder, then find that holder label."
    elif family=="latest_update":
        question=f"What is the label in the last update for item {item}?";instruction="Updates are ordered from oldest to newest. Use the final matching update."
    else:
        question="What is the label of the item whose group is north AND whose material is glass?";instruction="Both requested attributes must belong to the same record."

    def content(count,world):
        if family=="lookup":evidence=[(placement,f"Item {item}: label {answers[world]}.","decisive")]
        elif family=="linked_lookup":evidence=[(.2,f"Item {item}: holder {holder}.","link"),(.8,f"Holder {holder}: label {answers[world]}.","decisive")]
        elif family=="latest_update":evidence=[(.15,f"Update for item {item}: label {old[0]}.","old"),(.5,f"Update for item {item}: label {old[1]}.","old"),(.85,f"Update for item {item}: label {answers[world]}.","decisive")]
        else:evidence=[(placement,f"Item {item}: group north; material glass; label {answers[world]}.","decisive")]
        lines=list(background[:count]);tagged=[]
        for fraction,line,role in sorted(evidence,reverse=True):
            index=round(fraction*count);lines.insert(index,line);tagged.append((line,role,index))
        body=(instruction+" Use only the records below.\nAnswer with exactly one label word, without explanation.\n\nRecords:\n"
              +"\n".join(lines)+"\n\nQuestion: "+question+"\nAnswer:")
        return body,tagged

    def encode_body(body):
        rendered=tokenizer.apply_chat_template([{"role":"user","content":body}],tokenize=False,add_generation_prompt=True)
        return rendered,list(tokenizer.encode(rendered,add_special_tokens=False))
    target=length-MAX_NEW_TOKENS-8;left,right,selected=0,min(len(background),length//3),None
    while left<=right:
        middle=(left+right)//2;rendered,ids=encode_body(content(middle,0)[0])
        if len(ids)<=target:selected=middle;left=middle+1
        else:right=middle-1
    if selected is None:raise ValueError("template exceeds physical cap")
    rows=[]
    for world in (0,1):
        body,tagged=content(selected,world);rendered,ids=encode_body(body)
        if len(ids)+MAX_NEW_TOKENS>length or len(ids)<length-256:
            raise ValueError(f"{family}/{length} is not a near-cap intact prompt: {len(ids)}")
        spans=[]
        for line,role,line_index in tagged:
            start=rendered.find(line)
            if start<0 or rendered.find(line,start+1)>=0:raise ValueError("evidence line is absent or ambiguous")
            spans.append((start,start+len(line),role+f"_line_{line_index}"))
        group_id=stable_id(["e7-source",family,length,seed])
        rows.append({"row_id":stable_id([group_id,world]),"group_id":group_id,"source_seed":seed,
            "family":family,"task":family,"world":world,"length_cap":length,"answer":answers[world],
            "counterfactual_answers":answers,"question":question,"prompt_ids":ids,"input_tokens":len(ids),
            "max_new_tokens":MAX_NEW_TOKENS,"source_spans":token_spans(tokenizer,rendered,spans,ids),
            "source_position_policy":"same insertion indices across worlds; only the decisive evidence label changes"})
    if rows[0]["question"]!=rows[1]["question"] or rows[0]["input_tokens"]!=rows[1]["input_tokens"]:
        raise AssertionError("counterfactual worlds do not preserve query/token length")
    return rows


def prepare(args):
    from transformers import AutoTokenizer
    import numpy as np
    model=args.model.resolve();config=json.loads((model/"config.json").read_text())
    if config.get("model_type")!="olmo2" or config.get("hidden_size")!=2048 or config.get("num_hidden_layers")!=16:
        raise ValueError("requires OLMo-2-0425-1B-Instruct configuration")
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True);labels=equal_token_labels(tokenizer)
    source_tables=json.loads((args.e2_prepared/"tables.json").read_text());tables={}
    for name in ARMS:
        table=source_tables[name];values=np.asarray(table["values_float32"],dtype=np.float32)
        if values.shape!=(64,) or not np.isfinite(values).all() or not np.all(values>0) or not np.all(values[:-1]>values[1:]):
            raise ValueError(f"invalid E2 table values: {name}")
        tables[name]=table
    rows=[]
    for seed in SEEDS:
        for length in LENGTHS:
            for family in FAMILIES:rows.extend(build_group(tokenizer,family,length,seed,labels))
    if len(rows)!=256 or len({row["row_id"] for row in rows})!=256 or len({row["group_id"] for row in rows})!=128:
        raise AssertionError("unexpected fixed panel size")
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/"screen.jsonl").open("x") as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True)+"\n")
    write_json(args.out/"tables.json",tables)
    generation={"do_sample":False,"num_beams":1,"num_return_sequences":1,"use_cache":True,
                "eos_token_id":tokenizer.eos_token_id,"pad_token_id":tokenizer.pad_token_id,
                "min_new_tokens":0,"max_new_tokens":MAX_NEW_TOKENS}
    write_json(args.out/"generation_config.json",generation)
    write_json(args.out/"manifest.json",{"status":"E7_SOURCE_PREPARED_GPU_NOT_RUN","model_path":str(model),
        "rows":256,"groups":128,"families":list(FAMILIES),"lengths":list(LENGTHS),"source_seeds":list(SEEDS),
        "arms":list(ARMS),"rows_per_arm":256,"total_generations":768,"equal_token_labels":labels,
        "scope":"new constructed source-counterfactual mechanism panel; not full official RULER or full E7"})
    print(json.dumps({"status":"PREPARED","rows":256,"groups":128,"generations":768}))


def verify_runtime_table(model,table):
    import numpy as np
    expected=np.asarray(table["values_float32"],dtype=np.float32);actual=model.model.rotary_emb.inv_freq.detach().cpu().numpy()
    if actual.shape!=expected.shape or not np.array_equal(actual,expected) or float(model.model.rotary_emb.attention_scaling)!=float(table["gain"]):
        raise RuntimeError("installed table values/gain differ")


def run(args):
    prepared=args.prepared.resolve();manifest=json.loads((prepared/"manifest.json").read_text());rows=read_jsonl(prepared/"screen.jsonl")
    if manifest.get("status")!="E7_SOURCE_PREPARED_GPU_NOT_RUN" or len(rows)!=256 or len({row["row_id"] for row in rows})!=256:
        raise ValueError("prepared E7-source panel is incomplete")
    total_generations=len(rows)*len(manifest["arms"])
    print(json.dumps({"status":"DRY_RUN" if not args.execute else "STARTING","rows":256,"arms":manifest["arms"],"generations":total_generations}))
    if not args.execute:return
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    from scripts.experiments.olmo_fast_screen.runtime import install
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    if any(line.strip().isdigit() and int(line.strip())!=os.getpid() for line in active.splitlines()):
        raise RuntimeError("GPU already has a compute process")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability()!=(12,0) or "5090" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("requires idle RTX 5090 sm120 with BF16")
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    model=AutoModelForCausalLM.from_pretrained(manifest["model_path"],local_files_only=True,dtype=torch.bfloat16,device_map={"":"cuda"},attn_implementation="sdpa").eval()
    tokenizer=AutoTokenizer.from_pretrained(manifest["model_path"],local_files_only=True);decoding=GenerationConfig.from_dict(json.loads((prepared/"generation_config.json").read_text()))
    tables=json.loads((prepared/"tables.json").read_text());eos=decoding.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    args.out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for arm in manifest["arms"]:
            install(model,tables[arm]);verify_runtime_table(model,tables[arm]);path=args.out/f"{arm}.jsonl";saved=read_jsonl(path) if path.exists() else []
            for index,record in enumerate(saved):
                if index>=len(rows) or record["row_id"]!=rows[index]["row_id"] or record["arm"]!=arm:raise ValueError("resume row prefix differs")
            with path.open("a") as stream:
                for index,row in enumerate(rows[len(saved):],start=len(saved)):
                    ids=torch.tensor([row["prompt_ids"]],device="cuda");started=time.monotonic()
                    generated=model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=decoding,max_new_tokens=row["max_new_tokens"])[0,len(row["prompt_ids"]):].tolist()
                    ended=bool(generated and generated[-1] in eos);text=tokenizer.decode(generated[:-1] if ended else generated,skip_special_tokens=False)
                    result={key:row[key] for key in ("row_id","group_id","source_seed","family","world","length_cap","answer","source_spans","input_tokens","max_new_tokens")}
                    result.update(arm=arm,generated_ids=generated,output_text=text,whole_answer_exact=normalize_answer(text)==normalize_answer(row["answer"]),
                                  ended_eos=ended,hit_cap=len(generated)==row["max_new_tokens"] and not ended,seconds=time.monotonic()-started)
                    stream.write(json.dumps(result,sort_keys=True)+"\n");stream.flush();write_json(args.out/"live.json",{"arm":arm,"completed":index+1,"total":len(rows)})
            write_json(args.out/f"{arm}.json",{"status":"COMPLETE","rows":len(rows),"table":tables[arm]})
    write_json(args.out/"status.json",{"status":"COMPLETE","arms":manifest["arms"],"rows_per_arm":256,"total_generations":total_generations,"peak_cuda_bytes":int(torch.cuda.max_memory_allocated())})


def percentile(values,p):
    values=sorted(values);position=p*(len(values)-1);low=int(position);high=min(low+1,len(values)-1);fraction=position-low
    return values[low]*(1-fraction)+values[high]*fraction


def score(args):
    prepared=args.prepared.resolve();manifest=json.loads((prepared/"manifest.json").read_text());source={row["row_id"]:row for row in read_jsonl(prepared/"screen.jsonl")};arms={}
    generation=json.loads((prepared/"generation_config.json").read_text());eos=generation["eos_token_id"];eos=set(eos if isinstance(eos,list) else [eos])
    for arm in manifest["arms"]:
        values=read_jsonl(args.run/f"{arm}.jsonl")
        if len(values)!=256 or len({row["row_id"] for row in values})!=256 or {row["row_id"] for row in values}!=set(source):raise ValueError(f"incomplete arm {arm}")
        for row in values:
            expected=source[row["row_id"]];tokens=row["generated_ids"];ended=bool(tokens and tokens[-1] in eos)
            if row["arm"]!=arm or row["group_id"]!=expected["group_id"] or row["world"]!=expected["world"] or len(tokens)>expected["max_new_tokens"] or row["ended_eos"]!=ended:
                raise ValueError(f"raw output contract differs: {arm}/{row['row_id']}")
            hit_cap=len(tokens)==expected["max_new_tokens"] and not ended
            if row["hit_cap"]!=hit_cap or row["whole_answer_exact"]!=(normalize_answer(row["output_text"])==normalize_answer(expected["answer"])):
                raise ValueError(f"score/cap contract differs: {arm}/{row['row_id']}")
        arms[arm]={row["row_id"]:row for row in values}
    summary={};pair_values={}
    for arm,data in arms.items():
        cells=defaultdict(list);groups=defaultdict(list)
        for row in data.values():cells[row["length_cap"],row["family"]].append(row);groups[row["group_id"]].append(row)
        pair={key:float(len(rows)==2 and {row["world"] for row in rows}=={0,1} and all(row["whole_answer_exact"] and row["ended_eos"] for row in rows) and normalize_answer(rows[0]["output_text"])!=normalize_answer(rows[1]["output_text"])) for key,rows in groups.items()}
        group_meta={row["group_id"]:(row["length_cap"],row["family"]) for row in source.values()}
        pair_values[arm]=pair;summary[arm]={"by_length_family":{f"{length}/{family}":{"rows":len(rows),"whole_answer_exact":sum(row["whole_answer_exact"] for row in rows)/len(rows),"eos_rate":sum(row["ended_eos"] for row in rows)/len(rows),"cap_rate":sum(row["hit_cap"] for row in rows)/len(rows),"pair_follow_rate":sum(pair[key] for key,cell in group_meta.items() if cell==(length,family))/16} for (length,family),rows in sorted(cells.items())},"pair_follow_rate":{str(length):sum(value for key,value in pair.items() if group_meta[key][0]==length)/64 for length in LENGTHS}}
    differences={}
    if {"bm_g4","mrpro_g4"} <= set(pair_values):
        for length in LENGTHS:
            by_seed=[]
            for seed in SEEDS:
                keys={row["group_id"] for row in source.values() if row["length_cap"]==length and row["source_seed"]==seed}
                by_seed.append(sum(pair_values["bm_g4"][key]-pair_values["mrpro_g4"][key] for key in keys)/len(keys))
            rng=random.Random(20260912+length);draws=[sum(by_seed[rng.randrange(len(by_seed))] for _ in by_seed)/len(by_seed) for _ in range(20000)]
            differences[str(length)]={"bm_minus_mrpro_pair_follow":sum(by_seed)/len(by_seed),"source_seed_cluster_bootstrap_ci95":[percentile(draws,.025),percentile(draws,.975)],"seed_clusters":len(by_seed)}
    write_json(args.out,{"status":"COMPLETE","arms":summary,"bm_mrpro_paired":differences,"primary_terminal":"both counterfactual worlds whole-answer exact, terminal EOS, and source-conditioned outputs differ","scope":manifest["scope"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--model",type=Path,required=True);p.add_argument("--e2-prepared",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.set_defaults(func=prepare)
    p=sub.add_parser("run");p.add_argument("--prepared",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--execute",action="store_true");p.set_defaults(func=run)
    p=sub.add_parser("score");p.add_argument("--prepared",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.set_defaults(func=score)
    args=parser.parse_args();args.func(args)


if __name__=="__main__":main()
