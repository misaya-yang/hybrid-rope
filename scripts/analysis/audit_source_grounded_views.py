#!/usr/bin/env python3
"""CPU audit of realized evidence tokens, placement, truth graphs and split ownership."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.single_table_generation import rows,sha,write_json,canonical,tokenizer_identity
from scripts.data.prepare_source_grounded_transfer import truth_answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    manifest=json.loads(a.tasks.read_text());root=a.tasks.parent
    if manifest['tokenizer_files']!=tokenizer_identity(a.checkpoint):raise ValueError('tokenizer drift')
    tok=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True)
    proofs={r['semantic_id']:r for r in rows(root/'source_proofs.jsonl')}
    if sha(root/'source_proofs.jsonl')!=manifest['source_proofs_sha256']:raise ValueError('proof file drift')
    if sha(root/manifest['views_path'])!=manifest['views_sha256']:raise ValueError('view file drift')
    owners={};templates={};seen=set();paired={};verified=0;varying_width=set()
    for row in rows(root/manifest['views_path']):
        key=(row['semantic_id'],row['world'],row['layout'],row['length_cap'])
        if key in seen:raise ValueError('duplicate view')
        seen.add(key);c=proofs[row['semantic_id']];w=row['world']
        if truth_answer(c['proof'],w)!=c['worlds'][w]['answer']:raise ValueError('truth graph mismatch')
        for source in row['source_lineages']:
            if source in owners and owners[source]!=row['split']:raise ValueError('source crosses split')
            owners[source]=row['split']
        template=row['template_lineage']
        if template in templates and templates[template]!=row['split']:raise ValueError('template crosses split')
        templates[template]=row['split']
        if row['source_block']:
            start,length=row['source_block']
            expected=tok.encode('\n\n'.join(c['worlds'][w]['contexts'])+'\n',add_special_tokens=False)
            if row['prompt_ids'][start:start+length]!=expected:raise ValueError('realized evidence differs from verified source')
            if len(row['prompt_ids'])+row['generation_budget']!=row['length_cap']:raise ValueError('physical cap mismatch')
        verified+=1
        if row['split']=='validation':paired[key]=row
    comparisons=0
    for sid in sorted({k[0] for k in paired}):
        for world in (0,1):
            near=paired[(sid,world,'near',16384)];far=paired[(sid,world,'far',16384)]
            if Counter(near['prompt_ids'])!=Counter(far['prompt_ids']):raise ValueError('near/far token multiset changed')
            comparisons+=1
        a0=paired[(sid,0,'far',16384)];a1=paired[(sid,1,'far',16384)]
        start=a0['source_block'][0];width=max(a0['source_block'][1],a1['source_block'][1])
        if a0['source_block'][1]!=a1['source_block'][1]:varying_width.add(sid)
        if a0['prompt_ids'][:start]!=a1['prompt_ids'][:start] or a0['prompt_ids'][start+width:]!=a1['prompt_ids'][start+width:]:
            raise ValueError('far worlds differ outside the union evidence region')
    result={'status':'REALIZED_SOURCE_VIEW_AUDIT_PASS','task_manifest_sha256':sha(a.tasks),
            'verified_views':verified,'unique_source_articles':len(owners),'validation_near_far_multiset_pairs':comparisons,
            'validation_groups_with_different_world_source_widths':len(varying_width),'script_sha256':sha(__file__),
            'limits':'Near/far multisets are exact within each world. Counterfactual source widths can differ; no equal-evidence-token-count or near-world-only source-mask claim.'}
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
