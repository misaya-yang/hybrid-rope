"""Native document history, full second-order routing from question onset.

The onset is determined from frozen input text and exact tokenizer offsets.
Only this semantic boundary is tested; no answer labels or token threshold sweep.
Normal prefill chunks, native positions, final quotas and free decoding persist.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from . import run as base
from .binding_phase_cross import PhaseCrossSelector,ROWS

MARKER='What are all the special magic numbers for '


def question_boundary(row,tokenizer):
    text=row['prompt']
    if text.count(MARKER)!=1:raise ValueError('question onset must be unique')
    start=text.index(MARKER)
    matched=None
    for specials in (False,True):
        encoded=tokenizer(text,add_special_tokens=specials,return_offsets_mapping=True)
        if encoded['input_ids']==row['prompt_ids']:
            matched=(encoded,specials);break
    if matched is None:raise ValueError('full prompt retokenization differs from frozen IDs')
    encoded,specials=matched
    index=next(i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>start)
    lo,hi=encoded['offset_mapping'][index]
    if not lo<=start<hi or index>=len(row['prompt_ids'])-1:raise ValueError('invalid question token boundary')
    return {'switch_query_position':index,'question_start_char':start,
        'first_question_token_char_span':[lo,hi],'special_tokens_added':specials,
        'full_scored_prompt_tokens':len(row['prompt_ids'])-index,
        'prompt_ids_sha256':hashlib.sha256(json.dumps(row['prompt_ids']).encode()).hexdigest()}


class QuestionPhaseSelector(PhaseCrossSelector):
    def __init__(self,mode='native_question_full'):
        if mode not in ('native_question_full','full_covariance'):raise ValueError('unknown question phase arm')
        super().__init__(history='native' if mode=='native_question_full' else 'full_covariance',answer='full_covariance')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--reference-run',required=True)
    parser.add_argument('--confirmation-manifest')
    parser.add_argument('--queue-root',default='/root/autodl-tmp/position_overnight_20260909')
    args,remaining=parser.parse_known_args()
    locparser=argparse.ArgumentParser(add_help=False)
    for name in ('data','model','output'):locparser.add_argument('--'+name,required=True)
    locparser.add_argument('--row-ids')
    locparser.add_argument('--split');locparser.add_argument('--lengths',nargs='+',type=int)
    locparser.add_argument('--selectors',nargs='+')
    for name in ('topk','select-blocks','chunk-size','attention-query-chunk-size'):locparser.add_argument('--'+name,type=int)
    loc,_=locparser.parse_known_args(remaining)
    wanted=set(json.loads(Path(loc.row_ids).read_text())) if loc.row_ids else set(ROWS)
    allowed={f'ruler_dev_niah_multiquery_{length}_{i:03d}' for length in (8192,16384) for i in range(8)}
    confirmation=None
    if args.confirmation_manifest:
        confirmation=json.loads(Path(args.confirmation_manifest).read_text())
        if wanted!=set(confirmation['row_ids']) or len(wanted)!=4 or loc.split!='test' or loc.lengths!=[16384] or loc.selectors!=confirmation['selectors'] or confirmation.get('no_tuning_on_results') is not True:raise ValueError('confirmation scope mismatch')
        for name in ('topk','select_blocks','chunk_size','attention_query_chunk_size'):
            if getattr(loc,name)!=confirmation[name]:raise ValueError('confirmation configuration '+name)
        import ast
        nodes=[n for n in ast.parse(Path(__file__).read_text()).body if getattr(n,'name',None) in ('question_boundary','QuestionPhaseSelector')]
        identity=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
        if identity!=confirmation['operator_ast_sha256'] or hashlib.sha256(Path(__file__).with_name('binding_phase_cross.py').read_bytes()).hexdigest()!=confirmation['phase_source_sha256']:raise ValueError('frozen confirmation operator changed')
    elif not wanted or not wanted<=allowed:raise ValueError('only existing 8K/16K multiquery DEV inputs')
    ref=json.loads((Path(args.reference_run)/'contract.json').read_text())
    if hashlib.sha256(Path(loc.data).read_bytes()).hexdigest()!=ref['data_sha256'] or loc.model!=ref['model']:raise ValueError('model/data identity')
    for name in ('run.py','runtime.py','selector_controls.py','full_covariance_probe.py','exact_probe.py'):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=ref['source_hashes'][name]:raise ValueError('computation identity '+name)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(loc.model,local_files_only=True)
    selected=[r for r in map(json.loads,Path(loc.data).read_text().splitlines()) if r['row_id'] in wanted]
    if {r['row_id'] for r in selected}!=wanted:raise ValueError('missing requested DEV inputs')
    if confirmation and (any(r['split']!='test' or r['length_cap']!=16384 for r in selected) or ref['data_sha256']!=confirmation['data_sha256']):raise ValueError('confirmation inputs changed')
    receipts={r['row_id']:question_boundary(r,tokenizer) for r in selected}
    rows={tuple(r['prompt_ids']):r['row_id'] for r in selected}
    if not args.execute:print(json.dumps(receipts,indent=2));return
    old_generate,old_factory,old_modes,old_hashes,old_argv=base.generate,base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv
    def generate(model,ids,max_new_tokens,eos_ids,chunk_size,device):
        rid=rows.get(tuple(ids[0].tolist()))
        if rid is None:raise ValueError('only requested frozen binding examples')
        if chunk_size!=ref['chunk_size'] or model.settings.attention_query_chunk_size!=ref['attention_query_chunk_size'] or model.settings.topk!=ref['topk'] or model.settings.select_blocks!=ref['select_blocks']:raise ValueError('reader/quota mismatch')
        model.selector.threshold=receipts[rid]['switch_query_position']
        tokens,timing=old_generate(model,ids,max_new_tokens,eos_ids,chunk_size,device)
        return tokens,{**timing,**receipts[rid],
            'question_prompt_tokens':ids.numel()-model.selector.threshold,
            'full_scored_prompt_tokens':ids.numel() if model.selector.history_name=='full_covariance' else ids.numel()-model.selector.threshold,
            'history_selector':model.selector.history_name,
            'question_and_answer_selector':'full_covariance','prompt_chunks_unchanged':True}
    def hashes():
        files=[Path(__file__).with_name(n) for n in ('question_phase_full.py','binding_phase_cross.py','full_covariance_probe.py','exact_probe.py')]
        return {**old_hashes(),**{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    root=Path(args.queue_root);out=Path(loc.output)
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'STOP').exists() or out.exists():raise RuntimeError('STOP or existing output')
        out.mkdir(parents=True);base.write_json(out/'question_boundaries.json',receipts)
        if confirmation:base.write_json(out/'confirmation.json',confirmation)
        try:
            base.generate,base.BlockSummarySelector,base.MODES,base.source_hashes=generate,QuestionPhaseSelector,(*old_modes,'native_question_full','full_covariance'),hashes
            sys.argv=[str(Path(__file__)),*remaining];base.main()
        finally:
            base.generate,base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv=old_generate,old_factory,old_modes,old_hashes,old_argv

if __name__=='__main__':main()
