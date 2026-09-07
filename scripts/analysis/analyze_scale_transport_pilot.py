"""Standard-library receipt analysis and one untested, prior-art-based composition.

Does not run a model, tune task scores, or alter any original receipt.
"""
import argparse,collections,hashlib,json,math,struct
from pathlib import Path


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def read64(path):
    b=path.read_bytes()
    if b[:8]!=b'\x93NUMPY\x01\x00':raise ValueError('expected NPY v1')
    n=struct.unpack('<H',b[8:10])[0]
    import ast
    h=ast.literal_eval(b[10:10+n].decode().strip())
    if h['descr']!='<f4' or h['fortran_order'] or h['shape']!=(64,):raise ValueError('array layout')
    return list(struct.unpack('<64f',b[10+n:]))

def tensor_sha(x):return hashlib.sha256(struct.pack('<64f',*x)).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.receipts)
    rows=[]
    for run in ['unguarded_01','tail_01']:
        r=root/'runs'/run;raw=(r/'examples.jsonl').read_bytes();m=json.loads((r/'manifest.json').read_text())
        if hashlib.sha256(raw).hexdigest()!=m['examples_sha256']:raise ValueError('raw receipt hash')
        items=[json.loads(l) for l in raw.splitlines()]
        if len(items)!=m['rows']:raise ValueError('row count')
        rows+=items
    grouped=collections.defaultdict(list)
    for row in rows:grouped[(row['arm'],row['bucket'])].append(row)
    summary=[]
    for (arm,bucket),items in sorted(grouped.items()):
        summary.append(dict(arm=arm,bucket=bucket,rows=len(items),unique_contexts=len({r['context_sha256'] for r in items}),f1=sum(r['qa_f1'] for r in items)/len(items)))
    reference={r['row_id']:r for r in grouped[('Native','native')]};tailrows=grouped[('MrMiddleScaleTail','native')]
    contributions=[]
    for row in tailrows:
        ref=reference[row['row_id']]
        if row['context_sha256']!=ref['context_sha256']:raise ValueError('pairing')
        delta=row['qa_f1']-ref['qa_f1']
        if abs(delta)>1e-12:contributions.append(dict(row_id=row['row_id'],delta_f1=delta,contribution_to_mean=delta/len(tailrows)))
    T=read64(root/'derived_arrays/tail.npy')
    expected=json.loads((root/'runs/tail_01/intervention.json').read_text())['table_sha256']
    if tensor_sha(T)!=expected:raise ValueError('runtime tail table identity')
    proposal=json.loads((root/'runs/pilot_01/proposal.json').read_text())
    control=json.loads((root/'runs/pilot_01/visibility_control.json').read_text())
    residuals=[-math.log(proposal['slots'][c['slot']]['early']['raw_r']/c['fixed_content_crop']['raw_r'])/math.log(2) for c in control]
    # Same last-20-slot cosine rule as CoPE, evaluated in float64 then stored FP32.
    # This is an explicit reference convention, not claimed bit-identical to Torch cos.
    window=[1. if j<44 else .5*(1+math.cos(math.pi*(j-44)/19)) for j in range(64)]
    candidate=[f32(x*w) for x,w in zip(T,window)]
    if candidate[:44]!=T[:44] or candidate[-1]!=0 or any(x<0 or not math.isfinite(x) for x in candidate):raise ValueError('composition contract')
    if any(candidate[i]<candidate[i+1] for i in range(63)):raise ValueError('frequency crossing')
    max_delta=max(abs(x-y) for x,y in zip(candidate,T))
    rotation_bounds={str(L):2*math.sin(min(math.pi,L*max_delta)/2) for L in (32768,131072)}
    result=dict(status='CPU_ANALYSIS_AND_UNTESTED_COMPOSITION',scope='same eight-row development buckets; not new capability evidence',summary=summary,
        tail_vs_native_short_contributions=contributions,
        raw_scale_contrast=dict(min=min(residuals),max=max(residuals),slots_abs_above_half=sum(abs(x)>.5 for x in residuals),status='diagnostic only; not adopted as a frequency rule'),
        candidate=dict(name='MR_SCALE_TAIL_COPE20',status='UNTESTED',source_tail_tensor_sha256=expected,gain=1+.1*math.log(4),values_float32=candidate,tensor_sha256=tensor_sha(candidate),
            definition='Existing Mr-middle/scale-tail array times last-20-slot cosine taper; no refit, no task-based parameter choice',
            numerical_convention='Python float64 cosine, serialize IEEE float32; zero retained exactly',first_unchanged_slots=44,zero_slots=[63],
            equal_adjacent_slots=[i for i in range(63) if candidate[i]==candidate[i+1]],
            extra_rotation_operator_bounds=rotation_bounds,bound_scope='continuous distance interval, ungained block-diagonal relative rotation against tested tail table; not LM output',
            supported_claim='deterministic finite nonnegative nonincreasing array; first 44 slots unchanged',unsupported_claim='quality gain, optimality, novelty or SOTA'))
    out=Path(a.out);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','summary','tail_vs_native_short_contributions','raw_scale_contrast']},indent=2));print('candidate_sha256',result['candidate']['tensor_sha256'])
if __name__=='__main__':main()
