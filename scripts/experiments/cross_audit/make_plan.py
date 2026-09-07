"""Build a private, reviewable job plan from exact installed assets.

MrRoPE/YaRN remain frozen references. E2 adapts the project's Z arm only;
LoRA/repeats require their own scope. No model download or GPU execution here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contracts import sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('prepared','cpt','pool','scratch-root','legacy-repo','work-dir','gpu-name','out'):
        p.add_argument('--'+a,required=True)
    p.add_argument('--python',default=sys.executable);p.add_argument('--full-steps',type=int)
    a=p.parse_args();work=Path(a.work_dir).resolve();cwd=Path.cwd().resolve()
    if a.full_steps is not None and (a.full_steps<4 or a.full_steps%4):
        raise ValueError('full steps must be a positive multiple of four for matched budgets')
    jobs=[]
    def add(name,module,arguments,timeout,dependencies=(),blocked=None):
        output=work/'runs'/name
        jobs.append(dict(id=name,argv=[a.python,'-m','scripts.experiments.cross_audit.'+module,
            *map(str,arguments),'--out',str(output)],output=str(output),timeout_seconds=timeout,
            dependencies=list(dependencies),blocked_reason=blocked))
    common=['--prepared',str(Path(a.prepared).resolve())]
    add('E0_native_controls','evaluate',common+['--arm','Native','--groups-per-cell','32','--qualify-native','--layouts','compact','deleted'],900)
    add('E0_teacher','teacher',['cache',*common,'--pool',str(Path(a.pool).resolve())],1800,['E0_native_controls'])
    for arm in ('YaRN','MrPro','Z'):
        add(f'E0_{arm}_controls','evaluate',common+['--arm',arm,'--groups-per-cell','2'],1200,['E0_native_controls'])
    training=common+['--cpt',str(Path(a.cpt).resolve()),'--cpt-sha256',sha_file(a.cpt),
                    '--teacher-cache',str(work/'runs/E0_teacher'),'--lr','0.00002','--seed','137']
    for arm in ('Z',):
        add(f'E0_{arm}_full_probe','train',training+['--arm',arm,'--regime','full','--steps','4','--probe'],1800,
            ['E0_teacher',f'E0_{arm}_controls'])
    add('E1_scratch','scratch',['--root',str(Path(a.scratch_root).resolve()),'--legacy-repo',str(Path(a.legacy_repo).resolve()),
                              '--seeds','137','256'],3600)
    for arm in ('Native','YaRN','MrUni','MrPro','Z'):
        add(f'E1_{arm}_long','evaluate',common+['--arm',arm,'--groups-per-cell','32'],3600,['E0_native_controls'])
        add(f'E1_{arm}_native','teacher',['evaluate',*common,'--pool',str(Path(a.pool).resolve()),
            '--arm',arm,'--rows-per-stratum','64'],1800,['E0_native_controls'])
    blocked='Freeze own-arm steps/token budget after E0 throughput' if not a.full_steps else None
    for arm in ('Z',):
        name=f'E2_{arm}_full_s137'
        add(name,'train',training+['--arm',arm,'--regime','full','--steps',a.full_steps or 0],5*3600,
            [f'E0_{arm}_full_probe',f'E1_{arm}_long',f'E1_{arm}_native'],blocked)
        checkpoint=work/'runs'/name/f'step_{a.full_steps or 0}'
        add(name+'_long','evaluate',common+['--arm',arm,'--checkpoint',checkpoint,'--groups-per-cell','32'],3600,[name],blocked)
        add(name+'_native','teacher',['evaluate',*common,'--pool',str(Path(a.pool).resolve()),'--arm',arm,
            '--checkpoint',checkpoint,'--rows-per-stratum','64'],1440,[name],blocked)
    # Conditional stages are explicit missing decisions, never placeholder runs.
    conditional=[
        dict(stage='E3/repeat',status='SEPARATE_SCOPE',reason='Own-method LoRA/repeats only when explicitly planned; no automatically adapted MrRoPE/YaRN opponents'),
        dict(stage='E4',status='CONDITIONAL',reason='New scratch condition must follow E1 plus an independent prediction; no automatic sweep'),
        dict(stage='E5',status='INPUTS_REQUIRED',reason='Exact original-paper model and independent document/task instances absent; do not relabel old Qwen development as confirmation'),
    ]
    paths=list((cwd/'scripts/experiments/cross_audit').glob('*.py'))+list((cwd/'scripts/lib/rope').glob('*.py'))
    plan=dict(schema=1,source='RoPE_ICLR2027_Cross_Audit_Theory_and_Codex_Plan_20260906.md',
        source_sha256='eade4043ca4481a0f2f7a59da9ec1f5e8172808ee39d3890c745273ff688824a',
        cwd=str(cwd),state_dir=str(work/'job_state'),gpu_name=a.gpu_name,min_gpu_memory_gib=31,
        authorization='NOT GRANTED by this generated plan; operator needs exact user approval',
        code_files={str(x.relative_to(cwd)):sha_file(x) for x in paths},jobs=jobs,conditional_stages=conditional,
        contract_notes=['MrRoPE/YaRN are frozen references; only project Z is adapted. Report adaptation cost and compare Z before/after.',
            'Final trained weights plus per-step metrics only; no intermediate checkpoints or optimizer/RNG resume',
            'Four-step full probes use FP32 masters/Adam and BF16 compute, saving then removing only disposable probe snapshots',
            'The source audit does not freeze loss mixture; proposed update is CPT + paired answer SFT + Native KL, weights 1/1/1',
            'Equal deterministic sample order/step counts imply matched tokens; inspect actual token receipts',
            'Full checkpoints need substantially more disk than the old LoRA states',
            'No stage or dependency completion certifies scientific control validity; research lead reviews before next stage'])
    write_json(a.out,plan);print(json.dumps(dict(plan=a.out,sha256=sha_file(a.out),jobs=len(jobs))))


if __name__=='__main__':main()
