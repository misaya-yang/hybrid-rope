"""Run one explicit matched phase; prints exact commands without --execute."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def commands(root,phase,tokens):
    root=Path(root)
    common=[sys.executable,'-m']
    if phase=='smoke':
        return [common+['experiments.evq_recovery.train','--root',str(root),'--arm','Cosh','--smoke','--execute']]
    arms=['Cosh','YaRN'] if phase=='recovery' else ['Native','Exponential','Hybrid']
    jobs=[]
    for arm in (['Native']+arms if phase=='recovery' else arms):
        label=f'{arm}_lora_untrained_dev_core'
        if not (root/'evaluation'/label/'summary.json').exists():
            jobs.append(common+['experiments.evq_recovery.evaluate','--root',str(root),'--arm',arm,'--execute'])
    for arm in arms:
        checkpoint=root/'runs'/f'{arm}_lora'/f'checkpoint-{tokens}'
        if not (checkpoint/'state.json').exists():
            argv=common+['experiments.evq_recovery.train','--root',str(root),'--arm',arm,
                         '--until-cpt-tokens',str(tokens),'--execute']
            latest=root/'runs'/f'{arm}_lora'/'latest.json'
            if latest.exists():
                prior=json.loads(latest.read_text())
                if prior['cpt_tokens']>=tokens:raise ValueError('latest run is beyond missing requested checkpoint')
                argv+=['--resume',prior['checkpoint']]
            jobs.append(argv)
        label=f'{arm}_lora_checkpoint-{tokens}_dev_core'
        if not (root/'evaluation'/label/'summary.json').exists():
            jobs.append(common+['experiments.evq_recovery.evaluate','--root',str(root),'--arm',arm,
                                '--checkpoint',str(checkpoint),'--execute'])
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['smoke','recovery','shapes'],default='recovery')
    p.add_argument('--cpt-tokens',type=int,default=33554432)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    for command in commands(a.root.resolve(),a.phase,a.cpt_tokens):
        print(json.dumps({'argv':command,'execute':a.execute}),flush=True)
        if a.execute:subprocess.run(command,check=True)


if __name__=='__main__':main()
