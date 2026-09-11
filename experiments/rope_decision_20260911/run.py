"""Frozen-weight task-decision probe and local static-table calibration.

Every target is a full stock-greedy token sequence. Full-prefill derivatives
propose updates; stock generation and the original scorer judge task outcomes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from experiments.curvature_20260910.model import FrozenRoPE
from scripts.experiments.olmo_fast_screen.ruler_bench import score


def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)


class Runner:
    def __init__(self,args):
        self.args=args
        self.plan=json.loads(args.plan.read_text())
        self.root=args.root
        self.root.mkdir(parents=True,exist_ok=True)
        self.identity=hashlib.sha256(args.plan.read_bytes()).hexdigest()
        ident=self.root/'identity.json'
        receipt=dict(plan_sha256=self.identity,mode=args.mode,arms=args.arms,
                     max_steps=args.max_steps,start_from_sha256=hashlib.sha256(args.start_from.read_bytes()).hexdigest() if args.start_from else None)
        if ident.exists() and json.loads(ident.read_text())!=receipt:
            raise ValueError('existing output belongs to another plan or algorithm setting')
        atomic(ident,receipt)
        self.model=AutoModelForCausalLM.from_pretrained(self.plan['model'],
            local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},
            attn_implementation='sdpa').eval().requires_grad_(False)
        self.tok=AutoTokenizer.from_pretrained(self.plan['model'],local_files_only=True)
        self.gcfg=GenerationConfig.from_pretrained(self.plan['model'],local_files_only=True)
        # Transformers 5 serializes disabled/default options as None; generate
        # resolves these to ordinary greedy defaults, just as in the old runner.
        if self.gcfg.do_sample not in (None,False) or self.gcfg.num_beams not in (None,1) or self.gcfg.repetition_penalty not in (None,1):
            raise ValueError('probe currently requires ordinary greedy without repetition penalty')
        if getattr(self.model.config,'attention_dropout',0.)!=0.:
            raise ValueError('checkpointed gradient path requires zero attention dropout')
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.f=FrozenRoPE.__new__(FrozenRoPE)
        self.f.model,self.f.rotary=self.model,self.model.model.rotary_emb
        self.stock_rotary_forward=self.f.rotary.forward
        self.f.device,self.f.dtype='cuda',torch.bfloat16
        self.f._grad_patched=False
        self.tables=self.plan['tables']
        self.cases=self.plan['cases']
        self.base_nu=np.asarray(self.tables[self.plan['initial_arm']]['values_float32'],float)
        self.base_gain=float(self.tables[self.plan['initial_arm']]['gain'])
        self.events=self.root/'events.jsonl'
        import transformers
        atomic(self.root/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,
            model=self.plan['model'],generation=self.gcfg.to_dict(),plan_sha256=self.identity,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            gradient_source_sha256=hashlib.sha256(Path(__import__('experiments.curvature_20260910.model',fromlist=['x']).__file__).read_bytes()).hexdigest()))

    def event(self,data):
        data=dict(time=time.time(),**data)
        with self.events.open('a') as out:out.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
        atomic(self.root/'live.json',data)

    def table_at(self,x):
        nu=(self.base_nu*np.exp(-math.log(4)*np.asarray(x[:64]))).astype(np.float32)
        gain=self.base_gain*math.exp(float(x[64])/2)
        if not np.isfinite(nu).all() or np.any(nu<=0) or np.any(np.diff(nu)>=0):
            raise ValueError('nonpositive or unordered frequency proposal')
        return dict(values_float32=nu.tolist(),gain=gain)

    def install(self,table,grad=False):
        self.f.rotary.forward=self.stock_rotary_forward
        self.f._grad_patched=False
        nu=torch.tensor(table['values_float32'],device='cuda',dtype=torch.float32,requires_grad=grad)
        gain=torch.tensor(table['gain'],device='cuda',dtype=torch.float32,requires_grad=grad) if grad else float(table['gain'])
        self.f.rotary.inv_freq=nu
        self.f.rotary.original_inv_freq=nu.detach().clone()
        self.f.rotary.attention_scaling=gain
        if grad:self.f.enable_grad_path()
        return nu,gain

    def generate(self,table,case,label):
        folder=self.root/'generations'
        folder.mkdir(exist_ok=True)
        path=folder/f'{label}__{case["row_id"]}.json'
        table_hash=hashlib.sha256(json.dumps(table,sort_keys=True).encode()).hexdigest()
        if path.exists():
            old=json.loads(path.read_text())
            if old['table_hash']!=table_hash or old['plan_sha256']!=self.identity:
                raise ValueError('generation reuse identity mismatch')
            return old
        self.install(table)
        self.model.eval()
        ids=torch.tensor([case['prompt_ids']],device='cuda')
        self.event(dict(phase='GENERATE_START',label=label,row_id=case['row_id']))
        start=time.monotonic()
        with torch.inference_mode():
            tokens=self.model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=self.gcfg,
                                      max_new_tokens=case['max_new_tokens'])[0,ids.shape[1]:].tolist()
        eos=self.gcfg.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos or [])
        ended=bool(tokens and tokens[-1] in eos)
        text=self.tok.decode(tokens[:-1] if ended else tokens,skip_special_tokens=False)
        result=dict(row_id=case['row_id'],role=case['role'],label=label,generated_ids=tokens,
                    ended_eos=ended,output_text=text,correct=score(case,text),
                    table_hash=table_hash,plan_sha256=self.identity,
                    prompt_sha256=hashlib.sha256(np.asarray(case['prompt_ids'],dtype='<i8').tobytes()).hexdigest(),
                    seconds=time.monotonic()-start)
        atomic(path,result)
        self.event(dict(phase='GENERATE_DONE',row_id=case['row_id'],label=label,
                        correct=result['correct'],tokens=len(tokens),seconds=result['seconds']))
        del ids
        return result

    def margins(self,table,case,target,grad):
        nu,gain=self.install(table,grad)
        # Training mode only enables non-reentrant checkpointing; weights remain
        # frozen and dropout is exactly zero. Acceptance uses eval stock generate.
        self.model.train(grad)
        y=target['generated_ids']
        ids=torch.tensor([case['prompt_ids']+y[:-1]],device='cuda')
        desired=torch.tensor(y,device='cuda')
        self.event(dict(phase='MARGIN_START',row_id=case['row_id'],grad=grad,tokens=len(y)))
        start=time.monotonic()
        with torch.set_grad_enabled(grad):
            lg=self.model(ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(y)).logits[0].float()
            top=torch.topk(lg,2,dim=-1).indices
            competitor=torch.where(top[:,0]==desired,top[:,1],top[:,0])
            pos=torch.arange(len(y),device='cuda')
            margins=lg[pos,desired]-lg[pos,competitor]
            worst=int(margins.argmin())
            objective=margins[worst]
            result=dict(row_id=case['row_id'],role=case['role'],min_margin=float(objective.detach()),
                        negative_positions=int((margins.detach()<0).sum()),
                        zero_positions=int((margins.detach()==0).sum()),worst_position=worst,
                        desired_token=y[worst],competitor_token=int(competitor[worst]),
                        margins=margins.detach().cpu().tolist())
            if grad:
                gi,gg=torch.autograd.grad(objective,(nu,gain),allow_unused=False)
                result['gradient'] = np.concatenate((-math.log(4)*nu.detach().double().cpu().numpy()*gi.detach().double().cpu().numpy(),
                                                     [.5*float(gain.detach())*float(gg.detach())])).tolist()
        result['seconds']=time.monotonic()-start
        result['peak_gpu_mib']=torch.cuda.max_memory_allocated()/2**20
        self.model.eval()
        self.event({k:v for k,v in result.items() if k not in ('margins','gradient')}|dict(phase='MARGIN_DONE'))
        del ids,desired,lg,margins,objective,nu,gain
        torch.cuda.empty_cache()
        return result

    def probe(self):
        arms=self.args.arms.split(',')
        for arm in arms:
            for case in self.cases:self.generate(self.tables[arm],case,arm)
        parity=[]
        for case in self.cases:
            arm=case['target_arm']
            target=self.generate(self.tables[arm],case,arm)
            if target['correct']<.999:
                parity.append(dict(row_id=case['row_id'],status='TARGET_REPLAY_CHANGED',score=target['correct']))
                continue
            # Same table and same full teacher-forced input, before/after the
            # patched rotary. CPU tests also cover cached greedy parity.
            ordinary=self.margins(self.tables[arm],case,target,False)
            differentiable=self.margins(self.tables[arm],case,target,True)
            parity.append(dict(row_id=case['row_id'],forward_min_equal=ordinary['min_margin']==differentiable['min_margin'],
                               all_margins_equal=ordinary['margins']==differentiable['margins'],
                               ordinary=ordinary,differentiable=differentiable))
            atomic(self.root/'path_checks.json',parity)
        atomic(self.root/'status.json',dict(status='PROBE_COMPLETE',plan_sha256=self.identity))

    def evaluate(self):
        from collections import defaultdict
        summaries={}
        all_records={}
        for arm in self.args.arms.split(','):
            records=[self.generate(self.tables[arm],case,arm) for case in self.cases]
            cells=defaultdict(list)
            for case,record in zip(self.cases,records):
                cells[(case['length_cap'],case['task'])].append(record['correct'])
            result={}
            for cap in sorted({k[0] for k in cells}):
                tasks={task:float(np.mean(v)) for (length,task),v in cells.items() if length==cap}
                result[cap]=dict(task_scores=tasks,macro=float(np.mean(list(tasks.values()))))
            summaries[arm]=result
            all_records[arm]=records
            atomic(self.root/'scores.json',summaries)
        atomic(self.root/'status.json',dict(status='EVALUATION_COMPLETE',plan_sha256=self.identity,scores=summaries))

    def calibrate(self):
        from .solve import propose
        targets=[]
        for case in self.cases:
            target=self.generate(self.tables[case['target_arm']],case,case['target_arm'])
            if target['correct']+1e-12<case.get('minimum_target_score',.999):raise RuntimeError('selected target does not replay; revise case selection from actual outputs')
            targets.append(target)
        initial=[self.generate(self.table_at(np.zeros(65)),c,self.plan['initial_arm']) for c in self.cases]
        maxlen=max(len(c['prompt_ids']) for c in self.cases)
        radii=np.minimum(.05,.25/(maxlen*math.log(4)*self.base_nu))
        radii=np.append(radii,self.plan.get('gain_step_radius',.05))
        if self.args.mode=='gain-only':radii[:64]=0.
        evq_direction=None
        if self.args.mode=='evq-calibrate':
            from .tables import evq_midband_direction
            lo,hi=self.plan['evq_band']
            evq_direction=evq_midband_direction(self.base_nu,lo,hi,self.plan.get('evq_tau',1.))
            radii=np.zeros((65,2))
            radii[:64,0]=evq_direction*self.plan.get('evq_step_radius',.1)
            radii[64,1]=self.plan.get('gain_step_radius',.1)
        x=np.asarray(json.loads(self.args.start_from.read_text())['parameters'],float) if self.args.start_from else np.zeros(65)
        history=[]
        for iteration in range(self.args.max_steps):
            current=[self.margins(self.table_at(x),c,t,True) for c,t in zip(self.cases,targets)]
            atomic(self.root/f'gradients_{iteration:02}.json',current)
            gamma=np.array([r['min_margin'] for r in current])
            jac=np.array([r['gradient'] for r in current])
            desired_margin=np.where(np.array([c['role']=='short_repair' for c in self.cases]),.01,0.)
            # First minimize incompatible slack, then minimize movement at that
            # slack. Active constraints are worst full-output margins; the next
            # real forward discovers changed token/vocabulary competitors.
            bounds=None
            if evq_direction is not None:
                alpha=float(x[:64]@evq_direction/(evq_direction@evq_direction))
                alpha_step=float(radii[:64,0]@evq_direction/(evq_direction@evq_direction))
                bounds=[(max(-1.,-alpha/alpha_step),min(1.,(1.-alpha)/alpha_step)),(-1.,1.)]
            step,slack=propose(gamma,jac,radii,np.asarray(self.table_at(x)['values_float32']),desired_margin,bounds)
            prior_merit=float(np.maximum(desired_margin-gamma,0).sum())
            if np.max(np.abs(step))<1e-10:
                history.append(dict(iteration=iteration,status='NO_LOCAL_STEP',merit=prior_merit))
                break
            accepted=False
            for factor in (1.,.5,.25,.125):
                proposed=x+factor*step
                try:
                    table=self.table_at(proposed)
                except ValueError:
                    merit=None
                    continue
                vals=[self.margins(table,c,t,False) for c,t in zip(self.cases,targets)]
                merit=float(np.maximum(desired_margin-np.array([r['min_margin'] for r in vals]),0).sum())
                if merit < prior_merit-1e-5 or merit<=1e-5:
                    x=proposed
                    accepted=True
                    break
            event=dict(iteration=iteration,accepted=accepted,prior_merit=prior_merit,merit=merit,
                       predicted_slack=slack.tolist(),step_factor=factor,parameters=x.tolist())
            if accepted:
                results=[self.generate(self.table_at(x),c,f'candidate_{iteration:02}') for c in self.cases]
                event['scores']=[r['correct'] for r in results]
                event['exact_output_matches']=[r['generated_ids']==t['generated_ids'] for r,t in zip(results,targets)]
                atomic(self.root/f'candidate_{iteration:02}.json',dict(table=self.table_at(x),parameters=x.tolist(),event=event))
                task_win=all(r['correct']>=t['correct'] for r,t in zip(results,targets))
                history.append(event)
                atomic(self.root/'history.json',history)
                if task_win:
                    atomic(self.root/'status.json',dict(status='CALIBRATION_TASK_WIN',iteration=iteration,
                        initial_scores=[r['correct'] for r in initial],scores=event['scores'],
                        scope='Four development cases only; fresh-task evaluation required.',table=self.table_at(x)))
                    return
            else:
                history.append(event)
                atomic(self.root/'history.json',history)
                radii*=.5
        atomic(self.root/'status.json',dict(status='LOCAL_CALIBRATION_UNRESOLVED',history=history,
            scope='This local solver did not establish the requested task repair; not a no-go theorem.'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=['probe','calibrate','gain-only','evq-calibrate','evaluate'],default='probe')
    ap.add_argument('--arms',default='bm,b4wide,mrpro_archive,yarn_index,yarn_turns_paper,evq_endpoint_t1,evq_midpoint_t1')
    ap.add_argument('--max-steps',type=int,default=8)
    ap.add_argument('--start-from',type=Path)
    args=ap.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    atomic(args.root/'status.json',dict(status='RUNNING',pid=os.getpid(),start=time.time()))
    try:
        runner=Runner(args)
        if args.mode=='probe':runner.probe()
        elif args.mode=='evaluate':runner.evaluate()
        else:runner.calibrate()
    except BaseException as exc:
        atomic(args.root/'status.json',dict(status='FAILED',error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
