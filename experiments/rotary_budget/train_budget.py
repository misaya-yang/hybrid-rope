"""Budget adapter around the recovered exact-range model, loss, sampler and runtime.

Uses the existing server trainer primitives; adds explicit budget tables, periodic
same-target validation and resumable optimizer checkpoints.
"""
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from runtime import validate as validate_runtime
from build_tables import build_table, ARMS
from eval_inputs import make_example

CONFIG=dict(vocab_size=50304,hidden_size=768,num_layers=12,num_heads=12,head_dim=64,
            intermediate_size=3072,max_position_embeddings=8192)

def atomic_json(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)

def checkpoint(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)

def validation(model,docs,count=32,length=2048):
    values=[];model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for doc in docs[:count]:
            x,y,_=make_example(doc,length)
            logits=model(torch.as_tensor(x,device='cuda')[None])[:,-256:].float()
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,50304),torch.as_tensor(y,device='cuda'))
            values.append(float(loss))
    model.train()
    return float(np.mean(values))

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--updates',type=int,default=7629);p.add_argument('--micro-batch',type=int,default=8)
    p.add_argument('--probe',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--no-compile',action='store_true');p.add_argument('--compile-mode',default='default')
    a=p.parse_args()
    if 32%a.micro_batch:raise ValueError('microbatch must divide global32')
    if not 1<=a.updates<=7629:raise ValueError('invalid budget')
    sys.path.insert(0,str(a.original_root))
    from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import run_experiment as old
    runtime=validate_runtime(old)
    manifest=json.loads(a.data.read_text())
    class Dataset(old.FlatPrefixDataset):
        def _open(self):
            if self._array is None:
                self._array=np.load(self.path,mmap_mode='r',allow_pickle=False)
                if self._array.dtype not in (np.dtype('uint16'),np.dtype('int64')):raise ValueError('storage dtype')
                self._flat=self._array.reshape(-1)
    rows=a.updates*32
    order=old.deterministic_row_order(rows,seed=a.seed)
    dataset=Dataset(manifest['train']['path'],rows=rows,seq_len=2048)
    docs=np.load(manifest['validation']['path'],mmap_mode='r')
    old.seed_everything(a.seed)
    model=old.GPT(CONFIG,torch.from_numpy(build_table(a.arm)))
    count=sum(t.numel() for t in model.parameters())
    if count!=151898880:raise ValueError(f'Architecture changed: {count}')
    initial=old.trainable_state_sha256(model)
    scientific=dict(arm=a.arm,seed=a.seed,updates=a.updates,input_tokens=a.updates*65536,
      parameter_count=count,initial_trainable_sha256=initial,row_order_sha256=old.tensor_sha256(order),
      train_semantic_sha256=manifest['train']['semantic_int64_prefix_sha256'],
      validation_sha256=manifest['validation']['sha256'],table=build_table(a.arm).tolist(),config=CONFIG,
      optimizer=dict(name='AdamW',betas=[.9,.95],weight_decay=.01,peak_lr=.0006,min_lr=.00006,
        warmup=762 if a.updates==7629 else max(1,int(a.updates*.1)),schedule='original_cosine_final_step_floor',clipping=1.),
      loss='cross_entropy_on_batch_0_to_2046_predict_batch_1_to_2047',precision='FP32_master_BF16_autocast')
    signature=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    start=0;loaded=None
    if a.resume:
        loaded=torch.load(a.output/'resume.pt',map_location='cpu',weights_only=False)
        if loaded['signature']!=signature:raise ValueError('Resume scientific contract mismatch')
        model.load_state_dict(loaded['model']);start=loaded['completed_updates']
    model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,betas=(.9,.95),weight_decay=.01,fused=True)
    if loaded:
        optimizer.load_state_dict(loaded['optimizer'])
        torch.set_rng_state(loaded['torch_rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng'])
        np.random.set_state(loaded['numpy_rng']);random.setstate(loaded['python_rng'])
        del loaded
    loss_fn=old.CausalLanguageModelLoss(model)
    if not a.no_compile:loss_fn=torch.compile(loss_fn,mode=a.compile_mode,dynamic=False,fullgraph=False)
    loader=iter(DataLoader(dataset,batch_size=a.micro_batch,sampler=old.TensorOrderSampler(order[start*32:]),
                           num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2,drop_last=True))
    metadata={**scientific,'signature':signature,'runtime':runtime,'micro_batch':a.micro_batch,
              'compile_mode':None if a.no_compile else a.compile_mode,'status':'PROBE' if a.probe else 'RUNNING',
              'pid':os.getpid(),'discarded':a.probe,
              'adapter_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
              'original_trainer_code_sha256':old.code_fingerprint()}
    atomic_json(a.output/'status.json',metadata)
    warmup=762 if a.updates==7629 else max(1,int(a.updates*.1))
    total=10 if a.probe else a.updates
    started=time.monotonic();durations=[]
    for step in range(start,total):
        torch.cuda.synchronize();tick=time.monotonic()
        lr=.0006*(step+1)/warmup if step<warmup else .00006+.5*(.0006-.00006)*(1+math.cos(math.pi*(step-warmup)/max(1,a.updates-warmup-1)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for _ in range(32//a.micro_batch):
            batch=next(loader).cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/(32//a.micro_batch)).backward();loss_sum+=float(loss.detach())/(32//a.micro_batch)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite grad norm')
        optimizer.step();torch.cuda.synchronize();seconds=time.monotonic()-tick;durations.append(seconds)
        record={'completed_updates':step+1,'input_tokens':(step+1)*65536,'loss':loss_sum,'lr':lr,
                'grad_norm':float(norm),'seconds':seconds,'tokens_per_second':65536/seconds}
        if not a.probe and ((step+1) in (762,2048,4096,6144,a.updates)):
            record['validation_2k_32doc_tail_nll']=validation(model,docs)
        with open(a.output/'train.jsonl','a') as f:f.write(json.dumps(record)+'\n')
        if step==start or (step+1)%25==0 or a.probe:
            print(json.dumps(record),flush=True)
            atomic_json(a.output/'live.json',record)
        if not a.probe and ((step+1)%256==0 or step+1==a.updates):
            checkpoint(a.output/'resume.pt',dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
             completed_updates=step+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
             numpy_rng=np.random.get_state(),python_rng=random.getstate(),metadata=metadata))
    metadata.update(status='COMPLETE',elapsed_seconds=time.monotonic()-started,
      completed_updates=total,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if a.probe:
        # The first two updates include compile/startup; remaining updates load real data.
        steady=durations[2:];speed=65536/float(np.mean(steady))
        metadata.update(steady_tokens_per_second=speed,conservative_tokens_per_second=65536/max(steady),
            estimated_18_arm_training_hours=18*499974144/speed/3600,step_seconds=durations)
    else:
        checkpoint(a.output/'model.pt',{'model':model.state_dict(),'metadata':metadata})
    atomic_json(a.output/'status.json',metadata)
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
