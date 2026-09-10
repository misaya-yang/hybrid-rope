"""E10 small cutoff-supervised residual map; the language model stays frozen."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from .selector_controls import BlockSummarySelector
from .run import write_json


class CutoffResidual(nn.Module):
    def __init__(self,d=128,zdim=321):
        super().__init__()
        self.u=nn.Sequential(nn.Linear(d,64),nn.SiLU(),nn.Linear(64,8,bias=False))
        self.v=nn.Sequential(nn.Linear(zdim,64),nn.SiLU(),nn.Linear(64,8,bias=False))
        nn.init.zeros_(self.u[-1].weight)
        self.register_buffer('q_mean',torch.zeros(d));self.register_buffer('q_std',torch.ones(d))
        self.register_buffer('z_mean',torch.zeros(zdim));self.register_buffer('z_std',torch.ones(zdim))

    def query(self,q):return self.u((q-self.q_mean)/self.q_std)
    def descriptor(self,z):return self.v((z-self.z_mean)/self.z_std)


def features(summary):
    return torch.cat((summary.mean,summary.extra['var_x'],summary.extra['cov_xy'],
                      summary.extra['var_y'],summary.log_weight[...,None]),-1)


class LearnedCutoffSelector(BlockSummarySelector):
    def __init__(self,mode='e10_cutoff',**kwargs):
        super().__init__('pc2',**kwargs)
        path=os.environ.get('PC2_CUTOFF_MODEL')
        if not path:raise ValueError('PC2_CUTOFF_MODEL must identify the fixed calibration-trained residual')
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.residual=CutoffResidual(state['d'],state['zdim'])
        self.residual.load_state_dict(state['state_dict']);self.residual.eval()
        self.projected={};self.on_device=None
        self.metrics['e10_model_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @torch.no_grad()
    def logmass(self,context):
        if self.on_device!=context.q.device:
            self.residual.to(context.q.device);self.on_device=context.q.device
        if int(context.query_positions[0])==0:self.projected.pop(context.layer_idx,None)
        out=super().logmass(context)
        summary=self.cache.get(context.layer_idx)
        if summary is None:return out
        old=self.projected.get(context.layer_idx)
        done=old.shape[1] if old is not None else 0
        if summary.blocks>done:
            new=self.residual.descriptor(features(summary)[:,done:])
            old=torch.cat((old,new),1) if old is not None else new
            self.projected[context.layer_idx]=old
        h,_,d=context.k.shape
        q=context.q.float().reshape(h,-1,context.q.shape[1],d)/math.sqrt(d)
        correction=torch.einsum('hgqr,hbr->hgqb',self.residual.query(q),old)
        complete=torch.arange(summary.blocks,device=q.device)[None]<context.query_positions[:,None]//context.settings.block_size
        out[...,:summary.blocks]+=correction.masked_fill(~complete[None,None],0)
        self.metrics['max_metadata_bytes']=(sum(s.nbytes() for s in self.cache.values())+
                                           sum(v.nbytes for v in self.projected.values())+
                                           sum(p.nbytes for p in self.residual.parameters()))
        return out


def prepare_examples(raw):
    random=torch.Generator().manual_seed(20260910)
    examples=[]
    for item in raw:
        for head in range(item['q'].shape[0]):
            target=item['exact'][head].softmax(-1).sum(0)
            blocks=target.numel();current=item['full_blocks']
            ids=torch.arange(blocks)
            remote=ids[(ids>0)&(ids<current-16)]
            m=min(15,remote.numel())
            order=remote[torch.argsort(target[remote],descending=True,stable=True)]
            if order.numel()<=m or m==0:continue
            count=min(8,m,order.numel()-m)
            positive=order[m-count:m].repeat(2)
            boundary=order[m:m+count]
            outside=order[m:]
            global_negative=outside[torch.randint(outside.numel(),(count,),generator=random)]
            negative=torch.cat((boundary,global_negative))
            margin=target[positive]-target[negative]
            examples.append({'row_id':item['row_id'],'role':item['role'],'q':item['q'][head],
                             'z':item['z'][head],'pc2':item['pc2'][head],'target':target,
                             'positive':positive,'negative':negative,'margin':margin,
                             'full':item['full_blocks'],'remote':remote,'m':m})
    return examples


def predict(model,e):
    correction=model.query(e['q'])@model.descriptor(e['z']).T
    logits=e['pc2'].clone()
    logits[:,:e['full']]+=correction
    return logits.softmax(-1).sum(0)


@torch.no_grad()
def evaluate(model,examples,scale):
    losses=[];regrets=[]
    for e in examples:
        score=predict(model,e)
        losses.append(float(torch.relu(e['margin']-(score[e['positive']]-score[e['negative']])).mean()/scale))
        ids=e['remote'];m=e['m']
        reference=ids[torch.argsort(e['target'][ids],descending=True,stable=True)[:m]]
        selected=ids[torch.argsort(score[ids],descending=True,stable=True)[:m]]
        regrets.append(float(e['target'][reference].sum()-e['target'][selected].sum()))
    return {'normalized_hinge':sum(losses)/len(losses),'mean_cutoff_regret':sum(regrets)/len(regrets),'groups':len(examples)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20260910)
    examples=prepare_examples(torch.load(args.teacher,map_location='cpu',weights_only=True))
    fit=[e for e in examples if e['role']=='fit'];validation=[e for e in examples if e['role']=='validation']
    if not fit or not validation:raise ValueError('independent fit and validation documents are required')
    d,zdim=fit[0]['q'].shape[-1],fit[0]['z'].shape[-1]
    model=CutoffResidual(d,zdim)
    q=torch.cat([e['q'] for e in fit]);z=torch.cat([e['z'] for e in fit])
    model.q_mean.copy_(q.mean(0));model.q_std.copy_(q.std(0).clamp_min(1e-4))
    model.z_mean.copy_(z.mean(0));model.z_std.copy_(z.std(0).clamp_min(1e-4))
    scale=max(float(torch.cat([e['margin'] for e in fit]).mean()),1e-6)
    model.to(args.device)
    for e in examples:
        for key,value in e.items():
            if isinstance(value,torch.Tensor):e[key]=value.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    random=torch.Generator().manual_seed(20260910)
    trace=[];started=time.time()
    for step in range(501):
        if step%100==0:
            row={'step':step,'fit':evaluate(model,fit,scale),'validation':evaluate(model,validation,scale)}
            trace.append(row);print(json.dumps(row),flush=True)
        if step==500:break
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for index in torch.randint(len(fit),(8,),generator=random).tolist():
            e=fit[index];score=predict(model,e)
            losses.append(torch.relu(e['margin']-(score[e['positive']]-score[e['negative']])).mean()/scale)
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite cutoff loss')
        loss.backward();optimizer.step()
    model.cpu()
    torch.save({'d':d,'zdim':zdim,'state_dict':model.state_dict(),
                'teacher_sha256':hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest(),
                'updates':500,'answer_labels_used':False,'backbone_updated':False},root/'cutoff_model.pt')
    write_json(root/'training.json',{'status':'COMPLETE','updates':500,'trace':trace,'seconds':time.time()-started,
                                   'source_documents_fit':sorted({e['row_id'] for e in fit}),
                                   'source_documents_validation':sorted({e['row_id'] for e in validation}),
                                   'selection':'fixed final step500, no DEV-based checkpoint selection',
                                   'architecture':'global u/v maps, each one64-wide hidden layer,8-dimensional output',
                                   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__':main()
