"""Per-document matched-target evaluation; no table or gain modifications."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from eval_inputs import make_example
from runtime import validate as validate_runtime
from build_tables import build_table

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.original_root))
    from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import run_experiment as old
    validate_runtime(old)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    meta=checkpoint['metadata'];arm=meta['arm']
    checkpoint_hash=old.sha256_file(a.checkpoint)
    manifest=json.loads(a.data.read_text())
    if meta['validation_sha256']!=manifest['validation']['sha256']:raise ValueError('Different evaluation data')
    docs=np.load(manifest['validation']['path'],mmap_mode='r')
    model=old.GPT(meta['config'],torch.from_numpy(build_table(arm)))
    model.load_state_dict(checkpoint['model']);del checkpoint
    if not torch.equal(model.blocks[0].attention.rope.inv_freq,torch.from_numpy(build_table(arm))):raise ValueError('Table changed')
    model.cuda().eval();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    with open(a.output,'x') as out,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for length,condition in [(512,'intact'),(2048,'intact'),(4096,'intact'),(8192,'intact'),(4096,'remote_replaced')]:
            for i,doc in enumerate(docs):
                donor=docs[manifest['validation']['derangement'][i]] if condition=='remote_replaced' else None
                x,targets,loc=make_example(doc,length,donor)
                logits=model(torch.as_tensor(x,device='cuda')[None]).float()
                losses=torch.nn.functional.cross_entropy(logits[:,-256:].reshape(-1,50304),torch.as_tensor(targets,device='cuda'),reduction='none')
                row={'arm':arm,'training_seed':meta['seed'],'checkpoint_tokens':meta['input_tokens'],
                 'source_document_id':manifest['validation']['document_ids'][i],'document_index':i,
                 'context_length':length,'condition':condition,**loc,'sum_nll':float(losses.sum()),
                 'checkpoint_sha256':checkpoint_hash,'signature':meta['signature'],
                 'target_token_count':256,'target_sha256':hashlib.sha256(targets.astype('<i8').tobytes()).hexdigest()}
                if length==2048:
                    labels=np.asarray(doc[:8193][-2048:],dtype=np.int64)
                    whole=torch.nn.functional.cross_entropy(logits.reshape(-1,50304),torch.as_tensor(labels,device='cuda'),reduction='sum')
                    row['full_window_sum_nll']=float(whole);row['full_window_target_count']=2048
                if not np.isfinite(row['sum_nll']):raise RuntimeError('Nonfinite NLL')
                out.write(json.dumps(row)+'\n');out.flush()
            print(json.dumps({'length':length,'condition':condition,'documents':len(docs)}),flush=True)
if __name__=='__main__':main()
