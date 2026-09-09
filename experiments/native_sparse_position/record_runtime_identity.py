"""Bounded local-model fingerprint; no network, model forward or CUDA context."""
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
import torch,transformers

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 files=sorted(p for p in a.model.iterdir() if p.is_file() and (p.suffix in ('.json','.safetensors','.txt','.jinja')))
 hashes={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
 data={'model_path':str(a.model),'files':hashes,'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'torch_cuda':torch.version.cuda,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),'boundary':'Current local file identity, no claim of a freshly revalidated remote repository revision'}
 a.output.write_text(json.dumps(data,indent=2));print(json.dumps({'files':len(files),'total_bytes_hashed':sum(x['bytes'] for x in hashes.values()),'output':str(a.output),'gpu':data['gpu']}))
if __name__=='__main__':main()
