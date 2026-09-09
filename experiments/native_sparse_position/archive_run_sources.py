"""Copy only hash-matched run sources; explicitly inventory unavailable versions."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('runs',nargs='+');a=p.parse_args()
 for name in a.runs:
  run=a.root/name;meta=json.loads((run/'status.json').read_text());dst=run/'source';dst.mkdir(exist_ok=True);inventory=[]
  for filename,wanted in meta.get('code_sha256',{}).items():
   source=a.root/filename;found=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None;match=found==wanted
   if match:shutil.copy2(source,dst/filename)
   inventory.append({'filename':filename,'recorded_sha256':wanted,'current_sha256':found,'copied_exact_version':match})
  (run/'source_provenance.json').write_text(json.dumps({'files':inventory,'boundary':'Only source files exactly matching recorded run hashes were copied; missing versions are explicit'},indent=2))
  print(name,sum(r['copied_exact_version'] for r in inventory),len(inventory))
if __name__=='__main__':main()
