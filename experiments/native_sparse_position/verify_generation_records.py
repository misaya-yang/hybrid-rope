"""Re-decode every saved token stream and verify literal complete-answer metrics."""
import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('runs',type=Path,nargs='+');a=p.parse_args();tok=AutoTokenizer.from_pretrained(a.model)
 e=json.loads((a.model/'generation_config.json').read_text())['eos_token_id'];e=set(e if isinstance(e,list) else [e]);all_results=[]
 for run in a.runs:
  source=run/'outputs.jsonl';rows=[json.loads(x) for x in source.read_text().splitlines()];correct=0;canonical=0
  for r in rows:
   ids=r['generated_ids'];ended=ids[-1] in e
   assert ended==r['ended_eos'] and not any(i in e for i in ids[:-1])
   body=ids[:-1] if ended else ids;text=tok.decode(body,skip_special_tokens=False)
   assert text==r['output_text']
   exact=text==r['expected'];assert exact==r['text_exact'];assert (exact and ended)==r['full_exact_and_eos']
   correct+=exact and ended
   canonical+=ended and body==tok(r['expected'],add_special_tokens=False)['input_ids']
  result={'run':run.name,'rows':len(rows),'every_saved_body_matches_raw_token_decode':True,'every_raw_exact_and_eos_metric_matches':True,'correct_raw_body_and_eos':correct,'canonical_expected_token_ids_and_eos':canonical,'outputs_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
  (run/'raw_token_verification.json').write_text(json.dumps(result,indent=2));all_results.append(result)
 print(json.dumps(all_results,indent=2))
if __name__=='__main__':main()
