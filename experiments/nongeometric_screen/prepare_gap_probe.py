"""Freeze the user's GPT-5.6 Pro three-arm log-gap proposal, without search."""
import argparse
import json
import math
from pathlib import Path
import struct


def f32(x):return struct.unpack('f',struct.pack('f',x))[0]


def prepare(root,tables_path):
    root=Path(root);tables=json.loads(Path(tables_path).read_text())
    native=tables['Native']['values_float32'];base=tables['MrPro']['values_float32'];gain=tables['MrPro']['gain']
    candidates={'G1_gap_widen':(24,48),'G2_gap_narrow':(36,36),'G3_pair_shift_slow':(36,48)}
    receipt={'source':'User-provided GPT-5.6 Pro analysis, 2026-09-10',
        'baseline_m28_m29':[30/306,42/306],'step':6/306,'candidates':{},
        'interpretation':'Widen/narrow preserve the geometric-mean frequency, not arithmetic center. Adjacent exterior gaps also change; the total endpoint log-span stays fixed.'}
    for index,(name,numerators) in enumerate(candidates.items()):
        freq=list(base)
        for slot,num in zip((28,29),numerators):freq[slot]=f32(native[slot]*4**(-num/306))
        gaps=[math.log(freq[j]/freq[j+1])-math.log(base[j]/base[j+1]) for j in (27,28,29)]
        center=.5*math.log(freq[28]*freq[29]/(base[28]*base[29]))
        a,b=numerators[0]/306-30/306,numerators[1]/306-42/306
        expected=[a*math.log(4),(b-a)*math.log(4),-b*math.log(4)]
        assert max(abs(x-y) for x,y in zip(gaps,expected))<1e-6
        assert abs(center+(a+b)*math.log(4)/2)<1e-6
        assert abs(sum(gaps))<1e-6
        spec={'operator':'static','table':{'values_float32':freq,'gain':gain}}
        job={'id':name,'spec':spec,'panel':'full','nll_docs':16,'nll_lengths':[8192,16384,32768]}
        path=root/'queue'/f'044{chr(97+index)}_{name}.json'
        if path.exists():assert json.loads(path.read_text())==job
        else:path.write_text(json.dumps(job,indent=2)+'\n')
        receipt['candidates'][name]={'m28_m29':[n/306 for n in numerators],'spec':spec,
            'three_log_gap_changes':gaps,'log_geometric_center_change':center,'delta_m_L2':math.hypot(a,b)}
    out=root/'planned_controls/gap_probe.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    job={'id':'gap_probe_binding16','action':'module','module':'holdout_eval','prepared':str(root/'heldout_diverse_s20260910'),
        'cohort':'mixed_s20260910','methods':['MrPro',*candidates],'tasks':['niah_multikey_2','niah_multiquery'],
        'lengths':[131072],'per_cell':16}
    (root/'queue/046_gap_probe_binding16.json').write_text(json.dumps(job,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in data.items() if k!='spec'} for name,data in receipt['candidates'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tables',required=True)
    args=p.parse_args();prepare(args.root,args.tables)
