"""Paired, source-clustered contrasts; never pool QA, retrieval, and LM metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .data import write_json


def compare(candidate,baseline,bootstrap=2000):
    def read(folder):
        if not (folder/'summary.json').exists():raise ValueError(f'incomplete evaluation: {folder}')
        return {r['id']:r for r in (json.loads(l) for l in (folder/'generations.jsonl').read_text().splitlines())}
    ca=json.loads((candidate/'contract.json').read_text())
    cb=json.loads((baseline/'contract.json').read_text())
    if ca['data_manifest_sha256']!=cb['data_manifest_sha256']:
        raise ValueError('candidate and baseline do not share frozen input data')
    a,b=read(candidate),read(baseline)
    if a.keys()!=b.keys():raise ValueError('paired evaluation IDs differ')
    groups=defaultdict(list)
    for key,row in a.items():
        reference=b[key]
        if row.get('prompt_sha256')!=reference.get('prompt_sha256') or row['references']!=reference['references']:
            raise ValueError('same row ID has different inputs or answers')
        metric='official_recall' if row['suite']=='ruler' else 'f1'
        group=(row['suite'],row['task'],row.get('length_bucket',4096),metric)
        groups[group].append((row['source_id'],row[metric],reference[metric]))
    result={}
    for key,rows in sorted(groups.items()):
        by_source=defaultdict(list)
        for source,a_score,b_score in rows:by_source[source].append(a_score-b_score)
        clusters=list(by_source.values());rng=np.random.default_rng(20260910)
        distribution=[]
        for _ in range(bootstrap):
            selected=rng.integers(len(clusters),size=len(clusters))
            distribution.append(np.mean([d for index in selected for d in clusters[index]]))
        deltas=[a_score-b_score for _,a_score,b_score in rows]
        result[str(key)]=dict(rows=len(rows),source_clusters=len(clusters),
            candidate=float(np.mean([r[1] for r in rows])),baseline=float(np.mean([r[2] for r in rows])),
            delta=float(np.mean(deltas)),wins=sum(x>0 for x in deltas),losses=sum(x<0 for x in deltas),
            ties=sum(x==0 for x in deltas),source_cluster_bootstrap95=np.quantile(distribution,[.025,.975]).tolist())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    write_json(a.out,dict(comparisons=compare(a.candidate,a.baseline),
        scope='Within the supplied fixed inputs. Source clustering handles repeated QA papers; intervals are not training-seed uncertainty or universal model guarantees.'))


if __name__=='__main__':main()
