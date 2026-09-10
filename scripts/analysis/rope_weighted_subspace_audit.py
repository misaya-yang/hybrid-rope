"""CPU audit of learned Q/K operator response in source-weak directions.

Consumes existing static tables and the derived Q/K Gram. No model loading,
model execution, method fitting, or job submission is performed.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nongeometric_screen.smooth_budget import construct


def features(f, lag):
    phase = -np.asarray(lag)[:, None] * f[None, :]
    x = np.empty((len(lag), 2*len(f)))
    x[:, ::2], x[:, 1::2] = np.cos(phase), np.sin(phase)
    return x


def main():
    root = ROOT/'results/nongeometric_screen_20260909'
    tables = json.loads((root/'reference_tables.json').read_text())
    native = np.array(tables['Native']['values_float32'])
    mr = np.array(tables['MrPro']['values_float32'])
    full = json.loads((ROOT/'docs/research/ROPE_QWEN15_FULL_LAG_P2_CANDIDATE_20260907.json').read_text())
    fs = {'Native': native, 'PI': native/4, 'MrPro': mr,
          'BM': np.array(tables['MrProBM']['values_float32']),
          'P2': np.array(full['tables']['FullLagP2']['values_float32']),
          'Smooth': np.array(construct(tables,23,40)['Smooth_MrBudget']['values_float32'])}
    slots = np.flatnonzero((2*np.pi/mr >= 32768) & (2*np.pi/mr <= 131072))
    for label, direction in [('Slower',-1),('Faster',1)]:
        f = mr.copy(); f[slots] += direction/131072
        fs[label] = f.astype(np.float32).astype(float)
    x0 = features(native, np.arange(32768))/np.sqrt(32768)
    _, s, vh = np.linalg.svd(x0, full_matrices=False)
    h = np.load(root/'planned_controls/qk_operator_gram.npz')['per_layer']
    report = {'scope':'Source-weak rotary operator energy for independent unit-second-moment inputs; not measured activations or task performance',
              'source':'all integer causal lags 0..32767',
              'target':'all integer causal lags 32768..131071',
              'signed_separation':'key minus query; negative for positive causal lag',
              'attention_gain':1, 'thresholds':{}}
    for threshold in [1e-6,1e-8,1e-10]:
        v = vh[s <= s[0]*threshold]
        hv = np.einsum('ai,lij,bj->lab',v,h,v,optimize=True)
        block = {'discarded_directions':len(v),'methods':{}}
        for label,f in fs.items():
            covariance = np.zeros((len(v),len(v)))
            for start in range(32768,131072,8192):
                y = features(f,np.arange(start,min(start+8192,131072))) @ v.T
                covariance += y.T @ y / (131072-32768)
            energies = np.einsum('lab,ab->l',hv,covariance)
            block['methods'][label] = {'unweighted':float(np.trace(covariance)),
                'weighted_mean':float(np.mean(energies)),
                'weighted_per_layer':energies.tolist()}
        sm = np.array(block['methods']['Smooth']['weighted_per_layer'])
        mr_e = np.array(block['methods']['MrPro']['weighted_per_layer'])
        block['smooth_less_than_mr_layers'] = int(np.sum(sm<mr_e))
        report['thresholds'][str(threshold)] = block
        print(threshold, {k:round(v['weighted_mean'],9) for k,v in block['methods'].items()},flush=True)
    output = root/'planned_controls/weighted_source_subspace_audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
