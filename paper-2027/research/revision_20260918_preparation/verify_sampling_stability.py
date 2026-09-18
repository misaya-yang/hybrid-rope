"""Check the proposed generic stability interpretation using explicit features.

This checks one finite Fourier dictionary metric, not model/task performance or
all possible sampling models. No checkpoint, model state or CUDA is used.
"""
import json
from pathlib import Path
import numpy as np


def verify():
    n, scale, length = 17, 4, 32768
    q = np.arange(n+1, dtype=float)
    profiles = {
        'TailSpline': q*(3*n*n+3*n+1-q*q)/(n*(n+1)*(2*n+1)),
        'MrPro': q*(q+1)/(n*(n+1)),
    }
    native = 500000.**(-np.arange(18,36,dtype=float)/64)
    positions = np.arange(length,dtype=float)-(length-1)/2
    result = {}
    for name, profile in profiles.items():
        phase = positions[:,None]*(native*scale**(-profile))[None,:]
        features = np.concatenate((np.cos(phase),np.sin(phase)),axis=1)
        # Symmetric positions make within-pair cos/sin cross terms zero.
        features /= np.sqrt(np.sum(features*features,axis=0))[None,:]
        singular = np.linalg.svd(features,compute_uv=False)
        result[name] = {
            'lambda_min': float(singular[-1]**2),
            'lambda_max': float(singular[0]**2),
            'gram_condition_number': float((singular[0]/singular[-1])**2),
            'feature_condition_number': float(singular[0]/singular[-1]),
        }
    assert result['TailSpline']['lambda_min'] < result['MrPro']['lambda_min']
    assert result['TailSpline']['gram_condition_number'] > result['MrPro']['gram_condition_number']
    return {'status':'COMPLETE',
            'scope':'Public Llama mathematical grid; S4; centered discrete 32K; inclusive transition [18,35]; real full-pair block whitening; explicit 32768-by-36 feature SVD',
            'results':result,
            'conclusion':'MrPro is an admissible same-endpoint alternative with better lower eigenvalue and condition number under this declared metric. This refutes TailSpline optimality for these finite-dictionary stability objectives, not its task quality or every possible signal model.'}


if __name__ == '__main__':
    result = verify()
    Path(__file__).with_name('sampling_stability_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
