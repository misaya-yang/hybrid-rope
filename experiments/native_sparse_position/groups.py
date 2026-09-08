"""Static rotary-orbit farthest-first groups, per supplied PSR definition."""
import numpy as np

def phase_groups(omega,block_size=64,representatives=4):
    omega=np.asarray(omega,dtype=np.float64)
    if not 1<=representatives<=block_size:raise ValueError('Invalid representatives')
    if not np.any(omega):return np.minimum(np.arange(block_size)*representatives//block_size,representatives-1)
    delta=np.arange(block_size)[:,None]-np.arange(block_size)[None,:]
    dist=4*np.square(np.sin(delta[:,:,None]*omega[None,None,:]/2)).sum(-1)
    centers=[0];nearest=dist[:,0].copy()
    for _ in range(representatives-1):
        point=int(np.argmax(nearest));centers.append(point);nearest=np.minimum(nearest,dist[:,point])
    labels=np.argmin(dist[:,centers],axis=1)
    if len(np.unique(labels))!=representatives:raise ValueError('Degenerate phase centers')
    return labels

def controls(labels,seed=42):
    labels=np.asarray(labels);counts=np.bincount(labels)
    contiguous=np.repeat(np.arange(len(counts)),counts)
    rng=np.random.default_rng(seed)
    return {'PSR':labels,'Contiguous':contiguous,'Random':rng.permutation(contiguous)}
