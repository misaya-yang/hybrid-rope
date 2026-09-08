"""Identical targets under context length and remote-prefix interventions."""
import numpy as np

def make_example(document, length, donor=None, tail=256):
    doc=np.asarray(document,dtype=np.int64)
    if len(doc)<8193 or length not in (512,2048,4096,8192):
        raise ValueError('Need at least 8193 document tokens and a registered length')
    anchor=doc[:8193]
    segment=anchor[-(length+1):]
    inputs=segment[:-1].copy()
    labels=segment[1:].copy()
    if donor is not None:
        if length!=4096 or len(donor)<3584:
            raise ValueError('Remote replacement requires 4K and sufficient donor tokens')
        inputs[:-512]=np.asarray(donor,dtype=np.int64)[:length-512]
    return inputs,labels[-tail:],{'anchor_end':8193,'target_start':8193-tail,'target_end':8193}
