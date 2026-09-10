"""CPU-only exact Gram of rotary bilinear operators from checkpoint slices.

This reads Q/K projections, never instantiates or runs a language model.
It describes operator MSE for independent unit-second-moment inputs, not
language-model loss or the empirical activation distribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def operator_gram(q, k):
    """Half-split pairing; d=key_position-query_position in [cos, sin].

    q/k include a final bias coordinate. Positive causal query-minus-key
    lags must therefore use a negative d when evaluating this Gram.
    """
    pairs = q.shape[0] // 2
    gq, gk = q @ q.T, k @ k.T
    q00, q01 = gq[:pairs, :pairs], gq[:pairs, pairs:]
    q10, q11 = gq[pairs:, :pairs], gq[pairs:, pairs:]
    k00, k01 = gk[:pairs, :pairs], gk[:pairs, pairs:]
    k10, k11 = gk[pairs:, :pairs], gk[pairs:, pairs:]
    cc = q00*k00 + q01*k01 + q10*k10 + q11*k11
    dd = q11*k00 - q10*k01 - q01*k10 + q00*k11
    cd = q01*k00 - q00*k01 + q11*k10 - q10*k11
    h = np.empty((2*pairs, 2*pairs))
    h[::2, ::2], h[1::2, 1::2] = cc, dd
    h[::2, 1::2], h[1::2, ::2] = cd, cd.T
    return (h+h.T)/2


def check_formula():
    rng = np.random.default_rng(20260910)
    q, k = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    basis = []
    for j in range(3):
        basis += [np.outer(q[j], k[j])+np.outer(q[j+3], k[j+3]),
                  np.outer(q[j+3], k[j])-np.outer(q[j], k[j+3])]
    flat = np.stack(basis).reshape(6, -1)
    error = float(np.max(np.abs(operator_gram(q, k)-flat@flat.T)))
    if error > 1e-10:
        raise ValueError(f'operator Gram identity failed: {error}')
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    error = check_formula()
    # Explicit CPU tensors, no AutoModel or CUDA invocation.
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    cfg = json.loads((args.model/'config.json').read_text())
    index = json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
    heads, kv = cfg['num_attention_heads'], cfg['num_key_value_heads']
    dim = cfg['hidden_size']//heads
    values, matched_means, bias_means = [], [], []
    for layer in range(cfg['num_hidden_layers']):
        tensors = {}
        for projection in ('q_proj', 'k_proj'):
            for parameter in ('weight', 'bias'):
                key = f'model.layers.{layer}.self_attn.{projection}.{parameter}'
                if key not in index:
                    if parameter == 'bias':
                        continue
                    raise KeyError(key)
                with safe_open(str(args.model/index[key]), framework='pt', device='cpu') as f:
                    tensors[(projection, parameter)] = f.get_tensor(key).float().numpy().astype(np.float64)
        augmented = {}
        for projection, count in [('q_proj', heads), ('k_proj', kv)]:
            a = tensors[(projection, 'weight')]
            b = tensors.get((projection, 'bias'), np.zeros(a.shape[0]))
            augmented[projection] = np.concatenate([a, b[:, None]], axis=1).reshape(count, dim, -1)
        h = np.zeros((dim, dim))
        layer_means, layer_bias_means = [], []
        for head in range(heads):
            q = augmented['q_proj'][head]
            k = augmented['k_proj'][head//(heads//kv)]
            h += operator_gram(q, k)
            # Identical pre-projection hidden content with unit second moment:
            # E[h^T A_j h] = tr(A_j), including the augmented bias coordinate.
            # This is a specified content model, not observed token-pair means.
            pairs = dim//2
            mu = np.empty(dim)
            mu[::2] = np.sum(q[:pairs]*k[:pairs]+q[pairs:]*k[pairs:], axis=1)
            mu[1::2] = np.sum(q[pairs:]*k[:pairs]-q[:pairs]*k[pairs:], axis=1)
            layer_means.append(mu/np.sqrt(dim))
            mu_bias = np.empty(dim)
            mu_bias[::2] = q[:pairs,-1]*k[:pairs,-1]+q[pairs:,-1]*k[pairs:,-1]
            mu_bias[1::2] = q[pairs:,-1]*k[:pairs,-1]-q[:pairs,-1]*k[pairs:,-1]
            layer_bias_means.append(mu_bias/np.sqrt(dim))
        values.append(h/(heads*dim))
        matched_means.append(layer_means)
        bias_means.append(layer_bias_means)
        print('layer', layer, 'trace', np.trace(values[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, per_layer=np.stack(values), mean=np.mean(values, axis=0),
                        matched_content_mean_per_head=np.asarray(matched_means),
                        independent_content_mean_per_head=np.asarray(bias_means),
                        content_correlation_advantage_per_head=np.asarray(matched_means)-np.asarray(bias_means))
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        scope='Exact rotary bilinear-operator Gram from frozen Q/K projections including biases; half-split pairing and actual GQA sharing. Input second moments set to identity, including the bias coordinate. Not measured activation statistics or LM task loss.',
        layers=len(values), heads=heads, kv_heads=kv, head_dim=dim,
        attention_gain=1, dense_identity_max_error=error,
        matched_content_mean_scope='Exact mean under identical Q/K input hidden vectors with identity second moment; not measured activations or task-relevant token means.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
