"""Apply the existing YaRN/MrPro gain law to each query's visible length."""
import math

import torch


def query_factor(position_ids, *, native_length, installed_gain, coefficient=.1):
    if native_length < 1 or installed_gain <= 0 or coefficient < 0:
        raise ValueError('invalid gain geometry')
    n = position_ids.double()+1
    if torch.any(n < 1):
        raise ValueError('nonnegative positions required')
    gain = 1+coefficient*torch.log(torch.clamp(n/native_length, min=1.))
    return (gain.square()/installed_gain**2).float()


class CausalGain:
    """Full causal, unpadded Qwen2 only; keeps installed K rotations unchanged.

    Multiplying Q by g(n)^2/g(S)^2 gives effective scalar logit gain g(n)^2.
    Existing frequencies are unchanged. This is not full Native model parity.
    """

    def __init__(self, model, *, native_length, installed_gain, coefficient=.1):
        if (model.config.model_type != 'qwen2' or getattr(model.config, 'use_sliding_window', False)
                or not math.isfinite(installed_gain) or not math.isfinite(coefficient)):
            raise ValueError('requires qualified full-attention Qwen2 and finite gain')
        self.factor = None
        self.handles = []

        def positions(module, args, output):
            ids = args[1]
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError('only one unpadded prompt is supported')
            if not torch.equal(ids[0], torch.arange(ids[0, 0], ids[0, 0]+ids.shape[1], device=ids.device)):
                raise ValueError('noncontiguous positions are not visible-key counts')
            self.factor = query_factor(ids, native_length=native_length,
                installed_gain=installed_gain, coefficient=coefficient)

        def scale_query(module, args, output):
            if self.factor is None or output.shape[:2] != self.factor.shape:
                raise ValueError('query/position shape mismatch')
            return output*self.factor.to(output.dtype).unsqueeze(-1)

        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'q_norm'):
                raise ValueError('pre-norm query scaling is not qualified')
        self.handles.append(model.model.rotary_emb.register_forward_hook(positions))
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.q_proj.register_forward_hook(scale_query))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.factor = None
