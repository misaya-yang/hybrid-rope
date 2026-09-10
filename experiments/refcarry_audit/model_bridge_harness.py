"""One-token Qwen2/3 adapter harness for the prepared reference composition.

This is an instrumented experimental path, NOT an efficient inference backend.
Prefix prefill delegates to native SDPA. The optional support callback must return
the original experiment's selected key indices; this harness invents no selector.
Qwen3.5/GDN and native MiniCPM sparse-kernel integration are not qualified here.
"""
import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from .reference_bridge import ReferenceBridge


class ModelBridgeHarness:
    def __init__(self, model, writer_layer, reader_layer, inv_freq, support=None):
        if model.config.model_type not in ('qwen2', 'qwen3'):
            raise ValueError('only Qwen2/3 are qualified for this harness')
        layers = {m.layer_idx: m for m in model.modules()
                  if m.__class__.__name__ in ('Qwen2Attention', 'Qwen3Attention')}
        if writer_layer not in layers or reader_layer not in layers or writer_layer >= reader_layer:
            raise ValueError('a writer must precede the selected reader')
        if any(getattr(m, 'sliding_window', None) is not None for m in layers.values()):
            raise ValueError('sliding/cache-truncated positions need a separate qualified adapter')
        self.model, self.layers = model, layers
        self.writer, self.reader = writer_layer, reader_layer
        self.support = support
        self.mode = 'native'
        self.original = ALL_ATTENTION_FUNCTIONS['sdpa']
        self.name = 'reference_bridge_' + str(id(self))
        self.hidden = None
        self.calls = {'writer': 0, 'reader': 0}
        reader, writer = layers[reader_layer], layers[writer_layer]
        if hasattr(reader, 'reference_bias_adapter'):
            raise ValueError('reader already has a reference adapter')
        basis = inv_freq.to(device=next(reader.parameters()).device, dtype=torch.float32)
        self.bridge = ReferenceBridge(reader.config.hidden_size, writer.config.num_attention_heads,
                                      reader.config.num_attention_heads, basis)
        reader.add_module('reference_bias_adapter', self.bridge)
        self.hook = reader.register_forward_pre_hook(self._capture_hidden, with_kwargs=True)
        self.configs = list({id(m.config):m.config for m in layers.values()}.values())
        self.old_implementations = [c._attn_implementation for c in self.configs]
        if any(x != 'sdpa' for x in self.old_implementations):
            self.hook.remove()
            delattr(reader, 'reference_bias_adapter')
            raise ValueError('load this experiment with native sdpa before installing')
        ALL_ATTENTION_FUNCTIONS[self.name] = self.interface
        ALL_MASK_ATTENTION_FUNCTIONS[self.name] = ALL_MASK_ATTENTION_FUNCTIONS['sdpa']
        for config in self.configs:
            config._attn_implementation = self.name

    def _capture_hidden(self, module, args, kwargs):
        self.hidden = kwargs.get('hidden_states', args[0] if args else None)

    def interface(self, module, q, k, v, mask, **kwargs):
        index = module.layer_idx
        if index not in (self.writer, self.reader) or q.shape[-2] != 1:
            return self.original(module, q, k, v, mask, **kwargs)
        if q.shape[0] != 1:
            raise ValueError('one unpadded causal query per forward is supported')
        H, KV, N = q.shape[1], k.shape[1], k.shape[-2]
        position = N-1
        head_map = torch.arange(H, device=q.device)//(H//KV)
        if self.support is None:
            positions = torch.arange(N, device=q.device)[None].expand(H, -1)
        else:
            positions = self.support(module, q, k, position)
            if positions.ndim != 2 or positions.shape[0] != H:
                raise ValueError('support callback must return [query heads, selected keys]')
            if bool((positions < 0).any()) or bool((positions > position).any()):
                raise ValueError('support contains a future or invalid key')
            if any(p.unique().numel() != p.numel() for p in positions):
                raise ValueError('support duplicates physical keys')
        keys = k[0, head_map[:, None], positions]
        values = v[0, head_map[:, None], positions]
        selected_mask = None
        if mask is not None:
            if mask.ndim != 4:
                raise ValueError('expected native four-dimensional attention mask')
            mask_row = mask[0, :, -1, :N].expand(H, -1)
            selected_mask = mask_row.gather(1, positions)[None, :, None]
        if self.support is None:
            base_output, _ = self.original(module, q, k, v, mask, **kwargs)
        else:
            base_output, _ = self.original(module, q, keys[None], values[None], selected_mask, **kwargs)
        scale = float(module.scaling)
        logits = (q[0, :, 0].float()[:, None]*keys.float()).sum(-1)*scale
        if selected_mask is not None:
            selected = selected_mask[0, :, 0]
            logits = logits.masked_fill(~selected, -torch.inf) if selected.dtype == torch.bool else logits+selected
        probabilities = logits.softmax(-1)
        if index == self.writer:
            self.bridge.write(positions, probabilities, position)
            self.calls['writer'] += 1
            return base_output, None
        self.calls['reader'] += 1
        if self.mode == 'native':
            return base_output, None
        if self.hidden is None or self.hidden.shape[:2] != (1, 1):
            raise ValueError('reader hidden-state capture is missing')
        correction, _, _ = self.bridge.read(self.hidden[0, 0].float(), logits,
            values.float(), positions, position, mode=self.mode, return_delta=True)
        # Add a correction to the actual native kernel. Zero initialization is
        # exactly native even when manual and SDPA reductions differ slightly.
        delta = correction.to(base_output.dtype)[None, None]
        return base_output+delta, None

    def close(self):
        for config, previous in zip(self.configs, self.old_implementations):
            config._attn_implementation = previous
        del ALL_ATTENTION_FUNCTIONS[self.name]
        del ALL_MASK_ATTENTION_FUNCTIONS[self.name]
        self.hook.remove()
        self.bridge.clear()
        delattr(self.layers[self.reader], 'reference_bias_adapter')
