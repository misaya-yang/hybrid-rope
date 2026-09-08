"""Request-static per-layer MrPro/BM tables with the stock Qwen attention path."""
from copy import deepcopy

from .runtime import install, verify


class LayerTablePolicy:
    """Compute each table once per forward, select cos/sin at attention entry.

    The policy is fixed through prefill and decoding. It neither changes the
    attention mask nor reinterprets existing cache contents.
    """

    def __init__(self, model, tables, assignments):
        if len(assignments) != len(model.model.layers) or not set(assignments) <= tables.keys():
            raise ValueError('one known table per layer is required')
        if model.config.model_type != 'qwen2' or getattr(model.config, 'use_sliding_window', False):
            raise ValueError('this policy is qualified only for full-attention Qwen2')
        self.model = model
        self.handles = []
        self.embeddings = {}
        names = list(dict.fromkeys(assignments))
        self.rotaries = {}
        for name in names:
            install(model, tables[name]); verify(model, tables[name])
            self.rotaries[name] = deepcopy(model.model.rotary_emb)
        self.base = names[0]
        install(model, tables[self.base]); verify(model, tables[self.base])

        def cache_embeddings(module, args, output):
            self.embeddings = {self.base: output}
            for name, rotary in self.rotaries.items():
                if name != self.base:
                    self.embeddings[name] = rotary(*args)

        self.handles.append(model.model.rotary_emb.register_forward_hook(cache_embeddings))
        for layer, name in zip(model.model.layers, assignments):
            def choose(module, args, kwargs, table_name=name):
                if 'position_embeddings' not in kwargs or table_name not in self.embeddings:
                    raise ValueError('unexpected attention call contract')
                kwargs = dict(kwargs)
                kwargs['position_embeddings'] = self.embeddings[table_name]
                return args, kwargs
            self.handles.append(layer.self_attn.register_forward_pre_hook(choose, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.embeddings.clear()
        self.rotaries.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
