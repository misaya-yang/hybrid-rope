"""Prepared one-query bridge between two sparse/full softmax readers.

The caller supplies actual selected addresses, logits, and values. This does not
replace a model's selector, recurrent state, cache, or output-frame conversion.
No model integration or efficiency claim follows from the CPU checks.
"""
import torch
from torch import nn

from .normalized_reference import ReferenceBias, References, retain_references


class ReferenceBridge(nn.Module):
    def __init__(self, hidden_size, writer_heads, reader_heads, inv_freq,
                 references_per_head=2):
        super().__init__()
        self.writer_heads = writer_heads
        self.reader_heads = reader_heads
        self.count = references_per_head
        self.head_logits = nn.Parameter(torch.zeros(reader_heads, writer_heads,
            dtype=inv_freq.dtype, device=inv_freq.device))
        self.biases = nn.ModuleList([ReferenceBias(hidden_size, inv_freq)
                                    for _ in range(reader_heads)])
        self._written = None

    def write(self, positions, probabilities, query_position):
        """Record only this token's writer read, including its unnormalized tail."""
        if positions.ndim != 2 or positions.shape != probabilities.shape:
            raise ValueError("writer positions/probabilities must be [H,S]")
        if positions.shape[0] != self.writer_heads:
            raise ValueError("writer head count changed")
        refs = [retain_references(p, w, self.count, query_position)
                for p, w in zip(positions, probabilities)]
        self._written = (query_position, refs)

    def read(self, hidden, native_logits, values, positions, query_position,
             sink_logits=None, mode="arithmetic", return_delta=False):
        """Keep existing per-head support and value frame, perturb only scores."""
        if self._written is None or self._written[0] != query_position:
            raise ValueError("writer reference missing or belongs to a different query token")
        if native_logits.shape != positions.shape or native_logits.shape[0] != self.reader_heads:
            raise ValueError("reader logits/positions must be [H,S]")
        if values.shape[:2] != native_logits.shape:
            raise ValueError("reader values must match the existing selected keys")
        if bool((positions < 0).any()) or bool((positions > query_position).any()):
            raise ValueError("reader support must be causal")
        refs = self._written[1]
        source_positions = torch.cat([r.positions for r in refs])
        source_masses = torch.cat([r.masses for r in refs])
        source_heads = torch.cat([torch.full_like(r.positions, h) for h, r in enumerate(refs)])
        unique, inverse = source_positions.unique(sorted=True, return_inverse=True)
        head_weights = self.head_logits.softmax(-1)
        outputs, probabilities, retained = [], [], []
        for h, bias in enumerate(self.biases):
            # Equal addresses from different writer heads denote one alternative.
            mass = source_masses * head_weights[h, source_heads]
            combined = torch.zeros(unique.numel(), dtype=mass.dtype, device=mass.device)
            combined = combined.scatter_add(0, inverse, mass)
            chosen = retain_references(unique, combined, self.count, query_position)
            if mode == "point" and chosen.masses.numel():
                # MAP location, but the same total retained mass/gate as the
                # multi-reference method. This is not a second normalization
                # of the original writer distribution.
                selected = chosen.masses.argmax().reshape(1)
                chosen = References(chosen.positions[selected], chosen.masses.sum().reshape(1),
                                    chosen.residual_mass)
            elif mode == "current":
                chosen = References(torch.full_like(chosen.positions, query_position),
                                    chosen.masses, chosen.residual_mass)
            elif mode not in ("arithmetic", "geometric"):
                raise ValueError("unknown reference mode")
            sink = None if sink_logits is None else sink_logits[h]
            out, prob = bias(hidden, native_logits[h], values[h], positions[h], chosen, sink,
                            pooling="geometric" if mode == "geometric" else "arithmetic",
                            return_delta=return_delta)
            outputs.append(out)
            probabilities.append(prob)
            retained.append(chosen.masses.sum())
        return torch.stack(outputs), torch.stack(probabilities), torch.stack(retained)

    def clear(self):
        self._written = None
