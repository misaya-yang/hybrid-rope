"""Run the ten-direction candidates through the existing NOSA/QA runner."""
import hashlib
from pathlib import Path

from .nonlinear_gqa import NonlinearGQASelector
from .tail_value import TailValueSelector
from .bounded_int8 import BoundedInt8Selector
from .projected_distribution import ProjectedDistributionSelector


class TenSelector(NonlinearGQASelector):
    active = None

    def __init__(self, mode="exact_mass", **kwargs):
        self.value_impl = TailValueSelector(mode, **kwargs) if mode == "e08_tail_value" else None
        self.index_impl = BoundedInt8Selector(mode, **kwargs) if mode == "e01_int8" else None
        if mode in ProjectedDistributionSelector.modes:
            self.index_impl = ProjectedDistributionSelector(mode, **kwargs)
        super().__init__("exact_mass" if self.value_impl or self.index_impl else mode, **kwargs)

    def __call__(self, context):
        TenSelector.active = self
        if self.index_impl:
            result = self.index_impl(context)
            self.metrics = self.index_impl.metrics
            return result
        if self.value_impl:
            result = self.value_impl(context)
            self.metrics = self.value_impl.metrics
            return result
        return super().__call__(context)


def main():
    from . import run as base, runtime
    from experiments.pm_keep.run import score as qa_score
    import experiments.pm_keep.run as qa_module
    old_hashes, old_score = base.source_hashes, base.score_output
    old_reader = runtime.selected_causal_attention
    qa_hash = hashlib.sha256(Path(qa_module.__file__).read_bytes()).hexdigest()
    def hashes():
        names = ("ten_run.py", "nonlinear_gqa.py", "tail_value.py", "exact_probe.py", "bounded_int8.py", "projected_distribution.py")
        return {**old_hashes(), **{n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}}
    def score(row, generated, tokenizer, eos_ids):
        if row["score_contract"] == "longbench_qa_f1_context_first_v1":
            return {**qa_score(row, generated, tokenizer, eos_ids), "qa_source_sha256": qa_hash}
        return old_score(row, generated, tokenizer, eos_ids)
    def reader(context, selected):
        active = TenSelector.active
        if active is not None and active.value_impl is not None:
            return active.value_impl.read(context, selected, old_reader)
        return old_reader(context, selected)
    base.source_hashes, base.score_output = hashes, score
    base.MODES = (*base.MODES, "e02_nonlinear", "e08_tail_value", "e01_int8", *ProjectedDistributionSelector.modes)
    base.BlockSummarySelector = TenSelector
    runtime.selected_causal_attention = reader
    base.main()


if __name__ == "__main__":
    main()
