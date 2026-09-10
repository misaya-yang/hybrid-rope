"""Run the ten-direction candidates through the existing NOSA/QA runner."""
import hashlib
import argparse
import os
from pathlib import Path
import shutil
import sys

from .nonlinear_gqa import NonlinearGQASelector
from .tail_value import TailValueSelector
from .bounded_int8 import BoundedInt8Selector
from .projected_distribution import ProjectedDistributionSelector
from .temporal_response import TemporalResponseSelector
from .group_budget import GroupBudgetSelector
from .two_component import TwoComponentSelector
from .residual_sampling import ResidualSamplingSelector
from .learned_cutoff import LearnedCutoffSelector


class TenSelector(NonlinearGQASelector):
    active = None

    def __init__(self, mode="exact_mass", **kwargs):
        self.value_impl = TailValueSelector(mode, **kwargs) if mode == "e08_tail_value" else None
        self.index_impl = BoundedInt8Selector(mode, **kwargs) if mode == "e01_int8" else None
        if mode in ProjectedDistributionSelector.modes:
            self.index_impl = ProjectedDistributionSelector(mode, **kwargs)
        if mode == 'e03_temporal':
            self.index_impl = TemporalResponseSelector(mode, **kwargs)
        if mode == 'e07_group_budget':
            self.index_impl = GroupBudgetSelector(mode, **kwargs)
        if mode in TwoComponentSelector.modes:
            self.index_impl = TwoComponentSelector(mode, **kwargs)
        if mode == 'e06_residual_sampling':
            self.index_impl = ResidualSamplingSelector(mode, **kwargs)
        if mode == 'e10_cutoff':
            self.index_impl = LearnedCutoffSelector(mode, **kwargs)
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
    # Freeze each new process's import tree before loading the checkpoint.
    # Later candidate repairs in the staging directory cannot change this job.
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--output')
    options,_=pre.parse_known_args()
    if options.output and not os.environ.get('PC2_FROZEN_CODE'):
        destination=Path(options.output).resolve()/'launch_code'
        package=destination/'experiments'
        if not package.exists():
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copytree(Path(__file__).resolve().parents[1],package,
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (package/'__init__.py').touch(exist_ok=True)
        os.environ['PC2_FROZEN_CODE']=str(destination)
        os.chdir(destination)
        command=getattr(sys,'orig_argv',[sys.executable,'-m','experiments.nosa_position.ten_run',*sys.argv[1:]])
        os.execv(sys.executable,command)
    from . import run as base, runtime
    from experiments.pm_keep.run import score as qa_score
    import experiments.pm_keep.run as qa_module
    old_hashes, old_score, old_generate = base.source_hashes, base.score_output, base.generate
    old_reader = runtime.selected_causal_attention
    qa_hash = hashlib.sha256(Path(qa_module.__file__).read_bytes()).hexdigest()
    def hashes():
        names = ("ten_run.py", "nonlinear_gqa.py", "tail_value.py", "exact_probe.py", "bounded_int8.py", "projected_distribution.py", "temporal_response.py", "group_budget.py", "two_component.py", "residual_sampling.py", "learned_cutoff.py")
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
    def generate(model,*args,**kwargs):
        TenSelector.active=None
        result=old_generate(model,*args,**kwargs)
        implementation=getattr(model.selector,'index_impl',None)
        if hasattr(implementation,'finalize_metrics'):
            implementation.finalize_metrics()
        return result
    base.source_hashes, base.score_output = hashes, score
    base.generate = generate
    base.MODES = (*base.MODES, "e02_nonlinear", "e08_tail_value", "e01_int8", *ProjectedDistributionSelector.modes,
                  'e03_temporal', 'e07_group_budget', *TwoComponentSelector.modes, 'e06_residual_sampling', 'e10_cutoff')
    base.BlockSummarySelector = TenSelector
    runtime.selected_causal_attention = reader
    base.main()


if __name__ == "__main__":
    main()
