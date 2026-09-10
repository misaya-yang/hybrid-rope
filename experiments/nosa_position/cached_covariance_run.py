"""Bounded execution-parity/timing entry for the exact second-order cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import run as base
from .cached_full_covariance import CachedFullCovarianceSelector
from .full_covariance_probe import FullCovarianceSelector


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--execute", action="store_true")
    options, remaining = parser.parse_known_args()
    if not options.execute:
        print(json.dumps({"execute": False, "shadow": options.shadow, "arguments": remaining}))
        return

    class Factory(CachedFullCovarianceSelector):
        def __init__(self, mode="cached_full_covariance"):
            if mode != "cached_full_covariance":
                raise ValueError("this entry is for the cached full-covariance arm only")
            super().__init__()
            self.reference = FullCovarianceSelector()
            self.metrics.update(shadow_comparison=options.shadow, compared_head_query_rows=0,
                                changed_head_query_rows=0, changed_selected_block_slots=0)

        def __call__(self, context):
            selected = super().__call__(context)
            if options.shadow:
                reference = self.reference(context)
                changed = selected != reference
                self.metrics["compared_head_query_rows"] += selected.shape[0] * selected.shape[1]
                self.metrics["changed_head_query_rows"] += int(changed.any(-1).sum())
                self.metrics["changed_selected_block_slots"] += int(changed.sum())
            return selected

    previous = base.source_hashes
    def hashes():
        files = ("cached_full_covariance.py", "cached_covariance_run.py", "full_covariance_probe.py", "exact_probe.py")
        return {**previous(), **{name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in files}}
    base.MODES = (*base.MODES, "cached_full_covariance")
    base.source_hashes = hashes
    base.BlockSummarySelector = Factory
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
