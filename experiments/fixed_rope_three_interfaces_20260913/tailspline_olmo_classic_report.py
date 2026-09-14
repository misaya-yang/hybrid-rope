#!/usr/bin/env python3
"""Strict paired report for the TailSpline OLMo Full-13 + PPL46 contract."""
from __future__ import annotations

from . import tailspline_llama_classic_report as shared


def main() -> None:
    shared.LENGTHS = (4096, 8192, 16384)
    shared.COUNTS = {4096: 10, 8192: 10, 16384: 10}
    shared.EXPECTED_BAND = (14, 32)
    shared.PPL_CONTRACT = "TAILSPLINE_OLMO_PPL46_V1"
    shared.REPORT_STATUS = "TAILSPLINE_OLMO_CLASSIC_REPORT_V1"
    shared.main()


if __name__ == "__main__":
    main()
