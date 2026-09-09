"""Rebuild the exponent-allocation overview from the current evidence source."""
from make_exponent_revision_figures import build_overview, load_data

if __name__ == "__main__":
    build_overview(load_data())
