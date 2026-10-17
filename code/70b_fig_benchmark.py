"""Regenerate the benchmark figure on its own.

    python 70b_fig_benchmark.py

The figure itself is drawn by ``fig_benchmark`` in 70_figures.py, which builds
every figure in the paper; this entry point exists only so that one figure can
be redrawn without touching the rest.  An earlier version of this file carried
its own copy of the drawing code, which is how the pipeline and the one-off
script drifted apart and the two-panel figure came back.
"""

from __future__ import annotations

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIG_DIR  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "figures_pipeline", os.path.join(here, "70_figures.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = os.path.join(FIG_DIR, "fig2_benchmark.png")
    module.fig_benchmark(module.load_rows("main"), out)
    print("written:", out)


if __name__ == "__main__":
    main()
