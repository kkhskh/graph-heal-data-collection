#!/usr/bin/env python3
# Auto-generated shim – delegates to the legacy implementation to avoid code duplication.
"""Expose baseline_vs_graphheal under the importable *graph_heal.scripts* package.

The real implementation lives in the historical directory
``graph-heal/scripts/baseline_vs_graphheal.py``.  This thin wrapper simply
executes that file via ``runpy`` so callers can *import* it cleanly:

    python -m graph_heal.scripts.baseline_vs_graphheal --capture …
"""
from __future__ import annotations

import runpy
import pathlib
import sys

LEGACY_PATH = pathlib.Path(__file__).resolve().parents[2] / "graph-heal" / "scripts" / "baseline_vs_graphheal.py"
if not LEGACY_PATH.exists():
    raise ImportError("Legacy baseline_vs_graphheal script not found at %s" % LEGACY_PATH)

# Ensure project root is importable
_ROOT = LEGACY_PATH.parents[2]
import sys as _sys
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(LEGACY_PATH.parent))
    runpy.run_path(LEGACY_PATH.as_posix(), run_name="__main__") 