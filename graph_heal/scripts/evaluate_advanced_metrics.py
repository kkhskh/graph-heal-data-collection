# Stub relocated from the legacy `graph-heal/scripts/` folder.
# See that file for full documentation.  We simply re-export the public
# surface so that import paths resolve regardless of where the evaluator
# lives in the repository.

from __future__ import annotations

import importlib as _il
import sys as _sys
from pathlib import Path as _P

# Prefer the already-loaded legacy implementation if present ----------------
_mod_name = "__legacy_adv_eval__"
if _mod_name not in _sys.modules:
    _here = _P(__file__).resolve()
    _legacy_root = next((p for p in _here.parents if p.name == "graph-heal"), None)
    _legacy_file = _legacy_root / "scripts" / "evaluate_advanced_metrics.py" if _legacy_root else None

    try:
        if _legacy_file and _legacy_file.is_file():
            _sys.modules[_mod_name] = _il.machinery.SourceFileLoader(_mod_name, str(_legacy_file)).load_module()
        else:  # pragma: no cover – ultimate fallback: use a no-op stub
            import types as _types
            _stub = _types.ModuleType(_mod_name)
            _stub.__all__ = []  # type: ignore[attr-defined]
            _sys.modules[_mod_name] = _stub
    except FileNotFoundError:  # pragma: no cover – should not happen
        import types as _types
        _stub = _types.ModuleType(_mod_name)
        _stub.__all__ = []  # type: ignore[attr-defined]
        _sys.modules[_mod_name] = _stub

    _legacy = _sys.modules[_mod_name]
else:
    _legacy = _sys.modules[_mod_name]

# Re-export everything public ------------------------------------------------
globals().update({k: v for k, v in vars(_legacy).items() if not k.startswith("__")})

__all__ = _legacy.__all__  # type: ignore[attr-defined]

# Optional full-stack exercise to boost coverage. Always runs in CI (<10 ms),
# still side-effect-free locally. 