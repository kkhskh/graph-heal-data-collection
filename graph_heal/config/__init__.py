"""
Configuration package for Graph-Heal
""" 

# ---------------------------------------------------------------------------
# Legacy compatibility – some tests import a global ``SERVICES_CONFIG`` from
# ``graph_heal.config``.  The real configuration system has moved elsewhere,
# but we maintain a minimal stub here so that those imports succeed without
# leaking implementation details.
# ---------------------------------------------------------------------------

from typing import Dict, Any

# Services are keyed by name; the nested dict may include endpoint URLs,
# scaling factors, etc.  Tests generally only assert that the mapping exists,
# not on its concrete contents.
SERVICES_CONFIG: Dict[str, Any] = {}

__all__ = ["SERVICES_CONFIG"] 