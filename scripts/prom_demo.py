#!/usr/bin/env python3
"""Quick Prometheus-ingestion smoke-test for Graph-Heal.

This script subclasses ``PrometheusServiceMonitor`` with a tiny override that
parses *scalar(<number>)* PromQL expressions so we can demo the end-to-end
metric flow without needing a live Prometheus instance.

Run it from the repository root:

    python scripts/prom_demo.py

Expected output:

    Collected metrics: {'flow_rate': 3000.0, 'vibration': 0.02}
    Health summary: { ... }

Change the numbers in ``metric_map`` to cross your thresholds and watch
Graph-Heal raise anomalies automatically.
"""
from __future__ import annotations

import re
import time
import pprint
from typing import Optional

# Ensure we import the *full* monitoring module, not the legacy stub.
from graph_heal.graph_heal import GraphHeal  # type: ignore

try:
    from graph_heal.monitoring import PrometheusServiceMonitor  # type: ignore
except ImportError:
    # The stub package shadowed the real one – load it explicitly.
    import importlib.util, sys, pathlib

    _full_path = pathlib.Path(__file__).resolve().parents[1] / "graph_heal" / "monitoring.py"
    spec = importlib.util.spec_from_file_location("graph_heal.monitoring_full", _full_path)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _mod
        spec.loader.exec_module(_mod)  # type: ignore[arg-type]
        PrometheusServiceMonitor = _mod.PrometheusServiceMonitor  # type: ignore[attr-defined]


class DemoMon(PrometheusServiceMonitor):
    """Monitor that recognises *scalar(NUMBER)* queries and returns that number."""

    def _query(self, prom_query: str) -> Optional[float]:  # type: ignore[override]
        match = re.match(r".*?\(\s*([\d.]+)\s*\)", prom_query)
        if match:
            return float(match.group(1))
        return None


def main():
    gh = GraphHeal()
    gh.add_service("primary_pumps")

    metric_map = {
        "primary_pumps": {
            "flow_rate": "scalar(3000)",
            "vibration": "scalar(0.02)",
        }
    }

    mon = DemoMon(
        gh,
        prom_url="http://demo",  # not used in DemoMon
        metric_map=metric_map,
        poll_interval=1,
    )

    mon.start()
    time.sleep(2.5)  # allow a couple of polls
    mon.stop()

    print("Collected metrics:", gh.services["primary_pumps"].metrics)
    print("Health summary:")
    pprint.pp(gh.get_health_summary())


if __name__ == "__main__":
    main() 