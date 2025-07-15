class ServiceMonitor:
    """No-op replacement used only for test import compatibility."""
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        """Pretend to start monitoring – does nothing."""
        return True

    def stop(self):
        """Pretend to stop monitoring – does nothing."""
        return True


class GraphUpdater:
    """Dummy graph updater that records calls for verification."""

    def __init__(self) -> None:
        self._updates = []

    # The real implementation might push updates to a DAG or message bus.
    # We just remember them so tests can inspect behaviour.
    def push(self, update):
        self._updates.append(update)

    def get_updates(self):
        return list(self._updates)


# ---------------------------------------------------------------------------
# Optional – real-time Prometheus polling monitor
# ---------------------------------------------------------------------------

import threading
import time
import logging
from typing import Dict, Any, Mapping, Optional

try:
    import requests  # Heavyweight import guarded so that unit-tests without
except ImportError:  # network stack can still import this module.
    requests = None  # type: ignore


class PrometheusServiceMonitor(ServiceMonitor):
    """Poll Prometheus for service metrics and feed them into *GraphHeal*.

    Parameters
    ----------
    gh : GraphHeal
        An *already-built* GraphHeal instance whose ``update_metrics`` method
        will receive fresh samples each polling cycle.
    prom_url : str, default "http://localhost:9090"
        Base URL where Prometheus is reachable.
    metric_map : Mapping[str, Mapping[str, str]]
        Nested mapping of *service_id* → { *metric_alias*: <PromQL query> }.
        The alias becomes the key passed to ``update_metrics`` while the query
        string is sent to Prometheus' HTTP API.
    poll_interval : int, default 5
        Seconds between polling cycles.
    """

    def __init__(self,
                 gh: "GraphHeal",
                 prom_url: str = "http://localhost:9090",
                 metric_map: Optional[Mapping[str, Mapping[str, str]]] = None,
                 poll_interval: int = 5):
        super().__init__()
        self.gh = gh
        self.prom_url = prom_url.rstrip("/")
        self.metric_map: Dict[str, Dict[str, str]] = {
            k: dict(v) for k, v in (metric_map or {}).items()
        }
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="prom-monitor", daemon=True)

        self._log = logging.getLogger(__name__)

        if requests is None:
            self._log.warning("`requests` not installed – PrometheusServiceMonitor disabled.")

    # ------------------------------------------------------------------
    # Public control API – mirrors the stub for compatibility
    # ------------------------------------------------------------------

    def start(self):  # type: ignore[override]
        if requests is None:
            return False
        self._thread.start()
        return True

    def stop(self):  # type: ignore[override]
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.poll_interval + 1)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loop(self):
        while not self._stop_event.is_set():
            start = time.time()
            for svc_id, queries in self.metric_map.items():
                sample: Dict[str, float] = {}
                for alias, prom_query in queries.items():
                    val = self._query(prom_query)
                    if val is not None:
                        sample[alias] = val

                if sample:
                    self.gh.update_metrics(svc_id, sample)  # type: ignore[arg-type]

            # Sleep the *remaining* time so that drift stays low.
            elapsed = time.time() - start
            time_to_sleep = max(0.0, self.poll_interval - elapsed)
            self._stop_event.wait(time_to_sleep)

    def _query(self, prom_query: str) -> Optional[float]:
        # Quick path – recognise trivial *scalar(NUMBER)* expressions so that
        # developers can demo Graph-Heal without a running Prometheus instance.
        # This uses the same syntax accepted by Prometheus' expression browser
        # and is exploited by scripts like ``run_reactor_monitor.py`` which
        # intentionally inject a spike via "scalar(120)". Handling it here
        # avoids the need for a network round-trip and works even if the
        # provided ``prom_url`` is unreachable.
        import re

        match = re.fullmatch(r"scalar\(\s*([\d.]+)\s*\)", prom_query)
        if match:
            try:
                return float(match.group(1))
            except ValueError:  # malformed number – treat as missing metric
                return None

        try:
            resp = requests.get(
                f"{self.prom_url}/api/v1/query",
                params={"query": prom_query},
                timeout=2,
            )
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
            if data.get("status") != "success" or not data.get("data", {}).get("result"):
                return None
            # Vector results → [timestamp, value]
            val_str = data["data"]["result"][0]["value"][1]
            return float(val_str)
        except Exception as exc:  # noqa: BLE001 – broad but logged
            self._log.debug("Prometheus query '%s' failed: %s", prom_query, exc)
            return None


# Re-export for `from graph_heal.monitoring import PrometheusServiceMonitor`
__all__ = ["ServiceMonitor", "GraphUpdater", "PrometheusServiceMonitor"] 