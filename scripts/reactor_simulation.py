"""Standalone simulation harness for the reactor-service demo.

It takes special care to *force-load* the **full** GraphHeal implementation
(graph_heal/graph_heal.py).  On some machines the legacy stub that lives in
graph-heal/graph_heal/ shadows the real package, which means many advanced
attributes (like anomaly_thresholds or recovery helpers) are missing.  The
snippet below detects that situation and explicitly loads the right file.
"""

from __future__ import annotations

from importlib import import_module
import importlib.util
import pathlib
import sys
import pprint


# ---------------------------------------------------------------------------
# Import *full* GraphHeal, fall back to explicit file-load if we accidentally
# got the minimalist stub first.
# ---------------------------------------------------------------------------

GraphHeal = import_module("graph_heal.graph_heal").GraphHeal  # type: ignore

# If we ended up with the stub (no anomaly_thresholds attr), replace it with
# the full implementation sitting in the top-level graph_heal directory.
if not hasattr(GraphHeal, "anomaly_thresholds"):
    _full_path = pathlib.Path(__file__).resolve().parents[1] / "graph_heal" / "graph_heal.py"
    spec = importlib.util.spec_from_file_location("graph_heal.graph_heal_full", _full_path)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _mod  # so repeat imports pick it up
        spec.loader.exec_module(_mod)  # type: ignore[arg-type]
        GraphHeal = _mod.GraphHeal  # type: ignore[attr-defined]


def build_reactor_graph() -> GraphHeal:
    """Instantiate GraphHeal and populate it with the reactor-control service graph."""
    gh = GraphHeal()

    # 1. Build the Service Graph ---------------------------------------------------
    gh.add_service("SensorGateway", layer="network", dependencies=[])
    gh.add_service("TempSensorAggregator", dependencies=["SensorGateway"])
    gh.add_service("PressureSensorAggregator", dependencies=["SensorGateway"])
    gh.add_service("ReactorControl", dependencies=["TempSensorAggregator", "PressureSensorAggregator"])
    gh.add_service("CoolantPumpController", dependencies=["ReactorControl"])
    gh.add_service("SafetyShutdown", dependencies=["ReactorControl"])
    gh.add_service("AnalyticsEngine", dependencies=["TempSensorAggregator", "PressureSensorAggregator"])
    gh.add_service("OperatorDashboard", dependencies=["AnalyticsEngine", "ReactorControl"])

    # 2. Custom metric thresholds --------------------------------------------------
    # Some lightweight compatibility stubs used in legacy tests do **not** expose
    # ``anomaly_thresholds``.  Create it on-the-fly so the remainder of the
    # simulation works no matter which concrete GraphHeal class was imported.
    if not hasattr(gh, "anomaly_thresholds"):
        gh.anomaly_thresholds = {}

    # Overwrite / add domain-specific thresholds. Graph-Heal will treat any metric
    # name found here as a *direct* threshold; anything else can be handled by the
    # dynamic z-score logic inside Graph-Heal (if implemented).
    gh.anomaly_thresholds.update(
        {
            "temperature": 100.0,            # °C
            "pressure": 200.0,               # bar
            "control_loop_latency": 500.0,   # ms
            "pump_response_time": 200.0      # ms
        }
    )

    return gh


# ---------------------------------------------------------------------------
# 3. Fault-Injection Scenarios
# ---------------------------------------------------------------------------

def scenario_single_sensor_spike(gh: GraphHeal):
    """Scenario A: Single Temperature spike on one sensor"""
    print("\n[Scenario A] Single-Sensor Temperature Spike → expect quarantine only")
    gh.update_metrics("TempSensorAggregator", {"temperature": 120.0})  # 20 °C above naïve threshold
    summary = gh.get_health_summary()
    pprint.pp(_get_service_info(summary, "TempSensorAggregator"))


def scenario_cascading_overload(gh: GraphHeal):
    """Scenario B: Temperature + Pressure spike causing dependency anomaly"""
    print("\n[Scenario B] Cascading Overload → expect early dependency anomaly")
    gh.update_metrics("TempSensorAggregator", {"temperature": 105.0})
    gh.update_metrics("PressureSensorAggregator", {"pressure": 220.0})
    summary = gh.get_health_summary()
    pprint.pp(_get_service_info(summary, "ReactorControl"))


def scenario_control_loop_latency(gh: GraphHeal):
    """Scenario C: Control loop latency increase between ReactorControl → Pump"""
    print("\n[Scenario C] Control Loop Latency Attack – network slowness")
    gh.update_metrics("ReactorControl", {"control_loop_latency": 650.0})
    summary = gh.get_health_summary()
    pprint.pp(_get_service_info(summary, "CoolantPumpController"))


# ---------------------------------------------------------------------------
# 4. Entry-point & CLI convenience
# ---------------------------------------------------------------------------

def run_all():
    gh = build_reactor_graph()

    # Run scenarios sequentially
    scenario_single_sensor_spike(gh)
    scenario_cascading_overload(gh)
    scenario_control_loop_latency(gh)

    # Final system summary ---------------------------------------------------
    print("\nFinal health summary:")
    pprint.pp(gh.get_health_summary())


# ---------------------------------------------------------------------------
# Helper – adapt to both full GraphHeal and minimalist stub implementations
# ---------------------------------------------------------------------------


def _get_service_info(summary: dict, service_id: str):
    """Return a *uniform* per-service dict even when a stub is used.

    Full GraphHeal → ``summary['services'][service_id]``
    Stub implementation → ``{service_id: <state>} -> wrap in generic dict``.
    """
    if "services" in summary:
        return summary["services"].get(service_id, {})
    # Legacy stub – top-level keys are service IDs
    state = summary.get(service_id, "unknown")
    return {"health_state": state}


if __name__ == "__main__":
    run_all() 