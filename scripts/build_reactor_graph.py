"""Build a *production-scale* reactor-subsystem graph for Graph-Heal.

Phase-1 of the SCADA/PLC integration roadmap: we only *model* the
physical & control dependencies so that later phases (detection / healing)
can run unchanged.  Run the script and you should see every node marked as
`healthy`.

Usage
-----
$ python scripts/build_reactor_graph.py
"""

from __future__ import annotations

# Import the *full* implementation directly to avoid the lightweight stub
# masking the real class on some PYTHONPATH layouts.
from graph_heal.graph_heal import GraphHeal
import pprint


def construct_reactor_graph() -> GraphHeal:
    """Return a GraphHeal instance loaded with all reactor subsystems."""
    gh = GraphHeal()

    # ──────────────────────────────────────────────────────────────────
    # Core reactor loop
    # ──────────────────────────────────────────────────────────────────
    gh.add_service("fuel_assemblies")
    gh.add_service("control_rods", dependencies=["fuel_assemblies"])
    gh.add_service("primary_pumps", dependencies=["fuel_assemblies"])
    gh.add_service("pressurizer", dependencies=["primary_pumps"])

    # ──────────────────────────────────────────────────────────────────
    # Steam / turbine train
    # ──────────────────────────────────────────────────────────────────
    gh.add_service("steam_generator", dependencies=["primary_pumps"])
    gh.add_service("turbine", dependencies=["steam_generator"])
    gh.add_service("generator", dependencies=["turbine"])
    gh.add_service("condenser", dependencies=["turbine"])
    gh.add_service("feedwater_pumps", dependencies=["condenser"])

    # ──────────────────────────────────────────────────────────────────
    # Safety & Balance-of-Plant (BOP)
    # ──────────────────────────────────────────────────────────────────
    gh.add_service("diesel_generators")
    gh.add_service(
        "I&C_system",
        dependencies=[
            "fuel_assemblies",
            "primary_pumps",
            "turbine",
            "diesel_generators",
        ],
    )
    gh.add_service("containment", dependencies=["I&C_system"])

    return gh


# ──────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gh = construct_reactor_graph()

    summary = gh.get_health_summary()
    print("Reactor subsystem graph constructed. Service-count:", len(summary.get("services", summary)))
    print("All services should be healthy →")
    pprint.pp(summary) 