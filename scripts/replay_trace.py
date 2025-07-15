#!/usr/bin/env python3
"""Replay a time-stamped CSV of reactor metrics into Graph-Heal.

Usage
-----
    python scripts/replay_trace.py --trace data/simulation/reactor_trace.csv \
           --thresholds config/anomaly_thresholds.yaml \
           --recovery config/recovery_strategies.yaml

The CSV must contain:
    timestamp,<service>.<metric>,<service>.<metric>,...,fault_type
Example header:
    timestamp,TempSensorAggregator.temperature,PressureSensorAggregator.pressure,ReactorControl.control_loop_latency,fault_type

At each row the script feeds every ``service.metric`` value into
``GraphHeal.update_metrics(service, {metric: value})``.
It records the first timestamp where Graph-Heal raises an anomaly and prints
basic precision/recall stats against the *fault_type* column.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time
import yaml
import logging
import datetime as _dt
from collections import defaultdict

from graph_heal.graph_heal import GraphHeal
from scripts.reactor_simulation import build_reactor_graph  # reuse topology
from scripts.reactor_recovery import ReactorRecoveryAdapter

_LOG = logging.getLogger("trace-replay")


def parse_args():
    p = argparse.ArgumentParser(description="Replay reactor trace into Graph-Heal")
    p.add_argument("--trace", required=True, help="Path to CSV with timestamp + metrics + optional fault_type column")
    p.add_argument("--thresholds", help="YAML file with anomaly thresholds")
    p.add_argument("--recovery", help="YAML file with recovery strategies")
    p.add_argument("--adapter-endpoint", default="", help="OPC-UA endpoint; empty → dry-run mode")
    p.add_argument("--rate", type=float, default=1.0, help="Replay speed-up factor (1.0 = real-time, 0 = max speed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    gh = build_reactor_graph()

    # Load YAML configs if provided ------------------------------------------------
    if args.thresholds:
        gh.anomaly_thresholds.update(yaml.safe_load(pathlib.Path(args.thresholds).read_text()))
    if args.recovery:
        gh.recovery_strategies.update(yaml.safe_load(pathlib.Path(args.recovery).read_text()))

    # Inject recovery adapter ------------------------------------------------------
    gh.recovery_adapter = ReactorRecoveryAdapter(args.adapter_endpoint or None)

    # Statistics ------------------------------------------------------------------
    fault_start: float | None = None
    detect_time: float | None = None
    false_positives = 0

    # Track per-service state transitions ------------------------------------
    prev_state: dict[str, str] = {svc: "healthy" for svc in gh.services}
    detections: list[dict] = []  # each: service, metric?, t_fault, t_detected, t_recovered

    with open(args.trace, newline="") as f:
        reader = csv.DictReader(f)
        field_services = {
            col: tuple(col.split(".", 1))  # -> (service, metric)
            for col in reader.fieldnames if col not in {"timestamp", "fault_type"}
        }

        previous_ts: float | None = None
        for row in reader:
            ts = float(row["timestamp"])
            # Real-time replay sleep --------------------------------------------
            if args.rate > 0 and previous_ts is not None:
                dt = (ts - previous_ts) / args.rate
                if dt > 0:
                    time.sleep(dt)
            previous_ts = ts

            # Feed metrics ------------------------------------------------------
            per_service: defaultdict[str, dict] = defaultdict(dict)
            for col, (svc, metric) in field_services.items():
                val = float(row[col])
                per_service[svc][metric] = val
            for svc, sample in per_service.items():
                gh.update_metrics(svc, sample)  # type: ignore[arg-type]

            # Fault ground truth ------------------------------------------------
            fault_tag = row.get("fault_type", "")
            if fault_tag and fault_tag.lower() != "none":
                fault_start = fault_start or ts

            # ---------------------------------------------------------- state changes
            for svc_id, svc in gh.services.items():
                cur = getattr(svc, "health_state", "healthy")
                prev = prev_state.get(svc_id, "healthy")

                if cur in {"warning", "degraded", "critical"} and prev == "healthy":
                    # detection event
                    detections.append({
                        "service": svc_id,
                        "event": "detected",
                        "timestamp": ts,
                        "fault_active": bool(fault_start),
                    })
                elif cur == "healthy" and prev in {"warning", "degraded", "critical"}:
                    detections.append({
                        "service": svc_id,
                        "event": "recovered",
                        "timestamp": ts,
                        "fault_active": bool(fault_start),
                    })

                prev_state[svc_id] = cur

            # Detection latency for whole scenario ---------------------
            if fault_start and detect_time is None and any(v["event"]=="detected" for v in detections if v["timestamp"]==ts):
                detect_time = ts

            # False positives when no fault yet active ------------------
            if not fault_start and any(v["event"]=="detected" for v in detections if v["timestamp"]==ts):
                false_positives += 1

    # --------------------------------------------------------------------------
    if fault_start and detect_time:
        _LOG.info("Detection latency: %.2f s", detect_time - fault_start)
    if false_positives:
        _LOG.info("False positives before fault: %d", false_positives)
    _LOG.info("Replay finished. Total anomalies recorded: %d", sum(len(v) for v in gh.propagation_history.values()))

    # ------------------------------------------------------------------ write CSV
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"replay_results_{int(time.time())}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["service", "event", "timestamp", "fault_active"])
        writer.writeheader()
        writer.writerows(detections)
    _LOG.info("Detailed detection/recovery events written to %s", out_path)


if __name__ == "__main__":
    main() 