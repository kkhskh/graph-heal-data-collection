#!/usr/bin/env python3
"""Baseline Z-score detector replay for comparison with Graph-Heal.

For each metric column in the CSV it maintains a rolling window (default 30
samples) and raises an anomaly when the latest value's |z| exceeds
``z_threshold``.  Aggregated detection and false-positive statistics are
printed and detailed per-metric detection times are written to
``results/baseline_results_<ts>.csv``.

Usage
-----
    python scripts/baseline_replay.py \
        --trace data/simulation/reactor_trace.csv \
        --window 30 --z 3.0
"""
from __future__ import annotations

import argparse, csv, pathlib, time, logging
from collections import deque
from typing import Dict, Deque, List
import numpy as np

_LOG = logging.getLogger("baseline-replay")


def parse_args():
    p = argparse.ArgumentParser(description="Replay CSV into baseline z-score detector")
    p.add_argument("--trace", required=True, help="CSV file with timestamp + metrics + fault_type column")
    p.add_argument("--window", type=int, default=30, help="Rolling window size")
    p.add_argument("--z", type=float, default=3.0, help="Z-score threshold")
    p.add_argument("--rate", type=float, default=0.0, help="Replay speed-up (0 = max speed)")
    p.add_argument("--thresholds", help="YAML file with absolute metric thresholds (service.metric: value)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Per-metric history
    history: Dict[str, Deque[float]] = {}
    abs_thresholds: Dict[str, float] = {}
    detections: List[Dict] = []  # metric, timestamp, fault_active

    fault_start: float | None = None
    detect_time: float | None = None
    false_positives = 0

    if args.thresholds:
        import yaml, pathlib as _p
        abs_thresholds = yaml.safe_load(_p.Path(args.thresholds).read_text()) or {}

    with open(args.trace, newline="") as f:
        reader = csv.DictReader(f)
        metric_cols = [c for c in reader.fieldnames if c not in {"timestamp", "fault_type"}]

        prev_ts: float | None = None
        for row in reader:
            ts = float(row["timestamp"])
            if args.rate > 0 and prev_ts is not None:
                dt = (ts - prev_ts) / args.rate
                if dt > 0:
                    time.sleep(dt)
            prev_ts = ts

            # ground-truth fault tag
            fault_tag = row.get("fault_type", "").lower()
            if fault_tag and fault_tag != "none":
                fault_start = fault_start or ts

            # feed each metric into its window
            for col in metric_cols:
                val = float(row[col])
                buf = history.setdefault(col, deque(maxlen=args.window))

                # Absolute threshold check first ---------------------------------
                metric_name = col.split(".", 1)[-1]
                th_val = abs_thresholds.get(col) or abs_thresholds.get(metric_name)
                if th_val is not None and val > th_val:
                    detections.append({"metric": col, "timestamp": ts, "fault_active": bool(fault_start)})
                    if not fault_start:
                        false_positives += 1
                    if fault_start and detect_time is None:
                        detect_time = ts

                # Z-score check ---------------------------------------------------
                if len(buf) >= args.window - 1:
                    mean = np.mean(list(buf))
                    std = np.std(list(buf)) or 1e-9
                    z = abs((val - mean) / std)
                    if z > args.z:
                        detections.append({
                            "metric": col,
                            "timestamp": ts,
                            "fault_active": bool(fault_start),
                        })
                        if not fault_start:
                            false_positives += 1
                        if fault_start and detect_time is None:
                            detect_time = ts
                buf.append(val)

    if fault_start and detect_time:
        _LOG.info("Baseline detection latency: %.2f s", detect_time - fault_start)
    if false_positives:
        _LOG.info("Baseline false positives before fault: %d", false_positives)

    # write csv
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"baseline_results_{int(time.time())}.csv"
    with out.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["metric", "timestamp", "fault_active"])
        writer.writeheader()
        writer.writerows(detections)
    _LOG.info("Baseline events written to %s", out)


if __name__ == "__main__":
    main() 