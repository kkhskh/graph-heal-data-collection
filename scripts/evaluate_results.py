#!/usr/bin/env python3
"""Compare Graph-Heal vs. baseline detector on a replay trace.

Generates:
    results/comparison/
        detection_latency_cdf.png
        false_alarms.png
        recovery_time_cdf.png
        pr_curve.png
        summary.json

Run, for example:
    python scripts/evaluate_results.py \
        --trace data/simulation/reactor_trace.csv \
        --graphheal results/replay_results_1750278893.csv \
        --baseline results/baseline_results_1750279019.csv
"""
from __future__ import annotations

import argparse, pathlib, json, logging, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
_LOG = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_fault_periods(trace: pd.DataFrame) -> list[tuple[float, float]]:
    """Return list of (start, end) timestamps where *fault_type* != none."""
    periods: list[tuple[float, float]] = []
    in_fault = False
    start = 0.0
    for ts, tag in zip(trace["timestamp"], trace.get("fault_type", "")):
        active = isinstance(tag, str) and tag.lower() not in {"", "none"}
        if active and not in_fault:
            in_fault = True
            start = ts
        elif not active and in_fault:
            in_fault = False
            periods.append((start, ts))
    if in_fault:
        periods.append((start, float(trace["timestamp"].iat[-1])))
    return periods


def _detection_latency(detections: pd.Series, periods: list[tuple[float, float]]):
    latencies: list[float] = []
    for s, _ in periods:
        det_times = detections[detections >= s]
        if not det_times.empty:
            latencies.append(det_times.iloc[0] - s)
    return latencies


def _false_positives(detections: pd.Series, periods: list[tuple[float, float]]):
    fp = 0
    for t in detections:
        if not any(s <= t <= e for s, e in periods):
            fp += 1
    return fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--graphheal", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", default="results/comparison")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(args.trace)
    periods = _extract_fault_periods(trace)
    if not periods:
        _LOG.warning("Trace has no fault periods – metrics will be empty")

    gh = pd.read_csv(args.graphheal)
    base = pd.read_csv(args.baseline)

    # Detection timestamps --------------------------------------------------
    gh_det = gh[gh["event"] == "detected"]["timestamp"].sort_values()
    base_det = base["timestamp"].sort_values()

    gh_lat = _detection_latency(gh_det, periods)
    base_lat = _detection_latency(base_det, periods)

    gh_fp = _false_positives(gh_det, periods)
    base_fp = _false_positives(base_det, periods)

    # Recovery time distribution -------------------------------------------
    rec_times: list[float] = []
    grp = gh.groupby(["service"])
    for svc, df in grp:
        det_times = df[df["event"] == "detected"]["timestamp"].reset_index(drop=True)
        rec = df[df["event"] == "recovered"]["timestamp"].reset_index(drop=True)
        n = min(len(det_times), len(rec))
        for i in range(n):
            rec_times.append(rec[i] - det_times[i])

    # Precision / recall ----------------------------------------------------
    total_fault_seconds = sum(e - s for s, e in periods)
    recall_gh = len(gh_lat) / len(periods) if periods else math.nan
    recall_base = len(base_lat) / len(periods) if periods else math.nan
    precision_gh = len(gh_lat) / (len(gh_lat) + gh_fp) if gh_lat else 0
    precision_base = len(base_lat) / (len(base_lat) + base_fp) if base_lat else 0

    # ------------------------------------------------------------------ plots
    if gh_lat:
        sns.ecdfplot(gh_lat, label="Graph-Heal")
    if base_lat:
        sns.ecdfplot(base_lat, label="Baseline")
    plt.xlabel("Detection latency (s)")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "detection_latency_cdf.png")
    plt.close()

    # False alarms bar ------------------------------------------------------
    sns.barplot(x=["Graph-Heal", "Baseline"], y=[gh_fp, base_fp])
    plt.ylabel("False positives")
    plt.tight_layout()
    plt.savefig(out_dir / "false_alarms.png")
    plt.close()

    # Recovery times --------------------------------------------------------
    if rec_times:
        sns.ecdfplot(rec_times)
        plt.xlabel("Recovery time (s)")
        plt.ylabel("CDF")
        plt.tight_layout()
        plt.savefig(out_dir / "recovery_time_cdf.png")
        plt.close()

    # PR scatter ------------------------------------------------------------
    plt.scatter(recall_gh, precision_gh, label="Graph-Heal")
    plt.scatter(recall_base, precision_base, label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png")
    plt.close()

    # Summary JSON ----------------------------------------------------------
    summary = {
        "graphheal": {"latencies": gh_lat, "false_positives": gh_fp, "precision": precision_gh, "recall": recall_gh},
        "baseline": {"latencies": base_lat, "false_positives": base_fp, "precision": precision_base, "recall": recall_base},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _LOG.info("Evaluation complete – plots and summary written to %s", out_dir)


if __name__ == "__main__":
    main() 