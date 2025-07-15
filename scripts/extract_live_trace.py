#!/usr/bin/env python3
"""Extract timestamp, event_type, details from run_reactor_monitor logs.

Example
-------
    python scripts/extract_live_trace.py \
        --log logs/live_run_1750388929.log \
        --out data/evaluation/live_trace.csv
"""
from __future__ import annotations

import argparse, csv, pathlib, re
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Convert run_reactor_monitor logs to CSV")
    p.add_argument("--log", required=True, help="Path to .log file produced by run_reactor_monitor.py")
    p.add_argument("--out", default="data/evaluation/live_trace.csv", help="Destination CSV path")
    return p.parse_args()


LINE_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<level>\w+):(?P<logger>[^:]+):(?P<msg>.*)$")


def main() -> None:
    args = parse_args()
    in_path = pathlib.Path(args.log)
    if not in_path.exists():
        raise SystemExit(f"Log file {in_path} not found")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    with in_path.open() as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            ts_str = m.group("ts")
            msg = m.group("msg")
            # First word(s) up to colon or dash as event_type
            evt = re.split(r"[:-]", msg, maxsplit=1)[0].strip()
            records.append({"timestamp": ts_str, "event_type": evt, "details": msg})

    # Write CSV ------------------------------------------------------------
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "event_type", "details"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {out_path} with {len(records)} rows")


if __name__ == "__main__":
    main() 