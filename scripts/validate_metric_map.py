#!/usr/bin/env python3
"""Validate every PromQL expression in config/metric_map.yaml.

It performs an instant query against the Prometheus HTTP API and prints
SUCCESS / FAILURE for each mapping entry.  Requires the *requests* package.

Example
-------
    python scripts/validate_metric_map.py --prom http://localhost:9090
"""
from __future__ import annotations

import argparse, yaml, requests, urllib.parse, itertools, pathlib, sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="config/metric_map.yaml", help="YAML mapping file")
    p.add_argument("--prom", default="http://localhost:9090", help="Prometheus base URL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = yaml.safe_load(pathlib.Path(args.map).read_text()) or {}

    fail = False
    for svc, metrics in mapping.items():
        for metric, q in metrics.items():
            url = f"{args.prom}/api/v1/query?query=" + urllib.parse.quote_plus(q)
            try:
                resp = requests.get(url, timeout=5)
                ok = resp.status_code == 200 and resp.json().get("status") == "success"
            except Exception as exc:  # noqa: BLE001
                ok = False
            tag = "✔" if ok else "✖"
            print(f"[{tag}] {svc}.{metric}: {q}")
            fail |= not ok

    if fail:
        print("\nSome queries failed – check Prometheus labels / spelling.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 