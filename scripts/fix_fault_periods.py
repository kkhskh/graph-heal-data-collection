import os
import json
from datetime import datetime, timedelta

dir_path = 'results/processed'
for fname in os.listdir(dir_path):
    if not (fname.endswith('.json') and ('_4.json' in fname or '_5.json' in fname)):
        continue
    fpath = os.path.join(dir_path, fname)
    with open(fpath, 'r') as f:
        data = json.load(f)
    timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
    if not timestamps or 'fault_periods' not in data or not data['fault_periods']:
        print(f"Skipping {fname}: no timestamps or fault_periods")
        continue
    start = timestamps[0]
    # Find the timestamp 60 seconds after start, or the last timestamp
    end = start + timedelta(seconds=60)
    end_ts = next((ts for ts in timestamps if ts >= end), timestamps[-1])
    data['fault_periods'][0]['start'] = start.isoformat()
    data['fault_periods'][0]['end'] = end_ts.isoformat()
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {fname}: fault period {start.isoformat()} to {end_ts.isoformat()}") 