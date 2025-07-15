# Graph-Heal Data Collection

This repository is dedicated to running automated fault injection experiments for Graph-Heal.

## Quick Start

1. **Manual Run**: Clone this repo and run locally
   ```bash
   git clone <your-repo-url>
   cd graph-heal-data-collection
   pip install -r requirements.txt
   docker-compose up -d
   python scripts/cloud_fault_injector.py --hours 24
   ```

2. **GitHub Actions**: Use the "Fault Injection Data Collection" workflow
   - Go to Actions tab
   - Select "Fault Injection Data Collection"
   - Click "Run workflow"
   - Set hours and services as needed

## Results

After completion, download the artifacts from the GitHub Actions run:
- `fault_labels.csv` - Timestamps of all fault injections
- `cloud_fault_injection.log` - Detailed execution log
- `metric_data.csv` - Collected metrics during the experiment

## Configuration

Edit the workflow file `.github/workflows/fault-injection.yml` to customize:
- Duration of experiments
- Services to inject faults into
- Fault types and frequencies
