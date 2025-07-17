#!/usr/bin/env python3
"""
Script to set up a separate repository for Graph-Heal data collection experiments.
This creates a minimal version with only the necessary components for fault injection.
"""

import os
import shutil
import subprocess
from pathlib import Path

def create_data_collection_repo():
    """Create a minimal repository for data collection."""
    
    # Create the new directory
    repo_name = "graph-heal-data-collection"
    if os.path.exists(repo_name):
        print(f"Directory {repo_name} already exists. Please remove it first.")
        return
    
    os.makedirs(repo_name)
    os.chdir(repo_name)
    
    print(f"Creating data collection repository: {repo_name}")
    
    # Copy essential files
    essential_files = [
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile",
        "README.md"
    ]
    
    for file in essential_files:
        if os.path.exists(f"../{file}"):
            shutil.copy2(f"../{file}", file)
            print(f"Copied {file}")
    
    # Copy essential directories
    essential_dirs = [
        "services",
        "scripts",
        "graph_heal"
    ]
    
    for dir_name in essential_dirs:
        if os.path.exists(f"../{dir_name}"):
            shutil.copytree(f"../{dir_name}", dir_name)
            print(f"Copied {dir_name}/")
    
    # Create GitHub Actions workflow
    os.makedirs(".github/workflows", exist_ok=True)
    
    workflow_content = '''name: Fault Injection Data Collection

on:
  workflow_dispatch:
    inputs:
      num_experiments_per_job:
        description: 'Number of experiments per fault type per job'
        required: true
        default: '25'
      services:
        description: 'Services to inject faults into (space-separated)'
        required: true
        default: 'service_a service_b service_c service_d'

jobs:
  fault-injection:
    runs-on: ubuntu-latest
    timeout-minutes: 350
    strategy:
      fail-fast: false
      matrix:
        job_index: [0, 1, 2, 3]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Install Docker Compose
      run: |
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose

    - name: Build all services (force no-cache)
      run: |
        docker-compose build --no-cache

    - name: Start services
      run: |
        docker-compose up -d
        sleep 30

    - name: Show container status
      run: |
        docker-compose ps
        docker-compose logs service-a || true
        docker-compose logs service-b || true
        docker-compose logs service-c || true
        docker-compose logs service-d || true

    - name: Wait for all services to be healthy
      run: |
        for port in 5001 5002 5003 5004; do
          echo "Waiting for service on port $port to be ready..."
          until curl -sf http://localhost:$port/metrics; do sleep 2; done
          echo "Service on port $port is ready."
        done
        echo "All services are healthy."

    - name: Start monitoring
      run: |
        python scripts/run_monitoring.py &
        sleep 10

    - name: Run fault injection batch
      run: |
        START_ID=$(( ${{ matrix.job_index }} * ${{ github.event.inputs.num_experiments_per_job }} ))
        END_ID=$(( (${{ matrix.job_index }} + 1) * ${{ github.event.inputs.num_experiments_per_job }} ))
        echo "Running experiments from $START_ID to $END_ID"
        python scripts/run_experiments.py --start-id $START_ID --end-id $END_ID

    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: fault-injection-results-${{ matrix.job_index }}
        path: |
          results/
        retention-days: 30

    - name: Clean up
      if: always()
      run: |
        docker-compose down
        docker system prune -f
'''
    
    with open(".github/workflows/fault-injection.yml", "w") as f:
        f.write(workflow_content)
    
    # Create a minimal README for the data collection repo
    readme_content = '''# Graph-Heal Data Collection

This repository is dedicated to running automated fault injection experiments for Graph-Heal.

## Quick Start

1. **Manual Run**: Clone this repo and run locally
```bash
git clone <your-repo-url>
cd graph-heal-data-collection
pip install -r requirements.txt
docker-compose up -d

# Run experiments (example with 10 experiments per job)
python scripts/run_experiments.py --num-experiments 10 --services "service_a service_b service_c service_d" --duration 300
```

2. **GitHub Actions**: Use the "Fault Injection Data Collection" workflow
   * Go to Actions tab
   * Select "Fault Injection Data Collection"
   * Click "Run workflow"
   * Set number of experiments per job (default: 25)
   * Specify services to test (default: all services)

## Results

After completion, download the artifacts from the GitHub Actions run. For parallel jobs, there will be multiple artifacts:

* `fault-injection-results-0/` through `fault-injection-results-3/` - Results from each parallel job
* Each results directory contains:
  * `processed/` - JSON files with detailed experiment results and metrics
  * `fault_labels.csv` - Timestamps and details of all fault injections

## Configuration

Edit the workflow file `.github/workflows/fault-injection.yml` to customize:
* Number of experiments per job
* Services to inject faults into
* Fault types (CPU, memory, network)
* Job timeout (default: 350 minutes)

## Architecture

The system runs experiments in parallel jobs to stay within GitHub Actions time limits:
* Each job handles a subset of the total experiments
* Default: 4 parallel jobs, each running up to 25 experiments per fault type
* Each experiment tests all specified services with all fault types
* Results are collected and stored separately for each job
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    # Create .gitignore
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Logs
*.log
logs/

# Data files
*.csv
*.json
*.pkl
*.joblib

# Docker
.dockerignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
'''
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("\n✅ Data collection repository created successfully!")
    print(f"\nNext steps:")
    print(f"1. cd {repo_name}")
    print(f"2. git init")
    print(f"3. git add .")
    print(f"4. git commit -m 'Initial commit for data collection'")
    print(f"5. Create a new repository on GitHub")
    print(f"6. git remote add origin <your-new-repo-url>")
    print(f"7. git push -u origin main")
    print(f"\nThen you can run the GitHub Actions workflow for automated data collection!")

if __name__ == "__main__":
    create_data_collection_repo() 
