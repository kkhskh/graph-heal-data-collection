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
      hours:
        description: 'Hours to run'
        required: true
        default: '24'
      services:
        description: 'Services to inject faults into (space-separated)'
        required: true
        default: 'service_a service_b service_c service_d'

jobs:
  fault-injection:
    runs-on: ubuntu-latest
    timeout-minutes: 1440  # 24 hours max
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Start services
      run: |
        docker-compose up -d
        sleep 30
    
    - name: Start monitoring
      run: |
        python scripts/run_monitoring.py &
        sleep 10
    
    - name: Run fault injection
      run: |
        python scripts/cloud_fault_injector.py --hours ${{ github.event.inputs.hours }} --services ${{ github.event.inputs.services }} --label-file fault_labels.csv
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: fault-injection-results
        path: |
          fault_labels.csv
          cloud_fault_injection.log
          metric_data.csv
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