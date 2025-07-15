import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

class ExperimentalDataSplitter:
    def __init__(self, base_dir: str = 'results'):
        self.base_dir = Path(base_dir)
        self.splits_dir = self.base_dir / 'splits'
        self.raw_data_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        
        # Create necessary directories
        for dir_path in [self.splits_dir, self.raw_data_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_experimental_splits(self) -> Dict[str, List[str]]:
        """Create train/validation/test splits for experiments."""
        # Define experiment groups with 40/20/40 distribution
        experiments = {
            'cpu': {
                'train': ['cpu_experiment_1.json', 'cpu_experiment_2.json'],  # 40%
                'validation': ['cpu_experiment_3.json'],                       # 20%
                'test': ['cpu_experiment_4.json', 'cpu_experiment_5.json']    # 40%
            },
            'memory': {
                'train': ['memory_experiment_1.json', 'memory_experiment_2.json'],  # 40%
                'validation': ['memory_experiment_3.json'],                         # 20%
                'test': ['memory_experiment_4.json', 'memory_experiment_5.json']    # 40%
            },
            'network': {
                'train': ['network_experiment_1.json', 'network_experiment_2.json'],  # 40%
                'validation': ['network_experiment_3.json'],                           # 20%
                'test': ['network_experiment_4.json', 'network_experiment_5.json']    # 40%
            }
        }
        
        # Create split metadata
        splits_metadata = {
            'created_at': datetime.now().isoformat(),
            'splits': {
                'train': [],
                'validation': [],
                'test': []
            },
            'experiment_groups': experiments,
            'distribution': {
                'train': 0.4,      # 40%
                'validation': 0.2,  # 20%
                'test': 0.4        # 40%
            }
        }
        
        # Save split metadata
        with open(self.splits_dir / 'split_metadata.json', 'w') as f:
            json.dump(splits_metadata, f, indent=2)
        
        return experiments
    
    def validate_data_quality(self, data: Dict) -> Tuple[bool, List[str]]:
        """Validate the quality of experimental data."""
        issues = []
        
        # Check required fields
        required_fields = ['metrics', 'timestamps', 'fault_periods']
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
        
        # Validate metrics
        if 'metrics' in data:
            if not isinstance(data['metrics'], list):
                issues.append("Metrics must be a list")
            else:
                # Check metric consistency
                metric_keys = set()
                for entry in data['metrics']:
                    if not isinstance(entry, dict):
                        issues.append("Each metric entry must be a dictionary")
                        break
                    metric_keys.update(entry.keys())
                
                if len(metric_keys) < 3:  # At least 3 different metrics
                    issues.append("Insufficient metric variety")
        
        # Validate timestamps
        if 'timestamps' in data:
            if not isinstance(data['timestamps'], list):
                issues.append("Timestamps must be a list")
            else:
                try:
                    [datetime.fromisoformat(ts) for ts in data['timestamps']]
                except ValueError:
                    issues.append("Invalid timestamp format")
        
        return len(issues) == 0, issues
    
    def process_and_split_data(self):
        """Process raw data and create train/validation/test splits."""
        # Get experiment splits
        experiments = self.create_experimental_splits()
        
        # Process each experiment group
        for group, splits in experiments.items():
            for split_name, files in splits.items():
                for file_name in files:
                    # Read raw data
                    raw_file = self.raw_data_dir / file_name
                    if not raw_file.exists():
                        print(f"Warning: Raw data file {file_name} not found")
                        continue
                    
                    with open(raw_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate data
                    is_valid, issues = self.validate_data_quality(data)
                    if not is_valid:
                        print(f"Data quality issues in {file_name}:")
                        for issue in issues:
                            print(f"  - {issue}")
                        continue
                    
                    # Process and save data
                    processed_file = self.processed_dir / f"{group}_{split_name}_{file_name}"
                    with open(processed_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"Processed {file_name} -> {processed_file}")

def main():
    # Create splitter instance
    splitter = ExperimentalDataSplitter()
    
    # Process and split data
    splitter.process_and_split_data()
    
    print("\nData splitting complete!")
    print("Check the following directories:")
    print("  - results/splits/     : Split metadata and configuration")
    print("  - results/raw/        : Raw experimental data")
    print("  - results/processed/  : Processed and validated data")

if __name__ == '__main__':
    main() 