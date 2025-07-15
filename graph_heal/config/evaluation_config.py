#!/usr/bin/env python3

from typing import Dict, List
from datetime import timedelta

class EvaluationConfig:
    def __init__(self):
        """Initialize evaluation configuration with realistic parameters"""
        
        # Time periods
        self.periods = {
            'baseline': timedelta(minutes=1),     # 1 minute baseline
            'fault_injection': timedelta(minutes=5), # 5 minutes fault injection
            'recovery': timedelta(minutes=2),     # 2 minutes recovery
            'stability': timedelta(minutes=2)     # 2 minutes stability check
        }
        
        # Sampling intervals
        self.sampling = {
            'metrics': 1,      # Collect metrics every 1 second
            'snapshots': 10,   # Take snapshots every 10 seconds
            'evaluation': 30   # Evaluate every 30 seconds
        }
        
        # Fault injection parameters
        self.fault_injection = {
            'min_duration': 300,    # Minimum fault duration (5 minutes)
            'max_duration': 3600,   # Maximum fault duration (1 hour)
            'cooldown': 600,        # Cooldown between faults (10 minutes)
            'max_concurrent': 2     # Maximum concurrent faults
        }
        
        # Detection thresholds
        self.detection = {
            'z_score': 2.5,         # Z-score threshold
            'trend': 2.0,           # Trend threshold (% per second)
            'pattern': 0.8,         # Pattern matching threshold
            'confidence': 0.7       # Minimum confidence for detection
        }
        
        # Health state parameters
        self.health = {
            'healthy': 85,          # Healthy threshold
            'degraded': 60,         # Degraded threshold
            'warning': 30,          # Warning threshold
            'critical': 0           # Critical threshold
        }
        
        # Evaluation metrics
        self.metrics = {
            'detection_accuracy': True,
            'false_positive_rate': True,
            'detection_time': True,
            'recovery_time': True,
            'localization_accuracy': True,
            'impact_analysis': True
        }
        
        # Service dependencies
        self.dependencies = {
            'service_a': ['service_b', 'service_c'],
            'service_b': ['service_d'],
            'service_c': ['service_d'],
            'service_d': []
        }
        
        # Fault types and probabilities
        self.fault_types = {
            'cpu': 0.3,            # 30% probability
            'memory': 0.3,         # 30% probability
            'latency': 0.2,        # 20% probability
            'crash': 0.1,          # 10% probability
            'network': 0.1         # 10% probability
        }
        
        # Recovery parameters
        self.recovery = {
            'gradual': True,       # Enable gradual recovery
            'min_time': 30,        # Minimum recovery time (seconds)
            'max_time': 300,       # Maximum recovery time (seconds)
            'noise': 3.0           # Recovery noise (±3%)
        }
        
        # Cross-layer parameters
        self.cross_layer = {
            'enabled': True,       # Enable cross-layer faults
            'max_layers': 3,       # Maximum affected layers
            'correlation': 0.7     # Layer correlation factor
        }
        
        # Evaluation scenarios
        self.scenarios = [
            {
                'name': 'single_fault_cpu',
                'duration': 300,  # 5 minutes
                'fault_type': 'cpu',
                'target': 'service_a'
            },
            {
                'name': 'single_fault_memory',
                'duration': 300,  # 5 minutes
                'fault_type': 'memory',
                'target': 'service_a'
            },
            {
                'name': 'single_fault_latency',
                'duration': 300,  # 5 minutes
                'fault_type': 'latency',
                'target': 'service_a'
            },
            {
                'name': 'single_fault_network',
                'duration': 300,  # 5 minutes
                'fault_type': 'network',
                'target': 'service_a'
            },
            {
                'name': 'cascading_fault',
                'duration': 600,  # 10 minutes
                'fault_type': 'latency',
                'target': 'service_a',
                'propagation_delay': 60  # 1 minute
            },
            {
                'name': 'cross_layer_fault',
                'duration': 450,  # 7.5 minutes
                'fault_type': 'network',
                'target': 'service_b',
                'affected_layers': ['host', 'container', 'service']
            }
        ]
    
    def get_total_duration(self) -> int:
        """Calculate total evaluation duration in seconds"""
        total = sum(period.total_seconds() for period in self.periods.values())
        return int(total)
    
    def get_metrics_count(self) -> int:
        """Calculate total number of metric samples"""
        return int(self.get_total_duration() / self.sampling['metrics'])
    
    def get_snapshot_count(self) -> int:
        """Calculate total number of snapshots"""
        return int(self.get_total_duration() / self.sampling['snapshots'])
    
    def get_evaluation_count(self) -> int:
        """Calculate total number of evaluations"""
        return int(self.get_total_duration() / self.sampling['evaluation'])
    
    def get_scenario_duration(self, scenario_name: str) -> int:
        """Get duration for a specific scenario"""
        for scenario in self.scenarios:
            if scenario['name'] == scenario_name:
                return scenario['duration']
        return 0 