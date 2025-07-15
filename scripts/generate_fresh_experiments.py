import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

class ExperimentGenerator:
    def __init__(self, base_dir: str = 'results/raw'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Base metrics for normal operation
        self.base_metrics = {
            'cpu': {'mean': 30, 'std': 5},
            'memory': {'mean': 50, 'std': 8},
            'network': {'mean': 100, 'std': 15}
        }
        
        # Fault patterns
        self.fault_patterns = {
            'cpu': {
                'gradual': lambda t: 30 + 40 * (t / 300),  # Gradual increase
                'spike': lambda t: 30 + 60 * np.exp(-(t - 150)**2 / 1000),  # CPU spike
                'oscillating': lambda t: 30 + 40 * np.sin(t / 20)  # Oscillating pattern
            },
            'memory': {
                'leak': lambda t: 200 + 2 * t,  # Memory leak
                'spike': lambda t: 200 + 300 * np.exp(-(t - 150)**2 / 1000),  # Memory spike
                'fragmentation': lambda t: 200 + 100 * np.sin(t / 30)  # Memory fragmentation
            },
            'network': {
                'latency': lambda t: 50 + 100 * (t / 300),  # Increasing latency
                'packet_loss': lambda t: 50 + 150 * np.exp(-(t - 150)**2 / 1000),  # Packet loss
                'bandwidth': lambda t: 50 + 80 * np.sin(t / 25)  # Bandwidth oscillation
            }
        }
    
    def generate_metrics(self, fault_type, params):
        """Generate metrics with specified fault pattern"""
        if fault_type == 'cpu':
            return self._generate_cpu_metrics(params)
        elif fault_type == 'memory':
            return self._generate_memory_metrics(params)
        elif fault_type == 'network':
            return self._generate_network_metrics(params)
        elif fault_type == 'combined':
            return self._generate_combined_metrics(params)
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

    def _generate_cpu_metrics(self, params):
        """Generate CPU metrics with various patterns"""
        duration = params.get('duration', 300)
        pattern = params.get('pattern', 'spike')
        intensity = params.get('intensity', 'medium')
        
        # Base metrics
        base_cpu = np.random.normal(
            self.base_metrics['cpu']['mean'],
            self.base_metrics['cpu']['std'],
            duration
        )
        
        # Apply pattern
        if pattern == 'spike':
            fault = self._generate_spike(duration, intensity)
        elif pattern == 'gradual':
            fault = self._generate_gradual_increase(duration, intensity)
        elif pattern == 'oscillating':
            fault = self._generate_oscillating(duration, intensity)
        else:
            raise ValueError(f"Unknown CPU pattern: {pattern}")
            
        return np.clip(base_cpu + fault, 0, 100)

    def _generate_memory_metrics(self, params):
        """Generate memory metrics with various patterns"""
        duration = params.get('duration', 300)
        pattern = params.get('pattern', 'leak')
        leak_rate = params.get('leak_rate', 'medium')
        
        # Base metrics
        base_memory = np.random.normal(
            self.base_metrics['memory']['mean'],
            self.base_metrics['memory']['std'],
            duration
        )
        
        # Apply pattern
        if pattern == 'leak':
            fault = self._generate_memory_leak(duration, leak_rate)
        elif pattern == 'spike':
            fault = self._generate_spike(duration, leak_rate)
        elif pattern == 'fragmentation':
            fault = self._generate_fragmentation(duration)
        else:
            raise ValueError(f"Unknown memory pattern: {pattern}")
            
        return np.clip(base_memory + fault, 0, 100)

    def _generate_network_metrics(self, params):
        """Generate network metrics with various patterns"""
        duration = params.get('duration', 300)
        pattern = params.get('pattern', 'latency')
        
        # Base metrics
        base_network = np.random.normal(
            self.base_metrics['network']['mean'],
            self.base_metrics['network']['std'],
            duration
        )
        
        # Apply pattern
        if pattern == 'latency':
            fault = self._generate_latency_spike(duration)
        elif pattern == 'packet_loss':
            fault = self._generate_packet_loss(duration)
        elif pattern == 'bandwidth':
            fault = self._generate_bandwidth_fluctuation(duration)
        elif pattern == 'intermittent':
            fault = self._generate_intermittent_faults(duration)
        elif pattern == 'escalating':
            fault = self._generate_escalating_faults(duration)
        else:
            raise ValueError(f"Unknown network pattern: {pattern}")
            
        return np.clip(base_network + fault, 0, 1000)

    def _generate_combined_metrics(self, params):
        """Generate metrics for combined faults"""
        duration = params.get('duration', 300)
        fault_types = params.get('faults', ['cpu', 'memory'])
        
        metrics = {}
        for fault_type in fault_types:
            metrics[fault_type] = self.generate_metrics(fault_type, {'duration': duration})
            
        return metrics

    def _generate_spike(self, duration, intensity):
        """Generate a spike pattern"""
        intensity_map = {'low': 20, 'medium': 40, 'high': 60}
        spike = np.zeros(duration)
        spike[duration//2:duration//2 + 30] = intensity_map.get(intensity, 40)
        return spike

    def _generate_gradual_increase(self, duration, intensity):
        """Generate a gradual increase pattern"""
        intensity_map = {'low': 0.2, 'medium': 0.4, 'high': 0.6}
        return np.linspace(0, intensity_map.get(intensity, 0.4) * 100, duration)

    def _generate_oscillating(self, duration, intensity):
        """Generate an oscillating pattern"""
        intensity_map = {'low': 10, 'medium': 20, 'high': 30}
        t = np.linspace(0, 4*np.pi, duration)
        return intensity_map.get(intensity, 20) * np.sin(t)

    def _generate_memory_leak(self, duration, leak_rate):
        """Generate a memory leak pattern"""
        rate_map = {'slow': 0.1, 'medium': 0.2, 'gradual': 0.05}
        return np.linspace(0, rate_map.get(leak_rate, 0.2) * 100, duration)

    def _generate_fragmentation(self, duration):
        """Generate a memory fragmentation pattern"""
        t = np.linspace(0, 4*np.pi, duration)
        return 20 * np.sin(t) + 10 * np.sin(2*t)

    def _generate_latency_spike(self, duration):
        """Generate network latency spikes"""
        latency = np.zeros(duration)
        spike_times = np.random.choice(duration, size=5, replace=False)
        latency[spike_times] = np.random.uniform(200, 500, size=5)
        return latency

    def _generate_packet_loss(self, duration):
        """Generate packet loss pattern"""
        loss = np.zeros(duration)
        loss_periods = np.random.choice(duration, size=3, replace=False)
        for start in loss_periods:
            end = min(start + 20, duration)
            loss[start:end] = np.random.uniform(10, 30, size=end-start)
        return loss

    def _generate_bandwidth_fluctuation(self, duration):
        """Generate bandwidth fluctuation pattern"""
        t = np.linspace(0, 4*np.pi, duration)
        return 50 * np.sin(t) + 25 * np.sin(2*t)

    def _generate_intermittent_faults(self, duration):
        """Generate intermittent fault pattern"""
        fault = np.zeros(duration)
        fault_periods = np.random.choice(duration, size=5, replace=False)
        for start in fault_periods:
            end = min(start + 15, duration)
            fault[start:end] = np.random.uniform(30, 70, size=end-start)
        return fault

    def _generate_escalating_faults(self, duration):
        """Generate escalating fault pattern"""
        t = np.linspace(0, 1, duration)
        return 100 * t**2  # Quadratic escalation

    def generate_experiment(self,
                          experiment_name: str,
                          fault_type: str,
                          pattern: str,
                          duration: int,
                          intensity: float = 1.0) -> None:
        """Generate a complete experiment with metrics and metadata."""
        # Generate metrics
        metrics, timestamps = self.generate_metrics(
            fault_type, {'duration': duration, 'pattern': pattern, 'intensity': intensity}
        )
        
        # Create experiment data
        experiment_data = {
            'experiment_name': experiment_name,
            'fault_type': fault_type,
            'pattern': pattern,
            'duration': duration,
            'intensity': intensity,
            'timestamps': timestamps,
            'metrics': metrics,
            'fault_periods': [
                {
                    'start': timestamps[0],
                    'end': timestamps[-1],
                    'type': fault_type,
                    'pattern': pattern
                }
            ]
        }
        
        # Save experiment data
        output_file = self.base_dir / f"{experiment_name}.json"
        with open(output_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Generated experiment: {experiment_name}")

def create_new_fault_scenarios():
    """Create new fault scenarios for testing"""
    generator = ExperimentGenerator()
    scenarios = [
        # Different intensities
        {'type': 'cpu', 'pattern': 'spike', 'intensity': 'high', 'duration': 600},
        {'type': 'memory', 'pattern': 'leak', 'leak_rate': 'gradual', 'duration': 400},
        
        # Different patterns
        {'type': 'network', 'pattern': 'intermittent', 'duration': 500},
        {'type': 'network', 'pattern': 'escalating', 'duration': 300},
        
        # Combined faults
        {'type': 'combined', 'faults': ['cpu', 'memory'], 'duration': 450},
        {'type': 'combined', 'faults': ['network', 'cpu'], 'duration': 400},
        
        # Complex patterns
        {'type': 'cpu', 'pattern': 'oscillating', 'intensity': 'medium', 'duration': 550},
        {'type': 'memory', 'pattern': 'fragmentation', 'duration': 480},
        {'type': 'network', 'pattern': 'bandwidth', 'duration': 420}
    ]
    
    return scenarios

def main():
    """Generate new experiments with fresh scenarios"""
    scenarios = create_new_fault_scenarios()
    generator = ExperimentGenerator()
    
    print("Generating new test scenarios...")
    for i, scenario in enumerate(scenarios, 1):
        metrics = generator.generate_metrics(scenario['type'], scenario)
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(metrics, dict):
            metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in metrics.items()}
        elif isinstance(metrics, np.ndarray):
            metrics = metrics.tolist()
        
        # Save experiment
        experiment_data = {
            'scenario': scenario,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"results/raw/new_scenario_{i}.json"
        with open(filename, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        print(f"Generated scenario {i}: {scenario['type']} - {scenario.get('pattern', 'combined')}")

if __name__ == '__main__':
    main() 