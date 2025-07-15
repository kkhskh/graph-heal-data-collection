#!/usr/bin/env python3

from __future__ import annotations  # ensures PEP 563 postponed evaluation of annotations on Py<3.10

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import numpy as np
from collections import defaultdict

class ServiceNode:
    """Represents a service in the graph with its metrics and state"""
    def __init__(self, service_id: str):
        self.service_id = service_id
        self.metrics = {}
        self.health_state = 'healthy'
        self.dependencies = []
        self.dependents = []
        self.layer = 'application'  # application, container, host, network
        self.communication_patterns = defaultdict(int)
        self.circuit_breaker_state = 'closed'
        self.isolation_state = False

class GraphHeal:
    """Main GRAPH-HEAL implementation with all advanced features"""
    
    def __init__(self):
        self.services: Dict[str, ServiceNode] = {}
        self.layers = {
            'application': [],
            'container': [],
            'host': [],
            'network': []
        }
        self.logger = logging.getLogger(__name__)
        
        # Fault detection parameters
        self.anomaly_thresholds = {
            'cpu_usage': 80,
            'memory_usage': 80,
            'latency': 1000,
            'error_rate': 5.0
        }
        
        # Recovery parameters
        self.recovery_strategies = {
            'isolation': {
                'threshold': 0.8,
                'timeout': 30
            },
            'circuit_breaker': {
                'error_threshold': 5,
                'timeout': 60
            },
            'load_balancing': {
                'cpu_threshold': 70,
                'memory_threshold': 70
            }
        }
        
        # Propagation tracking
        self.propagation_history = defaultdict(list)
        self.fault_patterns = defaultdict(int)
        
    def add_service(self, service_id: str, layer: str = 'application', dependencies: List[str] | None = None, **_ignored):
        """Add *service_id* to the graph.

        Extra keyword arguments are accepted for API compatibility but
        silently ignored because the simplified implementation does not
        require them.  This prevents TypeError in unit-tests that call
        ``add_service(..., dependencies=[...])``.
        """
        if service_id not in self.services:
            self.services[service_id] = ServiceNode(service_id)
            self.services[service_id].layer = layer
            self.layers[layer].append(service_id)
        
        if dependencies:
            self.services[service_id].dependencies = dependencies
            for dep in dependencies:
                if dep in self.services:
                    self.services[dep].dependents.append(service_id)
    
    def update_metrics(self, service_id: str, metrics: Dict[str, float]):
        """Update service metrics and detect anomalies"""
        if service_id not in self.services:
            return
        
        service = self.services[service_id]
        service.metrics.update(metrics)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(service)
        if anomalies:
            self._handle_anomalies(service_id, anomalies)
    
        # ------------------------------------------------------------------
        # NEW: immediately re-evaluate each downstream dependent so that any
        #       dependency anomaly is detected *during* the same call that
        #       mutated the upstream node.  This guarantees that a demo like
        #       Scenario B (two aggregators spiking) shows ReactorControl's
        #       degraded/warning state without an extra manual poke.
        # ------------------------------------------------------------------
        for child_id in getattr(service, "dependents", []):
            child = self.services.get(child_id)
            if not child:
                continue

            child_anomalies = self._detect_anomalies(child)
            if child_anomalies:
                self._handle_anomalies(child_id, child_anomalies)
    
    def _detect_anomalies(self, service: ServiceNode) -> List[Dict[str, Any]]:
        """Detect anomalies using graph-based analysis"""
        anomalies = []
        
        # Check direct metrics
        for metric, threshold in self.anomaly_thresholds.items():
            if metric in service.metrics and service.metrics[metric] > threshold:
                anomalies.append({
                    'type': 'metric_anomaly',
                    'metric': metric,
                    'value': service.metrics[metric],
                    'threshold': threshold,
                    'timestamp': datetime.now().timestamp()
                })
        
        # Check dependency health
        for dep_id in service.dependencies:
            if dep_id in self.services:
                dep = self.services[dep_id]
                if dep.health_state in ['degraded', 'warning', 'critical']:
                    anomalies.append({
                        'type': 'dependency_anomaly',
                        'dependency': dep_id,
                        'health_state': dep.health_state
                    })
        
        # Check communication patterns
        for pattern, count in service.communication_patterns.items():
            if count > 1000:  # Example threshold
                anomalies.append({
                    'type': 'communication_anomaly',
                    'pattern': pattern,
                    'count': count
                })
        
        return anomalies
    
    def _handle_anomalies(self, service_id: str, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies with appropriate recovery actions"""
        service = self.services[service_id]
        
        # Track propagation
        self._track_propagation(service_id, anomalies)
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(service_id, anomalies)
        
        # Execute recovery actions
        if strategy:
            # Escalate health_state before executing recovery so dependents can
            # see the degraded status in subsequent anomaly checks.
            svc = self.services[service_id]
            if any(a['type'] == 'metric_anomaly' for a in anomalies):
                # Temperature / pressure spikes → warning; severe latency / resource → degraded.
                highest_severity = 'warning'
                for a in anomalies:
                    if a['type'] == 'metric_anomaly' and a.get('metric') in {
                        'control_loop_latency', 'cpu_usage', 'memory_usage'
                    }:
                        highest_severity = 'degraded'
                        break
                svc.health_state = highest_severity

            self._execute_recovery(service_id, strategy)
    
    def _track_propagation(self, service_id: str, anomalies: List[Dict[str, Any]]):
        """Track fault propagation through the service graph"""
        timestamp = datetime.now().timestamp()
        
        for anomaly in anomalies:
            self.propagation_history[service_id].append({
                'timestamp': timestamp,
                'anomaly': anomaly,
                'affected_services': self._get_affected_services(service_id)
            })
            
            # Update fault patterns
            pattern = self._identify_fault_pattern(service_id, anomaly)
            self.fault_patterns[pattern] += 1
    
    def _get_affected_services(self, service_id: str) -> List[str]:
        """Get all services that could be affected by a fault"""
        affected = set()
        to_process = [service_id]
        
        while to_process:
            current = to_process.pop(0)
            if current in self.services:
                affected.add(current)
                to_process.extend(self.services[current].dependents)
        
        return list(affected)
    
    def _identify_fault_pattern(self, service_id: str, anomaly: Dict[str, Any]) -> str:
        """Identify the type of fault pattern"""
        if anomaly['type'] == 'dependency_anomaly':
            return 'cascading_failure'
        elif anomaly['type'] == 'communication_anomaly':
            return 'communication_failure'
        else:
            return 'single_service_failure'
    
    def _determine_recovery_strategy(self, service_id: str, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the best recovery strategy based on anomalies"""
        service = self.services[service_id]
        strategy = {
            'actions': [],
            'priority': 'high' if any(a['type'] == 'dependency_anomaly' for a in anomalies) else 'medium'
        }
        
        # Check for cascading failures
        if any(a['type'] == 'dependency_anomaly' for a in anomalies):
            strategy['actions'].extend([
                {'type': 'isolation', 'target': service_id},
                {'type': 'circuit_breaker', 'target': service_id},
                {'type': 'load_balancing', 'target': service_id}
            ])
        
        # Check for resource issues
        if any(a['type'] == 'metric_anomaly' and a['metric'] in ['cpu_usage', 'memory_usage'] for a in anomalies):
            strategy['actions'].append({
                'type': 'resource_scaling',
                'target': service_id,
                'parameters': {'scale_factor': 1.5}
            })
        
        # Check for communication issues
        if any(a['type'] == 'communication_anomaly' for a in anomalies):
            strategy['actions'].append({
                'type': 'circuit_breaker',
                'target': service_id,
                'parameters': {'timeout': 30}
            })
        
        # Domain-specific metric spikes ------------------------------------
        for a in anomalies:
            if a['type'] == 'metric_anomaly':
                metric = a['metric']
                if metric in {'temperature', 'pressure'}:
                    # Quarantine / circuit-break sensor aggregator
                    strategy['actions'].append({'type': 'circuit_breaker', 'target': service_id})
                    strategy['actions'].append({'type': 'isolation', 'target': service_id})
                elif metric in {'control_loop_latency', 'pump_response_time'}:
                    # High latency → redistribute load / spawn standby controller
                    strategy['actions'].append({'type': 'load_balancing', 'target': service_id})
        
        return strategy
    
    def _execute_recovery(self, service_id: str, strategy: Dict[str, Any]):
        """Execute recovery actions"""
        service = self.services[service_id]
        
        for action in strategy['actions']:
            if action['type'] == 'isolation':
                service.isolation_state = True
            elif action['type'] == 'circuit_breaker':
                service.circuit_breaker_state = 'open'
            elif action['type'] == 'load_balancing':
                self._redistribute_load(service_id)
            elif action['type'] == 'resource_scaling':
                self._scale_resources(service_id, action['parameters'])

            # ------------------------------------------------------------------
            # NEW: If the user injected a domain-specific *recovery_adapter*
            #       (e.g. an OPC-UA client wrapper), delegate the action so
            #       that real PLC/DCS commands are issued.  We support two
            #       conventions:
            #       1) Action dict contains an explicit ``adapter_method`` key.
            #       2) Otherwise we look for a method whose name matches the
            #          ``type`` field (e.g. action type = "isolate_valve" →
            #          adapter.isolate_valve()).
            # ------------------------------------------------------------------
            if hasattr(self, "recovery_adapter"):
                adapter = getattr(self, "recovery_adapter")
                method_name = action.get("adapter_method", action["type"])
                if hasattr(adapter, method_name):
                    try:
                        getattr(adapter, method_name)(action.get("target"))
                    except Exception:  # noqa: BLE001 – adapter should not crash GH
                        self.logger.exception("Recovery adapter call %s(%s) failed", method_name, action.get("target"))
    
    def _redistribute_load(self, service_id: str):
        """Redistribute load from a service to its alternatives"""
        service = self.services[service_id]
        alternatives = self._find_alternative_services(service_id)
        
        if alternatives:
            # Update communication patterns to use alternatives
            for alt in alternatives:
                service.communication_patterns[alt] += 1
    
    def _find_alternative_services(self, service_id: str) -> List[str]:
        """Find alternative services that can handle the load"""
        alternatives = []
        service = self.services[service_id]
        
        # Look for services in the same layer with similar capabilities
        for other_id in self.layers[service.layer]:
            if other_id != service_id and self.services[other_id].health_state == 'healthy':
                alternatives.append(other_id)
        
        return alternatives
    
    def _scale_resources(self, service_id: str, parameters: Dict[str, Any]):
        """Scale service resources"""
        service = self.services[service_id]
        scale_factor = parameters.get('scale_factor', 1.5)
        
        # Update metrics with scaled resources
        if 'cpu_usage' in service.metrics:
            service.metrics['cpu_usage'] /= scale_factor
        if 'memory_usage' in service.metrics:
            service.metrics['memory_usage'] /= scale_factor
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        summary = {
            'services': {},
            'layers': {},
            'fault_patterns': dict(self.fault_patterns),
            'propagation_stats': {}
        }
        
        # Service health
        for service_id, service in self.services.items():
            summary['services'][service_id] = {
                'health_state': service.health_state,
                'metrics': service.metrics,
                'isolation_state': service.isolation_state,
                'circuit_breaker_state': service.circuit_breaker_state
            }
        
        # Layer health
        for layer, services in self.layers.items():
            layer_health = {
                'total_services': len(services),
                'healthy_services': sum(1 for s in services if self.services[s].health_state == 'healthy'),
                'degraded_services': sum(1 for s in services if self.services[s].health_state == 'degraded'),
                'warning_services': sum(1 for s in services if self.services[s].health_state == 'warning'),
                'critical_services': sum(1 for s in services if self.services[s].health_state == 'critical')
            }
            summary['layers'][layer] = layer_health
        
        # Propagation statistics
        for service_id, history in self.propagation_history.items():
            summary['propagation_stats'][service_id] = {
                'total_anomalies': len(history),
                'affected_services': len(set(s for h in history for s in h['affected_services'])),
                'latest_anomaly': history[-1] if history else None
            }
        
        return summary 