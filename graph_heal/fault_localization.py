import numpy as np
import pandas as pd
import time
import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque, defaultdict
import json
import os
import datetime
from graph_heal.service_graph import ServiceGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fault_localization')

class FaultLocalizer:
    """
    Base class for fault localization.
    """
    def __init__(self, data_dir: str = "data/faults"):
        """
        Initialize the fault localizer.
        
        Args:
            data_dir: Directory to store fault data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.faults: List[Dict[str, Any]] = []
        self.active_faults: Dict[str, Dict[str, Any]] = {}
    
    def localize_faults(self, service_statuses: Dict[str, Dict[str, Any]], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Localize faults based on service statuses and detected anomalies.
        
        Args:
            service_statuses: Current status of all services
            anomalies: List of detected anomalies
        
        Returns:
            List of localized faults
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def log_fault(self, fault: Dict[str, Any]):
        """
        Log a fault to the data directory.
        
        Args:
            fault: Fault information to log
        """
        try:
            timestamp = int(time.time())
            filename = f"fault_{fault['id']}_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(fault, f, indent=2)
                
            logger.info(f"Logged fault to {filepath}")
        except Exception as e:
            logger.error(f"Failed to log fault: {e}")

    def get_active_faults(self) -> List[Dict[str, Any]]:
        """
        Get currently active faults.
        
        Returns:
            List of active faults
        """
        return list(self.active_faults.values())
    
    def get_all_faults(self) -> List[Dict[str, Any]]:
        """
        Get all detected faults.
        
        Returns:
            List of all faults
        """
        return self.faults

class GraphBasedFaultLocalizer:
    def __init__(self, service_graph: ServiceGraph, patterns_config: Optional[Dict] = None):
        self.service_graph = service_graph
        # The patterns config is no longer needed for the new logic
        self.fault_patterns = {}

    def localize_faults(self, service_statuses: Dict, anomalies: List[Dict]) -> List[Dict]:
        """
        Localizes the fault by identifying the anomalous service that is the root
        of the dependency chain among all anomalous services.
        """
        if not anomalies:
            return []

        graph = self.service_graph.graph
        anomalous_services = {anomaly['service_id'] for anomaly in anomalies}

        if not anomalous_services:
            return []

        candidate_scores = {}

        # Score each anomalous service by the number of its anomalous dependencies.
        # A root cause should have zero anomalous dependencies.
        for candidate_service in anomalous_services:
            if candidate_service not in graph:
                candidate_scores[candidate_service] = float('inf')  # Penalize services not in graph
                continue

            descendants = nx.descendants(graph, candidate_service)
            anomalous_descendants = anomalous_services.intersection(descendants)
            candidate_scores[candidate_service] = len(anomalous_descendants)

        if not candidate_scores:
            logger.warning("Fault localization could not produce candidate scores.")
            return []

        # Find the minimum score (ideally 0)
        min_score = min(candidate_scores.values())

        # Identify all services that have this minimum score
        root_causes = [service for service, score in candidate_scores.items() if score == min_score]

        # If there's a tie, use a tie-breaker
        if len(root_causes) > 1:
            logger.info(f"Multiple root causes found with score {min_score}, applying tie-breaker: {root_causes}")
            tie_breaker_scores = {service: len(nx.ancestors(graph, service)) for service in root_causes}
            logger.info(f"Tie-breaker scores (more ancestors is better): {tie_breaker_scores}")
            
            # The service with the most ancestors is likely the "deepest" in the dependency chain
            best_candidate = max(tie_breaker_scores, key=tie_breaker_scores.get)
            root_causes = [best_candidate]
            logger.info(f"Tie-breaker selected: {root_causes[0]}")

        logger.info(f"Fault localization candidate scores (lower is better): {candidate_scores}")
        logger.info(f"Identified root cause(s): {root_causes}")

        if root_causes:
            return [{'service_id': root_causes[0], 'type': 'graph_based_root_cause'}]
        else:
            logger.warning("Could not identify a root cause.")
            return []

class NaiveFaultLocalizer(FaultLocalizer):
    """
    A simple fault localizer that blames the first service with an anomaly.
    This is used for the ablation study to represent a non-topological localization approach.
    """
    def localize_faults(self, service_statuses: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Localizes the fault to the first service that reported an anomaly.
        """
        if not anomalies:
            return []

        # Find the anomaly with the earliest timestamp
        # Note: Anomaly timestamps might not be perfectly synchronized.
        # A more robust implementation might consider a time window.
        earliest_anomaly = min(anomalies, key=lambda a: a.get('timestamp', float('inf')))
        
        faulty_service_id = earliest_anomaly.get("service_id")
        if not faulty_service_id:
            return []

        fault = {
            "id": f"fault_naive_{faulty_service_id}_{int(time.time())}",
            "type": "naive_localization",
            "service_id": faulty_service_id,
            "affected_services": [faulty_service_id],
            "confidence": 1.0,
            "timestamp": time.time(),
            "description": "Fault localized to the first service exhibiting an anomaly.",
            "related_anomalies": [a["id"] for a in anomalies if a.get("service_id") == faulty_service_id]
        }
        
        self.log_fault(fault)
        logger.info(f"Naively localized fault to {faulty_service_id} based on the earliest anomaly.")
        return [fault]

class FaultManager:
    """
    Manages fault localization and tracking.
    """
    def __init__(self, localizers: List[FaultLocalizer]):
        """
        Initialize the fault manager.
        
        Args:
            localizers: List of fault localizers to use
        """
        self.localizers = localizers
        self.active_faults: Dict[str, Dict[str, Any]] = {}
        self.fault_history: List[Dict[str, Any]] = []
    
    def process_system_state(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process the current system state to detect and localize faults.
        
        Args:
            system_state: Current system state including service statuses and metrics
        
        Returns:
            List of detected faults
        """
        # Get service statuses and anomalies from system state
        service_statuses = system_state.get("service_statuses", {})
        anomalies = system_state.get("anomalies", [])
        
        # Use all localizers to detect faults
        all_faults = []
        for localizer in self.localizers:
            try:
                faults = localizer.localize_faults(service_statuses, anomalies)
                all_faults.extend(faults)
            except Exception as e:
                logger.error(f"Error in fault localizer: {e}")
        
        # Update active faults
        current_time = time.time()
        for fault in all_faults:
            fault_id = fault["id"]
            if fault_id in self.active_faults:
                # Update existing fault
                self.active_faults[fault_id].update(fault)
                self.active_faults[fault_id]["last_seen"] = current_time
            else:
                # Add new fault
                fault["first_seen"] = current_time
                fault["last_seen"] = current_time
                self.active_faults[fault_id] = fault
        
        # Remove resolved faults
        resolved_faults = []
        for fault_id, fault in list(self.active_faults.items()):
            if current_time - fault["last_seen"] > 300:  # 5 minutes
                resolved_faults.append(fault)
                del self.active_faults[fault_id]
        
        # Update fault history
        self.fault_history.extend(resolved_faults)
        
        return all_faults
    
    def localize_faults(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Localize faults using all available localizers.
        
        Args:
            system_state: Current system state
        
        Returns:
            List of localized faults
        """
        all_faults = []
        
        for localizer in self.localizers:
            try:
                faults = localizer.localize_faults(
                    system_state.get("service_statuses", {}),
                    system_state.get("anomalies", {})
                )
                all_faults.extend(faults)
            except Exception as e:
                logger.error(f"Error in fault localizer: {e}")
        
        # Update active faults
        current_time = time.time()
        for fault in all_faults:
            fault_id = fault["id"]
            if fault_id in self.active_faults:
                # Update existing fault
                self.active_faults[fault_id].update(fault)
                self.active_faults[fault_id]["last_seen"] = current_time
            else:
                # Add new fault
                fault["first_seen"] = current_time
                fault["last_seen"] = current_time
                self.active_faults[fault_id] = fault
        
        # Remove resolved faults
        resolved_faults = []
        for fault_id, fault in list(self.active_faults.items()):
            if current_time - fault["last_seen"] > 300:  # 5 minutes
                resolved_faults.append(fault)
                del self.active_faults[fault_id]
        
        # Update fault history
        self.fault_history.extend(resolved_faults)
        
        return all_faults
    
    def get_active_faults(self, max_age_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get currently active faults.
        
        Args:
            max_age_seconds: Maximum age of faults to consider active
        
        Returns:
            List of active faults
        """
        current_time = time.time()
        return [
            fault for fault in self.active_faults.values()
            if current_time - fault["last_seen"] <= max_age_seconds
        ]
    
    def get_all_faults(self) -> List[Dict[str, Any]]:
        """
        Get all faults (active and historical).
        
        Returns:
            List of all faults
        """
        return list(self.active_faults.values()) + self.fault_history