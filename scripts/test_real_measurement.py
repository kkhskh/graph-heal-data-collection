#!/usr/bin/env python3
"""
Test script for the real measurement infrastructure.

This script tests the enhanced monitoring system and real experiment runner
to ensure they work correctly before running full experiments.
"""

import time
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_real_measurement')

def test_enhanced_monitor():
    """Test the enhanced monitoring system."""
    logger.info("Testing enhanced monitoring system...")
    
    try:
        # Import the enhanced monitor class directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_monitoring", 
            project_root / "scripts" / "run_monitoring.py"
        )
        run_monitoring = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_monitoring)
        
        # Create monitor instance
        monitor = run_monitoring.EnhancedMonitor()
        
        # Test metric parsing
        test_metrics = """
# HELP service_cpu_usage CPU usage percentage
# TYPE service_cpu_usage gauge
service_cpu_usage 45.2
service_memory_usage 67.8
service_response_time 0.15
"""
        
        parsed_metrics = monitor.parse_metrics(test_metrics)
        logger.info(f"Parsed metrics: {parsed_metrics}")
        
        # Test anomaly detection
        anomaly_data = monitor.detect_anomaly("service_a", parsed_metrics)
        logger.info(f"Anomaly detection result: {anomaly_data}")
        
        # Test health status determination
        health_status = monitor.determine_health_status(parsed_metrics)
        logger.info(f"Health status: {health_status}")
        
        # Test event logging
        monitor.log_detection_event("service_a", anomaly_data)
        monitor.log_localization_event("service_a", health_status, parsed_metrics)
        monitor.log_recovery_event("service_a", "restart", True)
        
        # Check event log
        events = monitor.get_event_stream()
        logger.info(f"Total events logged: {len(events)}")
        
        # Test event log saving
        log_file = monitor.save_event_log()
        logger.info(f"Event log saved to: {log_file}")
        
        logger.info("✅ Enhanced monitor test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_experiment_runner():
    """Test the real experiment runner."""
    logger.info("Testing real experiment runner...")
    
    try:
        # Import the real experiment runner class directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_real_controlled_experiment", 
            project_root / "scripts" / "run_real_controlled_experiment.py"
        )
        run_real_experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_real_experiment)
        
        # Create experiment runner instance
        experiment = run_real_experiment.RealControlledExperiment()
        
        # Test initialization
        logger.info(f"Experiment runner initialized with {len(experiment.services)} services")
        
        # Test metric parsing
        test_metrics_text = "service_cpu_usage 45.2\nservice_memory_usage 67.8"
        parsed_metrics = experiment._parse_metrics(test_metrics_text)
        logger.info(f"Metric parsing test: {parsed_metrics}")
        
        # Test anomaly detection
        anomaly_detected = experiment._detect_anomaly(parsed_metrics)
        logger.info(f"Anomaly detection test: {anomaly_detected}")
        
        # Test health status determination
        health_status = experiment._get_health_status(parsed_metrics)
        logger.info(f"Health status test: {health_status}")
        
        # Test event logging
        experiment._log_detection_event("service_a", parsed_metrics)
        experiment._log_localization_event("service_a", "degraded")
        
        logger.info(f"Detection events: {len(experiment.detected_anomalies)}")
        logger.info(f"Localization events: {len(experiment.localized_faults)}")
        
        logger.info("✅ Real experiment runner test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Real experiment runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_availability():
    """Test if services are available for real experiments."""
    logger.info("Testing service availability...")
    
    import requests
    
    services = {
        "service_a": "http://localhost:5001",
        "service_b": "http://localhost:5002",
        "service_c": "http://localhost:5003",
        "service_d": "http://localhost:5004"
    }
    
    available_services = []
    unavailable_services = []
    
    for service_name, url in services.items():
        try:
            # Try to get health endpoint
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                available_services.append(service_name)
                logger.info(f"✅ {service_name} is available")
            else:
                unavailable_services.append(service_name)
                logger.warning(f"⚠️ {service_name} returned status {resp.status_code}")
        except requests.RequestException:
            unavailable_services.append(service_name)
            logger.error(f"❌ {service_name} is not available")
    
    logger.info(f"Available services: {available_services}")
    logger.info(f"Unavailable services: {unavailable_services}")
    
    if len(available_services) >= 2:
        logger.info("✅ Sufficient services available for testing")
        return True
    else:
        logger.warning("⚠️ Insufficient services available for full testing")
        logger.info("💡 To start services, run: docker-compose up -d")
        return False

def test_evaluation_framework():
    """Test the evaluation framework integration."""
    logger.info("Testing evaluation framework...")
    
    try:
        from graph_heal.evaluation import Evaluator
        
        # Create evaluator instance
        evaluator = Evaluator()
        
        # Test test lifecycle
        evaluator.start_test("test_scenario", {"service": "service_a", "duration": 20})
        
        # Test event logging
        evaluator.log_event("test_event", {"message": "Test event"})
        
        # Test metric addition
        evaluator.add_metric("test_metric", 42.0)
        
        # Test evaluation methods
        test_faults = [{"target": "service_a", "timestamp": time.time()}]
        test_anomalies = [{"services": ["service_a"], "timestamp": time.time() + 5}]
        
        evaluator.evaluate_detection(test_faults, test_anomalies)
        evaluator.evaluate_localization(test_faults, [{"service": "service_a"}])
        
        # End test
        evaluator.end_test()
        
        logger.info("✅ Evaluation framework test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the real measurement infrastructure."""
    logger.info("🧪 Testing Real Measurement Infrastructure")
    logger.info("=" * 50)
    
    tests = [
        ("Enhanced Monitor", test_enhanced_monitor),
        ("Real Experiment Runner", test_real_experiment_runner),
        ("Service Availability", test_service_availability),
        ("Evaluation Framework", test_evaluation_framework)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Real measurement infrastructure is ready.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 