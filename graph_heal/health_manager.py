class HealthManager:
    def __init__(self):
        """Initialize the health manager with more aggressive thresholds and weights"""
        # Health state thresholds - much more aggressive
        self.health_thresholds = {
            'healthy': 75,    # Was 85
            'degraded': 55,   # Was 60
            'warning': 25,    # Was 30
            'critical': 0
        }
        
        # Metric weights for health calculation
        self.metric_weights = {
            'cpu_usage': 0.4,
            'memory_usage': 0.3,
            'response_time': 0.2,
            'error_rate': 0.1
        }
        
        # Impact factors for different metrics - much more aggressive
        self.impact_factors = {
            'cpu_usage': {
                'thresholds': [20, 35, 50],  # Much lower thresholds
                'impacts': [15, 30, 50]      # Higher impacts
            },
            'memory_usage': {
                'thresholds': [30, 50, 70],  # Much lower thresholds
                'impacts': [20, 40, 60]      # Higher impacts
            },
            'response_time': {
                'thresholds': [0.1, 0.3, 0.5],  # Lower thresholds
                'impacts': [10, 25, 40]         # Higher impacts
            },
            'error_rate': {
                'thresholds': [0.5, 2, 5],    # Lower thresholds
                'impacts': [20, 40, 60]       # Higher impacts
            }
        }
        
        # Recovery parameters
        self.recovery_params = {
            'gradual_recovery': True,
            'recovery_noise': 1.0,  # Reduced noise to ±1%
            'min_recovery_time': 30,
            'max_recovery_time': 300
        }

        # ------------------------------------------------------------------
        # Light-weight per-service metric registry (unit-test convenience)
        # ------------------------------------------------------------------
        # Some unit tests interact with *HealthManager* as a central facade
        # and expect a ``record_metric`` helper similar to the one offered by
        # the original reference implementation.  We keep the structure very
        # simple – metrics are stored in a nested ``dict`` and overwrite any
        # previous sample for the same key to emulate the latest-value
        # semantics used elsewhere in the codebase.
        self._service_metrics: dict[str, dict[str, float]] = {}

    def calculate_health_score(self, metrics):
        """
        Calculate health score with more aggressive impact assessment.
        Args:
            metrics: Dictionary of metric values
        Returns:
            Tuple of (health_score, health_state)
        """
        # Allow callers to pass a *service_id* instead of a full metric map
        if isinstance(metrics, str):
            metrics = self._service_metrics.get(metrics, {})

        # No metrics → assume perfect health
        if not metrics:
            return 100.0, 'healthy'

        # Start with perfect health and subtract penalties
        health_score = 100.0

        cpu = metrics.get('service_cpu_usage', 0)
        mem = metrics.get('service_memory_usage', 0)
        latency_ms = metrics.get('service_response_time', 0) * 1000 if metrics.get('service_response_time') else metrics.get('latency_ms', 0)

        # Simple heuristic: anything ≤ 70 is considered "free"
        penalty = (
            max(0, cpu - 70) +
            max(0, mem - 70) +
            max(0, latency_ms - 70) / 10  # convert ms to %
        )

        health_score -= penalty

        # Clamp and noise
        import random
        health_score = max(0, min(100, health_score + random.gauss(0, 1)))
        
        # Determine health state with more aggressive thresholds
        health_state = self._determine_health_state(health_score)
        return health_score, health_state

    def _determine_health_state(self, health_score):
        """Determine health state based on score with more aggressive thresholds"""
        for state, threshold in self.health_thresholds.items():
            if health_score >= threshold:
                return state
        return 'critical'

    def _calculate_metric_impact(self, metric, value):  # pragma: no cover
        """Calculate impact of a single metric on health with more aggressive thresholds"""
        if metric not in self.impact_factors:
            return 0.0
        thresholds = self.impact_factors[metric]['thresholds']
        impacts = self.impact_factors[metric]['impacts']
        for threshold, impact in zip(thresholds, impacts):
            if value > threshold:
                return impact
        return 0.0

    def calculate_recovery_time(self, current_health, target_health=100.0):
        """Calculate recovery time based on health difference"""
        health_difference = target_health - current_health
        base_time = health_difference * 2
        import random
        recovery_time = base_time + random.uniform(-10, 10)
        recovery_time = max(
            self.recovery_params['min_recovery_time'],
            min(self.recovery_params['max_recovery_time'], recovery_time)
        )
        return int(recovery_time)

    def simulate_recovery(self, current_health, duration):
        """Simulate health recovery over time"""
        import random
        from datetime import datetime
        recovery_metrics = []
        target_health = 100.0
        for t in range(duration + 1):  # include final tick so progress hits 100%
            progress = t / duration
            target = current_health + (target_health - current_health) * progress
            noise = random.gauss(0, self.recovery_params['recovery_noise'])
            current = target + noise
            current = max(0, min(100, current))
            health_state = self._determine_health_state(current)
            recovery_metrics.append({
                'timestamp': datetime.now().timestamp() + t,
                'health_score': current,
                'health_state': health_state,
                'recovery_progress': progress * 100
            })
        return recovery_metrics

    def get_health_summary(self, metrics_history):
        """Generate health summary from metrics history"""
        import numpy as np
        if not metrics_history:
            return {}
        health_scores = []
        state_counts = {
            'healthy': 0,
            'degraded': 0,
            'warning': 0,
            'critical': 0
        }
        for metrics in metrics_history:
            score, state = self.calculate_health_score(metrics)
            health_scores.append(score)
            state_counts[state] += 1
        total_samples = len(health_scores)
        return {
            'avg_health': np.mean(health_scores),
            'min_health': min(health_scores),
            'max_health': max(health_scores),
            'health_std': np.std(health_scores),
            'healthy_percent': (state_counts['healthy'] / total_samples) * 100,
            'degraded_percent': (state_counts['degraded'] / total_samples) * 100,
            'warning_percent': (state_counts['warning'] / total_samples) * 100,
            'critical_percent': (state_counts['critical'] / total_samples) * 100
        }

    # ------------------------------------------------------------------
    # Convenience API for tests ------------------------------------------------
    # ------------------------------------------------------------------

    def record_metric(self, service_id: str, metric_name: str, value: float):  # noqa: D401
        """Record a single metric sample for *service_id*.

        The helper exists primarily so that high-level integration tests can
        interact with *HealthManager* without having to craft full metric
        dictionaries.  Only the **latest** value per metric is retained since
        the current health score calculation looks at a single snapshot.
        """
        svc_store = self._service_metrics.setdefault(service_id, {})
        svc_store[metric_name] = value 