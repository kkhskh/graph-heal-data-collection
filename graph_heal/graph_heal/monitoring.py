    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            logger.info("Stopped service monitoring")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            for service in self.services:
                try:
                    # Check health
                    health_url = f"{service['url']}{service['health_endpoint']}"
                    health_response = requests.get(health_url, timeout=1)
                    is_healthy = health_response.status_code == 200

                    # Update health status
                    current_time = datetime.now()
                    self.service_status[service["id"]] = {
                        "name": service["name"],
                        "health": "healthy" if is_healthy else "unhealthy",
                        "last_check": current_time.isoformat()
                    }

                    # Update availability history
                    self.availability_history[service["id"]].append(is_healthy)
                    # Keep last hour of history (assuming 5s interval)
                    max_history = 720  # 3600s / 5s = 720 samples
                    if len(self.availability_history[service["id"]]) > max_history:
                        self.availability_history[service["id"]] = self.availability_history[service["id"]][-max_history:]

                    # Calculate availability percentage
                    history = self.availability_history[service["id"]]
                    availability = (sum(history) / len(history)) * 100 if history else 0
                    self.service_status[service["id"]]["availability"] = availability

                    # Get metrics if healthy
                    if is_healthy:
                        metrics_url = f"{service['url']}{service['metrics_endpoint']}"
                        metrics_response = requests.get(metrics_url, timeout=1)
                        if metrics_response.status_code == 200:
                            metrics = metrics_response.json()
                            # Add timestamp to metrics
                            metrics["timestamp"] = current_time.isoformat()
                            self.service_metrics[service["id"]] = metrics

                    self.last_check[service["id"]] = current_time.isoformat()

                except requests.RequestException as e:
                    logger.warning(f"Failed to check {service['name']}: {e}")
                    current_time = datetime.now()
                    self.service_status[service["id"]] = {
                        "name": service["name"],
                        "health": "unhealthy",
                        "last_check": current_time.isoformat(),
                        "error": str(e)
                    }

                    # Update availability history
                    self.availability_history[service["id"]].append(False)
                    if len(self.availability_history[service["id"]]) > 720:
                        self.availability_history[service["id"]] = self.availability_history[service["id"]][-720:]

                    # Calculate availability percentage
                    history = self.availability_history[service["id"]]
                    availability = (sum(history) / len(history)) * 100 if history else 0
                    self.service_status[service["id"]]["availability"] = availability

            # Sleep until next check
            time.sleep(self.poll_interval)

    def stop_updating(self):
        """Stop the graph update thread."""
        if self.updater_thread and self.updater_thread.is_alive():
            self.stop_event.set()
            self.updater_thread.join()
            logger.info("Stopped graph updating") 