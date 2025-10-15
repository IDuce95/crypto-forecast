"""Performance monitoring module - Fixed version"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np

# Make psutil optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class MonitoringConfig:
    enable_monitoring: bool = True
    metrics_window_size: int = 1000
    alert_check_interval: int = 60  # seconds
    resource_check_interval: int = 30  # seconds
    performance_log_interval: int = 300  # seconds
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "error_rate": 5.0,
        "response_time": 2000.0  # milliseconds
    })


class PerformanceMonitor:
    def __init__(self, config: MonitoringConfig):
        """Initialize performance monitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = defaultdict(lambda: deque(maxlen=config.metrics_window_size))
        self.alerts = []
        self.resource_metrics = {}
        self.lock = threading.RLock()
        
        # Start monitoring threads if enabled
        if config.enable_monitoring:
            self._start_monitoring_threads()
        
        self.logger.info("PerformanceMonitor initialized")
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        # Resource monitoring thread
        self.resource_thread = threading.Thread(
            target=self._resource_monitoring_loop,
            daemon=True
        )
        self.resource_thread.start()
        
        # Alert checking thread
        if self.config.enable_alerts:
            self.alert_thread = threading.Thread(
                target=self._alert_checking_loop,
                daemon=True
            )
            self.alert_thread.start()
    
    def _resource_monitoring_loop(self):
        """Background thread for resource monitoring"""
        while True:
            try:
                if PSUTIL_AVAILABLE:
                    # Get system resources
                    self.resource_metrics = {
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('/').percent,
                        "timestamp": datetime.now()
                    }
                else:
                    # Mock data when psutil is not available
                    self.resource_metrics = {
                        "cpu_percent": 50.0,
                        "memory_percent": 60.0,
                        "disk_percent": 40.0,
                        "timestamp": datetime.now()
                    }
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
            
            time.sleep(self.config.resource_check_interval)
    
    def _alert_checking_loop(self):
        """Background thread for checking alerts"""
        while True:
            try:
                # Check for alerts
                self._check_alerts()
            except Exception as e:
                self.logger.error(f"Error in alert checking: {e}")
            
            time.sleep(self.config.alert_check_interval)
    
    def _check_alerts(self):
        """Check for alert conditions"""
        with self.lock:
            # Check CPU usage
            if self.resource_metrics.get("cpu_percent", 0) > self.config.alert_thresholds["cpu_usage"]:
                self.alerts.append({
                    "type": "cpu_high",
                    "message": f"CPU usage is {self.resource_metrics['cpu_percent']:.1f}%",
                    "timestamp": datetime.now()
                })
            
            # Check memory usage
            if self.resource_metrics.get("memory_percent", 0) > self.config.alert_thresholds["memory_usage"]:
                self.alerts.append({
                    "type": "memory_high",
                    "message": f"Memory usage is {self.resource_metrics['memory_percent']:.1f}%",
                    "timestamp": datetime.now()
                })
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        with self.lock:
            self.metrics[metric_name].append({
                "value": value,
                "timestamp": time.time()
            })
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Metrics summary dictionary
        """
        with self.lock:
            summary = {}
            cutoff_time = time.time() - (hours * 3600)
            
            for metric_name, values in self.metrics.items():
                recent_values = [
                    v["value"] for v in values
                    if v["timestamp"] > cutoff_time
                ]
                
                if recent_values:
                    summary[metric_name] = {
                        "count": len(recent_values),
                        "mean": np.mean(recent_values),
                        "min": np.min(recent_values),
                        "max": np.max(recent_values),
                        "std": np.std(recent_values)
                    }
            
            # Add resource metrics
            summary["resources"] = self.resource_metrics
            
            return summary
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        with self.lock:
            return self.alerts[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts"""
        with self.lock:
            self.alerts = []
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage
        
        Returns:
            Resource usage dictionary
        """
        return self.resource_metrics.copy()
