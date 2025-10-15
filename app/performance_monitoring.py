import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import sys
from pathlib import Path
import smtplib
try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError:
    MIMEText = None
    MIMEMultipart = None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logger_manager import LoggerManager
from cache_manager import CacheManager


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"


@dataclass
class AlertRule:
    name: str
    metric_type: MetricType
    threshold: float
    comparison: str  # "gt", "lt", "eq", "ne"
    severity: AlertSeverity
    window_minutes: int = 5
    min_samples: int = 3
    enabled: bool = True
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["log"]


@dataclass
class Alert:
    timestamp: datetime
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MonitoringConfig:
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = LoggerManager().logger
        self.cache_manager = CacheManager()
        
        self.metrics_history: Dict[str, List[MetricSnapshot]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        self.alert_rules: Dict[str, AlertRule] = {}
        self._setup_default_alert_rules()
        
        self.monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        self.prediction_times: List[float] = []
        self.error_counts = 0
        self.total_predictions = 0
        self.last_data_update = datetime.now()
        
        self.logger.info("Performance monitor initialized")
    
    def _setup_default_alert_rules(self):
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        self.prediction_times.append(latency_ms)
        self.total_predictions += 1
        
        if error:
            self.error_counts += 1
        
        if accuracy is not None:
            self.add_metric(MetricType.ACCURACY, accuracy)
        
        self.add_metric(MetricType.LATENCY, latency_ms)
    
    def add_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_type != metric_type:
                continue
            
            key = metric_type.value
            if key not in self.metrics_history:
                continue
            
            recent_metrics = [
                m for m in self.metrics_history[key]
                if m.timestamp > datetime.now() - timedelta(minutes=rule.window_minutes)
            ]
            
            if len(recent_metrics) < rule.min_samples:
                continue
            
            values = [m.value for m in recent_metrics]
            avg_value = np.mean(values)
            
            should_alert = False
            if rule.comparison == "gt" and avg_value > rule.threshold:
                should_alert = True
            elif rule.comparison == "lt" and avg_value < rule.threshold:
                should_alert = True
            elif rule.comparison == "eq" and abs(avg_value - rule.threshold) < 1e-6:
                should_alert = True
            elif rule.comparison == "ne" and abs(avg_value - rule.threshold) > 1e-6:
                should_alert = True
            
            if should_alert:
                self._trigger_alert(rule, avg_value)
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        for channel in rule.notification_channels:
            try:
                if channel == "log":
                    self.logger.error(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
                elif channel == "email" and self.config.enable_email_alerts:
                    self._send_email_alert(alert)
                elif channel == "webhook" and self.config.enable_webhook_alerts:
                    self._send_webhook_alert(alert)
            except Exception as e:
                self.logger.error(f"Error sending alert notification via {channel}: {e}")
    
    def _send_email_alert(self, alert: Alert):
        Alert: {alert.rule_name}
        Severity: {alert.severity.value}
        Metric: {alert.metric_type.value}
        Current Value: {alert.current_value:.4f}
        Threshold: {alert.threshold:.4f}
        Time: {alert.timestamp.isoformat()}
        
        Message: {alert.message}
        import requests
        
        payload = {
            "alert": asdict(alert),
            "timestamp": alert.timestamp.isoformat()
        }
        
        for webhook_url in self.config.webhook_urls:
            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
            except Exception as e:
                self.logger.error(f"Error sending webhook to {webhook_url}: {e}")
    
    def resolve_alert(self, rule_name: str):
        now = datetime.now()
        
        recent_times = [t for t in self.prediction_times if t > 0]
        avg_latency = np.mean(recent_times) if recent_times else 0
        
        error_rate = self.error_counts / max(self.total_predictions, 1)
        
        data_age_hours = (now - self.last_data_update).total_seconds() / 3600
        
        throughput = len([t for t in self.prediction_times if t > 0]) / max(1, 
            (now - getattr(self, '_start_time', now)).total_seconds() / 60)
        
        health_score = self._calculate_health_score(avg_latency, error_rate, data_age_hours, throughput)
        
        return {
            "overall_health_score": health_score,
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy",
            "metrics": {
                "avg_latency_ms": avg_latency,
                "error_rate": error_rate,
                "throughput_per_minute": throughput,
                "data_age_hours": data_age_hours
            },
            "active_alerts": len(self.active_alerts),
            "total_predictions": self.total_predictions,
            "uptime_hours": (now - getattr(self, '_start_time', now)).total_seconds() / 3600
        }
    
    def _calculate_health_score(self, latency: float, error_rate: float, 
                               data_age: float, throughput: float) -> float:
        cutoff = datetime.now() - timedelta(hours=hours)
        
        summary = {}
        for metric_type, snapshots in self.metrics_history.items():
            recent_snapshots = [s for s in snapshots if s.timestamp > cutoff]
            
            if recent_snapshots:
                values = [s.value for s in recent_snapshots]
                summary[metric_type] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
        
        return summary
    
    def _cleanup_old_metrics(self):
        self._start_time = datetime.now()
        
        while not self._shutdown_event.is_set():
            try:
                self._collect_system_metrics()
                
                self._check_data_quality()
                
                health = self.get_system_health()
                self.add_metric(MetricType.SYSTEM_HEALTH, health["overall_health_score"])
                
                self.cache_manager.cache_processed_data(
                    "monitoring_status",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "health": health,
                        "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
                        "metrics_summary": self.get_metrics_summary(1)  # Last hour
                    },
                    ttl=self.config.metrics_collection_interval * 2
                )
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _collect_system_metrics(self):
        try:
            
            data_quality_score = 0.9  # Placeholder
            self.add_metric(MetricType.DATA_QUALITY, data_quality_score)
            
        except Exception as e:
            self.logger.error(f"Error checking data quality: {e}")


performance_monitor = PerformanceMonitor(MonitoringConfig())
