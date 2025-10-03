
import psutil
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}

    def record_metric(self, key, value):
        self.metrics[key] = value
        logger.info(f"Recorded {key}: {value}")

    def get_system_stats(self):
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }

    def monitor_image_processing(self, job_id):
        stats = self.get_system_stats()
        self.record_metric(f'image_job_{job_id}', stats)
        # ...expand with alerting and reporting

    def report(self):
        return self.metrics

# ...expand with additional monitoring features as needed
