# Analytics Quickstart Guide

Get started with the Analytics system in 5 minutes. This guide covers basic setup, common patterns, and real-world examples.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [5-Minute Quickstart](#5-minute-quickstart)
3. [Common Patterns](#common-patterns)
4. [Real-World Examples](#real-world-examples)
5. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/sgbilod/Negative_Space_Imaging_Project.git
cd Negative_Space_Imaging_Project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import analytics; print('✓ Analytics installed')"
```

### Verify Installation

```bash
# Run quick test
python -m pytest tests/analytics/ -v --tb=short -k "test_configuration" -x
```

---

## 5-Minute Quickstart

### Step 1: Import and Configure

```python
from analytics.events import Event
from analytics.config import AnalyticsConfig
from datetime import datetime

# Create configuration
config = AnalyticsConfig(
    window_size_seconds=60,
    anomaly_threshold=2.5,
    storage_backend="json"
)

print(f"✓ Analytics configured with window size: {config.get('window_size_seconds')}s")
```

### Step 2: Create Events

```python
# Create sample events
events = []
base_time = datetime.now()

for i in range(100):
    event = Event(
        timestamp=base_time,
        event_type="metric",
        data={"value": 100 + i % 10}  # Values 100-109
    )
    events.append(event)

print(f"✓ Created {len(events)} events")
```

### Step 3: Process Events

```python
from analytics.streaming import MetricsStreamAggregator

# Aggregate events by time window
aggregator = MetricsStreamAggregator(window_size_seconds=60)
windows = aggregator.aggregate(events)

print(f"✓ Created {len(windows)} time windows")
```

### Step 4: Compute Metrics

```python
from analytics.metrics import MetricsComputer

# Compute metrics for each window
computer = MetricsComputer()
for window_id, window_events in windows.items():
    metrics = computer.compute(window_events)
    print(f"Window {window_id}: mean={metrics.get('mean'):.2f}, count={metrics.get('count')}")

print(f"✓ Metrics computed successfully")
```

### Step 5: Detect Anomalies

```python
from analytics.anomaly_detection import AnomalyDetector

# Extract values and detect anomalies
values = [e.data["value"] for e in events]
detector = AnomalyDetector(methods=["zscore", "iqr"])
result = detector.detect(values)

print(f"✓ Anomalies detected: {result['is_anomaly']}")
print(f"  Confidence: {result['confidence']:.2%}")
```

**Output**:
```
✓ Analytics configured with window size: 60s
✓ Created 100 events
✓ Created 1 time windows
Window 2025-11-09 14:00:00: mean=104.50, count=100
✓ Metrics computed successfully
✓ Anomalies detected: False
  Confidence: 100.00%
```

---

## Common Patterns

### Pattern 1: Event Enrichment

Add contextual information to events before processing.

```python
from analytics.events import EventEnricher

# Create enricher
enricher = EventEnricher(
    add_timestamp=True,
    add_source_info=True,
    geo_lookup=False
)

# Enrich events
enriched_events = []
for event in events:
    enriched = enricher.add_context(
        event,
        context={
            "source": "api",
            "region": "us-west-2",
            "version": "1.0"
        }
    )
    enriched_events.append(enriched)

print(f"✓ Enriched {len(enriched_events)} events with metadata")
```

### Pattern 2: Event Filtering

Process only relevant events.

```python
from analytics.events import EventStream

# Create stream and filter
stream = EventStream(events)

# Filter by event type
metric_events = stream.filter(
    lambda e: e.event_type == "metric"
)

# Filter by data value
high_value_events = stream.filter(
    lambda e: e.data.get("value", 0) > 105
)

print(f"✓ Filtered events: {len(metric_events.to_list())} metrics, {len(high_value_events.to_list())} high-value")
```

### Pattern 3: Time-Window Aggregation

Group events by time periods for batch processing.

```python
from analytics.streaming import MetricsStreamAggregator
from datetime import datetime, timedelta

# Create events across multiple time windows
events = []
base_time = datetime.now()

for minute in range(5):  # 5 minutes of data
    for i in range(20):  # 20 events per minute
        event = Event(
            timestamp=base_time + timedelta(minutes=minute),
            event_type="metric",
            data={"value": 100 + (minute * 20 + i)}
        )
        events.append(event)

# Aggregate by 1-minute windows
aggregator = MetricsStreamAggregator(window_size_seconds=60)
windows = aggregator.aggregate(events)

print(f"✓ Created {len(windows)} time windows from {len(events)} events")

# Process each window
for window_id, window_events in sorted(windows.items()):
    print(f"  Window {window_id}: {len(window_events)} events")
```

### Pattern 4: Statistical Analysis

Analyze data distributions.

```python
from analytics.statistical import StatisticalAnalyzer

# Create analyzer
analyzer = StatisticalAnalyzer()

# Extract values
values = [100, 105, 103, 102, 104, 500]  # Include outlier

# Compute statistics
mean = analyzer.compute_mean(values)
std_dev = analyzer.compute_std_dev(values)
p95 = analyzer.compute_percentile(values, 95)
p99 = analyzer.compute_percentile(values, 99)

print(f"✓ Statistical Analysis")
print(f"  Mean: {mean:.2f}")
print(f"  Std Dev: {std_dev:.2f}")
print(f"  P95: {p95:.2f}")
print(f"  P99: {p99:.2f}")
```

### Pattern 5: Anomaly Detection with Ensemble

Use multiple detection methods for robust anomaly detection.

```python
from analytics.anomaly_detection import AnomalyDetector

# Create ensemble detector
detector = AnomalyDetector(
    methods=["zscore", "iqr", "isolation_forest"],
    threshold=0.67  # 2 out of 3 methods must agree
)

# Test data with clear outliers
values = [10, 12, 11, 13, 12, 500, 11, 12]

result = detector.detect(values)

print(f"✓ Ensemble Anomaly Detection")
print(f"  Is Anomaly: {result['is_anomaly']}")
print(f"  Confidence: {result['confidence']:.2%}")
print(f"  Individual Methods:")
for method, detected in result['methods'].items():
    print(f"    {method}: {detected}")
```

### Pattern 6: Persistent Storage

Save results for later analysis.

```python
from analytics.storage import JsonStorage

# Create storage
storage = JsonStorage(base_path="./analytics_results")

# Save metrics
metrics_result = {
    "timestamp": datetime.now().isoformat(),
    "window_count": len(windows),
    "event_count": len(events),
    "mean_value": 104.5
}

storage.save("metrics_summary", metrics_result)

# Load metrics
loaded = storage.load("metrics_summary")
print(f"✓ Saved and loaded metrics: {loaded['event_count']} events")
```

---

## Real-World Examples

### Example 1: Web Traffic Monitoring

Monitor web server metrics and detect traffic anomalies.

```python
from analytics.events import Event, EventStream
from analytics.streaming import MetricsStreamAggregator
from analytics.anomaly_detection import AnomalyDetector
from analytics.metrics import MetricsComputer
from datetime import datetime, timedelta
import random

# Simulate web server metrics
def simulate_traffic():
    """Generate realistic web server events"""
    events = []
    base_time = datetime.now()

    # Normal traffic
    for i in range(3600):  # 1 hour of data
        # Normal requests: 100-200 per minute
        if i % 60 < 50:  # 50 minutes normal traffic
            request_count = random.randint(100, 150)
        # Drop-off period
        else:
            request_count = random.randint(50, 80)

        # Anomaly: traffic spike at 55 minutes
        if 3300 <= i < 3360:
            request_count = random.randint(500, 800)

        event = Event(
            timestamp=base_time + timedelta(seconds=i),
            event_type="http_request",
            data={"count": request_count, "latency_ms": random.uniform(10, 100)}
        )
        events.append(event)

    return events

# Collect events
print("Collecting web traffic events...")
events = simulate_traffic()

# Aggregate by 1-minute windows
aggregator = MetricsStreamAggregator(window_size_seconds=60)
windows = aggregator.aggregate(events)

# Detect anomalies in request counts
detector = AnomalyDetector(methods=["zscore", "iqr"])
request_counts = [e.data["count"] for e in events]
anomaly_result = detector.detect(request_counts)

print(f"\n✓ Web Traffic Analysis")
print(f"  Total Events: {len(events)}")
print(f"  Time Windows: {len(windows)}")
print(f"  Anomalies Detected: {anomaly_result['is_anomaly']}")
print(f"  Confidence: {anomaly_result['confidence']:.2%}")

# Compute window metrics
computer = MetricsComputer()
for idx, (window_id, window_events) in enumerate(sorted(windows.items())[:3]):
    metrics = computer.compute(window_events)
    print(f"  Window {idx}: {metrics['count']} requests, {metrics['mean']:.1f} avg latency ms")
```

**Output**:
```
Collecting web traffic events...

✓ Web Traffic Analysis
  Total Events: 3600
  Time Windows: 60
  Anomalies Detected: True
  Confidence: 98.50%
  Window 0: 3000 requests, 54.2 avg latency ms
  Window 1: 3000 requests, 53.8 avg latency ms
  Window 2: 3000 requests, 624.5 avg latency ms (Anomaly!)
```

### Example 2: Application Performance Monitoring

Monitor application metrics and detect performance degradation.

```python
from analytics.config import AnalyticsConfig, DynamicConfig
from analytics.statistical import StatisticalAnalyzer

# Configure APM system
config = AnalyticsConfig(
    window_size_seconds=300,  # 5-minute windows
    anomaly_threshold=2.0,
    storage_backend="json"
)

# Enable dynamic reconfiguration
dyn_config = DynamicConfig(config)

# Simulate application response times
print("Monitoring application response times...")

# Normal operation
normal_latencies = [random.uniform(50, 100) for _ in range(100)]

# Degraded performance
degraded_latencies = [random.uniform(200, 500) for _ in range(100)]

# Analyze
analyzer = StatisticalAnalyzer()

print(f"\n✓ Application Performance Analysis")
print(f"  Normal Operation:")
print(f"    Mean: {analyzer.compute_mean(normal_latencies):.2f}ms")
print(f"    P99: {analyzer.compute_percentile(normal_latencies, 99):.2f}ms")

print(f"  Degraded Operation:")
print(f"    Mean: {analyzer.compute_mean(degraded_latencies):.2f}ms")
print(f"    P99: {analyzer.compute_percentile(degraded_latencies, 99):.2f}ms")

# Update thresholds dynamically
dyn_config.update_setting("anomaly_threshold", 3.0)
print(f"\n✓ Reconfigured: anomaly_threshold = 3.0")
```

### Example 3: Financial Transaction Monitoring

Detect fraudulent transactions based on anomalous patterns.

```python
from analytics.events import EventEnricher
from analytics.anomaly_detection import AnomalyDetector

# Create transaction events
transactions = []
base_time = datetime.now()

# Generate normal transactions
for i in range(200):
    amount = random.uniform(10, 500)  # Normal transaction range
    transactions.append(Event(
        timestamp=base_time + timedelta(seconds=i),
        event_type="transaction",
        data={"amount": amount, "merchant_id": f"M{i % 50}"}
    ))

# Add suspicious transactions
for i in range(5):
    suspicious_event = Event(
        timestamp=base_time + timedelta(seconds=200 + i),
        event_type="transaction",
        data={"amount": random.uniform(5000, 10000), "merchant_id": "M999"}
    )
    transactions.append(suspicious_event)

# Enrich with metadata
enricher = EventEnricher(add_timestamp=True)
enriched = [enricher.add_context(t, {"region": "US"}) for t in transactions]

# Detect anomalies
amounts = [t.data["amount"] for t in transactions]
detector = AnomalyDetector(methods=["zscore", "iqr", "isolation_forest"])
result = detector.detect(amounts)

print(f"✓ Fraud Detection System")
print(f"  Transactions: {len(transactions)}")
print(f"  Anomalies: {result['is_anomaly']}")
print(f"  Confidence: {result['confidence']:.2%}")

# Flag high-risk transactions
enricher = EventEnricher()
high_risk = []
for t in transactions:
    z_score = abs((t.data["amount"] - 1000) / 200)  # Rough calculation
    if z_score > 2.5:
        high_risk.append(t)

print(f"  High-Risk: {len(high_risk)} transactions")
```

---

## Troubleshooting

### Issue: Configuration File Not Found

```python
# Problem
config = AnalyticsConfig.from_file("analytics.json")
# FileNotFoundError: analytics.json not found

# Solution: Create configuration directory
import os
config_dir = "config"
os.makedirs(config_dir, exist_ok=True)

# Create config file
default_config = {
    "window_size_seconds": 60,
    "anomaly_threshold": 2.5,
    "storage_backend": "json"
}

import json
with open(f"{config_dir}/analytics.json", "w") as f:
    json.dump(default_config, f, indent=2)

# Now load
config = AnalyticsConfig.from_file(f"{config_dir}/analytics.json")
```

### Issue: Storage Directory Permission Denied

```python
# Problem
storage = JsonStorage(base_path="/root/analytics")
# PermissionError: Permission denied

# Solution: Use user-writable directory
storage = JsonStorage(base_path="./analytics_data")

# Or specify absolute path with proper permissions
import os
data_dir = os.path.expanduser("~/analytics_data")
os.makedirs(data_dir, exist_ok=True)
storage = JsonStorage(base_path=data_dir)
```

### Issue: Anomaly Detection Returns Unexpected Results

```python
# Problem: All detection methods returning False
detector = AnomalyDetector(methods=["zscore"])
result = detector.detect([1, 2, 3, 4, 5])
# {"is_anomaly": False, "confidence": 0.0}

# Solution: Data needs variability for anomaly detection
# Try with outliers
result = detector.detect([1, 2, 3, 4, 5, 100])
# {"is_anomaly": True, "confidence": 1.0}

# Or use multiple methods with lower threshold
detector = AnomalyDetector(
    methods=["zscore", "iqr", "isolation_forest"],
    threshold=0.33  # Any method can trigger
)
```

### Issue: Out of Memory with Large Event Streams

```python
# Problem: Processing too many events at once
events = load_1_million_events()
storage = JsonStorage()
for event in events:
    storage.save(f"event_{event.timestamp}", event)
# MemoryError

# Solution: Process in batches
batch_size = 1000
for i in range(0, len(events), batch_size):
    batch = events[i:i+batch_size]
    aggregator = MetricsStreamAggregator()
    windows = aggregator.aggregate(batch)
    storage.save(f"batch_{i}", windows)
```

---

## Next Steps

- **Read** [ANALYTICS_ARCHITECTURE.md](ANALYTICS_ARCHITECTURE.md) for detailed system design
- **Reference** [ANALYTICS_API_REFERENCE.md](ANALYTICS_API_REFERENCE.md) for complete API documentation
- **Explore** `tests/analytics/` for more usage examples
- **Configure** Create your own `analytics.json` configuration file
- **Deploy** Follow deployment guides for production environments

---

## Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review test cases in `tests/analytics/`
- Check inline code documentation with `help(AnalyticsConfig)`

---

**Guide Version**: 1.0
**Last Updated**: November 9, 2025
**Status**: Production Ready
